"""Re-voice a rendered Veo clip: keep its ambience/SFX but replace the (Veo-invented) spoken
voice with a consistent TTS voice generated from the shot's voice_over.

Flow (per clip):
    1. extract the clip's audio
    2. demucs two-stems → `vocals` (Veo's invented speech) + `no_vocals` (background = music + ambience)
    3. STT ON THE VOCALS TRACK — but only to find WHEN Veo starts speaking (segment start time);
       the transcribed TEXT is discarded, since the actual words come from voice_over (already
       known, authored by the script) not from what Veo happened to say.
    4. TTS(voice_over) via app.services.tts → trim silence → fit WITHIN the remaining clip time
       (from the detected start to the end of the clip)
    5. mix the TTS over the background AT the detected start offset
    6. mux the new audio back onto the original video (video copied, audio replaced)

demucs/torch/faster-whisper are lazy-imported (heavy) so importing this module stays cheap.
ffmpeg via subprocess, same as the rest of the app.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.core import observability as obs

FFMPEG = "ffmpeg"

# Cached across calls so repeated /revoice requests in the same process don't reload the model
# (a few seconds each) — same lazy-singleton shape as other credential/client caches in this app.
_whisper_model = None

# Trim ONLY leading + trailing silence (Gemini TTS pads both ends) — keep mid-sentence pauses.
_SILENCE_TRIM = (
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.03,"
    "areverse,"
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.03,"
    "areverse"
)


def _run(cmd: list[str]) -> None:
    # Capture output but surface it on failure — a bare CalledProcessError only says "exit status 1"
    # and hides the real cause (e.g. a missing optional dependency).
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip()[-800:]
        raise RuntimeError(f"{cmd[0]} failed (exit {p.returncode}): {tail}")


def _duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def _atempo_chain(ratio: float) -> str:
    """ffmpeg atempo chain for any ratio (each atempo is limited to 0.5–2.0)."""
    filters, remaining = [], ratio
    while remaining > 2.0:
        filters.append("atempo=2.0"); remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5"); remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def _extract_audio(video: Path, wav: Path) -> Path:
    _run([FFMPEG, "-y", "-i", video, "-vn", "-ac", "2", "-ar", "44100", wav])
    return wav


def _demucs_separate(audio: Path, work: Path) -> tuple[Path, Path]:
    """demucs two-stems → (vocals, background). `vocals` is Veo's own invented speech — used ONLY
    to detect WHEN it starts speaking (see _detect_speech_start), never mixed into the output.
    `background` (music + ambience) is what the new TTS gets mixed over."""
    import torch   # lazy — pulls ~2GB torch only when re-voicing runs

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with obs.span("revoice.demucs_separate", metadata={"device": device}):
        _run([sys.executable, "-m", "demucs.separate", "--two-stems=vocals",
              "-n", "htdemucs", "-d", device, "-o", str(work / "demucs"), str(audio)])
    stem_dir = work / "demucs" / "htdemucs" / audio.stem
    vocals, bg = stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"
    if not bg.exists():
        raise RuntimeError(f"demucs produced no background track ({bg})")
    return vocals, bg


def _detect_speech_span(vocals: Path, clip_sec: float) -> tuple[float, float]:
    """Find the WINDOW where Veo's own (invented) voice is speaking in `vocals` — via STT, but
    ONLY for the timestamps (first segment start → last segment end). The recognized TEXT is
    thrown away: the real words come from voice_over, not from whatever Veo happened to say.
    The new voice is then fitted into exactly this window so it lands while the mouth moves.
    Returns (0.0, clip_sec) — the whole clip, = old behavior — if detection fails or finds no
    speech, so a shaky detection never breaks the mix."""
    global _whisper_model
    try:
        with obs.span("revoice.whisper_speech_span", metadata={"clip_sec": round(clip_sec, 2)}) as sp:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                # ponytail: "small"/cpu/int8 — bump to a bigger model if timing is consistently off
                _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, _info = _whisper_model.transcribe(str(vocals), vad_filter=True)
            start, end = None, None
            for seg in segments:      # must drain the generator to reach the LAST segment's end
                if start is None:
                    start = float(seg.start)
                end = float(seg.end)
            if start is None or end is None:
                sp.update(output={"speech": False})
                return 0.0, clip_sec
            start = max(0.0, min(start, clip_sec - 0.15))
            end = min(clip_sec, max(end, start + 0.15))   # span at least 0.15s, within the clip
            sp.update(output={"start": round(start, 2), "end": round(end, 2)})
            return start, end
    except Exception as e:  # noqa: BLE001 — timing detection is best-effort, never fatal
        logger.warning(f"revoice: speech-span detection failed, using whole clip: {e}")
        return 0.0, clip_sec


def _merge_close(segs: list[tuple[float, float]], gap: float = 0.4) -> list[list[float]]:
    """Merge speech segments separated by less than `gap` (whisper over-splits within one utterance)."""
    if not segs:
        return []
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] < gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def _detect_speech_segments(vocals: Path, clip_sec: float, n: int) -> list[tuple[float, float]]:
    """Return exactly `n` speech windows (in order) — one per shot's voice_over — from Veo's own
    voice track. whisper VAD → merge close blocks → reconcile to n: len==n use as-is; len>n merge
    the closest-adjacent blocks until n remain (keeps the n-1 biggest gaps = shot boundaries);
    len<n (speech ran together) or any failure → n EVEN windows across the clip (safe fallback)."""
    def _even() -> list[tuple[float, float]]:
        w = clip_sec / n
        return [(round(i * w, 3), round((i + 1) * w, 3)) for i in range(n)]

    if n <= 1:
        return [_detect_speech_span(vocals, clip_sec)]
    global _whisper_model
    try:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _info = _whisper_model.transcribe(str(vocals), vad_filter=True)
        blocks = _merge_close(sorted((float(s.start), float(s.end)) for s in segments))
        if not blocks or len(blocks) < n:
            return _even()
        while len(blocks) > n:      # merge the two adjacent blocks with the smallest gap between them
            i = min(range(len(blocks) - 1), key=lambda k: blocks[k + 1][0] - blocks[k][1])
            blocks[i] = [blocks[i][0], blocks[i + 1][1]]
            del blocks[i + 1]
        out = []
        for s, e in blocks:         # clamp within the clip, each window ≥ 0.15s
            s = max(0.0, min(s, clip_sec - 0.15))
            out.append((s, min(clip_sec, max(e, s + 0.15))))
        return out
    except Exception as e:  # noqa: BLE001 — timing is best-effort, never fatal
        logger.warning(f"revoice: multi-segment detection failed → even split: {e}")
        return _even()


def _trim_silence(wav: Path, out: Path) -> Path:
    _run([FFMPEG, "-y", "-i", str(wav), "-af", _SILENCE_TRIM, str(out)])
    try:
        if _duration(out) < 0.05:      # trim ate everything (too quiet) → keep original
            shutil.copy(wav, out)
    except Exception:
        shutil.copy(wav, out)
    return out


def _tts_fit(voice_over: str, available_sec: float, work: Path, *, voice, style) -> Path:
    """TTS(voice_over) → trim silence → keep it WITHIN `available_sec` (the time remaining from
    the detected speech-start to the end of the clip): compress only if longer than that (atempo);
    if shorter, leave it natural (background plays under the tail)."""
    from app.services.tts import synthesize

    raw = synthesize(voice_over, out=work / "tts_raw.wav", voice=voice, style=style)
    trimmed = _trim_silence(raw, work / "tts_trim.wav")
    dur = _duration(trimmed)
    if available_sec > 0 and dur > available_sec + 0.02:      # too long → speed up to fit exactly
        fitted = work / "tts_fit.wav"
        _run([FFMPEG, "-y", "-i", str(trimmed), "-filter:a", _atempo_chain(dur / available_sec), str(fitted)])
        logger.info(f"revoice: TTS {dur:.2f}s > available {available_sec:.2f}s → compressed to fit")
        return fitted
    logger.info(f"revoice: TTS {dur:.2f}s ≤ available {available_sec:.2f}s → kept natural")
    return trimmed


def _mix_and_mux(video: Path, background: Path, tts_wav: Path, out: Path, *, offset_sec: float = 0.0) -> Path:
    """Mix TTS (delayed by offset_sec, to land on Veo's own detected speech-start) over the
    background, then mux onto the video (video copied, audio replaced). amix duration=first
    anchors the output to the background = clip length."""
    delay_ms = max(0, round(offset_sec * 1000))
    _run([FFMPEG, "-y", "-i", str(video), "-i", str(background), "-i", str(tts_wav),
          "-filter_complex",
          f"[2:a]adelay={delay_ms}|{delay_ms}[voiced];[1:a][voiced]amix=inputs=2:duration=first:normalize=0[a]",
          # 48kHz stereo AAC = Veo's own clip format, so the merged film's clips stay uniform (else a
          # -c copy concat drops audio on clips whose rate differs from the first).
          "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(out)])
    return out


def _mix_multi_over_bg(video: Path, background: Path, placed: list[tuple[float, Path]], out: Path) -> Path:
    """Mix N TTS clips (each delayed to its own window start) over the background, then mux onto the
    video. amix duration=first anchors the output to the background = clip length."""
    inputs: list[str] = ["-i", str(video), "-i", str(background)]   # 0=video, 1=background
    fc: list[str] = []
    labels = ["[1:a]"]
    for i, (off, wav) in enumerate(placed):
        inputs += ["-i", str(wav)]
        ms = max(0, round(off * 1000))
        fc.append(f"[{i + 2}:a]adelay={ms}|{ms}[v{i}]")
        labels.append(f"[v{i}]")
    fc.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[a]")
    _run([FFMPEG, "-y", *inputs, "-filter_complex", ";".join(fc),
          "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(out)])
    return out


def revoice_clip_multi(video: Path, voice_overs: list[str], *, voice: str | None = None,
                       style: str | None = None, out: Path | None = None) -> tuple[Path, list[tuple[float, float]]]:
    """Re-voice a GROUP clip (one mp4, several shots each with its own voice_over that Veo speaks in
    its own window): detect each shot's speech window, TTS each voice_over into its window, mix all
    over the ambience, and mux back. Keeps the clip's ambience. Falls back to revoice_clip for a
    single line. Raises if no non-empty voice_over.

    Returns (new_clip_path, spans) — `spans` is the ACTUAL [start, start+tts_duration] window each
    TTS line occupies in the output (not the whole detected window), for subtitle sync (see
    `_speech_timed_spans` in routes.py's /steps/merge)."""
    video = Path(video)
    vos = [v.strip() for v in (voice_overs or []) if (v or "").strip()]
    if not vos:
        raise ValueError("revoice_clip_multi: no non-empty voice_over")
    if len(vos) == 1:
        return revoice_clip(video, vos[0], voice=voice, style=style, out=out)
    if not video.exists():
        raise FileNotFoundError(f"clip not found: {video}")
    out = Path(out) if out else video.with_name(f"{video.stem}_voiced.mp4")

    with tempfile.TemporaryDirectory(prefix="revoice_") as tmp:
        work = Path(tmp)
        clip_sec = _duration(video)
        audio = _extract_audio(video, work / "audio.wav")
        vocals, background = _demucs_separate(audio, work)
        windows = (_detect_speech_segments(vocals, clip_sec, len(vos))
                   if settings.tts.detect_timestamp
                   else [(round(i * clip_sec / len(vos), 3), round((i + 1) * clip_sec / len(vos), 3)) for i in range(len(vos))])
        logger.info(f"revoice-multi: {len(vos)} segments → windows {[(round(s, 1), round(e, 1)) for s, e in windows]}")
        placed: list[tuple[float, Path]] = []
        spans: list[tuple[float, float]] = []
        for i, (vo, (s, e)) in enumerate(zip(vos, windows)):
            seg = work / f"seg{i}"
            seg.mkdir()
            wav = _tts_fit(vo, max(0.15, e - s), seg, voice=voice, style=style)
            placed.append((s, wav))
            spans.append((round(s, 3), round(min(clip_sec, s + _duration(wav)), 3)))
        _mix_multi_over_bg(video, background, placed, out)
    return out, spans


def revoice_clip(video: Path, voice_over: str, *, voice: str | None = None,
                 style: str | None = None, out: Path | None = None) -> tuple[Path, list[tuple[float, float]]]:
    """Replace a clip's spoken voice with TTS(voice_over), keeping its ambience, timed to land on
    Veo's own speech-onset (detected via STT — text discarded, timing only). Returns the new mp4.
    `out` defaults to a sibling `<name>_voiced.mp4`. Raises on empty voice_over / demucs failure.

    Returns (new_clip_path, spans) — see revoice_clip_multi's docstring."""
    video = Path(video)
    voice_over = (voice_over or "").strip()
    if not voice_over:
        raise ValueError("revoice_clip: voice_over is empty (nothing to speak)")
    if not video.exists():
        raise FileNotFoundError(f"clip not found: {video}")
    out = Path(out) if out else video.with_name(f"{video.stem}_voiced.mp4")

    with tempfile.TemporaryDirectory(prefix="revoice_") as tmp:
        work = Path(tmp)
        clip_sec = _duration(video)
        audio = _extract_audio(video, work / "audio.wav")
        vocals, background = _demucs_separate(audio, work)
        # Fit the new voice into the exact window where Veo is speaking [start, end] — so it lands
        # while the mouth moves, not lagging past it. Detection off → whole clip (old behavior).
        start, end = _detect_speech_span(vocals, clip_sec) if settings.tts.detect_timestamp else (0.0, clip_sec)
        available = max(0.15, end - start)
        logger.info(f"revoice: speech window [{start:.2f}s, {end:.2f}s] of {clip_sec:.2f}s clip")
        tts_wav = _tts_fit(voice_over, available, work, voice=voice, style=style)
        _mix_and_mux(video, background, tts_wav, out, offset_sec=start)
        span = (round(start, 3), round(min(clip_sec, start + _duration(tts_wav)), 3))
    return out, [span]


# ── "add" mode — the clip has NO spoken voice (or we don't want to touch it): overlay TTS while
#    KEEPING the clip's music + ambience untouched (no demucs, no STT). Also works on a fully
#    silent clip (e.g. Veo generate_audio=false). ──

def _has_audio(video: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def _extract_or_silent_audio(video: Path, wav: Path, clip_sec: float) -> Path:
    """The clip's ORIGINAL audio (music + ambience) as-is; or a silent track of clip length if the
    clip has no audio stream at all — nothing is removed either way."""
    if _has_audio(video):
        return _extract_audio(video, wav)
    _run([FFMPEG, "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
          "-t", f"{clip_sec:.3f}", str(wav)])
    return wav


def add_narration(video: Path, voice_over: str, *, voice: str | None = None,
                  style: str | None = None, out: Path | None = None) -> tuple[Path, list[tuple[float, float]]]:
    """Overlay TTS(voice_over) onto a clip that has NO spoken voice — keeping the clip's music +
    ambience fully intact (mixed, not replaced). The narration is fit to the clip length and placed
    CENTERED. No demucs / no STT (works even if demucs isn't installed). Raises on empty voice_over."""
    video = Path(video)
    voice_over = (voice_over or "").strip()
    if not voice_over:
        raise ValueError("add_narration: voice_over is empty (nothing to speak)")
    if not video.exists():
        raise FileNotFoundError(f"clip not found: {video}")
    out = Path(out) if out else video.with_name(f"{video.stem}_voiced.mp4")

    with tempfile.TemporaryDirectory(prefix="addvoice_") as tmp:
        work = Path(tmp)
        clip_sec = _duration(video)
        background = _extract_or_silent_audio(video, work / "audio.wav", clip_sec)
        tts_wav = _tts_fit(voice_over, clip_sec, work, voice=voice, style=style)
        tts_dur = min(clip_sec, _duration(tts_wav))
        start = max(0.0, (clip_sec - tts_dur) / 2)   # centered
        _mix_and_mux(video, background, tts_wav, out, offset_sec=start)
        span = (round(start, 3), round(start + tts_dur, 3))
    return out, [span]


def add_narration_multi(video: Path, voice_overs: list[str], *, voice: str | None = None,
                        style: str | None = None, out: Path | None = None) -> tuple[Path, list[tuple[float, float]]]:
    """Overlay several TTS lines onto a clip with NO spoken voice — one line per EVEN slice of the
    clip (no VAD needed, since there's no existing speech to detect), each centered in its slice,
    keeping the clip's music + ambience intact. Falls back to add_narration for a single line."""
    video = Path(video)
    vos = [v.strip() for v in (voice_overs or []) if (v or "").strip()]
    if not vos:
        raise ValueError("add_narration_multi: no non-empty voice_over")
    if len(vos) == 1:
        return add_narration(video, vos[0], voice=voice, style=style, out=out)
    if not video.exists():
        raise FileNotFoundError(f"clip not found: {video}")
    out = Path(out) if out else video.with_name(f"{video.stem}_voiced.mp4")

    with tempfile.TemporaryDirectory(prefix="addvoice_") as tmp:
        work = Path(tmp)
        clip_sec = _duration(video)
        background = _extract_or_silent_audio(video, work / "audio.wav", clip_sec)
        slice_sec = clip_sec / len(vos)
        placed: list[tuple[float, Path]] = []
        spans: list[tuple[float, float]] = []
        for i, vo in enumerate(vos):
            seg = work / f"seg{i}"
            seg.mkdir()
            wav = _tts_fit(vo, slice_sec, seg, voice=voice, style=style)
            dur = min(slice_sec, _duration(wav))
            start = i * slice_sec + max(0.0, (slice_sec - dur) / 2)
            placed.append((start, wav))
            spans.append((round(start, 3), round(start + dur, 3)))
        _mix_multi_over_bg(video, background, placed, out)
    return out, spans


def detect_speech_window(video: Path) -> tuple[float, float] | None:
    """Best-effort: find [start, end] where speech ACTUALLY occurs in `video`'s own audio track,
    via whisper VAD (no demucs — for use AFTER revoicing, when the spoken voice IS the TTS/Veo
    line, not mixed with a separate invented voice). Returns None on any failure/no-speech, so a
    caller (e.g. subtitle timing) can fall back to its own default span."""
    try:
        clip_sec = _duration(video)
        with tempfile.TemporaryDirectory(prefix="vad_") as tmp:
            wav = _extract_audio(video, Path(tmp) / "audio.wav")
            start, end = _detect_speech_span(wav, clip_sec)
        if (start, end) == (0.0, clip_sec):
            return None   # detection fell back to "whole clip" = no real signal
        return start, end
    except Exception as e:  # noqa: BLE001 — timing is best-effort, never fatal
        logger.warning(f"detect_speech_window failed for {video}: {e}")
        return None
