"""Text-to-speech for voice-over — Gemini TTS via Vertex AI.

Standalone (no ServiceContainer) so a notebook can `from app.services.tts import synthesize`
after putting the repo root on sys.path. Defaults come from settings.tts; every arg is overridable.
Uses the same Vertex AI service-account credentials (credentials.json) as the text-generation
client — no GEMINI_API_KEY needed.
Gemini TTS returns raw 16-bit mono PCM @ 24 kHz — we wrap it as a .wav.
"""
from __future__ import annotations

import base64
import hashlib
import json
import wave
from pathlib import Path

from app.core.config import ROOT_DIR, settings

_OUT_DIR = ROOT_DIR / "outputs" / "tts"


def _pcm_to_wav(pcm: bytes, path: Path, sample_rate: int) -> Path:
    """Wrap raw little-endian 16-bit mono PCM as a .wav file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return path


def _build_vertex_client():
    """Build a google-genai client backed by Vertex AI using the project's
    service-account credentials.json — same pattern as GeminiClient.__init__."""
    from google import genai
    from google.oauth2 import service_account

    creds_path = ROOT_DIR / settings.gemini.credentials_file
    if not creds_path.exists():
        raise FileNotFoundError(
            f"Gemini credentials file not found: {creds_path.resolve()}"
        )

    with open(creds_path) as f:
        creds_data = json.load(f)

    project = creds_data.get("project_id")
    if not project:
        raise ValueError("credentials file is missing project_id")

    credentials = service_account.Credentials.from_service_account_info(
        creds_data,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    location = settings.gemini.location

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials,
    )


def synthesize(text: str, *, out: str | Path | None = None, voice: str | None = None,
               style: str | None = None, model: str | None = None) -> Path:
    """Text → a .wav file via Gemini TTS on Vertex AI. Returns the saved path.

    voice/style/model default to settings.tts; `out` defaults to outputs/tts/<hash>.wav.
    Uses credentials.json (Vertex AI service account) — no GEMINI_API_KEY required.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("tts.synthesize: text is empty")
    cfg = settings.tts
    voice = voice or cfg.voice
    model = model or cfg.model
    style = cfg.style if style is None else style

    from google.genai import types
    from app.core import observability as obs

    prompt = f"{style.strip()}: {text}" if style.strip() else text
    client = _build_vertex_client()
    with obs.generation("tts.synthesize", model=model,
                        input={"voice": voice, "style": style, "chars": len(text), "text": text}) as gen:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            ),
        )
        pcm = resp.candidates[0].content.parts[0].inline_data.data
        gen.update(output={"chars": len(text)})
    if isinstance(pcm, str):           # some SDK paths hand back base64 text
        pcm = base64.b64decode(pcm)
    if not pcm:
        raise RuntimeError("Gemini TTS (Vertex AI) returned no audio")

    if out is None:
        out = _OUT_DIR / f"{hashlib.sha1((prompt + voice).encode()).hexdigest()[:16]}.wav"
    return _pcm_to_wav(pcm, Path(out), cfg.sample_rate)


def _demo() -> None:
    # ponytail: offline self-check of the PCM→WAV writer (no API). 0.1s of silence @ 24kHz.
    import tempfile
    p = _pcm_to_wav(b"\x00\x00" * 2400, Path(tempfile.mkdtemp()) / "t.wav", 24000)
    with wave.open(str(p)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == 24000 and w.getnframes() == 2400
    print("tts _pcm_to_wav OK:", p)


if __name__ == "__main__":
    _demo()
