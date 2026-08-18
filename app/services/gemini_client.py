import asyncio
import json
import math
import os
import re
import ssl
import time

import httpx

import vertexai
from google.oauth2 import service_account
from loguru import logger
from vertexai.generative_models import GenerativeModel, Part

from app.core import observability as obs
from app.core.config import ROOT_DIR, ScriptConfig, Settings
from app.core.usage import record_usage
from app.services.prompts import (
    TOOLS_GOLD_SCENE,
    TOOLS_RULE_BOTH,
    TOOLS_RULE_INGREDIENTS_ONLY,
    category_block,
    get_prompt,
)


def _extract_text(response) -> str:
    if not response.candidates:
        logger.warning("Response has no candidates")
        return ""

    candidate = response.candidates[0]
    content = candidate.content
    parts = content.parts if content else []

    logger.debug(f"finish_reason={candidate.finish_reason}, content_is_none={content is None}, parts_count={len(parts)}")

    for i, p in enumerate(parts):
        logger.debug(f"  part[{i}]: thought={getattr(p, 'thought', False)}, text_len={len(getattr(p, 'text', '') or '')}")

    try:
        text = response.text
        if text:
            return text
    except Exception as e:
        logger.debug(f"response.text raised: {e}")

    result = "".join(
        p.text for p in parts
        if getattr(p, "text", None) and not getattr(p, "thought", False)
    )

    if not result:
        logger.warning(f"_extract_text empty. finish_reason={candidate.finish_reason}, parts={len(parts)}, content={content}")

    return result


def _usage(response) -> dict[str, int] | None:
    """Map a Vertex usage_metadata to Langfuse usage_details (best-effort)."""
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return None
    try:
        return {
            "input": int(getattr(meta, "prompt_token_count", 0) or 0),
            "output": int(getattr(meta, "candidates_token_count", 0) or 0),
            "total": int(getattr(meta, "total_token_count", 0) or 0),
        }
    except Exception:
        return None


def _usage_lc(raw) -> dict[str, int] | None:
    """Map a LangChain AIMessage.usage_metadata to our usage dict (best-effort)."""
    meta = getattr(raw, "usage_metadata", None)
    if not meta:
        return None
    try:
        return {
            "input": int(meta.get("input_tokens", 0) or 0),
            "output": int(meta.get("output_tokens", 0) or 0),
            "total": int(meta.get("total_tokens", 0) or 0),
        }
    except Exception:
        return None


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


# What Omni will actually honour. The prompt asks for 4-10s; a model that answers 45 or "about six"
# must not put that number in front of the renderer.
CLIP_SECONDS_MIN = 4.0
CLIP_SECONDS_MAX = 10.0


def _clamp_clip_seconds(value) -> float:
    """A model-supplied clip length, coerced into the range Omni actually renders. 0.0 = unusable,
    which callers read as "no estimate" and fall back to the generic duration wording."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if seconds <= 0 or math.isnan(seconds) or math.isinf(seconds):
        return 0.0
    return round(min(max(seconds, CLIP_SECONDS_MIN), CLIP_SECONDS_MAX), 1)


# Measured from this project's own TTS on 8 real storyboard lines (12→118 characters), silence
# trimmed: 431 characters / 40.6 seconds. The rate held across all 8 (8.7–12.6).
NARRATION_CHARS_PER_SECOND = 10.6


def estimate_narration_seconds(text: str) -> float:
    """How long a narration line takes to speak, from its CHARACTER count.

    Not words: Thai writes without spaces, so `len(text.split())` reads a whole Thai sentence as
    two or three "words" and every estimate built on it collapses to its floor. Characters work
    for both scripts — 10.6 chars/s is also ~127 wpm in English — so one rule covers every
    language with no branch. Returns 0.0 for an empty line, which callers read as "no narration"."""
    chars = len((text or "").replace(" ", ""))
    return chars / NARRATION_CHARS_PER_SECOND if chars else 0.0


def _parse_json_list(raw: str) -> list:
    """Strip optional ```-fences and parse a JSON array; [] on failure."""
    try:
        data = json.loads(_strip_fences(raw))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_json_obj(raw: str) -> dict:
    """Strip optional ```-fences and parse a JSON object; {} on failure."""
    try:
        data = json.loads(_strip_fences(raw))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


_PROMPT_SHOT_DROP = frozenset({"image_plan", "prompt_full", "generated_img", "generated_video"})


def _shot_for_prompt(shot: dict | None) -> dict:
    """A shot with the PIPELINE'S OWN OUTPUT stripped, for embedding in a prompt.

    classify_shot_image / plan_shot_image dump the whole shot as JSON. The frontend sends back the
    live shot object, which for any already-rendered shot carries `image_plan`. Seeing it, the
    prompt author concluded the expected output was a finished, model-ready prompt: it copied the
    rule blocks the code appends anyway INTO its own prompt_img, three times over. Measured on run
    22 / shot 1.1-1 (POST /steps/image_plan, run_id bound), counting occurrences in the prompt that
    actually reaches the image model:

                                        with image_plan   without
      "COMPOSITION: the host stands…"          4             1
      "The host ONLY does what…"               4             1
      "Do NOT render any text…"                3             1
      total length                          6568          2883

    Three of those repeats were one identical 1576-char block emitted back-to-back, so the shot's
    real description — ~300 chars — was outweighed roughly 15:1 by boilerplate. Copying the block
    wholesale also carried over stray details from the previous render, which is why a plain
    "Generate" could not undo a bad image.

    `generated_img` is dropped for the same reason and NOT because the image leaks: both calls here
    are text-only and never receive pixels — removing it alone changed nothing (3/3 unchanged).

    Stripped server-side so the bulk path and any future caller are covered too.
    `prev_shot["image_plan"]["shows_dish"]` is read off the python dict in nodes.py, not from this
    JSON, so it is unaffected.
    """
    return {k: v for k, v in (shot or {}).items() if k not in _PROMPT_SHOT_DROP}


def _ref_instruction_gemini(refs: list[dict], prompt: str) -> str:
    """The TEXT part sent to the gemini image model (the ref images ride alongside as separate,
    labelled Parts). Mirrors generate_image's assembly so it can be previewed/edited as the
    'full prompt' before generation."""
    if not refs:
        return prompt
    # "Current image" = /steps/image/edit's base. Same situation as a previous frame — the first ref
    # IS the thing being edited — so it needs the strict reproduce-and-change-only-the-delta wrapper,
    # not the generic one that invites re-framing.
    if str(refs[0].get("label", "")).startswith(("Previous frame", "Current image")):
        # The first image is the TEMPLATE; the identity anchors that ride behind it are the CORRECTION.
        # Both are now sent on this route (resolve_shot_refs stopped skipping the kitchen anchor on
        # edit_state), but the wrapper never said what they were for — an unexplained extra image next
        # to "reproduce the first one identically" is one the model can only ignore or misread, which
        # is how the room and the host kept drifting on continuation shots.
        instr = (
            "EDIT the FIRST reference image: it is the TEMPLATE — reproduce its camera angle, crop, "
            "framing, surface, background, bowls/props, lighting and EVERY object not being changed. "
            "Do NOT redesign, rearrange or add objects, do NOT switch layout. Apply ONLY the single "
            "change described below.\n"
            "The OTHER references are ANTI-DRIFT ANCHORS, not new scenes or a new layout — they exist "
            "to stop the template's own errors from being copied forward. Where the template disagrees "
            "with an anchor, THE ANCHOR WINS: 'Character reference' fixes the host's face, hair, skin "
            "tone and outfit (if she looks different in the template, correct her to this reference); "
            "'Background reference' fixes the kitchen's fixtures, materials and colours; "
            "'Reference photo of <subject>' shows how the new or changed object AND its vessel should look.\n\n"
        )
    else:
        instr = "Generate the storyboard frame described below using the reference images above. "
        if any(str(r.get("label", "")).startswith("Character reference") for r in refs):
            instr += (
                "CRITICAL: the person MUST be the EXACT same person as the Character reference — "
                "identical face, facial features, hairstyle, skin tone, and the SAME outfit/clothing. "
                "Do NOT change their face or clothes; do NOT invent a different person. "
            )
        instr += (
            "The kitchen — its sink, faucet, stove, counters, cabinets and window — is FIXED by the "
            "Background reference: do NOT add, move, remove or invent any fixture; use the existing ones. "
            "Every object RESTS on a real surface — nothing floats in mid-air. FOOD IS NEVER LOOSE ON "
            "THE COUNTER: an ingredient, raw item or finished dish sits IN or ON a vessel (bowl, measuring "
            "cup, measuring spoon, plate, tray, cutting board, pan/pot, cooling rack); tools and equipment "
            "may rest on the bare counter. Lighting is soft and even, with NO rim light, backlight glow, "
            "halo or white outline around the subject (it blends naturally into the scene). For any "
            "'Reference photo of <subject>' images, render that subject AND the vessel holding it to "
            "match the real object's appearance shown — the same bowl/cup/plate, not a new one, unless "
            "the description below puts the food somewhere else. Only the pose, action and framing "
            "change to match the description.\n\n"
        )
    return instr + prompt


def _safe_int(v) -> int | None:
    """int(v) or None — the LLM sometimes echoes `"no": "shot 1"` / "" which would raise ValueError
    and take down the whole QA batch; skip that one item instead of crashing the step."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _download_gcs(uri: str, project: str, credentials) -> bytes | None:
    """Download bytes from a gs:// URI using the service-account credentials."""
    try:
        from google.cloud import storage

        bucket_name, _, blob_path = uri.replace("gs://", "").partition("/")
        client = storage.Client(project=project, credentials=credentials)
        return client.bucket(bucket_name).blob(blob_path).download_as_bytes()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"GCS video download failed for {uri}: {e}")
        return None


# transient network faults worth retrying a poll/download on (a blip must not kill a 7-min chain);
# an ApiError (400 / quota / bad arg) is NOT here — those are deterministic and must surface immediately.
_TRANSIENT_NET = (httpx.TransportError, ssl.SSLError, ConnectionError, TimeoutError)


def _net_retry(fn, *, tries: int = 4, base: float = 2.0):
    """Call fn(), retrying ONLY on a transient network fault (SSL/connect/timeout) with backoff.
    Use for IDEMPOTENT calls (operation polling, video download) — never for a submit (double render)."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except _TRANSIENT_NET as e:
            if attempt >= tries:
                logger.warning(f"network call failed after {tries} tries: {e}")
                raise
            logger.warning(f"transient network fault ({type(e).__name__}) — retry {attempt}/{tries - 1}")
            time.sleep(base * attempt)


def _veo_reasons(result, op=None) -> list[str]:
    """Why a finished Veo op produced no video — the RAI (safety/audio) filter, or a real op error.
    Returns [] when the result has a usable video. `op` (optional) lets a None result surface op.error
    instead of an opaque 'no response'. Mirrors the test scripts' _reasons()."""
    if result is None:
        err = getattr(op, "error", None) if op is not None else None
        return [f"no video (op error: {err})" if err else "no response"]
    if getattr(result, "generated_videos", None):
        return []
    return [str(r) for r in (getattr(result, "rai_media_filtered_reasons", None) or ["empty result (filtered)"])]


def _is_name_filter(reasons: list[str]) -> bool:
    """True when Veo rejected for a real person's name / celebrity / likeness — a DETERMINISTIC
    filter (retrying the same prompt won't help; the prompt must be rewritten to drop the name)."""
    s = " ".join(reasons).lower()
    return any(k in s for k in ("name", "celebrit", "likeness", "real people"))


def _is_audio_filter(reasons: list[str]) -> bool:
    """True when Veo rejected on the AUDIO track ('issue with the audio for your prompt') — a
    (near-)deterministic filter on on-camera lip-synced speech. Retrying the same prompt won't help;
    drop the lip-sync insistence (keep the dialogue) and retry once."""
    s = " ".join(reasons).lower()
    return "audio" in s


def _strip_lipsync(text: str) -> str:
    """Drop ONLY the on-camera lip-sync insistence, keeping the host SPEAKING on camera + the dialogue
    VERBATIM: 'The host, on camera, speaks the line herself — her lips move in sync with the words — in a
    {voice} voice: "..."' → 'The host, on camera, speaks the line herself, in a {voice} voice: "..."'.
    The host still visibly speaks (Veo animates "speaks the line herself"), just without the talking-head
    lip-sync pressure that trips Veo's audio filter more often — proven by render to raise the pass rate
    while keeping her speaking (test/test_audio_host_speak.py). Unchanged if the lip-sync clause is absent."""
    return re.sub(
        r'\s*[—–-]\s*(?:her|his|their)\s+lips?\s+move\s+in\s+sync\s+with\s+the\s+words\s*[—–-]\s*',
        ', ', text, flags=re.I)


def _validate_rebalanced_lines(lines: list[str], out, anchors: list[str] | None = None) -> list[str] | None:
    """Guard for the intro-VO rebalance: the LLM may ONLY move fragments across line boundaries.
    Reject (→ None) on wrong shape, changed line count, any word added/dropped/rewritten (the
    concatenation, whitespace-insensitive, must stay verbatim-identical) — or a line losing its own
    ITEM: each line is spoken over the shot showing anchors[i], so if the item's name was in the line
    before, it must still be there after (kills over-eager moves that drag a whole item sentence)."""
    if not isinstance(out, list) or len(out) != len(lines) or not all(isinstance(s, str) for s in out):
        return None
    if "".join(out).replace(" ", "") != "".join(lines).replace(" ", ""):
        return None
    for i, a in enumerate(anchors or []):
        key = (a or "").replace(" ", "")
        if key and key in lines[i].replace(" ", "") and key not in out[i].replace(" ", ""):
            return None
    return [s.strip() for s in out]


def _retry_backoff(cfg, attempt: int, reasons: list[str]) -> None:
    """Wait before a Veo re-attempt so we don't hammer a struggling backend. Linear in attempt;
    doubled for a code-13 INTERNAL error (Google literally says 'try again in a few minutes').
    EXCEPTION: the audio filter is an instant stochastic re-roll, not backend overload — a short
    fixed pause is enough, so the escalation ladder doesn't waste minutes riding the code-13 curve."""
    base = max(0, getattr(cfg, "retry_backoff_seconds", 0))
    if not base:
        return
    is_backend = any(("internal server" in r.lower() or "code: 13" in r.lower() or "'code': 13" in r.lower()) for r in reasons)
    if _is_audio_filter(reasons) and not is_backend:
        time.sleep(min(base, 3))     # audio re-roll — quick retry, no escalating wait
        return
    wait = base * attempt
    if is_backend:
        wait *= 2
    logger.info(f"waiting {wait}s before Veo retry (attempt {attempt}) — {'; '.join(reasons)[:80]}")
    time.sleep(wait)


def _is_omni(model: str) -> bool:
    """Gemini Omni Flash uses the Interactions API, not Veo's generate_videos."""
    return "gemini-omni" in (model or "").lower()


def _omni_ref_rule(kind: str, label: str) -> str:
    """The lock rule for one <IMAGE_REF_k>, by its kind — so Omni knows HOW to use each attached
    image (a person ref must not be redrawn, a scene ref must not gain fixtures, the shot's own
    generated frame is the step/state to reach, etc.)."""
    k = (kind or "").lower()
    if k == "person":
        return ("the on-camera person — WHENEVER the person or any part of them (face, hands, arms) appears in "
                "frame, MATCH their face, skin, hands, body, hair, outfit (including fabric pattern, texture, "
                "color, print and every visible detail of the clothing) and accessories EXACTLY as in this "
                "image; never restyle, age, re-dress, simplify patterns, or add/remove/replace people. "
                "Do NOT introduce a person who is not in the shot.")
    if k in ("kitchen", "scene"):
        # The structure lock is load-bearing (it stops the background drifting and growing fixtures)
        # and must stay. What is SCOPED here is how much of the setting has to be on screen: Omni was
        # reading "the fixed background for the whole clip" as licence to open the clip on this image,
        # so a tight insert of food on a board began on the empty kitchen.
        return ("the setting/background — whatever part of the setting is visible in this shot must match this "
                "image: keep its layout, fixtures and structure exactly consistent with it; do NOT distort it or "
                "invent extra/altered fixtures or objects. It describes how the room LOOKS and nothing more — it is "
                "NOT a frame of this clip, NOT the opening shot, and NOT a destination: never open on it, never cut "
                "to it, never move the camera or action toward it. If this shot is framed tight and little or none of "
                "the setting is in view, that is correct — do not widen, pan or re-frame to bring it into view.")
    if k == "prev_ref":
        return "the previous shot's ending frame — match its composition, framing and camera angle for a smooth match cut; the scene content may differ but the visual structure should feel continuous."
    if k == "still":
        return "this shot's OWN generated frame — the exact step/state the action should reach; keep dish/props/composition consistent with it."
    if k == "next":
        return "the NEXT shot's frame — end the clip settling toward this composition for a seamless join."
    if k in ("ingredient", "state", "keyword"):
        return f"reference for {label or 'this subject'} — render it to match this image; do not substitute or add extras."
    if k == "user":
        # The user attached this one by hand and said what it is; their words ARE the rule, because
        # nothing else in the pipeline knows what they wanted from it. Without the note it would be
        # indistinguishable from the auto refs, which is the same problem ref_notes solved on the
        # image side.
        return (f"a reference image the user attached — {label}. Use it as described; take only what "
                "that description asks for and nothing else from it."
                if label else
                "a reference image the user attached — take visual cues (subject / style / composition) "
                "from it; do not copy its background or unrelated objects.")
    return f"reference for {label or 'this subject'} — render to match this image."


def _parse_time_window(s: str) -> int | None:
    """A shot's time window 'mm:ss - mm:ss' → clip length in whole seconds (rounded), else None.
    Also accepts a bare 'ss' or 'mm:ss' single value (treated as the length itself)."""
    def _to_sec(t: str) -> int | None:
        t = t.strip()
        if not t:
            return None
        parts = t.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            if len(parts) == 1:
                return int(parts[0])
        except ValueError:
            return None
        return None

    if not s:
        return None
    m = re.split(r"\s*[-–—]\s*", s.strip(), maxsplit=1)
    if len(m) == 2:
        a, b = _to_sec(m[0]), _to_sec(m[1])
        if a is not None and b is not None and b > a:
            return b - a
    one = _to_sec(s)
    return one if (one and one > 0) else None


# The LLM sometimes writes its own (often truncated) adherence sentence — strip any variant so the
# code-appended OMNI_ADHERENCE_LINE is the single authoritative one.
_OMNI_ADHERENCE_RE = re.compile(r"^Only the subjects, actions and sounds[^\n]*\n?", re.M)


# Matches a bare `@Image3` OR a (possibly wrong) angle-bracket tag glued to it, e.g. `<IMAGE_REF_3>@Image3`,
# with optional surrounding backticks (markdown quoting degrades Omni generation — always stripped).
_OMNI_TAG_RE = re.compile(r"`?(?:<(?:FIRST_FRAME|IMAGE_REF_\d+)>)?@Image(\d+)`?")
_OMNI_BARE_TICK_RE = re.compile(r"`(<(?:FIRST_FRAME|IMAGE_REF_\d+)>)`")


# The Voice direction block is English prose sitting right under "reproduce the Thai script exactly
# as written", so a bare Thai word in a bullet risks being read aloud as narration. The UI picks
# ผู้หญิง/ผู้ชาย; this is the one place that turns any of it into the English the prompt uses.
_VOICE_GENDER = {"female": "female", "male": "male", "ผู้หญิง": "female", "ผู้ชาย": "male"}


def _voice_gender(value: str | None) -> str:
    """`female` unless the caller clearly asked for `male` — an unset or unrecognised value keeps
    the host every existing run was authored with."""
    return _VOICE_GENDER.get((value or "").strip().lower(), "female")


def _apply_reading_rules(vo: str) -> str:
    """Spell the narration the way Omni should READ it — see THAI_READING_RULES for the why.

    Applied only on the way into OMNI_VOICE_BLOCK, never to the stored voice_over: the DB text feeds
    the subtitle burn-in too, and "ปรา-กด-กาน" on screen would be a defect.

    One regex pass rather than a str.replace loop, because a loop re-reads its own output: with
    rules `ตากลม = ตาก ลม` and `ตาก = ต-าก`, the second rule rewrites the ตาก the first just
    produced, giving "ต-าก ลม". re.sub never rescans what it replaced, and alternation prefers the
    first branch that matches at a position — so listing terms longest-first also gives longest-match.
    """
    rules: dict[str, str] = {}
    for line in get_prompt("THAI_READING_RULES").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term, sep, repl = line.partition("=")
        if sep and term.strip():
            rules[term.strip()] = repl.strip()
    if not rules:
        return vo
    pattern = "|".join(re.escape(t) for t in sorted(rules, key=len, reverse=True))
    return re.sub(pattern, lambda m: rules[m.group(0)], vo)


def _normalize_omni_tags(text: str, source_header: str) -> str:
    """Rewrite the author's `@ImageN` references to the body style Omni responds to best:
    the @ImageN binding lives ONLY in the `[# Sources]` header — in the body a reference image is a
    bare `<IMAGE_REF_k>` and the FIRST_FRAME is plain `ImageN` (no tag, no @). Backticks around tags
    are stripped (they degrade generation). Deterministic "Use ImageN as the starting frame." /
    "Use ImageN as a reference for the video generation." lines are appended for every image not
    already covered. No-op when the header has no bindings."""
    canon: dict[str, str] = {}
    first_n: str | None = None
    ref_ns: list[str] = []
    for m in re.finditer(r"<(FIRST_FRAME|IMAGE_REF_\d+)>@Image(\d+)", source_header or ""):
        n = m.group(2)
        if m.group(1) == "FIRST_FRAME":
            canon[n] = f"Image{n}"
            first_n = n
        else:
            canon[n] = f"<{m.group(1)}>"
            ref_ns.append(n)
    if not canon:
        return text
    out = _OMNI_TAG_RE.sub(lambda m: canon.get(m.group(1), m.group(0)), text)
    out = _OMNI_BARE_TICK_RE.sub(r"\1", out)
    extra = []
    if first_n and f"Use Image{first_n} as the starting frame" not in out:
        extra.append(f"Use Image{first_n} as the starting frame.")
    for n in ref_ns:
        line = f"Use Image{n} as a reference for the video generation."
        if line not in out:
            extra.append(line)
    return out + ("\n" + "\n".join(extra) if extra else "")


def _omni_manifest_text(images: list[dict]) -> str:
    """Strong role-tag manifest for an Omni multi-ref prompt, prepended to the prompt IN SEND ORDER
    (Omni has no separate manifest channel — the model learns each image's job only from this block).
    FIRST_FRAME → <FIRST_FRAME> (Image1, the literal opening frame); the rest → <IMAGE_REF_k>@ImageN.
    Carries POC-strength rules: pixel-lock the opening frame (for seamless clip concatenation) and a
    per-kind lock rule for each reference (person appearance / scene structure / this shot's step)."""
    if not images:
        return ""
    # source header binding every tag to its image number, in send order
    header_parts, ref_i = [], 0
    for im in images:
        n = len(header_parts) + 1
        if (im.get("role") or "").upper() == "FIRST_FRAME":
            header_parts.append(f"<FIRST_FRAME>@Image{n}")
        else:
            header_parts.append(f"<IMAGE_REF_{ref_i}>@Image{n}")
            ref_i += 1
    lines = [f"[# Sources] {' '.join(header_parts)}",
             "These images are attached in THIS order — refer to each by its exact tag where you describe it:"]
    ref_i = 0
    for i, im in enumerate(images, start=1):
        role = (im.get("role") or "").upper()
        kind = im.get("kind", "")
        label = im.get("label", "")
        if role == "FIRST_FRAME":
            frm = ("the exact LAST FRAME of the previous clip" if kind == "prev"
                   else "this shot's own generated frame")
            lines.append(
                f"<FIRST_FRAME>@Image{i} — {frm}. The clip MUST OPEN on this frame pixel-for-pixel "
                "(same framing, crop, camera angle, lighting, colours and every object in place); all "
                "motion grows FROM it. Do NOT re-stage, re-frame, re-light or reinterpret the opening."
                + (" It is the join with the previous clip — if frame 0 differs, the assembled film jumps."
                   if kind == "prev" else ""))
        else:
            lines.append(f"<IMAGE_REF_{ref_i}>@Image{i} — {_omni_ref_rule(kind, label)}")
            ref_i += 1
    return "\n".join(lines)


def _chain_model(cfg) -> str:
    """Model to use for extension chaining. The `lite` tier does NOT support extension
    (Veo 400 "use case not supported"), so auto-switch to interpolation_model (standard).
    Omni can't chain at all → also fall back to the (Veo) interpolation_model."""
    m = cfg.model or ""
    if _is_omni(m):
        fallback = getattr(cfg, "interpolation_model", "") or ""
        if not fallback:
            raise RuntimeError(f"Omni model '{m}' can't do continuous chaining and no "
                               "interpolation_model (Veo) is set to fall back to")
        logger.warning(f"Omni chain: '{m}' can't extend → using Veo '{fallback}' instead")
        return fallback
    if "lite" in m.lower():
        fallback = getattr(cfg, "interpolation_model", "") or ""
        if fallback:
            logger.warning(f"Veo chain: model '{m}' (lite) can't extend → using '{fallback}' instead")
            return fallback
        raise RuntimeError(f"Veo extension chaining does not support the lite model '{m}', "
                           "and no interpolation_model is set to fall back to")
    return m


def _wait_video_active(client, video, *, timeout_s: int = 180, poll_s: int = 3) -> None:
    """Veo extension rejects a source clip until its Files-API file is ACTIVE — it 400s with
    '...must be a video that was generated by VEO that has been processed'. A just-generated clip's
    file can still be PROCESSING, so poll it to ACTIVE before extending. Best-effort: returns (lets
    the extend proceed) when the file name can't be resolved or queried."""
    uri = getattr(video, "uri", "") or ""
    m = re.search(r"/files/([^:/?]+)", uri)
    if not m:
        return
    name = f"files/{m.group(1)}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            f = _net_retry(lambda: client.files.get(name=name))   # ride out a transient blip while waiting
        except Exception as e:  # noqa: BLE001
            logger.debug(f"files.get({name}) failed, extending anyway: {e}")
            return
        state = str(getattr(f, "state", "") or "").upper()
        if "ACTIVE" in state:
            return
        if "FAIL" in state:
            raise RuntimeError(f"Veo source video {name} failed processing (state={state})")
        time.sleep(poll_s)
    logger.warning(f"Veo source video {name} not ACTIVE after {timeout_s}s — extending anyway")


def _veo_video_bytes(client, video, project, credentials) -> bytes | None:
    """Pull MP4 bytes out of a Veo result video (inline → files.download → GCS uri)."""
    data = getattr(video, "video_bytes", None)
    if not data:
        try:
            downloaded = client.files.download(file=video)
            data = downloaded if isinstance(downloaded, (bytes, bytearray)) else getattr(video, "video_bytes", None)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"files.download failed: {e}")
    if not data and getattr(video, "uri", None) and str(video.uri).startswith("gs://"):
        data = _download_gcs(video.uri, project, credentials)
    return data


class GeminiClient:
    """Vertex AI Gemini wrapper. All LLM calls are traced as Langfuse generations.

    Constructed with an explicit Settings object (no module-level singleton, no
    import-time side effects) so it can be built inside the app lifespan and
    swapped/mocked in tests.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        creds_path = ROOT_DIR / settings.gemini.credentials_file
        if not creds_path.exists():
            raise FileNotFoundError(f"Gemini credentials file not found: {creds_path.resolve()}")

        with open(creds_path) as f:
            creds_data = json.load(f)

        project = creds_data.get("project_id")
        if not project:
            raise ValueError("credentials file is missing project_id")

        location = settings.gemini.location

        credentials = service_account.Credentials.from_service_account_info(
            creds_data,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        vertexai.init(project=project, location=location, credentials=credentials)
        self.model = GenerativeModel(settings.gemini.model)
        self._project = project
        self._location = location
        self._credentials = credentials
        self._langchain_llm = None
        self._genai_client = None
        self._genai_image_client = None
        logger.info(f"Gemini client initialized: {settings.gemini.model} ({project}/{location})")

    @property
    def model_name(self) -> str:
        return self.settings.gemini.model

    async def generate_query_variants(self, topic: str, n: int = 3,
                                       temperature: float = 0.3, avoid: list[str] | None = None) -> list[str]:
        """Generate distinct YouTube search queries for `topic`. `avoid` (used at top-up) lists
        the queries already tried so the model diverges; `temperature` is raised for those rounds."""
        def _run() -> list[str]:
            avoid_block = get_prompt("QUERY_AVOID_BLOCK").format(used_queries="\n".join(f"  - {q}" for q in avoid)) if avoid else ""
            prompt = get_prompt("QUERY_VARIANTS_PROMPT").format(n=n, topic=topic, avoid_block=avoid_block)
            with obs.generation("gemini.query_variants", model=self.model_name, input={"topic": topic, "n": n, "temperature": temperature, "prompt": prompt}) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": 512,
                        "thinking_config": {"thinking_budget": 0},
                    },
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.query_variants", usage, (time.perf_counter() - _t0) * 1000)

            logger.debug(f"Query variant raw response: {raw!r}")
            try:
                variants: list[str] = json.loads(_strip_fences(raw))
                variants = [v for v in variants if isinstance(v, str) and v.strip()]
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Query variant parse failed, using original topic only. Raw: {raw!r}")
                variants = []
            # Only the FIRST round prepends the raw topic as query 0; top-up rounds (avoid set) must NOT
            # re-add it or every round wastes a slot on the same repeated search.
            used_lower = {q.lower() for q in (avoid or [])}
            base = [] if avoid else [topic]
            queries = base + [v for v in variants if v.lower() != topic.lower() and v.lower() not in used_lower]
            logger.info(f"Query variants for '{topic}' (temp={temperature}): {queries}")
            return queries[:n]

        return await asyncio.to_thread(_run)

    async def filter_videos(self, videos: list[dict], topic: str, n: int = 10) -> list[dict]:
        def _parse_verdicts(raw: str, count: int) -> tuple[list[int], list[tuple[int, str]]]:
            """Parse the per-title verdict array → (kept indices, [(rejected index, reason)]).
            Raises if the response has no usable verdict (wrong schema / old [0,3,7] format) so
            the caller can retry, then fail CLOSED — never silently keep the wrong videos."""
            data = json.loads(_strip_fences(raw))
            if not isinstance(data, list):
                raise TypeError(f"verdicts is not a list: {type(data).__name__}")
            kept: list[int] = []
            rejected: list[tuple[int, str]] = []
            seen: set[int] = set()
            for v in data:
                if not isinstance(v, dict):
                    continue
                i = v.get("index")
                if isinstance(i, bool) or not isinstance(i, int) or not (0 <= i < count) or i in seen:
                    continue
                seen.add(i)
                rel = v.get("relevant")
                if isinstance(rel, str):  # the model sometimes returns the boolean as a string
                    rel = rel.strip().lower() == "true"
                if rel is True:
                    kept.append(i)
                else:
                    rejected.append((i, str(v.get("reason", ""))))
            if not kept and not rejected:
                raise ValueError("no valid verdict objects in response (wrong schema?)")
            return kept, rejected

        def _run() -> list[dict]:
            candidates_json = json.dumps(
                [{"index": i, "title": v["title"], "duration_seconds": v.get("duration") or 0}
                 for i, v in enumerate(videos)],
                ensure_ascii=False,
            )
            prompt = get_prompt("FILTER_PROMPT").format(topic=topic, n=n, candidates=candidates_json)

            last_err: Exception | None = None
            for attempt in (1, 2, 3):  # retries if the response won't parse
                with obs.generation(
                    "gemini.filter_videos",
                    model=self.model_name,
                    input={"topic": topic, "n": n, "candidate_count": len(videos), "attempt": attempt, "prompt": prompt},
                ) as gen:
                    _t0 = time.perf_counter()
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.0,
                            "max_output_tokens": self.settings.filter.max_output_tokens,
                            "thinking_config": {"thinking_budget": self.settings.filter.thinking_budget},
                        },
                    )
                    raw = _extract_text(response).strip()
                    usage = _usage(response)
                    gen.update(output=raw, usage_details=usage)
                    record_usage("gemini.filter_videos", usage, (time.perf_counter() - _t0) * 1000)

                logger.debug(f"Filter raw response (attempt {attempt}): {raw!r}")
                try:
                    kept_idx, rejected = _parse_verdicts(raw, len(videos))
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    last_err = e
                    logger.warning(f"Filter parse failed (attempt {attempt}): {e}. Raw: {raw!r}")
                    continue

                filtered = [videos[i] for i in kept_idx][:n]
                for i, reason in rejected:
                    logger.info(f"Filter rejected [{i}] '{videos[i]['title'][:60]}' — {reason}")
                logger.info(f"Relevance filter: {len(videos)} candidates -> {len(filtered)} kept "
                            f"({len(rejected)} rejected) for topic '{topic}'")
                return filtered

            # fail-CLOSED: don't leak unfiltered videos into the pipeline — return nothing so top-up retries.
            logger.error(f"Filter parse failed 3 times, keeping NOTHING (fail-closed). Last error: {last_err}")
            from app.graph.state import emit  # local import: avoids a state->container->gemini_client cycle

            emit({
                "type": "progress",
                "message": f"⚠️ Relevance filter response couldn't be parsed after 3 attempts — treating this round as 0 relevant videos ({last_err}).",
            })
            return []

        return await asyncio.to_thread(_run)

    async def select_master_video(self, summaries: list[str], topic: str, video_meta: list[dict]) -> dict:
        """Auto-suggest which analyzed video should be the MASTER reference for synthesis (see
        `synthesize_from_summaries`), so the "one base method + supplementary tips" approach can be
        the DEFAULT even when the user didn't hand-pick a master. Fail-soft: any parse/API failure
        returns {"index": None} so the caller falls back to synthesize_from_summaries' own
        pick-the-best-method-itself behavior — this is a suggestion, never a hard requirement."""
        def _run() -> dict:
            candidates = "\n\n".join(
                f"[{i}] {m.get('title', '')} (~{int(m.get('duration_seconds') or 0)}s)\n"
                f"{(summaries[i] or '')[:1500]}"
                for i, m in enumerate(video_meta)
            )
            prompt = get_prompt("SELECT_MASTER_PROMPT").format(topic=topic, n=len(summaries), candidates=candidates)

            with obs.generation(
                "gemini.select_master_video",
                model=self.model_name,
                input={"topic": topic, "candidate_count": len(summaries), "prompt": prompt},
            ) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 512,
                        "thinking_config": {"thinking_budget": 512},
                    },
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.select_master_video", usage, (time.perf_counter() - _t0) * 1000)

            try:
                data = json.loads(_strip_fences(raw))
                idx = data.get("index")
                if isinstance(idx, bool) or not isinstance(idx, int) or not (0 <= idx < len(summaries)):
                    raise ValueError(f"index out of range or wrong type: {idx!r}")
                return {"index": idx, "reason": str(data.get("reason", "")).strip()}
            except Exception as e:  # noqa: BLE001 — best-effort suggestion, never blocks synthesis
                logger.warning(f"select_master_video parse failed, no auto-suggestion: {e}. Raw: {raw!r}")
                return {"index": None, "reason": ""}

        return await asyncio.to_thread(_run)

    async def extract_production_items(self, synthesis: str, topic: str) -> dict:
        """Pull structured {"ingredients": [...], "equipment": [...]} out of a research synthesis.

        Runs at the end of the research step so the run's Menu can be seeded before any script
        exists; the entry format mirrors SCRIPT_PROMPT rule 7 so the names match what the script's
        production block used to produce. Fail-soft: any parse/API failure returns empty lists —
        research must never fail over a missing shopping list (the script step then falls back to
        extracting from the synthesis itself, exactly today's behavior)."""
        def _run() -> dict:
            prompt = get_prompt("PRODUCTION_ITEMS_EXTRACT_PROMPT").format(topic=topic, synthesis=synthesis)
            with obs.generation(
                "gemini.extract_production_items",
                model=self.model_name,
                input={"topic": topic, "synthesis_chars": len(synthesis), "prompt": prompt},
            ) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 2048,
                        "thinking_config": {"thinking_budget": 0},
                    },
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.extract_production_items", usage, (time.perf_counter() - _t0) * 1000)

            def _clean(items) -> list[str]:
                out, seen = [], set()
                for x in items if isinstance(items, list) else []:
                    s = str(x).strip()
                    if s and s.lower() not in seen:
                        seen.add(s.lower())
                        out.append(s)
                return out

            try:
                data = json.loads(_strip_fences(raw))
                return {"ingredients": _clean(data.get("ingredients")), "equipment": _clean(data.get("equipment"))}
            except Exception as e:  # noqa: BLE001 — best-effort, never blocks research
                logger.warning(f"extract_production_items parse failed: {e}. Raw: {raw[:300]!r}")
                return {"ingredients": [], "equipment": []}

        return await asyncio.to_thread(_run)

    async def transform_image_queries(
        self, subjects: list[str], topic: str, theme: str = "", material_palette: str = ""
    ) -> dict[str, str]:
        """Batch-transform visual elements into image-search queries (one LLM call).

        Returns a {subject: query} map. Any subject the model drops falls back to
        using the subject itself as the query.
        """
        def _run() -> dict[str, str]:
            if not subjects:
                return {}
            prompt = get_prompt("IMAGE_QUERY_PROMPT").format(
                topic=topic,
                theme=theme or "-",
                material_palette=material_palette or "-",
                subjects=json.dumps(subjects, ensure_ascii=False),
            )
            with obs.generation(
                "gemini.image_queries", model=self.model_name, input={"count": len(subjects), "prompt": prompt}
            ) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 2048,
                        "thinking_config": {"thinking_budget": 0},
                    },
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.image_queries", usage, (time.perf_counter() - _t0) * 1000)

            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            mapping: dict[str, str] = {}
            try:
                for item in json.loads(raw):
                    if isinstance(item, dict) and item.get("subject"):
                        mapping[item["subject"]] = (item.get("query") or item["subject"]).strip()
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Image query transform parse failed, using subjects as-is. Raw: {raw!r}")

            for s in subjects:  # fallback for any missing element
                mapping.setdefault(s, s)
            logger.info(f"Transformed {len(subjects)} visual elements into image queries")
            return mapping

        return await asyncio.to_thread(_run)

    def _gen_with_retry(self, prompt, generation_config: dict, *, label: str = "gemini"):
        """generate_content with exponential backoff on Vertex 429 (RESOURCE_EXHAUSTED).

        A transient quota spike must RETRY — not kill the whole step. The storyboard breakdown
        fires one call per scene concurrently, so a single scene's 429 would otherwise fail the
        entire run via asyncio.gather. Re-raises after the last attempt / on any other error.
        """
        from google.api_core.exceptions import ResourceExhausted
        delay = 4.0
        for attempt in range(1, 6):
            try:
                return self.model.generate_content(prompt, generation_config=generation_config)
            except ResourceExhausted:
                if attempt == 5:
                    raise
                logger.warning(f"{label}: 429 quota exhausted (attempt {attempt}/5) — backoff {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def _validate_call(self, name: str, prompt: str, *, max_tokens: int = 4096) -> list:
        with obs.generation(name, model=self.model_name, input={"prompt": prompt}) as gen:
            _t0 = time.perf_counter()
            response = self._gen_with_retry(
                prompt,
                {
                    "temperature": 0.0,
                    "max_output_tokens": max_tokens,
                    "thinking_config": {"thinking_budget": 0},
                },
                label=name,
            )
            raw = _extract_text(response).strip()
            usage = _usage(response)
            gen.update(output=raw, usage_details=usage)
            record_usage(name, usage, (time.perf_counter() - _t0) * 1000)
        return _parse_json_list(raw)

    async def validate_image_subjects(self, items: list[dict], topic: str) -> dict[str, dict]:
        """Check each shot's image_subjects vs its context (and fix).

        items: [{"id", "motion_description", "on_screen_text", "prompt_img", "image_subjects"}]
        Returns {id: {"valid": bool, "image_subjects": [...]}}.
        """
        def _run() -> dict[str, dict]:
            if not items:
                return {}
            prompt = get_prompt("SUBJECTS_VALIDATE_PROMPT").format(topic=topic, scenes=json.dumps(items, ensure_ascii=False))
            out: dict[str, dict] = {}
            for item in self._validate_call("gemini.validate_subjects", prompt, max_tokens=8192):
                if isinstance(item, dict) and item.get("id"):
                    out[item["id"]] = {
                        "valid": bool(item.get("valid")),
                        "image_subjects": item.get("image_subjects") or [],
                    }
            return out

        return await asyncio.to_thread(_run)

    async def validate_image_queries(self, items: list[dict], topic: str, theme: str = "") -> dict[str, dict]:
        """Check each {subject, query} pair (and fix). Returns {subject: {"valid", "query"}}."""
        def _run() -> dict[str, dict]:
            if not items:
                return {}
            prompt = get_prompt("QUERIES_VALIDATE_PROMPT").format(
                topic=topic, theme=theme or "-", items=json.dumps(items, ensure_ascii=False)
            )
            out: dict[str, dict] = {}
            for item in self._validate_call("gemini.validate_queries", prompt):
                if isinstance(item, dict) and item.get("subject"):
                    out[item["subject"]] = {
                        "valid": bool(item.get("valid")),
                        "query": (item.get("query") or "").strip(),
                    }
            return out

        return await asyncio.to_thread(_run)

    async def validate_candidates(self, elements: list[dict], topic: str) -> dict[str, list[int]]:
        """Judge candidate relevance from descriptions. Returns {subject: [keep indices]}."""
        def _run() -> dict[str, list[int]]:
            if not elements:
                return {}
            prompt = get_prompt("CANDIDATES_VALIDATE_PROMPT").format(topic=topic, elements=json.dumps(elements, ensure_ascii=False))
            out: dict[str, list[int]] = {}
            for item in self._validate_call("gemini.validate_candidates", prompt):
                if isinstance(item, dict) and item.get("subject") is not None:
                    keep = [i for i in (item.get("keep") or []) if isinstance(i, int)]
                    out[item["subject"]] = keep
            return out

        return await asyncio.to_thread(_run)

    async def vision_filter_candidates(
        self, elements: list[dict], topic: str
    ) -> dict[str, list[int]]:
        """Multimodal relevance pass — Gemini LOOKS at the candidate images.

        elements: [{"subject": str, "images": [{"mime": str, "data": bytes}, ...]}]
            (the i-th image of an element maps to keep-index i)
        Returns {subject: [keep indices]}. Subjects absent from the reply are kept
        wholesale by the caller, so a model hiccup degrades to "keep all", never to
        dropping good images.
        """
        def _run() -> dict[str, list[int]]:
            payload = [e for e in elements if e.get("images")]
            if not payload:
                return {}

            # Interleave instruction text + per-element labels + inline images.
            parts: list = [get_prompt("CANDIDATES_VISION_PROMPT").format(topic=topic)]
            for el in payload:
                parts.append(f'\n=== Element: "{el["subject"]}" ===')
                for i, im in enumerate(el["images"]):
                    parts.append(f"image[{i}]:")
                    parts.append(Part.from_data(data=im["data"], mime_type=im["mime"]))

            with obs.generation("gemini.vision_filter_candidates", model=self.model_name,
                                input={"topic": topic, "elements": len(payload), "prompt": parts[0]}) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    parts,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 2048,
                        "thinking_config": {"thinking_budget": 0},
                    },
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.vision_filter_candidates", usage, (time.perf_counter() - _t0) * 1000)

            out: dict[str, list[int]] = {}
            for item in _parse_json_list(raw):
                if isinstance(item, dict) and item.get("subject") is not None:
                    out[item["subject"]] = [i for i in (item.get("keep") or []) if isinstance(i, int)]
            return out

        return await asyncio.to_thread(_run)

    def _get_genai_image_client(self):
        """google-genai client on the VERTEX backend for image generation — gives
        image_config.aspect_ratio (true 16:9, no letterbox) while keeping Vertex's
        permissive person policy (the Developer API blocks face/likeness)."""
        if self._genai_image_client is None:
            from google import genai

            self._genai_image_client = genai.Client(
                vertexai=True, project=self._project, location=self._location,
                credentials=self._credentials,
            )
            logger.info(f"Image (genai/Vertex) client initialized ({self._project}/{self._location})")
        return self._genai_image_client

    async def generate_image(self, prompt: str, references: list[dict] | None = None,
                             prebuilt: str | None = None) -> bytes | None:
        """Generate one image from a prompt, optionally guided by reference images.

        references: [{"label": str, "mime": str, "data": bytes}] — each is shown to the
        model (labelled) before the prompt so it keeps the character/setting/subjects
        consistent. prebuilt = a caller-assembled full prompt to send verbatim (user-edited
        "ครบ"), bypassing the ref-instruction wrapper. Returns PNG/JPEG bytes, or None on failure.
        """
        def _run() -> bytes | None:
            if not (prebuilt if prebuilt is not None else prompt).strip():
                return None
            from google.genai import types

            refs = references or []
            client = self._get_genai_image_client()
            aspect = self.settings.image_gen.gemini.aspect_ratio

            # genai contents: label text + image Part per ref, then the text part (instr + prompt).
            # prebuilt = the caller's already-assembled full prompt (user-edited "ครบ") → send verbatim.
            contents: list = []
            for r in refs:
                if r.get("label"):
                    contents.append(f"{r['label']}:")
                contents.append(types.Part.from_bytes(data=r["data"], mime_type=r.get("mime", "image/jpeg")))
            contents.append(prebuilt if prebuilt is not None else _ref_instruction_gemini(refs, prompt))

            config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect),
            )
            with obs.generation(
                "gemini.generate_image", model=self.settings.image_gen.gemini.model,
                # `refs` mirrors the video path: the source url + kind of each attached image, so a trace
                # identifies WHICH stored ref was sent. `references` keeps the inline media (needs
                # langfuse.log_media) for actually eyeballing them.
                input={"prompt": prompt, "aspect": aspect,
                       "refs": [{"kind": r.get("_kind", ""), "label": r.get("label", ""),
                                 "url": r.get("_url", "")} for r in refs],
                       "references": [
                    {"label": r.get("label", ""), "image": obs.media(r.get("data"), r.get("mime", "image/jpeg"))}
                    for r in refs
                ]},
            ) as gen:
                # The image endpoint throws transient 503 / "connection reset" ~randomly — retry.
                last_err = "no image in response"
                for attempt in range(1, 4):
                    try:
                        _t0 = time.perf_counter()
                        resp = client.models.generate_content(
                            model=self.settings.image_gen.gemini.model, contents=contents, config=config,
                        )
                        data = None
                        cand = resp.candidates[0] if resp.candidates else None
                        for p in (cand.content.parts if cand and cand.content else []):
                            idata = getattr(p, "inline_data", None)
                            if idata and getattr(idata, "data", None):
                                data = idata.data
                                break
                        if data:
                            usage = _usage(resp)
                            gen.update(output={"bytes": len(data), "image": obs.media(data, "image/png")}, usage_details=usage)
                            record_usage("gemini.generate_image", usage, (time.perf_counter() - _t0) * 1000)
                            return data
                        logger.warning(f"Image generation attempt {attempt}: empty response, retrying...")
                    except Exception as e:  # noqa: BLE001 — transient API errors → retry
                        last_err = str(e)
                        logger.warning(f"Image generation attempt {attempt} failed: {e}")
                    if attempt < 3:
                        time.sleep(1.5 * attempt)
                gen.update(level="ERROR", status_message=last_err)
                return None

        return await asyncio.to_thread(_run)

    async def breakdown_scene(self, scene: dict, production: dict, topic: str, director_prompt: str = "", aspect: str = "16:9", clip_seconds: int = 8, words_per_second: float = 2.3, shot_target: int = 0, intro_scene: bool = False, category: str = "food") -> list[dict]:
        """Break ONE script scene into storyboard shots (faithful split + image prompts).

        ``shot_target`` (>0) softly steers how many shots this scene produces — used by the per-part auto-fit
        to control runtime (duration = shots × clip length) and tame the shots-per-scene variance.
        """
        def _run() -> list[dict]:
            director_block = ""
            if director_prompt.strip():
                director_block = (
                    "\n\nDIRECTOR'S STYLE GUIDE — follow this visual style closely "
                    "(pacing, shots/framing, graphics, tone):\n"
                    f"{director_prompt.strip()}\n"
                )
            # Per-shot word budget only when a pace is configured (>0) — cleared → no word hint, seconds only.
            words_hint = ""
            if words_per_second and words_per_second > 0:
                words_hint = (f" — about {max(1, round(words_per_second * clip_seconds))} words, and never more "
                              f"than ~{max(1, round(words_per_second * 8))} words of speech")
            # Soft shot-count target for THIS scene (per-part auto-fit). Faithful: split finer / add cutaways of
            # the same action — never invent or repeat content, so produce fewer if the content can't fill it.
            shot_hint = ""
            if shot_target and shot_target > 0:
                shot_hint = (f"\n- SHOT COUNT TARGET — aim for about {shot_target} shots for THIS scene: split the "
                             f"scene's action and voice_over into ~{shot_target} single-beat shots (finer beats, or "
                             f"insert cutaways of the SAME action/step). Stay faithful — NEVER invent new content or "
                             f"repeat lines; if the content genuinely cannot fill {shot_target} shots, produce fewer.")
            prompt = get_prompt("STORYBOARD_PROMPT").format(
                topic=topic,
                shot_hint=shot_hint,
                character=(production.get("character_desc", "") or "").strip(),  # desc only — never the NAME (Veo blocks named people in visual prompts)
                theme=production.get("theme", "") or "-",
                lighting=production.get("lighting", "") or "-",
                mood=production.get("mood", "") or "-",
                material_palette=production.get("material_palette", "") or "-",
                start=scene.get("timecode_start", ""),
                end=scene.get("timecode_end", ""),
                aspect=aspect,
                duration=clip_seconds,
                words_hint=words_hint,
                dish=production.get("dish_appearance", "") or "-",
                ingredients=json.dumps(production.get("ingredients", []), ensure_ascii=False),
                equipment=json.dumps(production.get("equipment", []), ensure_ascii=False),
                scene=json.dumps(scene, ensure_ascii=False),
                category_rules=category_block(category, "storyboard"),
            ) + director_block
            # Ingredient/equipment INTRODUCTION scene → every shot is an object-only insert, so
            # motion_description + prompt_img must never mention the host (they feed prompt_video next).
            if intro_scene:
                prompt += (
                    "\n\nINTRODUCTION SCENE — EVERY shot in THIS scene is an OBJECT-ONLY insert (shot_kind='insert'): "
                    "the item(s) simply SIT on the counter and the camera reveals them. BOTH motion_description "
                    "AND prompt_img must show ONLY the item(s) plus a camera move — absolutely NO host, NO hand, "
                    "NO person, and NEVER phrases like 'ครูพี่เกศหยิบ...ขึ้นมาโชว์' / 'the host holds up / shows'. "
                    "Example motion_description: 'ถุงกรองชาวางนิ่งบนเคาน์เตอร์หินอ่อน กล้องค่อยๆ ดันเข้าหา'. "
                    "Tools/equipment rest DIRECTLY on the counter (not in a bowl); only loose ingredients may sit in their own small bowls."
                )
            with obs.generation(
                "gemini.storyboard_breakdown", model=self.model_name,
                input={"scene_id": scene.get("scene_id"), "prompt": prompt},
            ) as gen:
                _t0 = time.perf_counter()
                response = self._gen_with_retry(
                    prompt,
                    {
                        "temperature": 0.3,
                        # the ingredient-introduction scene emits one shot PER ingredient (+ overview) — can be
                        # 12+ shots each with a full prompt_img paragraph, so give it generous room (4096 truncated → 0 shots).
                        "max_output_tokens": max(8192, self.settings.gemini.max_output_tokens),
                        "thinking_config": {"thinking_budget": 0},
                    },
                    label="storyboard_breakdown",
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.storyboard_breakdown", usage, (time.perf_counter() - _t0) * 1000)
            shots = [s for s in _parse_json_list(raw) if isinstance(s, dict)]
            if not shots and raw:
                logger.warning(f"storyboard breakdown scene {scene.get('scene_id')} → 0 shots from {len(raw)} chars "
                               "(likely truncated/invalid JSON — raise gemini.max_output_tokens)")
            return shots

        return await asyncio.to_thread(_run)

    async def complete_intro_narration(self, item_captions: list[str], kind: str, topic: str,
                                       character_desc: str = "") -> dict:
        """Regenerate an intro scene's narration as ONE short line per item (in order) + a closing,
        so every canonical item is spoken (the storyboard splits it 1:1, no silent shots). ``kind`` is
        'ingredient' or 'equipment'. Returns {'voice_over': str, 'on_screen_text': str} (empty on failure)."""
        def _run() -> dict:
            if not item_captions:
                return {}
            prompt = get_prompt("INTRO_NARRATION_PROMPT").format(
                topic=topic,
                kind_th=("วัตถุดิบ" if kind == "ingredient" else "อุปกรณ์"),
                character=(character_desc or "").strip() or "-",
                items=json.dumps(item_captions, ensure_ascii=False),
            )
            with obs.generation("gemini.intro_narration", model=self.model_name,
                                input={"kind": kind, "count": len(item_captions), "prompt": prompt}) as gen:
                _t0 = time.perf_counter()
                response = self._gen_with_retry(
                    prompt,
                    {"temperature": 0.7, "max_output_tokens": 2048, "thinking_config": {"thinking_budget": 0}},
                    label="intro_narration",
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.intro_narration", usage, (time.perf_counter() - _t0) * 1000)
            obj = _parse_json_obj(raw)
            return obj if isinstance(obj, dict) else {}

        return await asyncio.to_thread(_run)

    async def rebalance_intro_vo(self, lines: list[str], anchors: list[str] | None = None) -> list[str] | None:
        """Semantic boundary fix for an intro scene's per-shot voice_over lines: the LLM moves any
        trailing lead-in fragment (announcing the NEXT item — any phrasing) to the start of the next
        line. ``anchors[i]`` = the item shot i shows. Code enforces the invariants the LLM can't be
        trusted with: same line count, concatenated words verbatim-identical, and every line keeps its
        own item's name — otherwise returns None and the caller keeps the deterministic result."""
        def _run() -> list[str] | None:
            if len(lines) < 2 or not any((s or "").strip() for s in lines):
                return None
            prompt = get_prompt("INTRO_VO_REBALANCE_PROMPT").format(
                lines=json.dumps(lines, ensure_ascii=False, indent=1),
                items=json.dumps(list(anchors or []), ensure_ascii=False))
            with obs.generation("gemini.intro_vo_rebalance", model=self.model_name,
                                input={"count": len(lines), "prompt": prompt}) as gen:
                _t0 = time.perf_counter()
                response = self._gen_with_retry(
                    prompt,
                    {"temperature": 0.1, "max_output_tokens": 2048, "thinking_config": {"thinking_budget": 0}},
                    label="intro_vo_rebalance",
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.intro_vo_rebalance", usage, (time.perf_counter() - _t0) * 1000)
            obj = _parse_json_obj(raw)
            out = _validate_rebalanced_lines(lines, obj.get("lines") if isinstance(obj, dict) else None, anchors)
            if out is None:
                logger.warning("intro VO rebalance rejected (count/verbatim/anchor guard) — keeping deterministic split")
            return out

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:  # noqa: BLE001 — fail-open: the deterministic split already ran
            logger.warning(f"intro VO rebalance failed (fail-open): {e}")
            return None

    async def generate_voice_desc(self, character_desc: str = "", vo_tone: str = "") -> str:
        """Synthesize a concise Thai voice-casting descriptor from the character + VO tone — used to
        PIN the narrator voice (ScriptConfig.voice_desc → {voice} in every prompt_video). Returns "" on failure."""
        def _run() -> str:
            if not (character_desc.strip() or vo_tone.strip()):
                return ""
            prompt = get_prompt("VOICE_DESC_PROMPT").format(
                character_desc=(character_desc or "").strip() or "-",
                vo_tone=(vo_tone or "").strip() or "-",
            )
            with obs.generation("gemini.voice_desc", model=self.model_name,
                                input={"prompt": prompt}) as gen:
                _t0 = time.perf_counter()
                response = self._gen_with_retry(
                    prompt,
                    {"temperature": 0.6, "max_output_tokens": 256, "thinking_config": {"thinking_budget": 0}},
                    label="voice_desc",
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.voice_desc", usage, (time.perf_counter() - _t0) * 1000)
            return raw.strip().strip('"').splitlines()[0].strip() if raw.strip() else ""

        return await asyncio.to_thread(_run)

    async def dish_state_timeline(
        self, shots: list[dict], production: dict, topic: str, research_summary: str = ""
    ) -> dict[int, str]:
        """Compute each shot's CURRENT dish/drink state across the WHOLE ordered timeline.

        shots: [{"id": int, "motion_description", "voice_over", "ingredient_refs"}] — the ENTIRE
        tutorial in time order. Seeing the full sequence lets the state progress monotonically (an
        ingredient joins the dish only at the step that adds it), fixing the "finished look leaks into
        an early step" class of error. Returns {id: dish_state} (missing/empty = no dish in that shot).
        """
        def _run() -> dict[int, str]:
            if not shots:
                return {}
            research_block = ""
            if research_summary.strip():
                research_block = ("Authoritative recipe method (use to decide WHEN each ingredient is "
                                  f"added/combined):\n{research_summary.strip()}\n")
            prompt = get_prompt("IMAGE_DISH_STATE_PROMPT").format(
                topic=topic,
                dish=production.get("dish_appearance", "") or "-",
                ingredients=json.dumps(production.get("ingredients", []), ensure_ascii=False),
                research_block=research_block,
                shots=json.dumps(shots, ensure_ascii=False),
            )
            with obs.generation("gemini.dish_state_timeline", model=self.model_name,
                                input={"prompt": prompt}) as gen:
                _t0 = time.perf_counter()
                response = self._gen_with_retry(
                    prompt,
                    {"temperature": 0.2,
                     "max_output_tokens": max(8192, self.settings.gemini.max_output_tokens),
                     "thinking_config": {"thinking_budget": 0}},
                    label="dish_state_timeline",
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.dish_state_timeline", usage, (time.perf_counter() - _t0) * 1000)
            out: dict[int, str] = {}
            for it in _parse_json_list(raw):
                if isinstance(it, dict) and "id" in it:
                    try:
                        out[int(it["id"])] = str(it.get("dish_state", "") or "")
                    except (TypeError, ValueError):
                        continue
            return out

        return await asyncio.to_thread(_run)

    async def generate_image_prompts(
        self, shots: list[dict], production: dict, topic: str, aspect: str = "16:9", category: str = "food",
        extra_ref_notes: list[str] | None = None
    ) -> dict[int, str]:
        """Re-write prompt_img for an already-broken-down scene (the "Image Prompt" step).

        shots: [{"no", "shot_kind", "motion_description", "voice_over", "on_screen_text",
                 "image_subjects", "ingredient_refs"}] — the shot set/text is FIXED.
        Returns {no: prompt_img}. Lets prompts be regenerated without re-running the text breakdown.

        `extra_ref_notes`: what the user said each of their own attached reference photos is. Same
        channel plan_shot_image gets on the Generate path — without it the "↻ Prompt" button rewrote
        the prompt blind to refs the user had just attached, so the new text never mentioned them
        and the render had nothing to hang them on. Applies to EVERY shot in this call, which is
        exact today because the only caller sends one shot at a time (useImagePromptGenApi); a
        whole-board regenerate would need these per shot instead.
        """
        def _run() -> dict[int, str]:
            if not shots:
                return {}
            _notes = [n.strip() for n in (extra_ref_notes or []) if n and n.strip()]
            refs_block = get_prompt("IMAGE_PROMPT_EXTRA_REFS_BLOCK").format(
                notes="\n".join(f"- {n}" for n in _notes)) if _notes else ""
            prompt = get_prompt("IMAGE_PROMPT_GEN_PROMPT").format(
                topic=topic,
                aspect=aspect,
                character=(production.get("character_desc", "") or "").strip(),  # desc only — never the NAME
                theme=production.get("theme", "") or "-",
                lighting=production.get("lighting", "") or "-",
                mood=production.get("mood", "") or "-",
                material_palette=production.get("material_palette", "") or "-",
                dish=production.get("dish_appearance", "") or "-",
                ingredients=json.dumps(production.get("ingredients", []), ensure_ascii=False),
                shots=json.dumps(shots, ensure_ascii=False),
                # Rides on category_rules rather than a new {placeholder}, same as plan_shot_image:
                # a template overridden in the Prompts panel keeps the old placeholder set, and
                # str.format drops unknown kwargs silently, so a new slot would vanish without error.
                category_rules=category_block(category, "image_prompt") + refs_block,
            )
            out: dict[int, str] = {}
            # generous room — a scene can be 12+ shots, each a full prompt_img paragraph.
            for item in self._validate_call("gemini.generate_image_prompts", prompt, max_tokens=max(8192, self.settings.gemini.max_output_tokens)):
                if isinstance(item, dict) and item.get("prompt_img") and (n := _safe_int(item.get("no"))) is not None:
                    out[n] = item["prompt_img"].strip()
            return out

        return await asyncio.to_thread(_run)

    async def classify_shot_image(self, shot: dict, prev_shot: dict | None, topic: str, scenes: list | None = None) -> dict:
        """Classify how to generate ONE shot's image (reuse_prev / use_ref_img / new_generate).
        Returns the classification dict (see IMAGE_CLASSIFY_PROMPT). When `scenes` (configured
        background locations) are given, also asks the LLM to pick the best-fitting one (scene_match)."""
        def _run() -> dict:
            scenes_block = ""
            if scenes:
                lines = "\n".join(
                    f'- id "{s.get("id")}": {s.get("name", "")}' + (f' — {s.get("desc")}' if s.get("desc") else "")
                    for s in scenes if s.get("id")
                )
                scenes_block = (
                    "\nCONFIGURED SCENES (the background locations available — pick the ONE whose purpose best fits "
                    "THIS shot's ACTION/setting):\n" + lines + "\nAlso output:\n"
                    "- scene_match (str): the exact `id` of the single best-fitting scene above. Read each scene's "
                    "description — it says what that place is used for — and match it to what the shot is DOING: "
                    "washing/rinsing → the sink scene; frying/boiling/anything on the stove → the stove scene; "
                    "everything else (talking, presenting, plating, mixing, seasoning, plain prep) → the general "
                    "counter/prep scene.\n"
                    "  Decide by ELIMINATION: if the shot clearly shows no washing, the sink scene is ruled out; if "
                    "it clearly shows no stove work, the stove scene is ruled out — then pick from what remains "
                    "instead of giving up.\n"
                    "  A close-up with no visible background STILL happens somewhere: choose where that action would "
                    'naturally take place. Never answer "" just because the background is not visible.\n'
                    '  Use "" ONLY if the list above has no scene related to this kind of work at all.\n'
                )
            prompt = get_prompt("IMAGE_CLASSIFY_PROMPT").format(
                topic=topic,
                has_prev=bool(prev_shot),
                prev_shot=json.dumps(_shot_for_prompt(prev_shot), ensure_ascii=False),
                shot=json.dumps(_shot_for_prompt(shot), ensure_ascii=False),
                scenes_block=scenes_block,
            )
            with obs.generation(
                "gemini.classify_shot_image", model=self.model_name,
                input={"shot_no": shot.get("no"), "prompt": prompt},
            ) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 1024,
                                       "thinking_config": {"thinking_budget": 0}},
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.classify_shot_image", usage, (time.perf_counter() - _t0) * 1000)
            return _parse_json_obj(raw)

        return await asyncio.to_thread(_run)

    async def extract_kitchen_fixtures(self, image: dict) -> list[str]:
        """Vision: list the fixed fixtures/appliances visible in the kitchen reference image
        (Thai + English terms). Returns [] on failure."""
        def _run() -> list[str]:
            if not image or not image.get("data"):
                return []
            parts = [get_prompt("KITCHEN_FIXTURES_PROMPT"), Part.from_data(data=image["data"], mime_type=image.get("mime", "image/jpeg"))]
            with obs.generation("gemini.extract_kitchen_fixtures", model=self.model_name,
                                input={"prompt": get_prompt("KITCHEN_FIXTURES_PROMPT")}) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    parts,
                    generation_config={"temperature": 0.0, "max_output_tokens": 1024,
                                       "thinking_config": {"thinking_budget": 0}},
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.extract_kitchen_fixtures", usage, (time.perf_counter() - _t0) * 1000)
            return [s for s in _parse_json_list(raw) if isinstance(s, str) and s.strip()]

        return await asyncio.to_thread(_run)

    async def qc_generated_image(self, image: dict, shot_ctx: dict, topic: str) -> dict:
        """C1 (output QC): Gemini LOOKS at a GENERATED shot image and passes/fails it against the
        shot's intent (person-in-insert, missing/extra subjects, anatomy artifacts, burned-in text,
        wrong framing). Returns {"pass": bool, "issues": [str], "fix_hint": str}.

        FAIL-OPEN by design: any API/parse error returns pass=True — QC may only ever reject an
        image it actually saw and judged; it must never block the pipeline on its own failure.
        image: {"mime": str, "data": bytes}; shot_ctx: {"shot_kind","framing","must_show",
        "description","expect_text"}."""
        def _run() -> dict:
            ok = {"pass": True, "issues": [], "fix_hint": ""}
            if not image or not image.get("data"):
                return ok
            must = [s for s in (shot_ctx.get("must_show") or []) if isinstance(s, str) and s.strip()]
            text_rule = (
                "text was EXPECTED in this frame — fail only on garbled/unreadable lettering."
                if shot_ctx.get("expect_text") else
                "ANY readable text, caption, label, number or logo rendered into the image (none was requested)."
            )
            prompt = get_prompt("IMAGE_OUTPUT_QC_PROMPT").format(
                topic=topic,
                shot_kind=shot_ctx.get("shot_kind") or "person",
                framing=shot_ctx.get("framing") or "unspecified",
                must_show=json.dumps(must, ensure_ascii=False) if must else "(none listed)",
                description=(shot_ctx.get("description") or "")[:1200],
                text_rule=text_rule,
            )
            parts = [prompt, Part.from_data(data=image["data"], mime_type=image.get("mime", "image/png"))]
            try:
                with obs.generation("gemini.qc_generated_image", model=self.model_name,
                                    input={"topic": topic, "shot_kind": shot_ctx.get("shot_kind"),
                                           "prompt": prompt}) as gen:
                    _t0 = time.perf_counter()
                    response = self.model.generate_content(
                        parts,
                        generation_config={"temperature": 0.0, "max_output_tokens": 1024,
                                           "thinking_config": {"thinking_budget": 0}},
                    )
                    raw = _extract_text(response).strip()
                    usage = _usage(response)
                    gen.update(output=raw, usage_details=usage)
                    record_usage("gemini.qc_generated_image", usage, (time.perf_counter() - _t0) * 1000)
                verdict = _parse_json_obj(raw)
                if not isinstance(verdict, dict) or "pass" not in verdict:
                    return ok                      # unparseable → fail-open
                return {
                    "pass": bool(verdict.get("pass")),
                    "issues": [str(i) for i in (verdict.get("issues") or []) if str(i).strip()][:6],
                    "fix_hint": str(verdict.get("fix_hint") or "").strip(),
                }
            except Exception as e:  # noqa: BLE001 — QC must never take the pipeline down
                logger.warning(f"image output QC failed (fail-open, keeping image): {e}")
                return ok

        return await asyncio.to_thread(_run)

    async def qc_generated_video(self, video_bytes: bytes, shot_ctx: dict, topic: str) -> dict:
        """Vision QC on a GENERATED Veo clip: Gemini WATCHES the video and fails it on state drift
        (clear water turning into tea), object morph, containment violations, foreign elements/scene
        changes at the tail, burned-in text, broken physics. Returns
        {"pass": bool, "issues": [str], "fix_hint": str} — fix_hint feeds the retry prompt.

        FAIL-OPEN like image QC: any API/parse error returns pass=True — QC may only reject a clip it
        actually watched. shot_ctx: {"shot_kind", "prompt_video"}."""
        def _run() -> dict:
            ok = {"pass": True, "issues": [], "fix_hint": ""}
            if not video_bytes:
                return ok
            prompt = get_prompt("VIDEO_OUTPUT_QC_PROMPT").format(
                topic=topic,
                shot_kind=shot_ctx.get("shot_kind") or "person",
                prompt_video=(shot_ctx.get("prompt_video") or "")[:2000],
            )
            parts = [prompt, Part.from_data(data=video_bytes, mime_type="video/mp4")]
            try:
                with obs.generation("gemini.qc_generated_video", model=self.model_name,
                                    input={"topic": topic, "shot_kind": shot_ctx.get("shot_kind"),
                                           "prompt": prompt, "video": obs.media(video_bytes, "video/mp4")}) as gen:
                    _t0 = time.perf_counter()
                    response = self.model.generate_content(
                        parts,
                        generation_config={"temperature": 0.0, "max_output_tokens": 1024,
                                           "thinking_config": {"thinking_budget": 0}},
                    )
                    raw = _extract_text(response).strip()
                    usage = _usage(response)
                    gen.update(output=raw, usage_details=usage)
                    record_usage("gemini.qc_generated_video", usage, (time.perf_counter() - _t0) * 1000)
                verdict = _parse_json_obj(raw)
                if not isinstance(verdict, dict) or "pass" not in verdict:
                    return ok                      # unparseable → fail-open
                return {
                    "pass": bool(verdict.get("pass")),
                    "issues": [str(i) for i in (verdict.get("issues") or []) if str(i).strip()][:6],
                    "fix_hint": str(verdict.get("fix_hint") or "").strip(),
                }
            except Exception as e:  # noqa: BLE001 — QC must never take the pipeline down
                logger.warning(f"video output QC failed (fail-open, keeping clip): {e}")
                return ok

        return await asyncio.to_thread(_run)

    async def describe_scene_layout(self, image: dict) -> str:
        """Draft a scene's `layout` text from its reference photo (Brand panel → ✨ button).

        `image`: {"data": bytes, "mime": str}. Returns prose for the operator to review and save —
        this never writes to the scene itself. See SCENE_LAYOUT_PROMPT for why it exists."""
        def _run() -> str:
            parts = [get_prompt("SCENE_LAYOUT_PROMPT"),
                     Part.from_data(data=image["data"], mime_type=image.get("mime", "image/png"))]
            with obs.generation("gemini.describe_scene_layout", model=self.model_name,
                                input={"bytes": len(image.get("data") or b"")}) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    parts,
                    generation_config={"temperature": 0.2, "max_output_tokens": 1024,
                                       "thinking_config": {"thinking_budget": 0}},
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.describe_scene_layout", usage, (time.perf_counter() - _t0) * 1000)
            return raw

        return await asyncio.to_thread(_run)

    async def plan_shot_image(self, shot: dict, classify: dict, keyword_pool: list[str], topic: str, aspect: str,
                              category: str = "food", scene_layout: str = "",
                              extra_ref_notes: list[str] | None = None,
                              vessel_from_ref: bool = False) -> dict:
        """Write prompt_img + pick ref_keywords for ONE shot, coupled (see IMAGE_PROMPT_PLAN_PROMPT).
        Returns {"prompt_img": str, "ref_keywords": [str]}.

        `scene_layout`: the chosen scene's geometry (camera angle, what is fore/background, where
        the host belongs). This call is text-only and never sees the scene photo, so without it the
        author had no idea a stove sits in the foreground and wrote prompts that let the model
        place the host beside it. Empty for scenes with no layout written yet — the block is then
        omitted entirely rather than sent blank.

        `extra_ref_notes`: what the user said each of their own attached reference photos is. Runs
        before the refs are resolved and never sees an image, so this text is the only way it can
        know an attachment exists at all — without it the prompt describes a generic bowl while a
        photo of a specific bowl rides along unmentioned.

        `vessel_from_ref`: this shot is NOT an ingredient/equipment introduction, so the containers
        it shows were already photographed and those photos are attached. The author is then told
        to name a container by type and leave its colour and material alone — one it invents can
        only contradict the photo. Off for intro and overview shots: those CREATE the photo, so
        they are exactly where the colour and material have to be written. The caller decides,
        because only it has the classifier's verdict."""
        def _run() -> dict:
            layout = (scene_layout or "").strip()
            layout_block = (
                "\n\nSCENE LAYOUT — the reference photo this shot is generated against, described. "
                "Place the subject and props to match it, and follow its rule for where the host "
                "stands; the image model receives that photo but is never told where anyone belongs:\n"
                f"{layout}\n"
            ) if layout else ""
            _notes = [n.strip() for n in (extra_ref_notes or []) if n and n.strip()]
            refs_block = get_prompt("IMAGE_PROMPT_EXTRA_REFS_BLOCK").format(
                notes="\n".join(f"- {n}" for n in _notes)) if _notes else ""
            vessel_block = get_prompt("IMAGE_PROMPT_VESSEL_FROM_REF_BLOCK") if vessel_from_ref else ""
            # Appended to category_rules rather than given its own {placeholder}: a template the user
            # has overridden in the Prompts panel keeps the old placeholder set, and str.format drops
            # unknown kwargs silently — the layout would vanish with no error. Riding on an existing
            # slot means an override can never lose it.
            prompt = get_prompt("IMAGE_PROMPT_PLAN_PROMPT").format(
                topic=topic, aspect=aspect,
                shot=json.dumps(_shot_for_prompt(shot), ensure_ascii=False),
                classify=json.dumps(classify, ensure_ascii=False),
                keyword_pool=json.dumps(keyword_pool, ensure_ascii=False),
                category_rules=category_block(category, "image_plan") + layout_block + refs_block + vessel_block,
            )
            with obs.generation(
                "gemini.plan_shot_image", model=self.model_name,
                input={"shot_no": shot.get("no"), "prompt": prompt},
            ) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.3, "max_output_tokens": 2048,
                                       "thinking_config": {"thinking_budget": 0}},
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.plan_shot_image", usage, (time.perf_counter() - _t0) * 1000)
            return _parse_json_obj(raw)

        return await asyncio.to_thread(_run)

    async def generate_video_prompt(self, scene: dict, topic: str, duration: int, director_prompt: str = "", mode_block: str = "", style_block: str = "", voice: str = "") -> list[dict]:
        """DEAD (Veo era) — nothing calls this. See VIDEO_PROMPT_PROMPT in prompts.py.

        Write a per-shot VIDEO (motion) prompt for one storyboard scene.

        Returns [{"no": int, "prompt_video": str}, ...] — how each shot's still frame
        moves over a `duration`-second clip (camera + subject action), in English.
        `style_block` is the optional cinematic-craft block (video_gen.cinematic_prompt).
        """
        def _run() -> list[dict]:
            director_block = ""
            if director_prompt.strip():
                director_block = (
                    "\nDIRECTOR'S STYLE GUIDE — apply this pacing and shot style to every motion prompt:\n"
                    f"{director_prompt.strip()}\n"
                )
            prompt = get_prompt("VIDEO_PROMPT_PROMPT").format(
                topic=topic,
                duration=duration,
                director_block=director_block,
                mode_block=("\n" + mode_block.strip() + "\n") if mode_block.strip() else "",
                style_block=("\n" + style_block.strip() + "\n") if style_block.strip() else "",
                voice=voice or "a warm, friendly Thai narrator",
                scene=json.dumps(scene, ensure_ascii=False),
            )
            with obs.generation(
                "gemini.video_prompt", model=self.settings.gemini.model,
                input={"scene_id": scene.get("scene_id"), "prompt": prompt},
            ) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.4,
                        "max_output_tokens": 4096,
                        "thinking_config": {"thinking_budget": 0},
                    },
                )
                raw = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.video_prompt", usage, (time.perf_counter() - _t0) * 1000)
            return [s for s in _parse_json_list(raw) if isinstance(s, dict)]

        return await asyncio.to_thread(_run)

    async def validate_video_prompts(self, shots: list[dict], topic: str, voice: str = "") -> dict[int, dict]:
        """DEAD (Veo era) — reached only from nodes._validate_video_prompts, which nothing calls.

        QA-check each shot's prompt_video and return corrected versions.

        shots: [{"no": int, "shot_kind": str, "voice_over": str, "prompt_img": str, "prompt_video": str}]
        Returns {no: {"valid": bool, "prompt_video": str}}.
        """
        def _run() -> dict[int, dict]:
            if not shots:
                return {}
            prompt = get_prompt("VIDEO_PROMPT_VALIDATE_PROMPT").format(
                topic=topic,
                voice=voice or "a warm, friendly Thai narrator",
                shots=json.dumps(shots, ensure_ascii=False),
            )
            out: dict[int, dict] = {}
            for item in self._validate_call("gemini.validate_video_prompts", prompt, max_tokens=8192):
                if isinstance(item, dict) and (n := _safe_int(item.get("no"))) is not None:
                    out[n] = {
                        "valid": bool(item.get("valid")),
                        "prompt_video": (item.get("prompt_video") or "").strip(),
                    }
            return out

        return await asyncio.to_thread(_run)

    async def normalize_prompt_img_consistency(
        self, shots: list[dict], production: dict, topic: str
    ) -> dict[int, str]:
        """Normalize prompt_img across shots in a scene so character/setting stay consistent.

        shots: [{"no": int, "prompt_img": str}]
        Returns {no: corrected_prompt_img}. Shots already consistent are returned unchanged.
        """
        def _run() -> dict[int, str]:
            if not shots:
                return {}
            prompt = get_prompt("PROMPT_IMG_CONSISTENCY_PROMPT").format(
                character=(production.get("character_desc", "") or "").strip(),  # desc only — never the NAME (Veo blocks named people in visual prompts)
                theme=production.get("theme", "") or "-",
                lighting=production.get("lighting", "") or "-",
                shots=json.dumps(shots, ensure_ascii=False),
            )
            out: dict[int, str] = {}
            for item in self._validate_call("gemini.normalize_prompt_img", prompt, max_tokens=8192):
                if isinstance(item, dict) and item.get("prompt_img") and (n := _safe_int(item.get("no"))) is not None:
                    out[n] = item["prompt_img"].strip()
            return out

        return await asyncio.to_thread(_run)

    def _get_genai_client(self):
        """Lazily build the google-genai client used for Veo. veo-3.1-lite lives on the
        Gemini Developer API (api_key), not Vertex — so default to that; `vertex` is
        available for models that exist in this project's Vertex Model Garden."""
        if self._genai_client is None:
            from google import genai

            if self.settings.video_gen.provider == "vertex":
                self._genai_client = genai.Client(
                    vertexai=True, project=self._project, location=self._location,
                    credentials=self._credentials,
                )
                logger.info(f"Veo (Vertex) client initialized ({self._project}/{self._location})")
            else:
                key = os.getenv("GEMINI_API_KEY", "").strip()
                if not key:
                    raise RuntimeError("GEMINI_API_KEY is not set (needed for video_gen.provider='gemini_api')")
                self._genai_client = genai.Client(api_key=key)
                logger.info("Veo (Gemini Developer API) client initialized")
        return self._genai_client

    def _strip_person_names(self, prompt: str) -> str:
        """Rewrite a video prompt to drop any real person's proper NAME / celebrity reference
        (Veo blocks named people in the visual), keeping action, camera, setting and quoted
        dialogue intact. Sync (runs inside the video _run thread). Falls back to the original."""
        try:
            instr = (
                "Rewrite this video prompt to REMOVE any real person's proper NAME or celebrity "
                "reference. Describe the person with a GENERIC visual descriptor instead (e.g. "
                "'a Thai woman cooking host in her 30s, warm and friendly'). Keep the camera, action, "
                "setting, and any QUOTED spoken dialogue EXACTLY as-is. Return ONLY the rewritten prompt.\n\n"
                f"PROMPT:\n{prompt}"
            )
            _t0 = time.perf_counter()
            resp = self.model.generate_content(instr)
            record_usage("gemini.strip_person_names", _usage(resp), (time.perf_counter() - _t0) * 1000)
            out = (_extract_text(resp) or "").strip()
            return out or prompt
        except Exception as e:  # noqa: BLE001
            logger.warning(f"name-strip rewrite failed, using original prompt: {e}")
            return prompt

    async def generate_video(self, prompt: str, image: dict | None = None,
                             last_image: dict | None = None, *, cfg=None) -> bytes | None:
        """Generate ONE video clip with Veo (long-running op). Blocking + slow — runs
        in a thread; the caller streams a heartbeat while it renders.

        image / last_image: {"mime": str, "data": bytes} — first frame (image-to-video)
        and optional last frame (for seamless clip-to-clip continuity; Veo only honours
        last_frame when a first-frame image is also given). `cfg` is an effective
        VideoGenConfig (request overrides merged); defaults to settings.video_gen.
        Returns MP4 bytes (inline from Vertex, else downloaded from the result GCS URI).
        RAISES on failure so the caller can surface the reason.
        """
        cfg = cfg or self.settings.video_gen
        if _is_omni(cfg.model):
            # Omni: single-clip only, Interactions API. last_image (interpolation) not supported.
            return await self._generate_video_omni(prompt, image, cfg=cfg)

        def _run() -> bytes | None:
            if not prompt.strip():
                return None
            from google.genai import types

            client = self._get_genai_client()
            # Veo bills by video seconds, not tokens — record the CALL + wall time (usage=None)
            # so step-7 accounting still shows what Veo cost in calls/time.
            _t_veo = time.perf_counter()

            # Per the Gemini API docs, pass prompt/image as TOP-LEVEL args (not via a
            # GenerateVideosSource — that shape is Vertex-only and 400s as "use case
            # not supported" on the Developer API).
            kwargs: dict = {"model": cfg.model, "prompt": prompt}
            if image and image.get("data"):
                kwargs["image"] = types.Image(
                    image_bytes=image["data"], mime_type=image.get("mime", "image/png")
                )
            cfg_kwargs: dict = dict(
                resolution=cfg.resolution,
                duration_seconds=cfg.duration_seconds,
                aspect_ratio=cfg.aspect_ratio,
                number_of_videos=1,
            )
            if getattr(cfg, "negative_prompt", ""):
                cfg_kwargs["negative_prompt"] = cfg.negative_prompt
            # person_generation by MODE (Veo rules): image-to-video / interpolation accept only
            # "allow_adult"; text-to-video accepts only "allow_all". Sending the wrong one 400s.
            # An explicit "dont_allow" is respected (no people in frame).
            if getattr(cfg, "person_generation", "") == "dont_allow":
                cfg_kwargs["person_generation"] = "dont_allow"
            else:
                cfg_kwargs["person_generation"] = "allow_adult" if image and image.get("data") else "allow_all"
            # last_frame (first+last-frame interpolation) — needs a first-frame image too,
            # and is only offered by the STANDARD model (fast/lite 400 "use case not
            # supported"), so switch this shot to interpolation_model.
            if last_image and last_image.get("data") and "image" in kwargs:
                cfg_kwargs["last_frame"] = types.Image(
                    image_bytes=last_image["data"], mime_type=last_image.get("mime", "image/png")
                )
                if getattr(cfg, "interpolation_model", ""):
                    kwargs["model"] = cfg.interpolation_model
                # Interpolation requires the standard model AND duration_seconds=8 (the
                # Gemini API rejects other durations for this use case → "not supported").
                cfg_kwargs["duration_seconds"] = 8
            # generate_audio + output_gcs_uri are Vertex/Enterprise-only — the Gemini
            # Developer API rejects them (audio is on by default there).
            if cfg.provider == "vertex":
                cfg_kwargs["generate_audio"] = cfg.generate_audio
                if cfg.output_gcs_uri:
                    cfg_kwargs["output_gcs_uri"] = cfg.output_gcs_uri
            gen_config = types.GenerateVideosConfig(**cfg_kwargs)
            # Full request captured for Langfuse + terminal (image bytes → bool, no raw bytes).
            req_input = {
                "model": kwargs["model"],
                "provider": cfg.provider,
                "prompt": prompt,
                "image": bool(kwargs.get("image")),
                "last_image": bool(last_image and last_image.get("data")),
                "first_frame_media": obs.media(image.get("data") if image else None, image.get("mime", "image/png") if image else "image/png"),
                "last_frame_media": obs.media(last_image.get("data") if last_image else None, last_image.get("mime", "image/png") if last_image else "image/png"),
                "config": {**{k: v for k, v in cfg_kwargs.items() if k != "last_frame"},
                           "last_frame": bool(cfg_kwargs.get("last_frame"))},
            }
            logger.info("Veo request → {}", {**req_input, "prompt": prompt[:200]})
            logger.info("Veo prompt (full) → {}", prompt)
            with obs.generation(
                "veo.generate_video", model=kwargs["model"], input=req_input,
            ) as gen:
                # Veo drops a clip on the safety/audio (RAI) filter probabilistically — retry
                # the whole generation a few times before giving up (rai_retries).
                retries = max(1, getattr(cfg, "rai_retries", 1))
                name_rewritten = False
                audio_stage = 0     # audio-filter ladder: 0=lip-sync · 1=host-speak · 2=host-speak+standard model
                try:
                    attempt, max_attempts = 0, retries
                    while attempt < max_attempts:
                        attempt += 1
                        op = client.models.generate_videos(config=gen_config, **kwargs)
                        while not op.done:
                            time.sleep(max(2, cfg.poll_interval_seconds))
                            op = _net_retry(lambda: client.operations.get(op))   # survive a transient poll blip

                        result = getattr(op, "response", None) or getattr(op, "result", None)
                        reasons = _veo_reasons(result, op)
                        if reasons:
                            err = getattr(op, "error", None)
                            logger.warning(f"Veo attempt {attempt}/{max_attempts} produced no video: {'; '.join(reasons)}")
                            # safety net: a real-name/celebrity filter is deterministic — rewrite the
                            # prompt once to drop the person's name (kept in dialogue), then retry.
                            if _is_name_filter(reasons) and not name_rewritten:
                                name_rewritten = True
                                new_prompt = self._strip_person_names(kwargs["prompt"])
                                if new_prompt and new_prompt != kwargs["prompt"]:
                                    logger.warning("Veo name filter → stripped person names from prompt, retrying")
                                    kwargs["prompt"] = new_prompt
                                    max_attempts += 1   # bonus attempt for the rewritten prompt
                                    _retry_backoff(cfg, attempt, reasons)
                                    continue
                            # audio filter → walk a LADDER that keeps the host SPEAKING at every rung:
                            # lip-sync → host-speak (drop the lip-sync clause) → host-speak on the standard model.
                            # The filter is stochastic, so each rung also just earns more retries within budget.
                            if _is_audio_filter(reasons):
                                if audio_stage == 0:
                                    audio_stage = 1
                                    new_prompt = _strip_lipsync(kwargs["prompt"])
                                    if new_prompt != kwargs["prompt"]:
                                        logger.warning("Veo audio filter → host-speak (dropped lip-sync clause), retrying")
                                        kwargs["prompt"] = new_prompt
                                        max_attempts += 1
                                elif audio_stage == 1 and getattr(cfg, "interpolation_model", "") \
                                        and kwargs["model"] != cfg.interpolation_model:
                                    audio_stage = 2
                                    logger.warning(f"Veo audio filter → swapping to standard model {cfg.interpolation_model}, retrying")
                                    kwargs["model"] = cfg.interpolation_model
                                    max_attempts += 1
                                # else: already host-speak on the standard model → keep retrying within budget
                            if attempt < max_attempts:
                                _retry_backoff(cfg, attempt, reasons)
                                continue
                            gen.update(output={"filtered": True, "rai_filtered_reasons": reasons,
                                               "op_error": str(err) if err else None})
                            raise RuntimeError(f"Veo filtered the video (safety): {'; '.join(reasons)}")

                        video = result.generated_videos[0].video
                        data = _net_retry(lambda: _veo_video_bytes(client, video, self._project, self._credentials))
                        if not data:
                            raise RuntimeError("Veo produced no retrievable video bytes")
                        gen.update(output={"filtered": False, "bytes": len(data),
                                           "duration_seconds": cfg_kwargs.get("duration_seconds"),
                                           "video_uri": getattr(video, "uri", None),
                                           "video": obs.media(data, "video/mp4")})
                        return data
                except Exception as e:
                    logger.warning(f"Video generation failed: {e}")
                    gen.update(level="ERROR", status_message=str(e))
                    raise
                finally:
                    record_usage("veo.generate_video", None, (time.perf_counter() - _t_veo) * 1000)

        return await asyncio.to_thread(_run)

    async def _generate_video_omni(self, prompt: str, image: dict | None, *, cfg) -> bytes | None:
        """Generate ONE clip with Gemini Omni Flash via the Interactions API (blocking).
        Omni ignores Veo-only knobs (duration/person_generation/negative_prompt/last-frame);
        only prompt + first-frame image + aspect_ratio apply. Returns MP4 bytes. RAISES on failure."""
        import base64

        def _run() -> bytes | None:
            if not prompt.strip():
                return None
            client = self._get_genai_client()
            _t = time.perf_counter()
            if image and image.get("data"):
                inp = [
                    {"type": "image",
                     "data": base64.b64encode(image["data"]).decode(),
                     "mime_type": image.get("mime", "image/png")},
                    {"type": "text", "text": prompt},
                ]
            else:
                inp = prompt
            logger.info("Gemini Omni request → model={} image={} prompt={}", cfg.model, bool(image and image.get("data")), prompt[:200])
            logger.info("Gemini Omni prompt (full) → {}", prompt)
            with obs.generation("omni.generate_video", model=cfg.model,
                                input={"model": cfg.model, "image": bool(image and image.get("data")),
                                       "aspect_ratio": cfg.aspect_ratio, "prompt": prompt}) as gen:
                try:
                    interaction = client.interactions.create(
                        model=cfg.model, input=inp,
                        response_format={"type": "video", "aspect_ratio": cfg.aspect_ratio},
                    )
                    ov = getattr(interaction, "output_video", None)
                    if ov is None:
                        raise RuntimeError(f"Omni returned no video (status={getattr(interaction, 'status', '?')})")
                    if getattr(ov, "data", None):
                        data = base64.b64decode(ov.data)
                    elif getattr(ov, "uri", None):
                        r = httpx.get(ov.uri, timeout=180, follow_redirects=True)
                        r.raise_for_status()
                        data = r.content
                    else:
                        raise RuntimeError("Omni video had neither inline data nor a uri")
                    if not data:
                        raise RuntimeError("Omni produced no video bytes")
                    gen.update(output={"bytes": len(data), "video": obs.media(data, "video/mp4")})
                    return data
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Omni video generation failed: {e}")
                    gen.update(level="ERROR", status_message=str(e))
                    raise
                finally:
                    record_usage("omni.generate_video", None, (time.perf_counter() - _t) * 1000)

        return await asyncio.to_thread(_run)

    async def generate_video_omni_multi(self, prompt: str, images: list[dict], *, cfg=None) -> bytes | None:
        """Generate ONE clip with Gemini Omni Flash from a prompt + ORDERED role-tagged reference
        images (Interactions API) — the multi-ref sibling of `_generate_video_omni` (which only
        takes a single first-frame image). `images`: [{"mime","data","role","label"}], role is
        "FIRST_FRAME" or "IMAGE_REF_N" (see `_omni_manifest_text`); sent in list order so the
        prompt's <FIRST_FRAME>/<IMAGE_REF_k> tags line up with what the model actually receives.
        A manifest block (built from `images`) is prepended to `prompt` so the model knows what
        each attached image is for. Returns MP4 bytes. RAISES on failure."""
        import base64

        cfg = cfg or self.settings.video_gen

        def _run() -> bytes | None:
            if not prompt.strip():
                return None
            client = self._get_genai_client()
            _t = time.perf_counter()
            # The Step5-authored prompt already carries its own [# Sources]/[# References] header +
            # inline tags (POC style) — don't prepend a second manifest in that case.
            if images and "[# Sources" not in prompt[:400]:
                full_prompt = f"{_omni_manifest_text(images)}\n\n{prompt}"
            else:
                full_prompt = prompt
            inp: list = []
            for im in images:
                data = im.get("data")
                if not data:
                    continue
                b64 = data if isinstance(data, str) else base64.b64encode(data).decode()
                inp.append({"type": "image", "data": b64, "mime_type": im.get("mime", "image/png")})
            inp.append({"type": "text", "text": full_prompt})
            # Tell Omni WHICH task this is — without it the model treats an attached FIRST_FRAME as a
            # loose style ref and re-invents the opening shot, so concatenated clips jump at the join.
            has_first = any((im.get("role") or "").upper() == "FIRST_FRAME" for im in images)
            n_img = len(inp) - 1
            # image_to_video accepts EXACTLY 1 image (Omni rejects >1 with a 400). With a first frame
            # PLUS person/scene refs, that's multi-image → reference_to_video (the FIRST_FRAME tag in
            # the prompt still drives the opening). Only a lone first frame uses image_to_video.
            if n_img <= 0:
                task = "text_to_video"
            elif n_img == 1:
                task = "image_to_video" if has_first else "reference_to_video"
            else:
                task = "reference_to_video"
            logger.info("Omni multi request → model={} images={} task={} prompt={}", cfg.model, n_img, task, full_prompt[:200])
            logger.info("Omni multi prompt (full) → {}", full_prompt)
            with obs.generation("omni.generate_video_multi", model=cfg.model,
                                input={"model": cfg.model, "images": n_img, "task": task,
                                       "aspect_ratio": cfg.aspect_ratio, "prompt": full_prompt,
                                       "refs": [{"role": im.get("role"), "kind": im.get("kind"),
                                                 "label": im.get("label"), "url": im.get("url")} for im in images]}) as gen:
                try:
                    _kwargs = dict(model=cfg.model, input=inp,
                                   response_format={"type": "video", "aspect_ratio": cfg.aspect_ratio})
                    try:
                        interaction = client.interactions.create(
                            **_kwargs, generation_config={"video_config": {"task": task}})
                    except Exception as e_task:  # noqa: BLE001
                        # video_config isn't in this SDK's typed params — if the backend rejects it,
                        # fall back to the plain call rather than failing the whole render. Surface the
                        # transient rejection on the trace (retry succeeds, so it wouldn't show otherwise).
                        logger.warning(f"Omni: video_config.task={task} rejected ({e_task}) → retry without it")
                        gen.update(level="WARNING", status_message=f"task={task} rejected → retried plain",
                                   metadata={"task_rejected": str(e_task)})
                        interaction = client.interactions.create(**_kwargs)
                    ov = getattr(interaction, "output_video", None)
                    if ov is None:
                        raise RuntimeError(f"Omni returned no video (status={getattr(interaction, 'status', '?')})")
                    if getattr(ov, "data", None):
                        vid = base64.b64decode(ov.data)
                    elif getattr(ov, "uri", None):
                        r = httpx.get(ov.uri, timeout=180, follow_redirects=True)
                        r.raise_for_status()
                        vid = r.content
                    else:
                        raise RuntimeError("Omni video had neither inline data nor a uri")
                    if not vid:
                        raise RuntimeError("Omni produced no video bytes")
                    gen.update(output={"bytes": len(vid), "video": obs.media(vid, "video/mp4")})
                    return vid
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Omni multi video generation failed: {e}")
                    gen.update(level="ERROR", status_message=str(e))
                    raise
                finally:
                    record_usage("omni.generate_video_multi", None, (time.perf_counter() - _t) * 1000)

        return await asyncio.to_thread(_run)

    # Extra rule appended for the region-edit modes — the author has to know the change is confined
    # (mask) or additive (outpaint), or it writes a whole-image instruction that fights the mask.
    _IMAGE_EDIT_MODE_RULES = {
        "mask": "- A MASK is attached: the change happens ONLY inside the masked area. Everything outside it —\n"
                "  pose, hands, background, lighting, framing — stays IDENTICAL. Say so explicitly.\n",
        "outpaint": "- This is an OUTPAINT: the frame is being extended and the new border area must be filled in.\n"
                    "  Describe how the scene continues outward with the same style, lighting and perspective;\n"
                    "  the ORIGINAL area stays exactly as it is. Do not describe changes inside it.\n",
    }

    async def generate_image_edit_prompt(self, instruction: str, *, original_desc: str = "",
                                         ref_notes: list[str] | None = None, mode: str = "") -> str:
        """Polish a rough image-edit note into ONE clear prompt (see IMAGE_EDIT_PROMPT_PROMPT).
        Best-effort: returns the original instruction unchanged on any failure.

        `ref_notes` describes the extra images attached to this edit, in send order. They are
        numbered from 2 because image 1 is always the current image being edited — matching the role
        list the provider wrapper builds, so a prompt saying "image 2" points at the real image 2."""
        def _run() -> str:
            notes = ref_notes or []
            refs_txt = "\n".join(
                f"- image {i + 2}: {(n or '').strip() or '(no description given)'}" for i, n in enumerate(notes)
            ) or "- (no extra reference images)"
            prompt = get_prompt("IMAGE_EDIT_PROMPT_PROMPT").format(
                instruction=instruction,
                original_desc=(original_desc or "").strip() or "(not described)",
                refs_manifest=refs_txt,
                mode_rule=self._IMAGE_EDIT_MODE_RULES.get(mode, ""),
            )
            with obs.generation("gemini.image_edit_prompt", model=self.settings.gemini.model,
                                input={"instruction": instruction, "refs": refs_txt}) as gen:
                _t0 = time.perf_counter()
                try:
                    response = self.model.generate_content(
                        prompt, generation_config={"temperature": 0.3, "max_output_tokens": 1024,
                                                    "thinking_config": {"thinking_budget": 0}})
                    out = (_extract_text(response) or "").strip()
                    usage = _usage(response)
                    gen.update(output=out, usage_details=usage)
                    record_usage("gemini.image_edit_prompt", usage, (time.perf_counter() - _t0) * 1000)
                    return out or instruction
                except Exception as e:  # noqa: BLE001 — polish is best-effort, never blocks the edit
                    logger.warning(f"image edit prompt polish failed, using original instruction: {e}")
                    return instruction

        return await asyncio.to_thread(_run)

    async def revise_image_prompt(self, full_prompt: str, feedback: str) -> str:
        """Rewrite the FULL image prompt to fix what the user says was wrong with the render it
        produced (see IMAGE_PROMPT_REVISE_PROMPT). Best-effort: returns `full_prompt` untouched on
        any failure, so a bad call can never leave the shot with a broken prompt.

        Works on the assembled full prompt rather than the shot's `prompt_img` on purpose. With
        `image_gen.dynamic_image` on, a render re-plans prompt_img from the shot and only falls back
        to the stored string, so a fix written there may never reach the image — whereas the full
        prompt goes to the provider verbatim as `prompt_override`.

        `max_output_tokens` is generous because the answer is the WHOLE prompt, not a diff: a full
        prompt with a ref manifest runs long, and truncating it would silently drop the rules at the
        end (which is exactly where the text/negative rules live)."""
        def _run() -> str:
            prompt = get_prompt("IMAGE_PROMPT_REVISE_PROMPT").format(
                full_prompt=full_prompt, feedback=feedback)
            with obs.generation("gemini.revise_image_prompt", model=self.settings.gemini.model,
                                input={"feedback": feedback, "chars_in": len(full_prompt)}) as gen:
                _t0 = time.perf_counter()
                try:
                    response = self.model.generate_content(
                        prompt, generation_config={"temperature": 0.2,
                                                   "max_output_tokens": max(8192, self.settings.gemini.max_output_tokens),
                                                   "thinking_config": {"thinking_budget": 0}})
                    out = (_extract_text(response) or "").strip()
                    usage = _usage(response)
                    gen.update(output=out, usage_details=usage)
                    record_usage("gemini.revise_image_prompt", usage, (time.perf_counter() - _t0) * 1000)
                    return out or full_prompt
                except Exception as e:  # noqa: BLE001 — never leave the caller without a usable prompt
                    logger.warning(f"image prompt revise failed, keeping the original: {e}")
                    return full_prompt

        return await asyncio.to_thread(_run)

    async def generate_omni_edit_prompt(self, instruction: str, refs: list[dict] | None = None) -> str:
        """Polish a user's rough edit note into ONE short Omni edit instruction (golden rule: short
        beats long for conversational video edit — see OMNI_EDIT_PROMPT_PROMPT). Best-effort: on any
        failure, returns the user's original instruction unchanged rather than blocking the edit.

        `refs` describes the images attached to THIS turn, in send order, so the instruction can cite
        <IMAGE_REF_k> by what each image is. Note these tags number the EDIT call's own attachments —
        unrelated to the generate path's manifest, which sends person/scene/first-frame instead."""
        def _run() -> str:
            refs_list = refs or []
            refs_txt = "\n".join(
                f"- <IMAGE_REF_{i}>: {(r.get('description') or '').strip() or '(no description given)'}"
                for i, r in enumerate(refs_list)) or "(none)"
            prompt = get_prompt("OMNI_EDIT_PROMPT_PROMPT").format(instruction=instruction, refs_manifest=refs_txt)
            with obs.generation("gemini.omni_edit_prompt", model=self.settings.gemini.model,
                                input={"instruction": instruction, "refs": refs_txt}) as gen:
                _t0 = time.perf_counter()
                try:
                    response = self.model.generate_content(
                        prompt, generation_config={"temperature": 0.3, "max_output_tokens": 512,
                                                    "thinking_config": {"thinking_budget": 0}})
                    out = (_extract_text(response) or "").strip()
                    usage = _usage(response)
                    gen.update(output=out, usage_details=usage)
                    record_usage("gemini.omni_edit_prompt", usage, (time.perf_counter() - _t0) * 1000)
                    return out or instruction
                except Exception as e:  # noqa: BLE001 — polish is best-effort, never blocks the edit
                    logger.warning(f"omni edit prompt polish failed, using original instruction: {e}")
                    return instruction

        return await asyncio.to_thread(_run)

    async def generate_omni_adapt_prompt(self, veo_prompt: str) -> str:
        """Auto-adapt a Step5-authored Veo-style structured prompt_video into Omni's plain-prose
        style — called right before rendering a shot with engine="omni" (see /steps/video and
        /steps/video_group), so a storyboard can mix Veo/Omni shots per-shot without needing a
        separate "which engine" setting back at Step5 (the prompt is always authored Veo-style;
        THIS is what makes it fit Omni instead, on demand). Best-effort: on any failure, returns the
        original Veo-style prompt unchanged — Omni can generally still parse it, just less natively,
        so a failed adapt should never block the render."""
        def _run() -> str:
            prompt = get_prompt("OMNI_ADAPT_PROMPT_PROMPT").format(veo_prompt=veo_prompt)
            with obs.generation("gemini.omni_adapt_prompt", model=self.settings.gemini.model,
                                input={"veo_prompt": veo_prompt}) as gen:
                _t0 = time.perf_counter()
                try:
                    response = self.model.generate_content(
                        prompt, generation_config={"temperature": 0.3, "max_output_tokens": 2048,
                                                    "thinking_config": {"thinking_budget": 0}})
                    out = (_extract_text(response) or "").strip()
                    usage = _usage(response)
                    gen.update(output=out, usage_details=usage)
                    record_usage("gemini.omni_adapt_prompt", usage, (time.perf_counter() - _t0) * 1000)
                    return out or veo_prompt
                except Exception as e:  # noqa: BLE001 — best-effort, never blocks the render
                    logger.warning(f"omni prompt adapt failed, using original Veo-style prompt: {e}")
                    return veo_prompt

        return await asyncio.to_thread(_run)

    @staticmethod
    def _opening_frame_text(prev_context: dict | None) -> str:
        """What the author should assume is already on screen at [0s].

        On a `cut` the clip opens on the shot's OWN still, which `prompt_img` already describes —
        say so, rather than leaving the slot blank, because a blank invites the author to guess.

        On `continuous`/`match_cut` the opening frame is the PREVIOUS clip's last frame, which the
        author has no other way to know: it sees only the manifest's role line ("the previous clip's
        last frame"), never the picture. Left uninformed it fills the gap from this shot's own
        prompt_img and writes "her hand is already pouring" over a frame that has no hand in it —
        which is the appearance defect, arriving in the very first second."""
        if not prev_context:
            return ("this shot's own still, exactly as the base image prompt above describes it — "
                    "whatever it names is on screen at [0s].")
        prev = (prev_context.get("prompt_video") or "").strip()
        motion = (prev_context.get("motion") or "").strip()
        # prompt_video is the fuller account and ends where this clip starts; motion is the short
        # human line and survives even when the previous shot's prompt generation failed.
        body = "\n".join(x for x in (f"previous shot's motion: {motion}" if motion else "",
                                     f"previous shot's video prompt: {prev}" if prev else "") if x)
        return ("the LAST FRAME of the PREVIOUS shot — you did not compose it. It shows wherever "
                "the previous shot ENDED, described here:\n" + (body or "(no description available)"))

    async def generate_omni_prompt(self, fields: dict, aspect_ratio: str, *, voice: dict | None = None,
                                   image_manifest: str = "", source_header: str = "", usage_block: str = "",
                                   prev_context: dict | None = None) -> tuple[str, float]:
        """Author ONE Omni-native video prompt directly (POC v2 approach) — the LLM writes the VISUAL
        half only, then the on-screen-text / narration / duration blocks are appended VERBATIM in code
        so those contracts can never be paraphrased away. Replaces the Veo-author→Omni-adapt round-trip.

        Returns `(prompt_video, target_seconds)`. Omni's API has NO duration parameter, so clip length
        is decided purely by what the prompt says; a fixed 4-10s range on its own made the model drift
        long, so the author now also estimates this shot's own length from its script and that number
        is written into the duration block. `target_seconds` is 0.0 when the model gave nothing usable,
        which the caller renders as the old generic wording.

        `fields`: {prompt_img, motion_description, voice_over, on_screen_text, key_message, time}.
        `voice`: {language, gender, vo_pace, tone, style}.

        `prev_context`: {prompt_video, motion} of the shot before this one — pass it ONLY when this
        shot's join_with_prev is continuous/match_cut, because only then does the clip open on that
        shot's last frame instead of its own still. See `_opening_frame_text`."""
        vc = voice or {}

        def _run() -> tuple[str, float]:
            prompt = get_prompt("OMNI_VIDEO_PROMPT_V2_PROMPT").format(
                image_manifest=(image_manifest.strip() or "(no reference images for this shot)"),
                prompt_img=fields.get("prompt_img", ""),
                motion=fields.get("motion_description", ""),
                opening_frame=self._opening_frame_text(prev_context),
                voice_over=fields.get("voice_over", ""),
                key_message=fields.get("key_message", ""),
                aspect_ratio=aspect_ratio or "16:9",
                # The default template now sizes the clip by characters, not words, so {vo_pace} is
                # unused here — still passed so an override saved against the older word-count
                # wording keeps formatting instead of raising.
                vo_pace=vc.get("vo_pace") or 2.5,
                chars_per_second=NARRATION_CHARS_PER_SECOND,
            )
            with obs.generation("gemini.omni_prompt", model=self.settings.gemini.model,
                                input={"prompt": prompt}) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    prompt, generation_config={"temperature": 0.5, "max_output_tokens": 8192,
                                                "thinking_config": {"thinking_budget": 0}})
                raw = (_extract_text(response) or "").strip()
                usage = _usage(response)
                gen.update(output=raw, usage_details=usage)
                record_usage("gemini.omni_prompt", usage, (time.perf_counter() - _t0) * 1000)
            # The author returns {"target_seconds", "prompt"}. Anything unparseable falls back to
            # treating the whole response as the prompt with no estimate — byte-identical to the
            # behaviour before the JSON contract existed, so a template override still on the old
            # prose format keeps working instead of producing an empty prompt.
            target_seconds = 0.0
            body = _strip_fences(raw)
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and isinstance(parsed.get("prompt"), str) and parsed["prompt"].strip():
                    raw = parsed["prompt"].strip()
                    target_seconds = _clamp_clip_seconds(parsed.get("target_seconds"))
                else:
                    logger.warning("omni_prompt: JSON had no usable 'prompt' — using the raw response")
            except (json.JSONDecodeError, TypeError):
                logger.warning("omni_prompt: response was not JSON — using it as the prompt verbatim")
            # Deterministically fix any tag-index drift (e.g. <IMAGE_REF_3>@Image3 → <IMAGE_REF_2>@Image3)
            # against the authoritative source_header, so body tags always match what's actually sent.
            fixed = _normalize_omni_tags(raw, source_header)
            if fixed != raw:
                logger.info("omni_prompt: normalized mismatched image-ref tags to match the manifest")
                raw = fixed
            # final = source header (POC @Image binding) → deterministic per-ref lock block → LLM body
            # (inline tags) → fixed blocks. The lock block guarantees each ref's rule even if the LLM
            # dropped it (e.g. omitting the person when only a hand is in frame).
            vo = (fields.get("voice_over") or "").strip()
            ost = (fields.get("on_screen_text") or "").strip()
            # The template says the clip must never be shorter than its narration, and a measured
            # round still had it short on a third of shots — so the floor is enforced here rather
            # than trusted, the same way the adherence block and @ImageN tags are.
            need = estimate_narration_seconds(vo)
            if need:
                target_seconds = _clamp_clip_seconds(max(target_seconds, need + 0.5))
                if need + 0.5 > CLIP_SECONDS_MAX:
                    logger.warning(f"omni_prompt: narration needs {need:.1f}s, past Omni's "
                                   f"{CLIP_SECONDS_MAX}s ceiling — this shot wants splitting, not a longer clip")
            # adherence guard: strip any LLM-written (often truncated) variant, then append the full
            # sentence verbatim — the only "generate nothing beyond the script" contract Omni gets.
            raw = _OMNI_ADHERENCE_RE.sub("", raw).strip() + "\n" + get_prompt("OMNI_ADHERENCE_BLOCK")
            parts = ([source_header.strip()] if source_header.strip() else [])
            if usage_block.strip():
                parts.append(usage_block.strip())
            parts.append(raw)
            parts.append(get_prompt("OMNI_ONSCREEN_BLOCK").format(on_screen_text=ost) if ost
                         else get_prompt("OMNI_ONSCREEN_NONE_BLOCK"))
            if vo:
                parts.append(get_prompt("OMNI_VOICE_BLOCK").format(
                    voice_over=_apply_reading_rules(vo),
                    language=(vc.get("language") or "Thai").strip(),
                    vo_pace=vc.get("vo_pace") or 2.5,
                    tone=(vc.get("tone") or "").strip(),
                    style=(vc.get("style") or "").strip(),
                    gender=_voice_gender(vc.get("gender")),
                ))
            # `.format` is called unconditionally: an override still on the placeholder-free text
            # simply ignores the kwarg, so an old override degrades to the previous wording instead
            # of raising.
            parts.append(get_prompt("OMNI_DURATION_BLOCK" if vo else "OMNI_DURATION_NO_VO_BLOCK").format(
                target_length=(f"Target length: about {target_seconds:g} seconds.\n\n" if target_seconds else "")
            ))
            return "\n\n".join(parts), target_seconds

        return await asyncio.to_thread(_run)

    async def edit_omni_video(self, prompt: str, *, video: dict | None = None, refs: list[dict] | None = None,
                              previous_interaction_id: str | None = None, cfg=None) -> tuple[bytes, str]:
        """Conversational video edit via Omni's Interactions API. Turn 1 (no `previous_interaction_id`)
        sends the base clip (`video`) + any reference images (`refs`) + the edit instruction; every
        later turn sends `previous_interaction_id` instead — Omni already has the clip and prior
        conversation, so re-sending it would be wasted bandwidth and could reset context. `refs` (new
        reference images for THIS turn only) may still be attached on a continuation turn.

        Returns (mp4_bytes, interaction_id) — the caller persists `interaction_id` on the shot (see
        StoryboardRepo.set_omni_interaction_id) to keep the conversation open for the NEXT edit.
        RAISES on failure (no video / no interaction id / Omni error)."""
        import base64

        cfg = cfg or self.settings.video_gen
        if not previous_interaction_id and not (video and video.get("data")):
            raise ValueError("edit_omni_video: first turn needs a base `video` (no previous_interaction_id to continue)")

        def _run() -> tuple[bytes, str]:
            client = self._get_genai_client()
            _t = time.perf_counter()
            inp: list = []
            if not previous_interaction_id and video and video.get("data"):
                vdata = video["data"]
                b64 = vdata if isinstance(vdata, str) else base64.b64encode(vdata).decode()
                inp.append({"type": "video", "data": b64, "mime_type": video.get("mime", "video/mp4")})
            for r in (refs or []):
                data = r.get("data")
                if not data:
                    continue
                b64 = data if isinstance(data, str) else base64.b64encode(data).decode()
                inp.append({"type": "image", "data": b64, "mime_type": r.get("mime", "image/png")})
            # Scope guard, appended here rather than trusted to the prompt author: the ✨ polish step
            # is optional, so a hand-typed instruction reaches Omni without ever seeing
            # OMNI_EDIT_PROMPT_PROMPT. Skipped when already present (the polished path adds it too).
            guard = get_prompt("OMNI_EDIT_SCOPE_BLOCK").strip()
            text = prompt if guard.lower() in prompt.lower() else f"{prompt.rstrip()} {guard}"
            inp.append({"type": "text", "text": text})
            model = cfg.omni_model
            # log/trace `text`, not `prompt` — the guard is part of what Omni actually receives, so a
            # trace showing the pre-guard string would misrepresent the request when debugging an edit
            logger.info("Omni edit request → model={} turn={} inputs={} prompt={}",
                       model, "continue" if previous_interaction_id else "first", len(inp) - 1, text[:200])
            with obs.generation("omni.edit_video", model=model,
                                input={"model": model, "turn": "continue" if previous_interaction_id else "first",
                                       "has_video": bool(video and video.get("data")), "prompt": text,
                                       "refs": [{"mime": r.get("mime"), "url": r.get("url"), "label": r.get("label")} for r in (refs or [])]}) as gen:
                try:
                    kwargs: dict = {"model": model, "input": inp,
                                    "response_format": {"type": "video", "aspect_ratio": cfg.aspect_ratio}}
                    if previous_interaction_id:
                        kwargs["previous_interaction_id"] = previous_interaction_id
                    interaction = client.interactions.create(**kwargs)
                    interaction_id = getattr(interaction, "id", None) or ""
                    if not interaction_id:
                        raise RuntimeError("Omni edit returned no interaction id — can't continue this conversation later")
                    ov = getattr(interaction, "output_video", None)
                    if ov is None:
                        raise RuntimeError(f"Omni edit returned no video (status={getattr(interaction, 'status', '?')})")
                    if getattr(ov, "data", None):
                        vid = base64.b64decode(ov.data)
                    elif getattr(ov, "uri", None):
                        r2 = httpx.get(ov.uri, timeout=180, follow_redirects=True)
                        r2.raise_for_status()
                        vid = r2.content
                    else:
                        raise RuntimeError("Omni edit video had neither inline data nor a uri")
                    if not vid:
                        raise RuntimeError("Omni edit produced no video bytes")
                    gen.update(output={"bytes": len(vid), "interaction_id": interaction_id, "video": obs.media(vid, "video/mp4")})
                    return vid, interaction_id
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Omni edit failed: {e}")
                    gen.update(level="ERROR", status_message=str(e))
                    raise
                finally:
                    record_usage("omni.edit_video", None, (time.perf_counter() - _t) * 1000)

        return await asyncio.to_thread(_run)

    async def generate_video_chain(self, segments: list[str], image: dict, *, cfg=None, on_progress=None) -> tuple[bytes | None, int]:
        """Render a CONTINUOUS group of shots as ONE Veo take via extension chaining.

        segments: each shot's prompt_video, in order (>=1). image: first frame
        {"mime","data"} for the opening shot (image-to-video). Each extension's output
        is the CUMULATIVE clip. PARTIAL-SALVAGE: if a clip fails after retries the chain STOPS
        and returns the cumulative clip through the LAST good shot — the caller renders the rest
        individually. Returns (mp4_bytes | None, covered) where covered = shots in the returned clip
        (0 if even the opening failed). Only the opening-model failure (_chain_model) still raises.
        """
        cfg = cfg or self.settings.video_gen
        segments = [s for s in segments if (s or "").strip()]
        if not segments or not (image and image.get("data")):
            return None, 0

        def _run() -> tuple[bytes | None, int]:
            from google.genai import types

            client = self._get_genai_client()
            retries = max(1, getattr(cfg, "rai_retries", 1))
            try:
                model = _chain_model(cfg)   # lite → interpolation_model (extension unsupported on lite)
            except Exception as e:          # log before raising, per the pattern used elsewhere
                logger.warning(f"Video chain failed: {e}")
                raise

            def _generate(**call_kwargs):
                """One generate_videos op + poll, retried on the RAI filter. On a name/celebrity
                filter, rewrite the prompt once to drop person names, then retry. Returns result.video obj."""
                _t_clip = time.perf_counter()
                try:
                    return _generate_inner(**call_kwargs)
                finally:
                    # Veo bills by seconds, not tokens — count the clip + wall time for step-7 accounting
                    record_usage("veo.generate_video_chain", None, (time.perf_counter() - _t_clip) * 1000)

            def _generate_inner(**call_kwargs):
                reasons: list[str] = []
                name_rewritten = False
                audio_softened = False
                attempt, max_attempts = 0, retries
                while attempt < max_attempts:
                    attempt += 1
                    op = client.models.generate_videos(model=model, **call_kwargs)
                    while not op.done:
                        time.sleep(max(2, cfg.poll_interval_seconds))
                        op = _net_retry(lambda: client.operations.get(op))   # a poll blip must not kill the chain
                    result = getattr(op, "response", None) or getattr(op, "result", None)
                    reasons = _veo_reasons(result, op)
                    if not reasons:
                        return result.generated_videos[0]
                    logger.warning(f"Veo chain attempt {attempt}/{max_attempts}: {'; '.join(reasons)}")
                    # safety net: deterministic name/celebrity filter → strip person names once, then retry
                    if _is_name_filter(reasons) and not name_rewritten and call_kwargs.get("prompt"):
                        name_rewritten = True
                        new_prompt = self._strip_person_names(call_kwargs["prompt"])
                        if new_prompt and new_prompt != call_kwargs["prompt"]:
                            logger.warning("Veo chain name filter → stripped person names, retrying")
                            call_kwargs["prompt"] = new_prompt
                            max_attempts += 1   # bonus attempt for the rewritten prompt
                    # audio filter → host-speak (drop the lip-sync clause, keep the host speaking + dialogue),
                    # then keep retrying (stochastic). A chain can't swap model mid-extend; if host-speak still
                    # fails to end-of-budget we raise → the caller salvages the good prefix and renders the rest
                    # individually via the single path (which has the full ladder incl. the standard-model swap).
                    if _is_audio_filter(reasons) and not audio_softened and call_kwargs.get("prompt"):
                        audio_softened = True
                        new_prompt = _strip_lipsync(call_kwargs["prompt"])
                        if new_prompt != call_kwargs["prompt"]:
                            logger.warning("Veo chain audio filter → host-speak (dropped lip-sync clause), retrying")
                            call_kwargs["prompt"] = new_prompt
                            max_attempts += 1
                    if attempt < max_attempts:
                        _retry_backoff(cfg, attempt, reasons)   # wait before re-attempt (esp. code-13 backend)
                raise RuntimeError(f"Veo filtered the clip (safety): {'; '.join(reasons)}")

            with obs.generation("veo.generate_video_chain", model=model,
                                input={"segments": len(segments), "segments_text": segments}) as gen:
                n = len(segments)
                # 1. opening shot — image-to-video from the first frame. If it fails after retries there's
                # nothing to salvage → return (None, 0) so the caller renders every shot individually.
                if on_progress:
                    on_progress(1, n)
                logger.info(f"Veo chain: clip 1/{n} (opening, image-to-video) → rendering…")
                _t = time.perf_counter()
                try:
                    with obs.span("veo.chain.clip", input={"index": 1, "total": n, "mode": "image-to-video", "prompt": segments[0]}) as cs:
                        vid = _generate(
                            prompt=segments[0],
                            image=types.Image(image_bytes=image["data"], mime_type=image.get("mime", "image/png")),
                            config=types.GenerateVideosConfig(
                                number_of_videos=1, resolution=cfg.resolution,
                                aspect_ratio=cfg.aspect_ratio, person_generation="allow_adult",
                                duration_seconds=cfg.duration_seconds,   # opening clip = config length (not Veo's 8s default)
                                **({"negative_prompt": cfg.negative_prompt} if getattr(cfg, "negative_prompt", "") else {})),
                        )
                        cs.update(output={"done": True, "seconds": round(time.perf_counter() - _t, 1)})
                except Exception as e:
                    logger.warning(f"Veo chain: opening clip failed → nothing to salvage: {e}")
                    gen.update(level="ERROR", status_message=str(e))
                    return None, 0
                covered = 1
                logger.info(f"Veo chain: clip 1/{n} done ({time.perf_counter() - _t:.0f}s)")
                # 2. extend with each subsequent shot — video + prompt only. STOP at the first failure and
                # salvage the cumulative clip through the last good shot (the caller renders the rest solo).
                # Wait for the source clip's file to be ACTIVE first — Veo 400s if it's still PROCESSING.
                for i, seg in enumerate(segments[1:], start=2):
                    _wait_video_active(client, vid.video)
                    if on_progress:
                        on_progress(i, n)
                    logger.info(f"Veo chain: clip {i}/{n} (extend) → rendering…")
                    _t = time.perf_counter()
                    try:
                        with obs.span("veo.chain.clip", input={"index": i, "total": n, "mode": "extend", "prompt": seg}) as cs:
                            vid = _generate(
                                video=vid.video, prompt=seg,
                                config=types.GenerateVideosConfig(
                                    number_of_videos=1, resolution=cfg.resolution,
                                    **({"negative_prompt": cfg.negative_prompt} if getattr(cfg, "negative_prompt", "") else {})),
                            )
                            cs.update(output={"done": True, "seconds": round(time.perf_counter() - _t, 1)})
                    except Exception as e:
                        logger.warning(f"Veo chain broke at clip {i}/{n} ({e}) — salvaging {covered} shot(s), rest render individually")
                        break
                    covered += 1
                    logger.info(f"Veo chain: clip {i}/{n} done ({time.perf_counter() - _t:.0f}s)")
                data = _net_retry(lambda: _veo_video_bytes(client, vid.video, self._project, self._credentials))
                if not data:
                    logger.warning("Veo chain: no retrievable bytes from the cumulative clip")
                    gen.update(level="ERROR", status_message="no retrievable bytes")
                    return None, 0
                logger.info(f"Veo chain: salvaged {covered}/{n} shots → combined {len(data)} bytes")
                gen.update(output={"bytes": len(data), "segments": covered, "video": obs.media(data, "video/mp4")})
                return data, covered

        return await asyncio.to_thread(_run)

    async def generate_director_prompt(self, url: str, title: str = "") -> dict:
        """Watch one reference video and reverse-engineer a directorial style guide.

        Multimodal (1 YouTube URL per request, like summarize_video). Returns a dict
        with 'summary' + the 7 style sections. Empty dict on failure.
        """
        def _run() -> dict:
            video_part = Part.from_uri(uri=url, mime_type="video/mp4")
            prompt = get_prompt("DIRECTOR_EXTRACT_PROMPT").format(title=title or "(unknown)")
            logger.info(f"Extracting director style from: {title[:60] or url}")
            with obs.generation(
                "gemini.director_prompt",
                model=self.model_name,
                input={"url": url, "title": title, "prompt": prompt},
            ) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    [video_part, prompt],
                    generation_config={
                        "temperature": self.settings.gemini.temperature,
                        "max_output_tokens": self.settings.gemini.max_output_tokens,
                        "thinking_config": {"thinking_budget": 1024},
                    },
                )
                text = _extract_text(response).strip()
                usage = _usage(response)
                gen.update(output=text, usage_details=usage)
                record_usage("gemini.director_prompt", usage, (time.perf_counter() - _t0) * 1000)
            return _parse_json_obj(text)

        return await asyncio.to_thread(_run)

    async def summarize_video(self, video: dict, topic: str, idx: int, total: int) -> dict:
        """Summarize a single video via YouTube URL — Gemini allows exactly 1 URL per request.

        Returns {"notes": str, "duration_seconds": int, "phases": dict|None, "voice_samples": list[str],
        "presentation_style": str} — `notes` is the plain-text instructional summary (same shape
        callers relied on before); `phases` is a best-effort intro/prep/main/finish % breakdown;
        `voice_samples` are a few short lines quoting the real host's natural speaking style;
        `presentation_style` is an analytical description of HOW they run the show (pacing,
        transitions, engagement technique) — distinct from voice_samples, which is WHAT they say.
        All parsed off a trailing marker block the prompt asks for. Used to ground the Script
        step's pacing, voice, AND hosting flow in the actual source videos instead of only a
        generic fixed example.
        """
        def _run() -> dict:
            duration_s = int(video.get("duration") or 0)
            duration_str = f"{duration_s // 60}:{duration_s % 60:02d}" if duration_s else "unknown"
            video_part = Part.from_uri(uri=video["url"], mime_type="video/mp4")
            prompt = get_prompt("BATCH_PROMPT").format(topic=topic, n=idx, total=total, duration=duration_str)
            logger.info(f"Summarizing video {idx}/{total}: {video['title'][:60]}")
            with obs.generation(
                "gemini.summarize_video",
                model=self.model_name,
                input={"topic": topic, "url": video["url"], "title": video.get("title"), "prompt": prompt},
            ) as gen:
                _t0 = time.perf_counter()
                response = self.model.generate_content(
                    [video_part, prompt],
                    generation_config={
                        "temperature": self.settings.gemini.temperature,
                        "max_output_tokens": self.settings.gemini.max_output_tokens,
                    },
                )
                text = _extract_text(response)
                usage = _usage(response)
                gen.update(output=text, usage_details=usage)
                record_usage("gemini.summarize_video", usage, (time.perf_counter() - _t0) * 1000)

            notes = text
            phases: dict | None = None
            voice_samples: list[str] = []
            presentation_style = ""
            marker = "---PHASE_BREAKDOWN---"
            if marker in text:
                notes_part, _, json_part = text.partition(marker)
                notes = notes_part.strip()
                try:
                    parsed = json.loads(_strip_fences(json_part.strip()))
                    phases = {
                        k: float(parsed[k])
                        for k in ("intro", "prep", "main", "finish")
                        if isinstance(parsed.get(k), (int, float))
                    } or None
                    voice_samples = [s.strip() for s in (parsed.get("voice_samples") or []) if isinstance(s, str) and s.strip()][:4]
                    presentation_style = str(parsed.get("presentation_style") or "").strip()
                except Exception as e:  # noqa: BLE001 — phase breakdown is best-effort, never blocks the notes
                    logger.debug(f"Phase breakdown parse failed for video {idx}/{total}: {e}")

            return {
                "notes": notes, "duration_seconds": duration_s, "phases": phases,
                "voice_samples": voice_samples, "presentation_style": presentation_style,
            }

        return await asyncio.to_thread(_run)

    async def synthesize_from_summaries(
        self, summaries: list[str], topic: str, master_index: int | None = None, duration_hint: str = "",
    ) -> str:
        """``master_index`` (0-based, into ``summaries``) — when given, that video is treated as
        the MASTER reference: the tutorial's core method/sequence/quantities are built primarily
        from it, and the other videos only contribute supplementary tips/refinements. None (the
        default) still commits to a SINGLE base method (never a blend) — it just picks it itself.

        ``duration_hint`` — optional pre-formatted sentence scoping how much content to synthesize
        (see ``nodes._duration_budget_hint``), so a fixed target runtime set at the Research step
        doesn't force the Script step to pad or compress content it wasn't scoped for."""
        def _run() -> str:
            notes = "\n\n---\n\n".join(
                f"Video {i + 1} Notes{' [MASTER REFERENCE]' if i == master_index else ''}\n{s}"
                for i, s in enumerate(summaries)
            )
            if master_index is not None and 0 <= master_index < len(summaries):
                n = master_index + 1
                method_block = (
                    f"Video {n} is the MASTER reference — build the tutorial's core method, sequence and "
                    f"quantities PRIMARILY from Video {n}'s notes. Use the OTHER videos' notes only to ADD "
                    "complementary tips, refinements, warnings, or alternative techniques that genuinely "
                    f"improve on the master's method — do NOT replace Video {n}'s core approach with a "
                    "different one taken from another video. When another video's point conflicts with the "
                    f"master's method, prefer Video {n}'s way unless the other point is a clear safety or "
                    "quality improvement — in that case, fold it in as a tip rather than overriding the "
                    "master's structure."
                )
            else:
                method_block = (
                    "Before writing, analyze all the methods and approaches found across the notes. "
                    "Identify the ONE best method — the one most commonly recommended, most practical, "
                    "and most likely to succeed for the average person. Commit to that single method as "
                    "your BASE, exactly as if it had been marked MASTER: build the core steps, sequence "
                    "and quantities from that one method only, and use every other video's notes ONLY to "
                    "add complementary tips, warnings or refinements — never to replace or blend into the "
                    "base method's own steps or quantities. State plainly at the top of your synthesis "
                    "which single approach you chose as the base. Do not present multiple methods, "
                    "alternatives, or variations as if they were equally valid."
                )
            if duration_hint.strip():
                method_block += f"\n\n{duration_hint.strip()}"
            prompt = get_prompt("SYNTHESIS_HEADER").format(topic=topic, method_block=method_block) + notes

            logger.info(f"Synthesizing from {len(summaries)} video summaries for topic '{topic}' (prompt_len={len(prompt):,} chars)")

            generation_config = {
                "temperature": self.settings.gemini.temperature,
                "max_output_tokens": self.settings.gemini.max_output_tokens,
                "thinking_config": {"thinking_budget": 2048},
            }

            with obs.generation(
                "gemini.synthesize",
                model=self.model_name,
                input={"topic": topic, "summary_count": len(summaries), "prompt": prompt},
            ) as gen:
                last_text = ""
                for attempt in range(1, 4):
                    _t0 = time.perf_counter()
                    response = self.model.generate_content(prompt, generation_config=generation_config)
                    logger.debug(f"Synthesis attempt {attempt}: {response}")
                    result = _extract_text(response)
                    if result:
                        usage = _usage(response)
                        gen.update(output=result, usage_details=usage)
                        record_usage("gemini.synthesize", usage, (time.perf_counter() - _t0) * 1000)
                        logger.info(f"Synthesis result length: {len(result):,} chars")
                        return result
                    last_text = result
                    logger.warning(f"Synthesis attempt {attempt} returned empty, retrying...")
                gen.update(output=last_text, level="ERROR", status_message="empty after 3 attempts")

            raise RuntimeError("Synthesis returned empty after 3 attempts")

        return await asyncio.to_thread(_run)

    def _get_script_llm(self):
        """LangChain LLM used for structured script generation.

        Given a larger output budget than the rest of the pipeline (a full
        storyboard is big) and a bounded thinking budget so the model doesn't
        spend the whole token budget thinking and then truncate the JSON.
        Uses `script_temperature` (higher than the pipeline default) — this is a
        creative-writing call (hook, voice, pacing), unlike the mostly-deterministic
        classify/filter/synthesize calls that want the low default temperature.
        """
        if self._langchain_llm is None:
            from langchain_google_vertexai import ChatVertexAI
            self._langchain_llm = ChatVertexAI(
                model=self.settings.gemini.model,
                project=self._project,
                location=self._location,
                credentials=self._credentials,
                temperature=self.settings.gemini.script_temperature,
                max_output_tokens=self.settings.gemini.script_max_output_tokens,
                thinking_budget=self.settings.gemini.script_thinking_budget,
            )
            logger.info(
                "LangChain ChatVertexAI initialized for script "
                f"(temperature={self.settings.gemini.script_temperature}, "
                f"max_output_tokens={self.settings.gemini.script_max_output_tokens}, "
                f"thinking_budget={self.settings.gemini.script_thinking_budget})"
            )
        return self._langchain_llm

    async def generate_script(
        self, synthesis: str, topic: str, cfg: ScriptConfig, director_prompt: str = "", brand_block: str = "",
        voice_samples: list[str] | None = None, presentation_notes: list[str] | None = None,
        menu_block: str = "",
    ):
        """Generate the storyboard script in one LLM call.

        Returns a ``ScriptDocument`` of pure structured data. Markdown is NOT
        produced here — it's a derived view rendered from this document on demand
        (see ``app.services.script_render.render_markdown``).

        ``director_prompt`` is an optional directorial style guide (from a reference
        video) injected so the script matches that professional style.

        ``voice_samples`` are real host-voice quotes pooled from the research step's reference
        videos (routes._aggregate_voice_samples) — when present, they ground voice_over's tone in
        an ACTUAL cooking video instead of only the prompt's fixed generic gold-standard example.

        ``presentation_notes`` are real "how they run the show" observations pooled from the same
        videos (routes._aggregate_presentation_notes) — distinct from voice_samples (WHAT they
        say): this is pacing/transitions/engagement technique, grounding the hook/open-loop/rhythm
        craft rules in what real reference videos on this exact topic actually do.
        """
        from app.models.script import ScriptDocument

        def _run():
            director_block = ""
            if voice_samples:
                quotes = "\n".join(f'  - "{q}"' for q in voice_samples)
                director_block += (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "REAL HOST VOICE — quotes captured from actual reference videos on this exact topic. The\n"
                    "GOLD-STANDARD EXAMPLE further below is for STRUCTURE/depth/field-shape ONLY — for VOICE and\n"
                    "tone, follow these real quotes instead (natural phrasing, rhythm, how a real host actually\n"
                    "talks), blended with the character's persona (given elsewhere in this prompt). Do NOT copy\n"
                    "these lines verbatim — absorb the STYLE, not the words:\n"
                    f"{quotes}\n"
                )
            if presentation_notes:
                notes_block = "\n".join(f"  - {n}" for n in presentation_notes)
                director_block += (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "REAL PRESENTATION FLOW — how the hosts in the actual reference videos on this exact topic\n"
                    "RUN THE SHOW (pacing, transitions, how they build anticipation, how they recap/close) —\n"
                    "apply this concretely to the HOOK, SENTENCE RHYTHM and OPEN LOOP rules above instead of\n"
                    "generic versions of those techniques. This is about presentation STRUCTURE, not word choice\n"
                    "(voice_samples above covers that):\n"
                    f"{notes_block}\n"
                )
            if director_prompt.strip():
                director_block += (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "DIRECTOR'S STYLE GUIDE — follow this directorial style closely "
                    "(tone, pacing, shots, narration, hook/CTA):\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{director_prompt.strip()}\n"
                )
            if brand_block.strip():
                director_block += f"\n\n{brand_block.strip()}\n"

            tools_rule = TOOLS_RULE_BOTH if cfg.include_tools_scene else TOOLS_RULE_INGREDIENTS_ONLY
            tools_example = TOOLS_GOLD_SCENE if cfg.include_tools_scene else ""

            # duration_per_part is the duration PER PART (min–max minutes for EACH part). Frame it for the LLM
            # and give a concrete shot target (video = fixed ~Ns clips) so each part lands in its window.
            clip_s = max(1, self.settings.video_gen.duration_seconds)
            def _secs(t: str) -> int:
                mm, _, ss = t.strip().partition(":")
                return int(mm) * 60 + int(ss or 0)
            def _mmss(s: float) -> str:
                s = round(s)
                return f"{s // 60}:{s % 60:02d}"
            # Word/pace guidance is OPTIONAL — a cleared (0) field drops that clause from the prompt entirely.
            wpp = cfg.words_per_part or 0
            wps = cfg.words_per_second or 0
            min_sp = cfg.min_shots_per_part or 0
            min_sc = cfg.min_scenes_per_part or 0
            pace_note = f" Voice_over is spoken at a natural pace of ~{wps} words/second." if wps > 0 else ""

            if cfg.duration_mode == "source" and cfg.source_avg_duration_seconds > 0:
                # "source" mode: derive the target from the reference videos' OWN pacing (research's
                # `pacing` block) instead of a fixed config window, with a soft ±30% guard rail instead
                # of a hard floor — this is the fix for scripts feeling padded/dragged out when the
                # source material is naturally shorter than a fixed duration_per_part window.
                per_part = cfg.source_avg_duration_seconds / max(1, cfg.parts)
                lo, hi = per_part * 0.7, per_part * 1.3
                duration_per_part_display = f"{_mmss(lo)}–{_mmss(hi)} (ตามต้นฉบับ)"
                phase_note = ""
                if cfg.source_phase_pct:
                    shape = ", ".join(f"{k} ~{v:.0f}%" for k, v in cfg.source_phase_pct.items())
                    phase_note = f" Follow this rough pacing SHAPE observed across the reference videos: {shape}."
                if wpp > 0:
                    phase_note += f" Aim for roughly {wpp} words of spoken voice_over per part, but let actual content depth decide — do not stretch to hit this number."
                duration_part_note = (
                    f"Duration PER PART: base the pacing on the SOURCE reference videos (~{_mmss(per_part)} per part "
                    f"on average across {cfg.parts} parts). Stay within {_mmss(lo)}–{_mmss(hi)} per part (mm:ss) — "
                    f"this is a SOFT guard rail, not a fixed target: follow the natural flow and depth of the "
                    f"tutorial content. Do NOT pad with filler to reach the upper bound, and do NOT rush or cut "
                    f"content short just to land near the lower bound — let the content itself decide, within this "
                    f"range.{phase_note}{pace_note} Each part is assembled from ~{clip_s}s clips (one per shot), so "
                    f"plan roughly {round(lo / clip_s)}-{round(hi / clip_s)} shots per part."
                )
            else:
                try:
                    lo_s, _, hi_s = cfg.duration_per_part.partition("-")
                    lo, hi = _secs(lo_s), _secs(hi_s or lo_s)
                    duration_per_part_display = cfg.duration_per_part
                    if wpp > 0 and wps > 0:
                        words_sent = f" Write about {wpp} words of spoken voice_over PER PART at a natural pace of ~{wps} words/second."
                    elif wpp > 0:
                        words_sent = f" Write about {wpp} words of spoken voice_over PER PART."
                    else:
                        words_sent = pace_note
                    each = f" (~{round(wps * clip_s)} words each)" if wps > 0 else ""
                    min_clause = f" — and AT LEAST {min_sp} shots per part (≈ {min_sp * clip_s}s minimum)" if min_sp > 0 else ""
                    scenes_clause = f" Each part MUST contain AT LEAST {min_sc} scenes (scenes drive the shot count)." if min_sc > 0 else ""
                    duration_part_note = (
                        f"Duration PER PART: EACH of the {cfg.parts} parts must run LONGER than {lo_s.strip()} but NO LONGER "
                        f"than {hi_s.strip()} (mm:ss).{words_sent} Each part is assembled from ~{clip_s}s clips (one per shot), "
                        f"so EACH part should total roughly {lo // clip_s}-{hi // clip_s} shots{each}{min_clause}.{scenes_clause} "
                        f"Reach AT LEAST the lower bound — never pad with filler, and never exceed the upper bound."
                    )
                except (ValueError, ZeroDivisionError):
                    duration_per_part_display = cfg.duration_per_part
                    mn = f" (at least {min_sp} shots per part)" if min_sp > 0 else ""
                    sn = f" (at least {min_sc} scenes per part)" if min_sc > 0 else ""
                    duration_part_note = (
                        f"Duration per part (each of the {cfg.parts} parts): {cfg.duration_per_part}{mn}{sn}."
                    )

            prompt = get_prompt("SCRIPT_PROMPT").format(
                topic=topic,
                parts=cfg.parts,
                duration_per_part=duration_per_part_display,
                duration_part_note=duration_part_note,
                character_name=cfg.character_name or "ไม่ระบุ",
                character_desc=cfg.character_desc or "ไม่ระบุ",
                theme=cfg.theme or "ไม่ระบุ",
                mood=cfg.mood or "ไม่ระบุ",
                material_palette=cfg.material_palette or "ไม่ระบุ",
                lighting=cfg.lighting or "ไม่ระบุ",
                tools_rule=tools_rule,
                tools_example=tools_example,
                finished_look_rule=category_block(cfg.category, "finished_look"),
            ) + director_block + menu_block + synthesis

            logger.info(
                f"Generating structured script for topic '{topic}' "
                f"({cfg.parts} parts, {cfg.duration_per_part} per part)"
            )

            # include_raw=True so we can inspect finish_reason / parsing errors
            # when the model truncates instead of silently retrying blind.
            structured = self._get_script_llm().with_structured_output(ScriptDocument, include_raw=True)

            with obs.generation(
                "gemini.generate_script",
                model=self.model_name,
                input={"topic": topic, "parts": cfg.parts, "prompt": prompt},
            ) as gen:
                last_err: str | None = None
                for attempt in range(1, 4):
                    try:
                        _t0 = time.perf_counter()
                        out = structured.invoke(prompt)
                    except Exception as exc:  # noqa: BLE001
                        last_err = str(exc)
                        logger.warning(f"Script attempt {attempt} raised: {exc}; retrying...")
                        continue

                    doc = out.get("parsed") if isinstance(out, dict) else out
                    raw = out.get("raw") if isinstance(out, dict) else None
                    parse_err = out.get("parsing_error") if isinstance(out, dict) else None
                    finish = (getattr(raw, "response_metadata", {}) or {}).get("finish_reason")

                    if doc and doc.parts:
                        logger.info(
                            f"Script generated: {len(doc.parts)} parts, "
                            f"{sum(len(p.scenes) for p in doc.parts)} scenes"
                        )
                        usage = _usage_lc(raw)
                        gen.update(usage_details=usage)
                        record_usage("gemini.generate_script", usage, (time.perf_counter() - _t0) * 1000)
                        gen.update(output={
                            "title": doc.title,
                            "parts": len(doc.parts),
                            "scenes": sum(len(p.scenes) for p in doc.parts),
                        })
                        return doc

                    # Diagnose why it was empty/incomplete.
                    last_err = f"finish_reason={finish}, parsing_error={parse_err}"
                    logger.warning(
                        f"Script attempt {attempt} empty/incomplete "
                        f"(finish_reason={finish}, parsing_error={parse_err}). "
                        f"If finish_reason=MAX_TOKENS, raise gemini.script_max_output_tokens."
                    )

                gen.update(level="ERROR", status_message=last_err or "empty")

            raise RuntimeError(f"Script generation returned empty after 3 attempts ({last_err})")

        return await asyncio.to_thread(_run)

    async def regenerate_script_part(self, full_script, part_number: int, synthesis: str, topic: str, cfg: ScriptConfig, target_duration: str, target_scenes: int | None = None, min_shots: int | None = None, director_prompt: str = "", brand_block: str = "", menu_block: str = ""):
        """Regenerate ONE part of an existing ``ScriptDocument`` (staircase per-part auto-fit).

        Keeps the other parts. Feeds the full ``overview`` + the previous part as read-only context so the
        new part stays continuous and doesn't duplicate other parts. Returns a validated ``ScriptPart``
        (the caller renumbers scene_ids and merges it back).
        """
        from app.models.script import ScriptPart

        def _run():
            director_block = ""
            if director_prompt.strip():
                director_block = (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "DIRECTOR'S STYLE GUIDE — follow this directorial style closely "
                    "(tone, pacing, shots, narration, hook/CTA):\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{director_prompt.strip()}\n"
                )
            if brand_block.strip():
                director_block += f"\n\n{brand_block.strip()}\n"

            tools_rule = TOOLS_RULE_BOTH if cfg.include_tools_scene else TOOLS_RULE_INGREDIENTS_ONLY
            prod = full_script.production

            # per-part duration + scene target sentences (single-part variant of duration_part_note)
            clip_s = max(1, self.settings.video_gen.duration_seconds)
            def _secs(t: str) -> int:
                mm, _, ss = t.strip().partition(":")
                return int(mm) * 60 + int(ss or 0)
            try:
                lo_s, _, hi_s = target_duration.partition("-")
                lo, hi = _secs(lo_s), _secs(hi_s or lo_s)
                duration_sentence = (
                    f"Part {part_number} must run LONGER than {lo_s.strip()} but NO LONGER than {hi_s.strip()} "
                    f"(mm:ss) — roughly {lo // clip_s}-{hi // clip_s} shots total. Reach at least the lower bound; never pad, never exceed the upper bound."
                )
            except (ValueError, ZeroDivisionError):
                duration_sentence = f"Part {part_number} should run about {target_duration}."
            sc_target = target_scenes if (target_scenes and target_scenes > 0) else (cfg.min_scenes_per_part or 0)
            scenes_sentence = f" Output about {sc_target} scenes (scenes drive the shot count)." if sc_target > 0 else ""
            if min_shots and min_shots > 0:
                scenes_sentence += (f" This part must contain enough content to break into AT LEAST {min_shots} shots total"
                                    f" — write each scene RICH (several distinct beats / sub-steps) so it splits into multiple shots, not thin one-liners.")

            overview_block = "\n".join(
                f"* PART {o.number} — {o.title}: {o.summary}" for o in (full_script.overview or [])
            ) or "(no overview available)"

            prev_block = ""
            if part_number > 1:
                prev = next((p for p in full_script.parts if p.number == part_number - 1), None)
                if prev:
                    lines = "\n".join(
                        f"  - {s.name}: {s.key_message} | VO: {s.voice_over}" for s in prev.scenes
                    )
                    prev_block = (
                        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"PREVIOUS PART {prev.number} — {prev.title} "
                        f"(READ-ONLY — Part {part_number} must continue SEAMLESSLY from here; do NOT repeat it)\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"{lines}\n"
                    )

            prompt = get_prompt("SCRIPT_REGENERATE_PART_PROMPT").format(
                topic=topic,
                part_number=part_number,
                character_name=cfg.character_name or prod.character_name or "ไม่ระบุ",
                character_desc=cfg.character_desc or prod.character_desc or "ไม่ระบุ",
                theme=cfg.theme or prod.theme or "ไม่ระบุ",
                mood=cfg.mood or prod.mood or "ไม่ระบุ",
                material_palette=cfg.material_palette or prod.material_palette or "ไม่ระบุ",
                lighting=cfg.lighting or prod.lighting or "ไม่ระบุ",
                overview_block=overview_block,
                prev_block=prev_block,
                duration_sentence=duration_sentence,
                scenes_sentence=scenes_sentence,
                clip_s=clip_s,
                tools_rule=tools_rule,
            ) + director_block + menu_block + synthesis

            logger.info(f"Regenerating part {part_number} (target {target_duration}, ~{sc_target or '?'} scenes)")

            structured = self._get_script_llm().with_structured_output(ScriptPart, include_raw=True)

            with obs.generation(
                "gemini.regenerate_script_part",
                model=self.model_name,
                input={"topic": topic, "part": part_number, "prompt": prompt},
            ) as gen:
                last_err: str | None = None
                for attempt in range(1, 4):
                    try:
                        _t0 = time.perf_counter()
                        out = structured.invoke(prompt)
                    except Exception as exc:  # noqa: BLE001
                        last_err = str(exc)
                        logger.warning(f"Regen part {part_number} attempt {attempt} raised: {exc}; retrying...")
                        continue

                    part = out.get("parsed") if isinstance(out, dict) else out
                    raw = out.get("raw") if isinstance(out, dict) else None
                    parse_err = out.get("parsing_error") if isinstance(out, dict) else None
                    finish = (getattr(raw, "response_metadata", {}) or {}).get("finish_reason")

                    if part and part.scenes:
                        usage = _usage_lc(raw)
                        gen.update(usage_details=usage)
                        record_usage("gemini.regenerate_script_part", usage, (time.perf_counter() - _t0) * 1000)
                        gen.update(output={"part": part_number, "scenes": len(part.scenes)})
                        logger.info(f"Regenerated part {part_number}: {len(part.scenes)} scenes")
                        return part

                    last_err = f"finish_reason={finish}, parsing_error={parse_err}"
                    logger.warning(
                        f"Regen part {part_number} attempt {attempt} empty/incomplete "
                        f"(finish_reason={finish}, parsing_error={parse_err})."
                    )

                gen.update(level="ERROR", status_message=last_err or "empty")

            raise RuntimeError(f"Part {part_number} regeneration returned empty after 3 attempts ({last_err})")

        return await asyncio.to_thread(_run)
