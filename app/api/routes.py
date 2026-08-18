import asyncio
import base64
import io
import os
import re
import statistics
import subprocess
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import ROOT_DIR, settings as _settings
from loguru import logger
from pydantic import ValidationError

from app.api.schemas import (
    DirectorCreateRequest,
    ImageEditPromptRequest,
    ImagePromptReviseRequest,
    ImageEditRequest,
    ImagePromptsStepRequest,
    ImageCropRequest,
    ImageUndoRequest,
    ImageRegionEditRequest,
    ImageStepRequest,
    KitchenFixturesRequest,
    ImagesStepRequest,
    MergeRequest,
    PdfExportRequest,
    RegenEventRequest,
    ResearchStepRequest,
    RunCreateRequest,
    RunUpdateRequest,
    CharacterMetaRequest,
    SaveResearchRequest,
    SaveScriptRequest,
    SaveStoryboardRequest,
    ScriptRegeneratePartRequest,
    ScriptStepRequest,
    ShotPromptVideoUpdateRequest,
    ShotVideoRefsUpdateRequest,
    StoryboardStepRequest,
    RefManifestRequest,
    RevoiceRequest,
    SubtitleRequest,
    SynthesizeRequest,
    VideoEditPromptRequest,
    VideoEditRequest,
    VideoGroupRequest,
    VideoUndoRequest,
    VideoCropRequest,
    VideoTrimRequest,
    VideoPromptsStepRequest,
    VideoStepRequest,
    sse,
)
from app.core.config import ScriptConfig
from app.db.base import get_db
from app.db.models import StoryboardShot as DBStoryboardShot, iso_utc
from app.graph.nodes import (
    _MEDIA_DIR, _prompt_has_person, _provider_generate, _ref_src, _slug, extract_kitchen_fixtures_from_scene,
    generate_shot_dynamic, load_base_image_refs, load_kitchen_fixtures, render_shot_image, save_kitchen_fixtures,
)
from app.services.image_fetch import load_reference
from app.services import openai_image
from app.core.usage import UsageTracker, set_tracker
from app.core import observability as obs
from app.graph.runner import run_pipeline, run_step
from app.models.director import SECTION_KEYS
from app.models.script import ScriptDocument
from app.models.storyboard import StoryboardDocument
from app.repositories import (
    BrandRepo,
    CharacterRepo,
    CostRateRepo,
    DirectorRepo,
    MediaAssetRepo,
    MenuRepo,
    NotFoundError,
    RegenRepo,
    ResearchRepo,
    RunRepo,
    ScriptRepo,
    StoryboardRepo,
    UsageRepo,
)
from app.services import (active_config, character_store, cost_estimate, director_store, prompts,
                          scene_store, storage as _storage, subject_ref_store)
from app.services.video_search import fetch_video_meta

router = APIRouter()


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# Config reference images are served from stable paths but their FILE can change (replace/re-upload),
# so tell the browser never to reuse a cached copy — else the old image lingers after a swap.
_NO_STORE = {"Cache-Control": "no-store"}


def _asset_newer(db: Session, newer_id: str | None, older_id: str | None) -> bool:
    """Was `newer_id` created after `older_id`? False unless both exist.

    Used for the per-shot "the image changed after this clip was made" warning. Reads the
    media_assets rows' own created_at — no extra column, and it stays correct however the image
    got replaced (generate, edit, upload).
    """
    if not newer_id or not older_id:
        return False
    repo = MediaAssetRepo(db)
    a, b = repo.get(newer_id), repo.get(older_id)
    return bool(a and b and a.created_at and b.created_at and a.created_at > b.created_at)


def _resolve_asset_url(db: Session, asset_id: str | None) -> str:
    """DB id -> resolved S3 URL, same shape v1 returned inline (a plain image/video URL string)
    instead of leaking the internal media_assets.id to API responses."""
    if not asset_id:
        return ""
    asset = MediaAssetRepo(db).get(asset_id)
    return (asset.s3_url if asset else "") or ""


def _persist_usage(db: Session, run_id: str | None, step: str, summary: dict) -> None:
    """Write one usage_records row per agent for this step call. New in v2 — v1 only ever
    emitted usage over the SSE stream (app/core/usage.py, unchanged) with no durable per-run
    record; skips silently when there's no run_id since usage_records.run_id is NOT NULL."""
    if not run_id:
        return
    try:
        repo = UsageRepo(db)
        for agent, a in (summary.get("agents") or {}).items():
            repo.create(
                run_id=run_id,
                step=step,
                agent=agent,
                prompt_tokens=int(a.get("input", 0) or 0),
                completion_tokens=int(a.get("output", 0) or 0),
                total_tokens=int(a.get("total", 0) or 0),
                duration_ms=round(a.get("duration_ms", 0) or 0),
            )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(f"failed to persist usage for run {run_id!r} step {step!r}")


def _media_key_from_url(url: str) -> str:
    """Best-effort object key for a URL a shot's generated_img/generated_video field carries —
    either an S3 URL (`storage.key_from_url`) or a local `/api/media/...` fallback path. Falls
    back to the URL itself so MediaAssetRepo still gets a queryable value either way."""
    key = _storage.key_from_url(url)
    if key:
        return key
    if "/api/media/" in url:
        return url.split("/api/media/", 1)[-1]
    return url


def _persist_bulk_storyboard(db: Session, run_id: str, dumped: dict, *, step: str) -> None:
    """Save a bulk-generated StoryboardDocument (from /steps/images, /steps/video_prompts, ...)
    as a new version AND route each shot's generated_img/generated_video through MediaAssetRepo —
    otherwise a shot rendered by a bulk pass has no queryable/superseded-on-regen history and its
    media is lost the moment the page reloads (only /steps/image and /steps/video, the per-shot
    on-demand endpoints, did this before)."""
    doc = StoryboardRepo(db).save_new_version(
        run_id, title=dumped.get("title", ""), ingredients=dumped.get("ingredients", []),
        equipment=dumped.get("equipment", []), production=dumped.get("production", {}),
        scenes=dumped.get("scenes", []),
    )
    shots_by_key = {(sc.scene_id, sh.no): sh for sc in doc.scenes for sh in sc.shots}
    media = MediaAssetRepo(db)
    board = StoryboardRepo(db)
    for scene in dumped.get("scenes", []):
        for shot_data in scene.get("shots", []):
            shot_row = shots_by_key.get((scene.get("scene_id"), shot_data.get("no")))
            if shot_row is None:
                continue
            img_url = (shot_data.get("generated_img") or "").strip()
            if img_url:
                asset = media.create(kind="image", s3_key=_media_key_from_url(img_url), s3_url=img_url, run_id=run_id)
                board.set_shot_image(shot_row.id, asset.id)
            vid_url = (shot_data.get("generated_video") or "").strip()
            if vid_url:
                asset = media.create(kind="video", s3_key=_media_key_from_url(vid_url), s3_url=vid_url, run_id=run_id)
                board.set_shot_video(shot_row.id, asset.id)
            # Persist the resolved ref plan too — generate_images already computed this per shot in
            # memory (nodes.py's dynamic-image loop sets sh.image_plan), it just never got saved,
            # so the "Refs" thumbnails always went blank again the next time the storyboard loaded.
            plan_data = shot_data.get("image_plan") or {}
            if plan_data.get("refs_used") or plan_data.get("image_generate_type"):
                # Whole plan, filtered by upsert's `allowed` set — hand-picking fields here is how
                # the classifier verdicts (has_human, scene_match, …) never reached the DB.
                board.upsert_image_plan(
                    shot_row.id,
                    refs_used=plan_data.get("refs_used"),
                    **{k: v for k, v in plan_data.items() if k != "refs_used"},
                )
    db.commit()
    logger.info(f"persisted bulk storyboard ({step}) for run {run_id}: version {doc.version}")


def _aggregate_pacing(video_pacing: list[dict], master_index: int | None = None) -> dict | None:
    """Roll up per-video {duration_seconds, phases} into one pacing summary for the research
    result — Script's "source" duration_mode reads this to derive its target instead of a fixed
    config.yaml window. None when no video had usable timing (pacing feature degrades gracefully).

    When ``master_index`` points at a video with usable timing, ITS pacing becomes the summary
    directly instead of an average across every analyzed video — the script is built primarily
    from the master's method (see synthesize_from_summaries), so the master's own runtime/phase
    shape is what "ตามต้นฉบับ" (duration_mode="source") should actually track, not a group average
    that a tangential supplementary-tip video could skew."""
    durations = [v["duration_seconds"] for v in video_pacing if v.get("duration_seconds")]
    if not durations:
        return None
    if master_index is not None and 0 <= master_index < len(video_pacing) and video_pacing[master_index].get("duration_seconds"):
        master = video_pacing[master_index]
        master_phases = master.get("phases") or {}
        return {
            "avg_duration_seconds": round(master["duration_seconds"]),
            "video_count": len(durations),
            "phase_pct": {k: round(v, 1) for k, v in master_phases.items()} or None,
            "from_master": True,
        }
    phases_list = [v["phases"] for v in video_pacing if v.get("phases")]
    phase_pct = None
    if phases_list:
        keys = ("intro", "prep", "main", "finish")
        phase_pct = {k: round(sum(p.get(k, 0) for p in phases_list) / len(phases_list), 1) for k in keys}
    return {
        "avg_duration_seconds": round(sum(durations) / len(durations)),
        "video_count": len(durations),
        "phase_pct": phase_pct,
        "from_master": False,
    }


def _aggregate_voice_samples(video_pacing: list[dict], cap: int = 8) -> list[str]:
    """Pool the real host-voice quotes captured per video (see BATCH_PROMPT) into one list — Script
    generation uses these, when present, as a REAL reference for how an actual video host talks,
    instead of relying only on the fixed generic gold-standard example. Capped so the script prompt
    doesn't balloon when many videos were analyzed; round-robins across videos rather than draining
    one video first, so the sample set isn't dominated by a single source."""
    per_video = [v.get("voice_samples") or [] for v in video_pacing]
    out: list[str] = []
    i = 0
    while len(out) < cap and i < max((len(s) for s in per_video), default=0):
        for samples in per_video:
            if i < len(samples):
                out.append(samples[i])
                if len(out) >= cap:
                    break
        i += 1
    return out


def _aggregate_presentation_notes(video_pacing: list[dict], cap: int = 4) -> list[str]:
    """Pool the per-video "how they run the show" analysis (see BATCH_PROMPT's presentation_style)
    — distinct from voice_samples (WHAT they say): this is pacing/transitions/engagement technique,
    used to ground the Script step's hook/open-loop/pacing craft in what real reference videos on
    this topic actually do, not just a generic fixed example."""
    out = [v["presentation_style"] for v in video_pacing if v.get("presentation_style")]
    return out[:cap]


def _merge_script_config(container, req_cfg: ScriptConfig | None,
                         db: Session | None = None, run_id: str = "") -> ScriptConfig:
    """config.yaml is the base; the request overrides ONLY the fields the client
    explicitly sent — so settings the UI doesn't expose keep their config.yaml value
    instead of silently reverting to model defaults.

    `category` gets one extra layer: the RUN's own `type`, between config.yaml and the request.
    Only Steps 1 and 2 ever send a script_config, so every later step used to fall through to
    config.yaml — which has no `category:` key at all, meaning storyboard and image prompts were
    built with FOOD rules no matter what the run was. The run row is the single source of truth for
    what a run is about (Step 2's chips are locked to it), so reading it here fixes every step at
    once instead of asking three more frontend call sites to remember to send it.

    `character_name`/`character_desc` get the same treatment from the run's CHARACTER, which is why
    the client never sends them: the host is picked per run now, so reading the run's character here
    is the only thing that lets two runs be made with two different people.

    Priority: what the request explicitly set  >  the run's own row  >  config.yaml.
    """
    cfg = container.settings.script
    sent = req_cfg.model_fields_set if req_cfg is not None else set()
    if db is not None and run_id:
        if "category" not in sent:
            run = RunRepo(db).get(run_id)
            # "" (never typed) and any unexpected value leave config.yaml's default in place.
            if run is not None and run.type in ("food", "drink", "other"):
                cfg = cfg.model_copy(update={"category": run.type})
        # resolve() answers even when the run never chose one (it returns the default character),
        # so a run created before characters existed keeps the host it always had.
        character = CharacterRepo(db).resolve(run_id)
        if character is not None:
            update = {}
            if "character_name" not in sent and (character.name or "").strip():
                update["character_name"] = character.name.strip()
            if "character_desc" not in sent and (character.description or "").strip():
                update["character_desc"] = character.description.strip()
            if update:
                cfg = cfg.model_copy(update=update)
    if sent:
        cfg = cfg.model_copy(update={k: getattr(req_cfg, k) for k in sent})
    return cfg


def _image_category(db: Session, run_id: str, req_category: str | None, container) -> str:
    """Category for a per-shot image request: request > run.type > config.yaml.

    Same layering _merge_script_config gives the bulk /steps/images route — without the middle
    layer, a per-shot regenerate on a drink run silently used config.yaml's category (food) and
    got the wrong vessel rules. The frontend stopped sending category in the refactor, so the
    run's own type is what usually decides here."""
    if req_category:
        return req_category
    if run_id:
        run = RunRepo(db).get(run_id)
        if run is not None and run.type in ("food", "drink", "other"):
            return run.type
    return container.settings.script.category


# The voice fields a character may carry; anything else in the stored blob is ignored.
_VOICE_FIELDS = ("language", "gender", "vo_pace", "tone", "style")


def _merge_voice_config(container, req: "VideoPromptsStepRequest",
                        db: Session | None = None, run_id: str = ""):
    """The voice of the RUN'S character is the base; the request overrides only the fields it sent.

    Each character carries their own voice, so this is what makes two runs with two hosts sound
    different. config.yaml stays the floor underneath for any field a character left blank.

    Step 5 sends none of these (its panel is a read-only preview) — the escape hatch stays open for
    tests and POC scripts. `is not None` rather than truthiness because 0 and "" are values a caller
    could plausibly mean for vo_pace/tone."""
    voice = container.settings.voice
    if db is not None and run_id:
        character = CharacterRepo(db).resolve(run_id)
        stored = (character.voice if character else None) or {}
        fold = {k: v for k, v in stored.items()
                if k in _VOICE_FIELDS and v not in ("", None)}
        if fold:
            voice = voice.model_copy(update=fold)
    overrides = {k: v for k, v in {
        "language": req.language, "vo_pace": req.vo_pace, "tone": req.tone,
        "style": req.style, "gender": req.gender,
    }.items() if v is not None}
    return voice.model_copy(update=overrides)


def _merge_video_config(container, req: VideoStepRequest):
    """config.yaml video_gen is the base; apply the request's non-None overrides."""
    overrides = {
        k: v for k, v in {
            "duration_seconds": req.duration_seconds,
            "resolution": req.resolution,
            "aspect_ratio": req.aspect_ratio,
            "generate_audio": req.generate_audio,
            "person_generation": req.person_generation,
        }.items() if v is not None
    }
    return container.settings.video_gen.model_copy(update=overrides)


def _s3_url_to_local(url: str) -> Path | None:
    """Map an S3/CDN URL back to its local copy under _MEDIA_DIR.
    Phase 2 always writes locally before uploading, so the file should exist."""
    from urllib.parse import unquote
    aws_base = os.getenv("AWS_URL", "").rstrip("/")
    if not aws_base or not url.startswith(aws_base + "/"):
        return None
    key = unquote(url[len(aws_base) + 1:])   # URL is percent-encoded (upload_bytes quotes); on-disk name is raw
    try:
        p = (_MEDIA_DIR / key).resolve()
        p.relative_to(_MEDIA_DIR.resolve())
    except (ValueError, OSError):
        return None
    return p if p.exists() else None


def _fetch_bytes(url: str) -> bytes | None:
    """Fetch raw bytes from any HTTP/HTTPS URL (S3, CDN). Used as fallback when no local copy."""
    import httpx
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True)
        r.raise_for_status()
        return r.content
    except Exception as exc:
        logger.warning(f"_fetch_bytes failed {url}: {exc}")
        return None


def _safe_media_path(url: str) -> Path | None:
    """Resolve a media URL to a local Path, blocking path traversal.
    Handles both /api/media/ local URLs and S3 URLs (reconstructed from AWS_URL prefix)."""
    if not url:
        return None
    if "/api/media/" in url:
        rel = url.split("/api/media/", 1)[-1]
        try:
            p = (_MEDIA_DIR / rel).resolve()
            p.relative_to(_MEDIA_DIR.resolve())   # raises if p escapes _MEDIA_DIR
        except (ValueError, OSError):
            logger.warning(f"rejected media path outside media dir: {rel!r}")
            return None
        return p if p.exists() else None
    # S3/CDN URL: reconstruct local path (Phase 2 writes locally before uploading)
    return _s3_url_to_local(url)


def _ref_preview_bytes(url: str) -> dict | None:
    """Resolve the config character/scene ref preview URL → its bytes. refs_used stores person/scene
    as the STABLE endpoint URL `/api/refs/{character,scene}/preview` (not the underlying source, which
    can be swapped per brand/upload), so the render path must resolve it the same way `ref_preview`
    does — otherwise Omni silently loses the person + scene locks. `_settings` is the global settings
    singleton (== container.settings, mutated by upload_ref), so this reflects uploaded refs too."""
    kind = {"/api/refs/character/preview": "character",
            "/api/refs/scene/preview": "scene"}.get((url or "").split("?")[0])
    if not kind:
        return None
    # "character" resolves per run (the run's chosen host, else the default one); "scene" is still
    # the single global fallback that a brand's own scene normally replaces upstream.
    src = (character_store.active_character_ref() if kind == "character"
           else getattr(_settings.image_gen, _REF_ATTR[kind], "")) or ""
    if not src:
        src = getattr(_settings.image_gen, _REF_ATTR[kind], "") or ""
    if not src:
        return None
    if src.startswith(("http://", "https://")) or "/api/media/" in src:
        return _media_bytes(src)   # S3 / http / S3-off local-copy fallback
    p = Path(src) if Path(src).is_absolute() else ROOT_DIR / src   # config.yaml default, e.g. docs/teacher.png
    if not p.exists():
        return None
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return {"mime": mime, "data": p.read_bytes()}


_MEDIA_MIME_BY_EXT = {".png": "image/png", ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}


def _mime_for(suffix: str) -> str:
    """MIME from a file extension. Most callers here read images, but /steps/video/edit reads a CLIP
    to hand to Omni — without the video entries it was shipping an mp4 labelled `image/jpeg`."""
    return _MEDIA_MIME_BY_EXT.get(suffix.lower(), "image/jpeg")


def _media_bytes(url: str) -> dict | None:
    """Read media bytes → {"mime", "data"}.
    Supports /api/media/ local URLs, S3 URLs (local copy), generic HTTP URLs (fetch), and the
    config character/scene ref preview URLs (/api/refs/*/preview → the configured ref file)."""
    if (url or "").startswith("/api/refs/"):
        return _ref_preview_bytes(url)
    path = _safe_media_path(url)
    if path is not None:
        return {"mime": _mime_for(path.suffix), "data": path.read_bytes()}
    if (url or "").startswith(("http://", "https://")):
        data = _fetch_bytes(url)
        if data is None:
            return None
        clean = url.lower().split("?")[0]
        return {"mime": _mime_for("." + clean.rsplit(".", 1)[-1] if "." in clean else ""), "data": data}
    return None


def _extract_last_frame_bytes(video_data: bytes) -> bytes | None:
    """The literal last frame of a rendered clip, as a PNG — this is what the Omni POC actually
    chains shot-to-shot with (gemini_omni.md: "this frame becomes the first frame of the next
    clip"), not a separately-generated still image. Best-effort: returns None on any ffmpeg failure
    (short/corrupt clip, missing binary) rather than raising, since a missing continuity frame
    should fall back to the shot's own still image, never block the render."""
    with tempfile.TemporaryDirectory(prefix="omni_lastframe_") as tmp:
        video_path = Path(tmp) / "clip.mp4"
        video_path.write_bytes(video_data)
        out_path = Path(tmp) / "last.png"
        try:
            # Seek 2s from the end (safely inside almost any clip Omni/Veo produce, which run
            # several seconds) and grab the LAST decodable frame from that point on.
            subprocess.run(
                [_FFMPEG_BIN, "-y", "-sseof", "-2", "-i", str(video_path), "-update", "1", "-q:v", "2", str(out_path)],
                capture_output=True, check=True, timeout=30,
            )
            return out_path.read_bytes() if out_path.exists() else None
        except Exception as e:  # noqa: BLE001 — best-effort, caller falls back to the shot's still image
            logger.warning(f"extract last frame failed: {e}")
            return None


def _probe_duration_bytes(video_data: bytes) -> float:
    """Real duration (seconds) of already-in-memory video bytes — Omni has no duration KNOB (unlike
    Veo's config.yaml video_gen.duration_seconds), it renders "roughly 4-10s" on its own, so an
    Omni clip's persisted MediaAsset.duration_seconds must come from probing the actual output, not
    from Veo's configured value. Returns 0.0 on any failure (never blocks persisting the clip)."""
    with tempfile.TemporaryDirectory(prefix="omni_probe_") as tmp:
        video_path = Path(tmp) / "clip.mp4"
        video_path.write_bytes(video_data)
        try:
            r = subprocess.run(
                [_FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True, timeout=15,
            )
            return float(r.stdout.strip())
        except Exception:  # noqa: BLE001 — best-effort, never blocks persisting the clip
            return 0.0


# Omni's real single-clip cap, observed in production (~10s; the model just stops there). Batches
# render under a safety margin so a slightly-under estimate doesn't get truncated/rejected.
OMNI_SAFE_CLIP_SECONDS = 8.5


def _estimate_shot_seconds(voice_over: str) -> float:
    """Rough PRE-render duration estimate for a shot, from its voice_over — used only to decide how
    many consecutive shots can be bin-packed into ONE Omni clip (see OMNI_SAFE_CLIP_SECONDS above);
    never trust this as an actual render duration. _probe_duration_bytes reports what Omni actually
    produced, after the fact.

    Character-based via estimate_narration_seconds: the old words/2.5 rule read every Thai line as
    two or three words and pinned this at its 2.0s floor, so scenes packed three and four shots into
    one clip whose narration could never fit."""
    from app.services.gemini_client import estimate_narration_seconds   # local, like every use here

    if not (voice_over or "").strip():
        return 3.0  # silent/action beat — assume a short default rather than 0
    return max(2.0, estimate_narration_seconds(voice_over))


async def _omni_prev_frame_ref(prev_video_url: str) -> dict | None:
    """The PREVIOUS shot's rendered clip's actual last frame, ready to use as this shot's
    FIRST_FRAME — the real Omni continuity mechanism (see _extract_last_frame_bytes). None if there
    is no previous video, it can't be fetched, or frame extraction fails."""
    if not prev_video_url:
        return None
    prev_video = _media_bytes(prev_video_url)
    if not prev_video or not prev_video.get("data"):
        return None
    frame = await asyncio.to_thread(_extract_last_frame_bytes, prev_video["data"])
    return {"mime": "image/png", "data": frame} if frame else None


def _resolve_ref_url(url: str, db: Session) -> str:
    """Map a refs_used preview-endpoint URL → the direct reachable URL it points at (so logs/refs
    show the real S3 link, not the indirection). brand-scene preview → the scene's asset.s3_url;
    /api/refs/{kind}/preview → the configured character/scene ref; anything else unchanged."""
    u = (url or "").split("?")[0]
    if u.startswith("/api/brands/") and u.endswith("/preview") and "/scenes/" in u:
        try:
            parts = u.strip("/").split("/")   # api brands {bid} scenes {sid} preview
            brand_id, scene_id = parts[2], int(parts[4])
            scene = next((s for s in BrandRepo(db).list_scenes(brand_id) if s.id == scene_id), None)
            asset = MediaAssetRepo(db).get(scene.image_asset_id) if (scene and scene.image_asset_id) else None
            if asset and asset.s3_url:
                return asset.s3_url
        except Exception as e:  # noqa: BLE001
            logger.warning(f"_resolve_ref_url brand-scene failed for {url}: {e}")
        return url
    if u == "/api/refs/character/preview":
        return character_store.active_character_ref() or _settings.image_gen.character_ref or url
    if u == "/api/refs/scene/preview":
        return _settings.image_gen.scene_ref or url
    return url


def _resolve_ref_bytes(url: str, db: Session) -> dict | None:
    """Load a refs_used URL → bytes, resolving internal preview-endpoint URLs the render can't fetch
    directly. brand-scene preview → the scene's real S3 URL; everything else (direct S3/http, and
    /api/refs/* which _media_bytes handles via _ref_preview_bytes incl. the local docs/ default) goes
    straight to _media_bytes."""
    u = (url or "").split("?")[0]
    if u.startswith("/api/brands/") and u.endswith("/preview") and "/scenes/" in u:
        s3 = _resolve_ref_url(url, db)
        return _media_bytes(s3) if s3 != url else None
    return _media_bytes(url)


def _omni_images_for_shot(
    image: dict | None, last_image: dict | None, shot_id: int | None, db: Session, *,
    prev_frame: dict | None = None, match_cut: bool = False,
) -> list[dict]:
    """Ordered role-tagged image list for the Omni multi-ref engine.

    FIRST_FRAME is the literal opening frame Omni must open on — when `prev_frame` (the actual last
    frame extracted from the PREVIOUS shot's rendered clip, see _omni_prev_frame_ref) is available,
    THAT is FIRST_FRAME for real shot-to-shot continuity, and the shot's own still image (from
    Step4) is demoted to an IMAGE_REF describing what this shot's action should arrive at. Without a
    previous clip (first shot in a scene, or the previous shot has no rendered video yet), the
    shot's own still image is FIRST_FRAME instead, same as before.

    match_cut=True keeps that SAME order — only the opening frame's rule differs (compositional match
    rather than pixel-for-pixel). Putting the prev frame in an IMAGE_REF slot instead was tried and
    failed in practice: Omni read it as a DESTINATION, opening on this shot's own frame and drifting
    toward the previous one — the reverse of a match cut.

    Then the person/scene refs selected by `_omni_shot_ref_urls` (storyboard-state rules with config
    fallbacks — the same list the 🖼 strip and the prompt header are built from). `last_image`
    (the NEXT shot's still frame, when use_last_frame is on) is appended as a continuity IMAGE_REF."""
    images: list[dict] = []
    seen_urls: set[str] = set()
    if match_cut and prev_frame and prev_frame.get("data"):
        images.append({**prev_frame, "role": "FIRST_FRAME", "kind": "prev_ref",
                       "label": "the previous shot's ending frame — open matching its composition and framing"})
        if image and image.get("data"):
            images.append({**image, "role": "IMAGE_REF", "kind": "still",
                           "label": "this shot's own generated frame — the step/state the action should reach"})
    elif prev_frame and prev_frame.get("data"):
        # continuous: the previous clip's real last frame is the hard opening frame (FIRST_FRAME)
        images.append({**prev_frame, "role": "FIRST_FRAME", "kind": "prev",
                       "label": "the exact last frame of the previous shot's clip — continue from here"})
        if image and image.get("data"):
            images.append({**image, "role": "IMAGE_REF", "kind": "still",
                           "label": "this shot's own generated frame — the step/state the action should reach"})
    elif image and image.get("data"):
        # non-continuous: the shot's own generated frame IS the literal opening frame (FIRST_FRAME)
        images.append({**image, "role": "FIRST_FRAME", "kind": "still", "label": "this shot's own generated frame"})
    if shot_id is not None:
        # SAME selection as the 🖼 strip / header (_omni_shot_ref_urls) — person/scene per the
        # storyboard-state rules, incl. config fallbacks — so strip == header == images actually sent.
        for r in _omni_shot_ref_urls(shot_id, db):
            kind, url = r["kind"], r["url"]
            if url in seen_urls:
                logger.info(f"omni ref shot={shot_id} DROP kind={kind!r} (dup) url={url}")
                continue
            ref_bytes = _resolve_ref_bytes(url, db)
            if not ref_bytes:
                logger.warning(f"omni ref shot={shot_id} DROP kind={kind!r} (bytes unreadable) url={url}")
                continue
            seen_urls.add(url)
            disp_url = _resolve_ref_url(url, db)   # show the real S3 link in logs/Langfuse, not the preview endpoint
            images.append({**ref_bytes, "role": "IMAGE_REF", "kind": kind,
                           "url": disp_url,
                           "label": f"{kind}: {r['label']}" if r["label"] else kind})
            logger.info(f"omni ref shot={shot_id} KEEP kind={kind!r} url={disp_url}")

        # The user's own attachments go LAST so they never displace the identity/structure locks or
        # the opening frame, and so adding one only ever appends a tag instead of renumbering the
        # ones an already-authored prompt cites.
        for r in _omni_user_refs(shot_id, db):
            url = r["url"]
            if url in seen_urls:
                logger.info(f"omni ref shot={shot_id} DROP kind='user' (dup) url={url}")
                continue
            ref_bytes = _resolve_ref_bytes(url, db)
            if not ref_bytes:
                logger.warning(f"omni ref shot={shot_id} DROP kind='user' (bytes unreadable) url={url}")
                continue
            seen_urls.add(url)
            disp_url = _resolve_ref_url(url, db)
            images.append({**ref_bytes, "role": "IMAGE_REF", "kind": "user", "url": disp_url,
                           "label": r["label"] or "user reference"})
            logger.info(f"omni ref shot={shot_id} KEEP kind='user' url={disp_url}")
    logger.info(f"omni images for shot={shot_id}: sent {len(images)} "
                f"roles={[(i['role'], i.get('kind')) for i in images]}")
    return images


def _refs_used_of(images: list[dict] | None) -> list[dict] | None:
    """The ref manifest of ONE render, trimmed to what is worth keeping on the asset row.

    Drops the `data` bytes and keeps the four fields that answer "what was this clip made from" —
    the video analogue of ShotImagePlan.refs_used. None when there is nothing to record, so the
    column stays NULL rather than an empty list for renders that attached no references.
    """
    out = [{"kind": i.get("kind", ""), "label": i.get("label", ""),
            "url": i.get("url", ""), "tag": i.get("tag", "")}
           for i in (images or [])]
    return out or None


def _omni_user_refs(shot_id: int | None, db: Session) -> list[dict]:
    """Reference images the user attached to this shot by hand, in their chosen order.

    `label` is their own note — the only thing that tells the prompt author what the image is for
    (see _omni_ref_rule's "user" branch). Mirrors the image side's extra_refs + ref_notes, but
    persisted per shot because the authored prompt_video cites each ref by position tag.
    """
    if not shot_id:
        return []
    # Slot rows (kind person/kitchen) live in the same table but are answers ABOUT the automatic
    # refs, not images to append — _omni_shot_ref_urls consumes those.
    return [{"url": r.url, "label": (r.note or "").strip()}
            for r in StoryboardRepo(db).list_video_refs(shot_id)
            if (r.kind or "user") == "user" and (r.url or "").strip()]


# Omni image refs = ONLY the person + the scene/background (the shot's own generated frame is added
# separately). Ingredient/keyword/state refs are Step-4 image-gen aids Omni doesn't need.
_OMNI_REF_KINDS = {"person", "kitchen", "scene"}

# Short human role label per kind — for the Step5 "what does each tag map to" inspection panel.
_OMNI_KIND_ROLE = {
    "prev": "เฟรมสุดท้ายคลิปก่อน (ต่อเนื่อง)", "prev_ref": "เฟรมสุดท้ายคลิปก่อน (match cut)",
    "still": "รูป generate ของช็อตนี้ (ขั้นตอน)",
    "person": "อ้างอิงคน (ล็อกหน้าตา/แต่งกาย)", "kitchen": "อ้างอิงฉาก (ล็อกโครงสร้าง)",
    "scene": "อ้างอิงฉาก (ล็อกโครงสร้าง)", "ingredient": "อ้างอิงวัตถุดิบ",
    "state": "อ้างอิงสถานะอาหาร", "keyword": "อ้างอิง subject", "next": "เฟรมช็อตถัดไป (ปิดท้าย)",
    "user": "รูปที่แนบเอง",
}


def _omni_shot_ref_urls(shot_id: int, db: Session) -> list[dict]:
    """เลือก ref คน/ฉากของช็อตตามกฎ storyboard state — แหล่งเดียวที่ทั้ง 🖼 strip, header/listing/lock
    block และ render ใช้ร่วมกัน. Returns [{kind, url, label}] in send order.

    UNION ของสองแหล่ง — กฎเป็นตัว "เติม" ไม่ใช่ตัว "ตัด":
    (ก) ref ที่ storyboard ใช้สร้างรูปช็อตนี้จริง (`refs_used` kind person/kitchen/scene) → ไปครบเสมอ
        เพราะวิดีโอต้องอ้างของชุดเดียวกับที่รูปถูกสร้างมา
    (ข) กฎ storyboard state เติมเมื่อ (ก) ขาด: มีคน (shot_kind=person หรือ prompt เอ่ยถึงคน/มือ) → เติม
        config character ref; ช็อตมีคน → เติม brand scene ref ตาม scene_match/default"""
    repo = StoryboardRepo(db)
    shot = repo.get_shot(shot_id)
    plan = repo.get_image_plan(shot_id)
    if shot is None:
        return []
    src: dict[str, str] = {}   # kind → url ที่ storyboard สร้างมา (แหล่ง URL)
    for r in (plan.refs_used if plan else []):
        kind = (r.kind or "").lower()
        if kind in _OMNI_REF_KINDS and r.url and kind not in src:
            src[kind] = r.url
    person_needed = shot.shot_kind == "person" or _prompt_has_person(f"{shot.prompt_img} {shot.motion_description}")
    refs: list[dict] = []
    if "person" in src or person_needed:
        _char = character_store.active_character_ref() or _settings.image_gen.character_ref or ""
        url = src.get("person") or (_char if _char.startswith(("http://", "https://")) else "/api/refs/character/preview")
        refs.append({"kind": "person", "url": url, "label": "person"})
    _scene_url = src.get("kitchen") or src.get("scene")
    if _scene_url or person_needed or shot.shot_kind == "person":
        if not _scene_url:
            _scenes = scene_store.active_scenes()
            _match = (plan.scene_match if plan else "") or ""
            _sel = next((s for s in _scenes if s.get("id") == _match), None) or scene_store.active_default_scene()
            _scene_url = (_sel or {}).get("image") or "/api/refs/scene/preview"
        refs.append({"kind": "kitchen", "url": _scene_url, "label": "kitchen"})
    # (ค) ผู้ใช้ตัดสินท้ายสุด — the UNION above is what the RULE wants; a slot row is what the user
    # wants, and it wins. url → replaces that ref, empty url → drops it, no row → rule stands. This
    # is the video counterpart of the image side's ref_person/ref_scene + removed_refs, except it is
    # persisted: prompt_video cites each ref by position tag, so a session-local choice would leave
    # the stored prompt pointing at an image the render no longer sends.
    slots = {r.kind: (r.url or "").strip()
             for r in repo.list_video_refs(shot_id) if (r.kind or "user") != "user"}
    if slots:
        refs = [{**r, "url": slots[r["kind"]], "source": "user"} if slots.get(r["kind"]) else r
                for r in refs if r["kind"] not in slots or slots[r["kind"]]]
    return refs


def _prev_last_frame_url(prev_shot, db: Session) -> str | None:
    """The prev shot's clip's last frame as an IMAGE url. Assets rendered before last_frame_s3_url
    existed have none — extract it from the stored clip once, on first ask, and keep it.

    A clip's S3 key is fixed per shot (`<scene>_<no>.mp4`, no timestamp — unlike images), so a
    re-render OVERWRITES it and every render of the same shot yields the identical URL. The `?v=`
    carries the asset id so the URL still changes: without it the browser would keep serving the
    previous render's frame from cache, and the strip — which exists to show what will actually be
    sent — would quietly show the wrong one."""
    asset = prev_shot.current_video_asset if prev_shot else None
    if asset is None:
        return None
    if asset.last_frame_s3_url:
        return f"{asset.last_frame_s3_url}?v={asset.id[:8]}"
    # ponytail: blocking download+ffmpeg on the request thread — one-off legacy backfill only;
    # move to a background job if it ever runs on more than a handful of old assets.
    try:
        clip = _media_bytes(asset.s3_url)
        if not clip or not clip.get("data"):
            return None
        frame = _extract_last_frame_bytes(clip["data"])
        if not frame:
            return None
        key = asset.s3_key.rsplit(".", 1)[0] + "_lastframe.png"
        url = _persist_media(key, frame, "image/png")
        asset.last_frame_s3_url = url
        db.commit()
        return f"{url}?v={asset.id[:8]}"
    except Exception:  # noqa: BLE001 — a missing thumbnail must not blank the panel
        db.rollback()
        logger.warning(f"last-frame backfill failed for asset {asset.id}", exc_info=True)
        return None


def _omni_ref_manifest_preview(shot_id: int, db: Session, *, prev_shot_id: int | None,
                               match_cut: bool = False) -> list[dict]:
    """URL-based mirror of _omni_images_for_shot ORDER (no bytes) for the UI inspection panel.

    Same tag order the render will send, so what the user checks == what Omni gets. FIRST_FRAME:
    continuous OR match_cut (prev_shot_id given) → the previous shot's clip, shown pending until it
    is rendered, then auto-fills; scene-opening → this shot's own generated image. MUST stay in sync
    with _omni_images_for_shot above (kept parallel, not shared, to avoid touching the working render
    path)."""
    repo = StoryboardRepo(db)
    shot = repo.get_shot(shot_id)
    own_url = shot.current_image_asset.s3_url if (shot and shot.current_image_asset) else ""
    items: list[dict] = []
    if match_cut and prev_shot_id is not None:
        prev_lf = _prev_last_frame_url(repo.get_shot(prev_shot_id), db)
        items.append({"role": "FIRST_FRAME", "kind": "prev_ref", "url": prev_lf, "pending": prev_lf is None,
                      "label": "previous shot's ending frame (match cut)"})
        if own_url:
            items.append({"role": "IMAGE_REF", "kind": "still", "url": own_url, "pending": False})
    elif prev_shot_id is not None:
        prev_lf = _prev_last_frame_url(repo.get_shot(prev_shot_id), db)
        items.append({"role": "FIRST_FRAME", "kind": "prev", "url": prev_lf, "pending": prev_lf is None})
        if own_url:
            items.append({"role": "IMAGE_REF", "kind": "still", "url": own_url, "pending": False})
    elif own_url:
        # non-continuous: the shot's own generated frame IS the opening frame
        items.append({"role": "FIRST_FRAME", "kind": "still", "url": own_url, "pending": False})
    for r in _omni_shot_ref_urls(shot_id, db):   # กฎเลือก ref เดียวกับ render (person/scene + fallback)
        items.append({"role": "IMAGE_REF", "kind": r["kind"], "url": r["url"],
                      "pending": False, "label": r["label"]})
    # Same trailing position as the render path — the two orders must match or the tags the author
    # writes point at different images than the ones actually sent. The dedup mirrors the render's
    # `seen_urls`: now that the attach picker offers the very frames the rules already send, an
    # attached duplicate would be listed here, dropped at render, and every later tag would shift.
    _seen = {it["url"] for it in items if it.get("url")}
    for r in _omni_user_refs(shot_id, db):
        if r["url"] in _seen:
            continue
        _seen.add(r["url"])
        items.append({"role": "IMAGE_REF", "kind": "user", "url": r["url"],
                      "pending": False, "label": r["label"]})
    _omni_assign_tags(items)
    for it in items:
        it["role_desc"] = _OMNI_KIND_ROLE.get(it["kind"], "อ้างอิง")
    return items


def _omni_assign_tags(items: list[dict]) -> list[dict]:
    """Set each item's `tag` (`<FIRST_FRAME>@ImageN` / `<IMAGE_REF_k>@ImageN`) from its position in
    SEND ORDER — the single tag authority for both the 🖼 strip and the render-time header/lock block,
    so ImageN always matches the image actually sent in that slot. Mutates + returns `items`."""
    ref_i = 0
    for i, it in enumerate(items, start=1):
        if it.get("role") == "FIRST_FRAME":
            it["tag"] = f"<FIRST_FRAME>@Image{i}"
        else:
            it["tag"] = f"<IMAGE_REF_{ref_i}>@Image{i}"
            ref_i += 1
    return items


def _omni_lock_block(items: list[dict]) -> str:
    """Deterministic per-ref usage/lock rules for the FINAL prompt. The LLM often drops a ref it should
    honor (e.g. omits the person even when a hand is in frame → the appearance lock is lost). This block
    is code-generated from the manifest items (correct tags) + the shared POC-strength `_omni_ref_rule`,
    so every attached image ALWAYS carries its explicit lock instruction, whatever the LLM wrote."""
    if not items:
        return ""
    from app.services.gemini_client import _clamp_clip_seconds, _omni_ref_rule
    lines = ["Reference usage — these images are attached; honor EACH one exactly, in addition to the description above:"]
    for it in items:
        # body style: @ImageN binding lives only in the [# Sources] header — here the FIRST_FRAME is
        # plain "ImageN" and a reference is its bare "<IMAGE_REF_k>" (matches _normalize_omni_tags)
        full = it.get("tag", "")                       # "<IMAGE_REF_2>@Image3"
        img_n = full.split("@", 1)[-1]                 # "Image3"
        if it.get("role") == "FIRST_FRAME":
            if it.get("kind") == "prev_ref":
                # match cut: same opening slot as continuous, but a compositional match rather than a
                # literal continuation — the scene may change, the visual structure must not jump
                lines.append(f"- {img_n} = OPENING FRAME (the previous clip's last frame); the clip opens on its "
                             "composition, framing and camera angle so the join reads as a match cut, and all motion "
                             "grows from there.")
            else:
                src = ("the previous clip's last frame" if it.get("kind") == "prev"
                       else "this shot's own generated frame")
                lines.append(f"- {img_n} = OPENING FRAME ({src}); the clip opens on it pixel-for-pixel and all motion grows from it.")
        else:
            bare = full.split("@", 1)[0]               # "<IMAGE_REF_2>"
            lines.append(f"- {bare} ({img_n}) = {_omni_ref_rule(it.get('kind', ''), it.get('label', ''))}")
    # Nothing above says a reference image may NOT be frame 0 — only that the OPENING FRAME one is.
    # Omni read that gap as permission and opened a tight insert on the kitchen ref. Stated once here,
    # in code, so it holds whatever the author wrote. Must stay a "- " line: _OMNI_USAGE_BLOCK_RE only
    # strips the header plus consecutive "- " lines, and the block is rebuilt from scratch each render.
    if len(items) > 1:
        lines.append("- ONLY the OPENING FRAME image above may be frame 0. Every other attached image is a lock "
                     "applied INSIDE the shot (identity, structure) — never open on one, never cut to one, and "
                     "never treat one as a frame of this clip.")
    return "\n".join(lines)


def _omni_source_header(items: list[dict]) -> str:
    """POC-style source header binding every tag to its image number, in send order:
    `[# Sources <FIRST_FRAME>@Image1] [# References <IMAGE_REF_0>@Image2 <IMAGE_REF_1>@Image3]`."""
    if not items:
        return ""
    sources = [it["tag"] for it in items if it.get("role") == "FIRST_FRAME"]
    refs = [it["tag"] for it in items if it.get("role") != "FIRST_FRAME"]
    parts = []
    if sources:
        parts.append(f"[# Sources {' '.join(sources)}]")
    if refs:
        parts.append(f"[# References {' '.join(refs)}]")
    return " ".join(parts)


# The locked, code-owned regions of an Omni prompt — stripped and rebuilt on every render so they can
# never drift from the images actually sent (see _omni_apply_manifest).
_OMNI_HEADER_LINE_RE = re.compile(r"^\[# Sources[^\n]*\n?", re.M)
_OMNI_USAGE_BLOCK_RE = re.compile(r"^Reference usage —[^\n]*(?:\n- [^\n]*)*\n?", re.M)
_OMNI_USE_LINE_RE = re.compile(
    r"^Use Image\d+ as (?:the starting frame|a reference for the video generation)\.\n?", re.M)


_OMNI_OPENING_LINE_RE = re.compile(r"^- Image\d+ = OPENING FRAME \(([^)]*)\)", re.M)

# Each shot's prompt carries its own "Target length: about N seconds." (see OMNI_DURATION_BLOCK). A
# group render joins several of those into ONE clip, where N conflicting lines tell the renderer
# nothing — so they are stripped and a single line for the batch total fronts the joined prompt,
# exactly as _omni_apply_manifest does for the [# Sources] header.
_OMNI_TARGET_LEN_RE = re.compile(r"^Target length: about [\d.]+ seconds\.\n?\n?", re.M)


def _omni_prompt_is_stale(shot_id: int, db: Session, items: list[dict]) -> bool:
    """True when the shot's saved prompt_video was authored against a different ref set than `items`.
    _omni_apply_manifest rebuilds the header and lock block at render time, but the body's inline
    <IMAGE_REF_k> keep whatever they meant when the LLM wrote them — so drift there is unrepairable
    and the prompt has to be re-authored. Two checks, because either can move independently:

    - the header, which catches a ref being added or removed (tag count/order changes);
    - what the OPENING FRAME is, which catches a swap that leaves the header identical — e.g. a
      match_cut shot whose opening flipped from its own still to the previous clip's last frame.

    False when there is nothing to compare (no prompt yet, or a prompt predating these blocks)."""
    shot = StoryboardRepo(db).get_shot(shot_id)
    if shot is None or not (shot.prompt_video or "").strip():
        return False
    m = _OMNI_HEADER_LINE_RE.search(shot.prompt_video)
    if not m:
        return False
    if m.group(0).strip() != _omni_source_header(items).strip():
        return True
    was = _OMNI_OPENING_LINE_RE.search(shot.prompt_video)
    now = _OMNI_OPENING_LINE_RE.search(_omni_lock_block(items))
    return bool(was and now and was.group(1) != now.group(1))


def _omni_apply_manifest(prompt: str, images: list[dict]) -> str:
    """Force the LOCKED header + reference-usage blocks onto `prompt`, rebuilt from the images that
    are ACTUALLY being sent to Omni. The Step5-authored prompt carries the blocks as frozen text, so
    any change since authoring (image regen, brand scene swap, hand-edited prompt) would leave a
    stale `[# Sources …]` describing images that no longer line up. Stripping and re-emitting them
    here makes header == 🖼 strip == images sent, always. Idempotent."""
    body = _OMNI_USE_LINE_RE.sub("", _OMNI_USAGE_BLOCK_RE.sub("", _OMNI_HEADER_LINE_RE.sub("", prompt))).strip()
    if not images:
        return body
    _omni_assign_tags(images)
    use_lines = []
    for it in images:
        img_n = it.get("tag", "").split("@", 1)[-1]      # "Image2"
        use_lines.append(f"Use {img_n} as the starting frame." if it.get("role") == "FIRST_FRAME"
                         else f"Use {img_n} as a reference for the video generation.")
    return "\n\n".join([_omni_source_header(images), _omni_lock_block(images), body, "\n".join(use_lines)])


def _omni_manifest_listing(items: list[dict]) -> str:
    """Turn the ordered ref slots into a tag→role listing the prompt author weaves inline
    (POC style). English so it goes straight into the LLM prompt."""
    if not items:
        return "(no attached reference images for this shot)"
    # each ref: what it is (lock rule) + WHEN/WHERE it applies in the clip (used_when) — so the
    # author weaves the tag inline at the right beat, POC-style.
    role_en = {
        "still": "this shot's OWN generated frame — MATCH its subject/props/composition/framing (the intended look of the shot); keep dish/props consistent with it",
        "person": "the on-camera person — lock face/body/outfit/accessories exactly (identity lock); use throughout, wherever the host is on screen",
        # "the fixed background for the whole clip" used to end these two lines and read as "this image
        # IS the clip's background", which invited the author to stage the shot in it — see the matching
        # note in _omni_ref_rule. The structure lock itself is unchanged.
        "kitchen": "the setting/background — lock its layout & structure, add no new or altered fixtures (structure lock), for whatever part of it is visible in THIS shot; it is not a frame of the clip and never the opening — do not widen or re-frame to show more of it",
        "scene": "the setting/background — lock its layout & structure, add no new or altered fixtures (structure lock), for whatever part of it is visible in THIS shot; it is not a frame of the clip and never the opening — do not widen or re-frame to show more of it",
    }
    # User-attached refs carry no fixed role — their meaning is whatever note the user wrote, which is
    # the only channel telling the author this image exists and what to do with it.
    def _role_for(it: dict) -> str:
        if it.get("kind") == "user":
            note = (it.get("label") or "").strip()
            return (f"a reference image the user attached — {note}; use it as described"
                    if note else "a reference image the user attached — take visual cues from it")
        return role_en.get(it.get("kind", ""), "a reference image; render it to match")
    lines = []
    for it in items:
        # The author references each image by its simple @ImageN token; the system rewrites those to
        # the authoritative <FIRST_FRAME>/<IMAGE_REF_k> role tags — so the LLM never has to track the
        # 0-based ref index (which it kept mis-numbering as the 1-based Image number).
        img = "@" + it.get("tag", "@").split("@", 1)[-1]   # "<IMAGE_REF_2>@Image3" → "@Image3"
        if it.get("role") == "FIRST_FRAME":
            if it.get("kind") == "prev_ref":
                lines.append(f"{img} = OPENING FRAME (the previous clip's last frame) at [0s]; the clip MUST open on its "
                             "composition, framing and camera angle so the join reads as a match cut — state this in your "
                             "very first sentence, then grow the motion out of it.")
            else:
                src = ("the previous clip's last frame" if it.get("kind") == "prev"
                       else "this shot's own generated frame")
                lines.append(f"{img} = OPENING FRAME ({src}) at [0s]; the clip MUST open on this exact frame and all motion grows from it.")
        else:
            lines.append(f"{img} = {_role_for(it)}")
    return "\n".join(lines)


def _persist_media(s3_key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Store generated media and return its URL. When S3 is configured, upload and return the
    S3 URL and keep NO local copy (so outputs/media never grows). Otherwise write it under
    _MEDIA_DIR and return a /api/media/ URL (local fallback). Callers no longer write the file
    themselves — this owns the write."""
    if _storage.is_configured():
        return _storage.upload_bytes(s3_key, data, content_type)
    path = _MEDIA_DIR / s3_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return f"/api/media/{s3_key}"


@router.get("/download")
async def media_download(url: str):
    """Same-origin download proxy — a browser ignores <a download> on a cross-origin S3 link, so
    "save this clip" needs the bytes served from our own origin with Content-Disposition. Ported
    from the Omni POC's /omni/download, allow-list and all: only OUR media (AWS_URL prefix or
    /api/media/) may be fetched, so this can never be used as an open proxy."""
    aws = os.getenv("AWS_URL", "").rstrip("/")
    if not ((aws and url.startswith(aws + "/")) or "/api/media/" in (url or "")):
        raise HTTPException(status_code=400, detail="url not allowed")
    p = _safe_media_path(url)
    data = await asyncio.to_thread(p.read_bytes) if p else await asyncio.to_thread(_fetch_bytes, url)
    if not data:
        raise HTTPException(status_code=404, detail="file not found")
    name = (url.split("?")[0].rsplit("/", 1)[-1]) or "download"
    media_type = _mime_for("." + name.rsplit(".", 1)[-1] if "." in name else "")
    return Response(content=data, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


def _local_media_file(url: str, work_dir: Path) -> Path | None:
    """A real local file for a media URL — the local copy if it exists, else the S3/HTTP copy
    downloaded into `work_dir` (caller owns work_dir cleanup). None if unavailable. Needed by
    ffmpeg flows (merge/revoice) once media lives only on S3."""
    p = _safe_media_path(url)
    if p is not None:
        return p
    if (url or "").startswith(("http://", "https://")):
        data = _fetch_bytes(url)
        if data:
            work_dir.mkdir(parents=True, exist_ok=True)
            dest = work_dir / (_safe_name_part((url.split("?")[0].rsplit("/", 1)[-1]) or "clip") or "clip")
            dest.write_bytes(data)
            return dest
    return None


def _safe_name_part(s: str) -> str:
    """Sanitize a value (e.g. scene_id) before using it in a filename — no path separators
    or traversal. A distinct original that sanitizes to a collision keeps a short hash suffix."""
    clean = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s)).strip("._") or "x"
    if clean != str(s):
        clean = f"{clean}_{abs(hash(str(s))) % 10000:04d}"
    return clean


def _img_ext(http_request: Request) -> str:
    ct = http_request.headers.get("content-type", "")
    return "jpg" if ("jpeg" in ct or "jpg" in ct) else "png"


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ─── Runs (v2: first-class row replacing v1's implicit sheet-row-id / active_config.json job) ──

@router.post("/runs")
async def create_run(request: RunCreateRequest, db: Session = Depends(get_db)) -> dict:
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")
    try:
        run_id = _slug(topic)
        run = RunRepo(db).create(id=run_id, topic=topic, title=request.title or topic)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"run": {"id": run.id, "topic": run.topic, "title": run.title, "status": run.status}}


def _run_progress(run_id: str, db: Session) -> dict:
    """Per-category completion flags for one run, computed from what's ACTUALLY persisted —
    ported from v1's JobPicker (research/script/storyboard_text/storyboard_image/video_prompt/
    video_result dots read straight from the Sheet row). v2's `Run.status` field is set once at
    creation and never updated after (nothing calls RunRepo.update_status), so it always freezes
    at "research" regardless of real progress — that's the "ไม่ตรงกับความจริง" bug. Deriving these
    flags fresh from the actual documents/shots every time means there's no separate status value
    to fall out of sync in the first place."""
    research = ResearchRepo(db).get_current(run_id) is not None
    script = ScriptRepo(db).get_current(run_id) is not None
    storyboard_doc = StoryboardRepo(db).get_current(run_id)
    storyboard = storyboard_doc is not None
    shots = [sh for sc in (storyboard_doc.scenes if storyboard_doc else []) for sh in sc.shots]
    return {
        "research": research,
        "script": script,
        "storyboard": storyboard,
        "image": any(sh.current_image_asset_id for sh in shots),
        "video_prompt": any((sh.prompt_video or "").strip() for sh in shots),
        "video": any(sh.current_video_asset_id for sh in shots),
    }


def _run_out(run, db: Session) -> dict:
    """One shape for a run everywhere it is returned. Was duplicated between the list and the
    detail route, so adding a field meant remembering both."""
    return {"id": run.id, "topic": run.topic, "title": run.title, "status": run.status,
            "type": run.type, "avatar": run.avatar,
            "brand_id": run.brand_id, "menu_id": run.menu_id, "character_id": run.character_id,
            "director_prompt_id": run.director_prompt_id,
            "progress": _run_progress(run.id, db)}


@router.get("/runs")
async def list_runs(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)) -> dict:
    return {"runs": [_run_out(r, db) for r in RunRepo(db).list(limit=limit, offset=offset)]}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = RunRepo(db).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run": _run_out(run, db)}


@router.patch("/runs/{run_id}")
async def update_run(run_id: str, request: RunUpdateRequest, db: Session = Depends(get_db)) -> dict:
    """Edit a run's descriptive fields and its active brand/character. Omitted fields are left alone, so the
    UI can send one changed cell. `id` is not editable — see RunRepo.update."""
    repo = RunRepo(db)
    if repo.get(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    given = request.model_dump(exclude_unset=True)
    if "topic" in given and not str(given["topic"]).strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")
    try:
        run = repo.update(run_id, **{k: v for k, v in given.items()
                                     if k in ("topic", "title", "type", "avatar")})
        # brand_id/character_id are pointers, not descriptive fields — they go through
        # set_active_pointers so its partial-update sentinel stays the one place that owns them.
        pointers = {k: (given[k] or None) for k in ("brand_id", "character_id") if k in given}
        if pointers:
            run = repo.set_active_pointers(run_id, **pointers)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"run": _run_out(run, db)}


@router.get("/runs/{run_id}/children")
async def run_children(run_id: str, db: Session = Depends(get_db)) -> dict:
    """What a delete would take with it, per table — the confirm dialog shows this rather than
    asking the operator to delete blind."""
    repo = RunRepo(db)
    if repo.get(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "children": repo.count_children(run_id)}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    """Delete a run and everything under it — research, script, storyboard, media rows, usage and
    regen history. Permanent; the caller is expected to have confirmed against /children."""
    repo = RunRepo(db)
    children = repo.count_children(run_id)
    try:
        ok = repo.delete(run_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    if not ok:
        raise HTTPException(status_code=404, detail="run not found")
    return {"deleted": run_id, "children_deleted": children}


@router.get("/runs/{run_id}/character")
async def get_run_character(run_id: str, db: Session = Depends(get_db)) -> dict:
    """The character this run will actually be made with — its own, or the default one.

    The UI's read-only previews (Step 2's name/description, Step 4's ref photo, Step 5's voice) all
    need the RESOLVED answer, and they cannot use `/refs/character/preview` for it: that route reads
    the per-request contextvar `active_config` sets, which only exists while a pipeline step is
    running. A plain GET from the browser would silently get the default every time."""
    c = CharacterRepo(db).resolve(run_id)
    return {"run_id": run_id, "character": _character_dict(c) if c else None}


@router.get("/runs/{run_id}/research")
async def get_run_research(run_id: str, db: Session = Depends(get_db)) -> dict:
    doc = ResearchRepo(db).get_current(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"No research found for run {run_id}")
    return {"run_id": run_id, "version": doc.version, "research": doc.payload}


@router.get("/runs/{run_id}/script")
async def get_run_script(run_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        doc = ScriptRepo(db).require_current(run_id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # script_config rides along so reopening a run can restore the knobs that produced this script —
    # it was persisted from day one but only the versions list ever returned it.
    return {"run_id": run_id, "version": doc.version, "script": ScriptRepo.to_dict(doc),
            "script_config": doc.script_config or {}}


@router.get("/runs/{run_id}/script/versions")
async def list_run_script_versions(run_id: str, db: Session = Depends(get_db)) -> dict:
    """History view for a run's ScriptDocument — replaces v1's DriveDocBar (manual save/load by
    Drive id); v2 already versions every save, this just lists them so the UI can browse back."""
    versions = ScriptRepo(db).list_versions(run_id)
    return {"run_id": run_id, "versions": [
        {
            "version": v.version, "title": v.title, "is_current": v.is_current,
            "created_at": iso_utc(v.created_at),
            "script_config": v.script_config or {},
        }
        for v in versions
    ]}


@router.get("/runs/{run_id}/script/versions/{version}")
async def get_run_script_version(run_id: str, version: int, db: Session = Depends(get_db)) -> dict:
    doc = ScriptRepo(db).get_version(run_id, version)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"script version {version} not found for run {run_id!r}")
    return {"run_id": run_id, "version": doc.version, "script": ScriptRepo.to_dict(doc)}


@router.post("/runs/{run_id}/script/versions/{version}/activate")
async def activate_run_script_version(run_id: str, version: int, db: Session = Depends(get_db)) -> dict:
    """Switch the run back to an earlier script version — and to that version's own storyboard.

    Storyboards belong to a script version, so the two move together: activating a script points
    the run at that script's newest board, or at none when it was never broken down. `storyboard`
    is null in that second case and Step 3 has to generate — the same state a fresh script
    regenerate leaves behind, and for the same reason.
    """
    doc = ScriptRepo(db).activate_version(run_id, version)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"script version {version} not found for run {run_id!r}")
    board = StoryboardRepo(db).activate_for_script(run_id, version)
    db.commit()
    return {
        "run_id": run_id,
        "version": doc.version,
        "script": ScriptRepo.to_dict(doc),
        "storyboard_version": board.version if board is not None else None,
        "storyboard": _storyboard_to_dict(board, db) if board is not None else None,
    }


@router.get("/runs/{run_id}/storyboard")
async def get_run_storyboard(run_id: str, db: Session = Depends(get_db)) -> dict:
    """The board for the run's CURRENT script version. 404 when that script has never been broken
    down — which is the normal state right after regenerating a script, not an error."""
    try:
        doc = StoryboardRepo(db).require_current(run_id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"run_id": run_id, "version": doc.version, "script_version": doc.script_version,
            "storyboard": _storyboard_to_dict(doc, db)}


@router.get("/runs/{run_id}/storyboard/versions")
async def list_run_storyboard_versions(run_id: str, db: Session = Depends(get_db)) -> dict:
    """History view for a run's StoryboardDocument — same rationale as the script versions
    endpoint above (replaces v1's DriveDocBar manual save/load-by-id)."""
    # Only the current script's boards: a board built from different script text is not a version
    # of this one, and offering it would put the two out of step again.
    script = ScriptRepo(db).get_current(run_id)
    versions = StoryboardRepo(db).list_versions(run_id, script_version=script.version if script else None)
    return {"run_id": run_id, "versions": [
        {"version": v.version, "title": v.title, "is_current": v.is_current,
         "script_version": v.script_version, "created_at": iso_utc(v.created_at)}
        for v in versions
    ]}


@router.get("/runs/{run_id}/storyboard/versions/{version}")
async def get_run_storyboard_version(run_id: str, version: int, db: Session = Depends(get_db)) -> dict:
    doc = StoryboardRepo(db).get_version(run_id, version)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"storyboard version {version} not found for run {run_id!r}")
    return {"run_id": run_id, "version": doc.version, "storyboard": _storyboard_to_dict(doc, db)}


@router.post("/runs/{run_id}/storyboard/versions/{version}/activate")
async def activate_run_storyboard_version(run_id: str, version: int, db: Session = Depends(get_db)) -> dict:
    """Switch the run back to an earlier version — its text AND the images/videos generated while
    it was current (each version owns its shot rows, which hold their own asset ids).

    The GET above only READS an old version; anything generated afterwards still writes against
    whatever row is current, so browsing history and then hitting Generate used to write renders
    onto a version the app no longer considers current. Activating first keeps the shot ids the UI
    holds and the current document in agreement.
    """
    doc = StoryboardRepo(db).activate_version(run_id, version)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"storyboard version {version} not found for run {run_id!r}")
    db.commit()
    return {"run_id": run_id, "version": doc.version, "storyboard": _storyboard_to_dict(doc, db)}


# ── Per-step manual save ─────────────────────────────────────────────────────────────────────────
# The step endpoints persist what they GENERATE; anything typed into the UI lived only in React
# state and evaporated on refresh. These accept the edited document back. Same save_new_version as
# the generate path (so DocVersionBar keeps working), with three additions the generate path does
# not need: a no-op when nothing changed (a save button gets clicked "just in case"), a row lock on
# the run (two tabs saving at once used to make the loser vanish behind a 200 via the
# IntegrityError-swallowing rollback), and — storyboard only — a 409 before an edit that would
# orphan a shot's generated image/video.


def _lock_run(db: Session, run_id: str) -> None:
    """Serialize concurrent saves for one run. save_new_version computes next_version from a read,
    so two concurrent saves collide on UniqueConstraint(run_id, version) — the second would roll
    back and its edits silently drop. FOR UPDATE on the run row makes the second wait its turn.
    (SQLite ignores FOR UPDATE and relies on its single-writer lock — same net effect.)"""
    from app.db.models import Run
    from sqlalchemy import select
    row = db.execute(select(Run).where(Run.id == run_id).with_for_update()).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")


def _storyboard_content(sb: StoryboardDocument) -> dict:
    """The saveable projection of a storyboard — exactly the fields save_new_version/_build_shot
    consume, so the dedupe comparison can't be defeated by fields that never persist anyway
    (shot_id, generated_img, image_plan…)."""
    return {
        "title": sb.title, "ingredients": sb.ingredients, "equipment": sb.equipment,
        "production": sb.production if isinstance(sb.production, dict) else sb.production.model_dump(),
        "scenes": [{
            "scene_id": sc.scene_id, "name": sc.name, "transition_in": sc.transition_in, "music": sc.music,
            "shots": [{
                "no": sh.no, "time": sh.time, "motion_description": sh.motion_description,
                "voice_over": sh.voice_over, "on_screen_text": sh.on_screen_text,
                "key_message": sh.key_message, "shot_kind": sh.shot_kind,
                "join_with_prev": sh.join_with_prev, "screen_direction": sh.screen_direction,
                "prompt_img": sh.prompt_img, "prompt_full": sh.prompt_full, "dish_state": sh.dish_state,
                "prompt_video": sh.prompt_video, "target_seconds": sh.target_seconds,
                "speed": sh.speed, "ken_burns": sh.ken_burns,
                "ingredient_refs": list(sh.ingredient_refs or []),
                "equipment_refs": list(sh.equipment_refs or []),
                "image_subjects": list(sh.image_subjects or []),
            } for sh in sc.shots],
        } for sc in sb.scenes],
    }


@router.put("/runs/{run_id}/storyboard")
async def save_run_storyboard(run_id: str, request: SaveStoryboardRequest, db: Session = Depends(get_db)) -> dict:
    """Persist a UI-edited storyboard as the new current version. No-op when identical to current;
    409 (unless force) when the edit would orphan generated media — links carry forward keyed on
    (scene_id, shot.no), so a renumbered/removed shot loses its image/video with no warning."""
    try:
        sb = StoryboardDocument.model_validate(request.storyboard)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid storyboard JSON: {e}")

    _lock_run(db, run_id)
    repo = StoryboardRepo(db)
    current = repo.get_current(run_id)

    payload = _storyboard_content(sb)
    if current is not None:
        from app.models.storyboard import StoryboardDocument as SBModel
        current_model = SBModel.model_validate(_storyboard_to_dict(current, db))
        if _storyboard_content(current_model) == payload:
            return {"run_id": run_id, "version": current.version, "saved": False}

        if not request.force:
            new_keys = {(sc["scene_id"], sh["no"]) for sc in payload["scenes"] for sh in sc["shots"]}
            lost = [
                {"scene_id": scn.scene_id, "no": sh.no,
                 "has_image": bool(sh.current_image_asset_id), "has_video": bool(sh.current_video_asset_id)}
                for scn in current.scenes for sh in scn.shots
                if (sh.current_image_asset_id or sh.current_video_asset_id)
                and (scn.scene_id, sh.no) not in new_keys
            ]
            if lost:
                raise HTTPException(status_code=409, detail={"lost_media": lost})

    doc = repo.save_new_version(
        run_id, title=payload["title"], ingredients=payload["ingredients"],
        equipment=payload["equipment"], production=payload["production"], scenes=payload["scenes"],
    )
    db.commit()
    return {"run_id": run_id, "version": doc.version, "saved": True}


@router.put("/runs/{run_id}/script")
async def save_run_script(run_id: str, request: SaveScriptRequest, db: Session = Depends(get_db)) -> dict:
    """Persist a UI-edited script as the new current version (no-op when unchanged). No media hangs
    off script rows, so there is no loss check to do."""
    script = request.script or {}
    _lock_run(db, run_id)
    repo = ScriptRepo(db)
    current = repo.get_current(run_id)
    if current is not None:
        same_config = request.script_config is None or (current.script_config or {}) == request.script_config
        if same_config and ScriptRepo.to_dict(current) == script:
            return {"run_id": run_id, "version": current.version, "saved": False}
    doc = repo.save_new_version(
        run_id,
        title=script.get("title", ""),
        production=script.get("production", {}) or {},
        overview=script.get("overview", []) or [],
        parts=script.get("parts", []) or [],
        # an edit-only save has no fresh config — keep the one that produced the current script
        script_config=request.script_config or (current.script_config if current else None),
    )
    db.commit()
    return {"run_id": run_id, "version": doc.version, "saved": True}


@router.put("/runs/{run_id}/research")
async def save_run_research(run_id: str, request: SaveResearchRequest, db: Session = Depends(get_db)) -> dict:
    """Persist a UI-edited research doc as the new current version (no-op when unchanged)."""
    _lock_run(db, run_id)
    repo = ResearchRepo(db)
    current = repo.get_current(run_id)
    if current is not None and current.payload == request.payload:
        return {"run_id": run_id, "version": current.version, "saved": False}
    doc = repo.save_new_version(run_id, payload=request.payload)
    db.commit()
    return {"run_id": run_id, "version": doc.version, "saved": True}


@router.put("/shots/{shot_id}/prompt_video")
async def update_shot_prompt_video(shot_id: int, request: ShotPromptVideoUpdateRequest, db: Session = Depends(get_db)) -> dict:
    """Persist ONE shot's prompt_video in place — no new StoryboardDocument version. Step 5's bulk
    /steps/video_prompts call is what versions the whole storyboard; the per-shot '↻ Prompt รายช็อต'
    regenerate (and manual inline edits) deliberately skip that (see its frontend comment) to avoid
    spamming a new version per click, but were never saved AT ALL as a result — lost on refresh and
    never visible in Step 6. This gives them a real, cheap save path."""
    from app.services.gemini_client import _clamp_clip_seconds   # local, like every other use here
    try:
        board = StoryboardRepo(db)
        shot = board.require_shot(shot_id)
        # Log the OUTGOING text before overwriting it — in-place saves were the one path with no way
        # back, since activating an older storyboard version reverts every other shot too.
        board.log_prompt_video(shot_id, shot.prompt_video, source=request.source or "manual")
        shot.prompt_video = request.prompt_video
        if request.target_seconds is not None:
            shot.target_seconds = _clamp_clip_seconds(request.target_seconds)
        db.commit()
    except NotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        db.rollback()
        raise
    return {"shot_id": shot_id, "prompt_video": shot.prompt_video, "target_seconds": shot.target_seconds}


@router.get("/shots/{shot_id}/video_history")
async def list_shot_video_history(shot_id: int, db: Session = Depends(get_db)) -> dict:
    """Every clip ever rendered for this shot, newest first, with the refs each was made from.

    The shot itself only points at two (current + the one step of undo), but clip keys carry a
    timestamp so nothing was ever overwritten — this exposes the rest. Clips rendered before
    media_assets carried a shot_id are not listed; there is no way to attribute them after the fact.
    """
    try:
        shot = StoryboardRepo(db).require_shot(shot_id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    rows = MediaAssetRepo(db).history_for_shot(shot_id)
    return {"shot_id": shot_id, "current_asset_id": shot.current_video_asset_id,
            "prev_asset_id": shot.prev_video_asset_id,
            "clips": [{"id": a.id, "url": a.s3_url, "status": a.status,
                       "duration_seconds": a.duration_seconds, "created_at": iso_utc(a.created_at),
                       "refs_used": a.refs_used or []} for a in rows]}


@router.post("/shots/{shot_id}/video_history/{asset_id}/activate")
async def activate_shot_video(shot_id: int, asset_id: str, db: Session = Depends(get_db)) -> dict:
    """Point the shot back at an earlier clip. Goes through set_shot_video, so the clip being
    replaced becomes the undo target exactly as a fresh render would — this is a normal switch,
    not a special restore path."""
    try:
        repo = MediaAssetRepo(db)
        asset = repo.get(asset_id)
        if asset is None or asset.shot_id != shot_id:
            raise HTTPException(status_code=404, detail=f"clip {asset_id!r} is not a clip of shot {shot_id}")
        shot = StoryboardRepo(db).set_shot_video(shot_id, asset_id)
        repo.set_status(asset_id, "generated")   # it was superseded when it was replaced
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except NotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        db.rollback()
        raise
    return {"shot_id": shot_id, "current_asset_id": shot.current_video_asset_id,
            "prev_asset_id": shot.prev_video_asset_id}


@router.get("/shots/{shot_id}/video_refs")
async def list_shot_video_refs(shot_id: int, db: Session = Depends(get_db)) -> dict:
    """Reference images the user attached to this shot for VIDEO generation."""
    try:
        repo = StoryboardRepo(db)
        repo.require_shot(shot_id)   # an unknown shot is a 404, not an empty list
        refs = repo.list_video_refs(shot_id)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"shot_id": shot_id,
            "refs": [{"url": r.url, "note": r.note, "kind": r.kind} for r in refs]}


@router.put("/shots/{shot_id}/video_refs")
async def update_shot_video_refs(shot_id: int, request: ShotVideoRefsUpdateRequest,
                                 db: Session = Depends(get_db)) -> dict:
    """Replace the shot's attached video refs. Order is meaningful — it decides each image's tag at
    render, and the authored prompt_video cites those tags, so reordering an existing set will make
    /steps/video/ref_manifest report the prompt stale (which is correct)."""
    try:
        refs = StoryboardRepo(db).set_video_refs(
            shot_id, [{"url": r.url, "note": r.note, "kind": r.kind} for r in request.refs])
        db.commit()
    except NotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        db.rollback()
        raise
    return {"shot_id": shot_id,
            "refs": [{"url": r.url, "note": r.note, "kind": r.kind} for r in refs]}


@router.get("/runs/{run_id}/usage")
async def get_run_usage(run_id: str, db: Session = Depends(get_db)) -> dict:
    """Per-run token/cost report — replaces v1's POST /report/usage_xlsx (which built a Drive-
    uploaded Excel workbook from ephemeral SSE-only usage). v2 persists one usage_records row per
    step call (see _persist_usage above), so this is a durable, queryable report instead of a
    one-off file: per-call records plus totals rolled up by step and by agent."""
    records = UsageRepo(db).list_for_run(run_id)
    rates = CostRateRepo(db).as_dict()   # editable — see /cost_rates; seeded on first read

    def _bucket() -> dict:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration_ms": 0, "calls": 0, "estimated_cost_usd": 0.0}

    totals = _bucket()
    by_step: dict[str, dict] = {}
    by_agent: dict[str, dict] = {}
    for r in records:
        # Estimated PER RECORD (using its own agent's rate/flat-fee) — never from an already-summed
        # bucket, since a step/agent bucket can mix token-billed (Gemini text) and flat-fee (Veo,
        # gpt-image-2) agents whose token counts aren't comparable dollar-for-dollar.
        cost = cost_estimate.estimate_usd(r.agent, r.prompt_tokens, r.completion_tokens, calls=1, rates=rates)
        for bucket in (totals, by_step.setdefault(r.step, _bucket()), by_agent.setdefault(r.agent, _bucket())):
            bucket["prompt_tokens"] += r.prompt_tokens
            bucket["completion_tokens"] += r.completion_tokens
            bucket["total_tokens"] += r.total_tokens
            bucket["duration_ms"] += r.duration_ms
            bucket["calls"] += 1
            bucket["estimated_cost_usd"] += cost

    return {
        "run_id": run_id,
        "records": [
            {
                "step": r.step, "agent": r.agent, "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens, "total_tokens": r.total_tokens,
                "duration_ms": r.duration_ms, "created_at": iso_utc(r.created_at),
            }
            for r in records
        ],
        "totals": totals,
        "by_step": by_step,
        "by_agent": by_agent,
        # Best-effort estimate ONLY — see app/services/cost_estimate.py's module docstring. Never
        # treat this as an actual bill; provider prices change (edit them at GET/PUT /cost_rates)
        # and video/image agents use a flat per-call guess instead of real per-request settings.
        "cost_disclaimer": "ตัวเลขค่าใช้จ่ายเป็นการประมาณการคร่าวๆ จากอัตราที่ตั้งไว้ใน /cost_rates ไม่ใช่ยอดเรียกเก็บเงินจริง — แก้อัตราได้ถ้าราคาผู้ให้บริการเปลี่ยน",
        "usd_to_thb_rate": rates.get("usd_to_thb", 36.0),
    }


@router.get("/cost_rates")
async def list_cost_rates(db: Session = Depends(get_db)) -> dict:
    """Editable $/unit rates the Usage Report's cost estimate is computed from — seeded once from a
    pricing snapshot (see cost_estimate.DEFAULT_RATES), editable here from then on so a provider
    price change never needs a code deploy to fix."""
    rates = CostRateRepo(db).list_all()
    db.commit()   # persists the seed-on-first-read, if this call is what triggered it
    return {"rates": [
        {"key": r.key, "label": r.label, "unit": r.unit, "value": r.value,
         "default_value": cost_estimate.DEFAULT_RATES[r.key][2] if r.key in cost_estimate.DEFAULT_RATES else None,
         "updated_at": iso_utc(r.updated_at)}
        for r in rates
    ]}


@router.put("/cost_rates/{key}")
async def update_cost_rate(key: str, value: float = Body(..., embed=True), db: Session = Depends(get_db)) -> dict:
    try:
        rate = CostRateRepo(db).update(key, value)
        db.commit()
    except KeyError:
        db.rollback()
        raise HTTPException(status_code=404, detail=f"unknown cost rate key: {key!r}")
    except Exception:
        db.rollback()
        raise
    return {"key": rate.key, "label": rate.label, "unit": rate.unit, "value": rate.value,
            "updated_at": iso_utc(rate.updated_at)}


@router.post("/runs/{run_id}/regen")
async def log_regen_event(run_id: str, request: RegenEventRequest, db: Session = Depends(get_db)) -> dict:
    """Fire-and-forget log of a user-initiated regenerate/edit/re-upload action — how often an
    AI-produced result WASN'T good enough as-is. Never call this for a first-time generate (there's
    nothing to regenerate yet); the frontend only fires it when the target already had a result."""
    try:
        RegenRepo(db).create(
            run_id=run_id, step=request.step, action=request.action,
            scene_id=request.scene_id, shot_no=request.no, note=request.note,
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(f"failed to log regen event for run {run_id}")
    return {"status": "logged"}


@router.get("/runs/{run_id}/regen_report")
async def get_regen_report(run_id: str, db: Session = Depends(get_db)) -> dict:
    """Per-run 'how much did the AI need fixing' report — counts of regenerate/edit/upload-replace
    actions grouped by step and by action, plus the individual shots hit most often (the clearest
    signal of exactly which shots/steps are weakest)."""
    events = RegenRepo(db).list_for_run(run_id)

    by_step: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_shot: dict[str, dict] = {}
    for e in events:
        by_step[e.step] = by_step.get(e.step, 0) + 1
        by_action[e.action] = by_action.get(e.action, 0) + 1
        if e.scene_id and e.shot_no is not None:
            shot_key = f"{e.scene_id}.{e.shot_no}"
            bucket = by_shot.setdefault(shot_key, {"scene_id": e.scene_id, "no": e.shot_no, "count": 0, "steps": {}})
            bucket["count"] += 1
            bucket["steps"][e.step] = bucket["steps"].get(e.step, 0) + 1

    top_shots = sorted(by_shot.values(), key=lambda b: b["count"], reverse=True)[:20]

    return {
        "run_id": run_id,
        "total": len(events),
        "by_step": by_step,
        "by_action": by_action,
        "top_shots": top_shots,
        "events": [
            {
                "step": e.step, "action": e.action, "scene_id": e.scene_id, "no": e.shot_no,
                "note": e.note, "created_at": iso_utc(e.created_at),
            }
            for e in events
        ],
    }


def _image_plan_to_dict(plan) -> dict | None:
    """ShotImagePlan ORM row → the shape the frontend's ShotImagePlan type expects. None when the
    shot has never actually been through generate_shot_dynamic (plan row doesn't exist yet)."""
    if plan is None:
        return None
    return {
        "image_generate_type": plan.image_generate_type,
        "refs_used": [
            {"kind": r.kind, "label": r.label, "url": r.url, "source": r.source} for r in plan.refs_used
        ],
        "qc_passed": plan.qc_passed,
        "qc_issues": plan.qc_issues,
        # persisted since a7b8c9d0e1f2 — lets the "ครบ" editor show the last render's exact prompt
        # after a reload instead of needing a paid dry-run; "" for shots rendered before then
        "full_prompt": plan.full_prompt,
        # classifier verdict — debugging aid (was invisible: rows sat at column defaults because the
        # per-shot route cherry-picked 2 fields; see upsert_image_plan's allowed set for the source)
        "classified": plan.classified,
        "has_human": plan.has_human,
        "is_process": plan.is_process,
        "shot_scale": plan.shot_scale,
        "scene_match": plan.scene_match,
    }


def _scene_key(scene_id: str) -> list:
    """Numeric sort key for a dotted scene id: "1.10" → [(0,1),(0,10)] so it lands AFTER "1.2",
    not before it the way a string sort puts it. Non-numeric parts fall back to a string compare
    within the same position (the 0/1 tag keeps ints and strings from ever being compared)."""
    return [(0, int(p)) if p.isdigit() else (1, p) for p in str(scene_id or "").split(".")]


def _storyboard_to_dict(doc, db: Session) -> dict:
    """Round-trip a StoryboardDocument row back to the StoryboardDocument Pydantic shape.

    `generated_img`/`generated_video` are resolved here from current_image_asset_id/
    current_video_asset_id (v1's shot model carried these as plain URL strings directly).

    Scenes are numerically re-sorted here rather than trusting row order: boards written before
    ScriptPart.scenes' order_by was fixed were INSERTED in string order (scene 1.10 before 1.2),
    so a plain read-back — with or without an ORDER BY — replays that mistake. Sorting at the one
    serializer every endpoint uses also repairs those existing boards."""
    return {
        "title": doc.title,
        "ingredients": doc.ingredients,
        "equipment": doc.equipment,
        "production": doc.production,
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "name": scene.name,
                "transition_in": scene.transition_in,
                "music": scene.music,
                "shots": [
                    {
                        "no": shot.no, "time": shot.time, "motion_description": shot.motion_description,
                        "voice_over": shot.voice_over, "on_screen_text": shot.on_screen_text,
                        "key_message": shot.key_message, "shot_kind": shot.shot_kind,
                        "join_with_prev": shot.join_with_prev, "screen_direction": shot.screen_direction,
                        "prompt_img": shot.prompt_img, "prompt_full": shot.prompt_full,
                        "dish_state": shot.dish_state, "prompt_video": shot.prompt_video,
                        "target_seconds": shot.target_seconds,
                        "speed": shot.speed, "ken_burns": shot.ken_burns,
                        "ingredient_refs": shot.ingredient_refs, "equipment_refs": shot.equipment_refs,
                        "image_subjects": shot.image_subjects,
                        "generated_img": _resolve_asset_url(db, shot.current_image_asset_id),
                        "generated_video": _resolve_asset_url(db, shot.current_video_asset_id),
                        "shot_id": shot.id,
                        "image_plan": _image_plan_to_dict(shot.image_plan),
                        # true once this shot has an open Omni edit conversation (see /steps/video/edit)
                        # — the UI uses it to label the button "แก้ต่อ" vs "แก้ไขคลิปด้วย AI" (first turn).
                        "omni_chain": bool(shot.omni_interaction_id),
                        # a clip this one replaced still exists → the undo button is worth showing
                        "has_prev_video": bool(shot.prev_video_asset_id),
                        # same for the image: Step 4 shows its undo only when there is a way back
                        "has_prev_image": bool(shot.prev_image_asset_id),
                        # the image was re-rendered AFTER the clip was made, so the clip animates a
                        # picture that no longer exists. Re-rendering one shot's image deliberately
                        # keeps its clip (it may still be fine) — this is how the UI says otherwise.
                        "video_stale": _asset_newer(db, shot.current_image_asset_id, shot.current_video_asset_id),
                    }
                    for shot in scene.shots
                ],
            }
            for scene in sorted(doc.scenes, key=lambda s: _scene_key(s.scene_id))
        ],
    }


@router.post("/synthesize")
async def synthesize(request: SynthesizeRequest, http_request: Request) -> StreamingResponse:
    topic = request.topic.strip()
    container = http_request.app.state.container
    graph = http_request.app.state.pipeline
    script_config = _merge_script_config(container, request.script_config)
    active_config.set_current_run(request.run_id or "")

    async def stream():
        if not topic:
            yield sse({"type": "error", "message": "Topic must not be empty."})
            return
        async for event in run_pipeline(graph, container, topic, script_config, run_id=request.run_id):
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


# ─── Per-step pipeline (each step fed the previous step's JSON) ─────────────────

@router.post("/steps/research")
async def step_research(request: ResearchStepRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    container = http_request.app.state.container
    cfg = _merge_script_config(container, request.script_config, db, request.run_id or "")
    topic = request.topic.strip()
    active_config.set_current_run(request.run_id or "")

    run_id = request.run_id or ""
    reuse = [r for r in (request.reuse_references or []) if str(r.get("url") or "").strip()]

    def build_result(st: dict) -> dict:
        result = {
            "topic": topic,
            "synthesis": st.get("synthesis", ""),
            "summaries": st.get("summaries", []),
            "references": [
                {"title": v.get("title", ""), "url": v.get("url", "")}
                for v in st.get("confirmed", [])
            ],
            "pacing": _aggregate_pacing(st.get("video_pacing", []), st.get("master_index")),
            "voice_samples": _aggregate_voice_samples(st.get("video_pacing", [])),
            "presentation_notes": _aggregate_presentation_notes(st.get("video_pacing", [])),
            # Structured lists the synthesize node extracted — these seed the run's Menu below,
            # which from then on is the authoritative source for the script's production lists.
            "ingredients": st.get("ingredients", []),
            "equipment": st.get("equipment", []),
        }
        if run_id:
            try:
                ResearchRepo(db).save_new_version(run_id, payload=result)
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist research for run {run_id}")
            # Seed the Menu the moment research lands (was: after the first script generation).
            # PRUNES: re-running research REPLACES the recipe, so a subject the new lists don't
            # mention is gone — otherwise leftovers from the previous research sit in the menu
            # forever and end up back in the script. Items that are still listed keep their row and
            # their uploaded photo. Skipped entirely when both lists are empty: an empty extraction
            # is an LLM miss, not an instruction to wipe the operator's menu.
            # Never raises — research already saved; a missing menu is a nuisance, not data loss.
            if result["ingredients"] or result["equipment"]:
                try:
                    _synced = _sync_menu_subjects(db, run_id, topic, result["ingredients"],
                                                  result["equipment"], prune=True)
                    RunRepo(db).set_active_pointers(run_id, menu_id=run_id)
                    db.commit()
                    if _synced["n_removed"]:
                        logger.info(f"run {run_id}: research re-seed removed {_synced['n_removed']} "
                                    f"menu subject(s) the new lists no longer mention")
                except Exception:
                    db.rollback()
                    logger.exception(f"failed to seed config menu for run {run_id}")
        return result

    async def stream():
        if not topic:
            yield sse({"type": "error", "message": "Topic must not be empty."})
            return

        if reuse:
            # "ใช้ source link เดิม" — re-analyze the SAME videos instead of searching again.
            # Backfill duration/title (best-effort — yt-dlp metadata only, no LLM cost) so the
            # reused videos still feed the pacing/voice-samples features correctly.
            graph = http_request.app.state.steps["research_reuse"]
            confirmed = []
            for r in reuse:
                url = str(r["url"]).strip()
                try:
                    meta = await fetch_video_meta(url)
                except Exception:  # noqa: BLE001 — best-effort; fall back to the caller's own title
                    meta = {}
                confirmed.append({
                    "id": meta.get("id") or url,
                    "url": url,
                    "title": (meta.get("title") or r.get("title") or "").strip(),
                    "duration": meta.get("duration") or 0,
                })
            yield sse({"type": "status", "message": f"Reusing {len(confirmed)} source video(s) — skipping search..."})
            initial = {
                "topic": topic, "confirmed": confirmed, "script_config": cfg,
                "master_url": (request.master_url or "").strip(),
            }
        else:
            graph = http_request.app.state.steps["research"]
            initial = {
                "topic": topic, "target": container.settings.video.max_results, "script_config": cfg,
                "master_url": (request.master_url or "").strip(),
            }

        async for event in run_step(graph, container, initial, build_result, run_id=request.run_id, step="research"):
            if event.get("type") == "usage":
                _persist_usage(db, request.run_id, "research", event)
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


def _run_menu_lists(db: Session, run_id: str) -> tuple[bool, list[str], list[str]]:
    """(menu_locked, ingredients, equipment) for a run — the script step's authoritative lists.

    A menu ROW existing is what locks: its subject names become the script's production lists
    verbatim (generate_script force-overwrites), so a Manage-panel deletion sticks across script
    regenerations. A locked-but-empty menu means the operator deleted everything — respected as
    empty lists. No menu row at all (legacy run researched before menus were seeded at research
    time) = not locked: the LLM extracts from the synthesis as it always did."""
    if not run_id or MenuRepo(db).get_menu(run_id) is None:
        return False, [], []
    refs = MenuRepo(db).list_subject_refs(run_id)
    return (True,
            [r.name for r in refs if r.kind == "ingredient"],
            [r.name for r in refs if r.kind == "equipment"])


def _sync_run_menu(db: Session, run_id: str, script_dump: dict) -> None:
    """Seed the run's config menu from a script's `production` block — LEGACY fallback only.

    The menu is normally seeded the moment research finishes (see step_research's build_result);
    this runs only for a run that has no menu row at all, so old runs keep working. It fires at
    most once: the seed it performs creates the menu row, and every later script generation sees
    `menu_locked` and skips it — which is what lets a Manage-panel deletion survive regeneration.
    The menu is keyed on the RUN ID (picking a run picks its menu via `active_config`).

    Additive: `_menu_from_script_doc` skips subjects that already exist, so images already uploaded
    against old entries survive.

    Never raises. A missing menu is a nuisance; a script that failed to save because its menu could
    not be built is data loss. The caller commits."""
    if not run_id:
        return
    try:
        _menu_from_script_doc(db, run_id, {"script": script_dump})
        RunRepo(db).set_active_pointers(run_id, menu_id=run_id)
    except Exception:  # noqa: BLE001 — the script must persist regardless
        logger.exception(f"failed to sync config menu for run {run_id}")


@router.post("/steps/script")
async def step_script(request: ScriptStepRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    container = http_request.app.state.container
    graph = http_request.app.state.steps["script"]
    cfg = _merge_script_config(container, request.script_config, db, request.run_id or "")
    topic = request.topic.strip()
    run_id = request.run_id or ""
    active_config.set_current_run(run_id)
    # Menu-authoritative lists, read from the DB (not the request): the Menu is curated in the
    # Manage panel, which shares no state with Step 2 — a payload copy could be stale the moment
    # the operator deletes a ref and clicks Generate. No menu row at all = legacy run → the LLM
    # extracts from the synthesis as before and the menu is seeded once below.
    menu_locked, menu_ing, menu_eq = _run_menu_lists(db, run_id)

    def build_result(st: dict) -> dict:
        doc = st.get("script_doc_obj")
        result = {"topic": topic, "script": doc.model_dump() if doc is not None else None}
        if doc is not None and run_id:
            try:
                dumped = doc.model_dump()
                ScriptRepo(db).save_new_version(
                    run_id, title=dumped.get("title", ""), production=dumped.get("production", {}),
                    overview=dumped.get("overview", []), parts=dumped.get("parts", []),
                    script_config=cfg.model_dump(),
                )
                if not menu_locked:   # legacy one-time seed; from then on the Menu is authoritative
                    _sync_run_menu(db, run_id, dumped)
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist script for run {run_id}")
        return result

    async def stream():
        if not request.synthesis.strip():
            yield sse({"type": "error", "message": "synthesis must not be empty."})
            return
        initial = {
            "topic": topic, "synthesis": request.synthesis, "script_config": cfg,
            "voice_samples": request.voice_samples, "presentation_notes": request.presentation_notes,
            "menu_locked": menu_locked, "menu_ingredients": menu_ing, "menu_equipment": menu_eq,
        }
        async for event in run_step(graph, container, initial, build_result, run_id=run_id, step="script"):
            if event.get("type") == "usage":
                _persist_usage(db, run_id, "script", event)
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/script/regenerate_part")
async def step_script_regenerate_part(
    request: ScriptRegeneratePartRequest, http_request: Request, db: Session = Depends(get_db)
) -> StreamingResponse:
    """Regenerate ONE part of an existing script (staircase per-part auto-fit).

    Returns the full ScriptDocument with only ``part_number`` replaced (scene_ids renumbered N.1..N.k);
    other parts are byte-identical. Global timecodes are left to the frontend to reconcile.
    """
    container = http_request.app.state.container
    graph = http_request.app.state.steps["script_regenerate_part"]
    cfg = _merge_script_config(container, request.script_config, db, request.run_id or "")
    topic = request.topic.strip()
    run_id = request.run_id or ""
    active_config.set_current_run(run_id)
    # Same menu-authoritative read as the full-script route — doubly needed here, because the
    # request carries the frontend's script doc whose production block can be stale.
    menu_locked, menu_ing, menu_eq = _run_menu_lists(db, run_id)

    def build_result(st: dict) -> dict:
        doc = st.get("script_doc_obj")
        result = {"topic": topic, "script": doc.model_dump() if doc is not None else None}
        if doc is not None and run_id:
            try:
                dumped = doc.model_dump()
                ScriptRepo(db).save_new_version(
                    run_id, title=dumped.get("title", ""), production=dumped.get("production", {}),
                    overview=dumped.get("overview", []), parts=dumped.get("parts", []),
                    script_config=cfg.model_dump(),
                )
                if not menu_locked:   # legacy one-time seed, same as the full-script route
                    _sync_run_menu(db, run_id, dumped)
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist regenerated script for run {run_id}")
        return result

    async def stream():
        try:
            doc = ScriptDocument.model_validate(request.script)
        except ValidationError as e:
            yield sse({"type": "error", "message": f"Invalid script JSON: {e}"})
            return
        if not (1 <= request.part_number <= len(doc.parts)):
            yield sse({"type": "error", "message": f"part_number {request.part_number} out of range (1..{len(doc.parts)})"})
            return
        if not request.synthesis.strip():
            yield sse({"type": "error", "message": "synthesis must not be empty."})
            return
        initial = {
            "topic": topic, "synthesis": request.synthesis, "script_config": cfg,
            "script_doc_obj": doc, "part_number": request.part_number,
            "target_duration": request.target_duration, "target_scenes": request.target_scenes or 0,
            "min_shots": request.min_shots or 0,
            "menu_locked": menu_locked, "menu_ingredients": menu_ing, "menu_equipment": menu_eq,
        }
        async for event in run_step(graph, container, initial, build_result, run_id=run_id, step="script_regenerate_part"):
            if event.get("type") == "usage":
                _persist_usage(db, run_id, "script_regenerate_part", event)
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/storyboard")
async def step_storyboard(request: StoryboardStepRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    container = http_request.app.state.container
    graph = http_request.app.state.steps["storyboard"]
    cfg = _merge_script_config(container, request.script_config, db, request.run_id or "")
    topic = request.topic.strip()
    run_id = request.run_id or ""
    active_config.set_current_run(run_id)

    def build_result(st: dict) -> dict:
        sb = st.get("storyboard_obj")
        result = {"topic": topic, "storyboard": sb.model_dump() if sb is not None else st.get("storyboard")}
        if sb is not None and run_id:
            try:
                dumped = sb.model_dump()
                StoryboardRepo(db).save_new_version(
                    run_id, title=dumped.get("title", ""), ingredients=dumped.get("ingredients", []),
                    equipment=dumped.get("equipment", []), production=dumped.get("production", {}),
                    scenes=dumped.get("scenes", []),
                    # A regenerate rewrites every shot; the shot now sitting at a given
                    # (scene_id, no) is frequently a different shot, so its predecessor's render
                    # would contradict its own prompt. The previous version keeps those renders and
                    # `POST …/versions/{n}/activate` brings them back.
                    media="drop_all",
                )
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist storyboard for run {run_id}")
        return result

    async def stream():
        try:
            doc = ScriptDocument.model_validate(request.script)
        except ValidationError as e:
            yield sse({"type": "error", "message": f"Invalid script JSON: {e}"})
            return
        initial = {"topic": topic, "script_config": cfg, "script_doc_obj": doc, "target_shots": request.target_shots or 0}
        async for event in run_step(graph, container, initial, build_result, run_id=run_id, step="storyboard"):
            if event.get("type") == "usage":
                _persist_usage(db, run_id, "storyboard", event)
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/image_prompts")
async def step_image_prompts(request: ImagePromptsStepRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    container = http_request.app.state.container
    graph = http_request.app.state.steps["image_prompts"]
    topic = request.topic.strip()
    cfg = _merge_script_config(container, request.script_config, db, request.run_id or "")   # category: request > run.type > config.yaml
    run_id = request.run_id or ""
    active_config.set_current_run(run_id)

    def build_result(st: dict) -> dict:
        sb = st.get("storyboard_obj")
        result = {"topic": topic, "storyboard": sb.model_dump() if sb is not None else st.get("storyboard")}
        if sb is not None and run_id:
            try:
                dumped = sb.model_dump()
                StoryboardRepo(db).save_new_version(
                    run_id, title=dumped.get("title", ""), ingredients=dumped.get("ingredients", []),
                    equipment=dumped.get("equipment", []), production=dumped.get("production", {}),
                    scenes=dumped.get("scenes", []),
                    media="drop_all",   # prompt_img is rewritten → every existing render is off-prompt
                )
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist storyboard (image_prompts) for run {run_id}")
        return result

    async def stream():
        try:
            sb = StoryboardDocument.model_validate(request.storyboard)
            doc = ScriptDocument.model_validate(request.script) if request.script else None
        except ValidationError as e:
            yield sse({"type": "error", "message": f"Invalid storyboard/script JSON: {e}"})
            return
        if doc is None and not (sb.production or {}):
            yield sse({"type": "error", "message": "Need the Script (Step 2) — this storyboard has no embedded production spec."})
            return
        initial = {"topic": topic, "storyboard_obj": sb, "script_doc_obj": doc, "script_config": cfg,
                   "ref_notes": request.ref_notes}
        async for event in run_step(graph, container, initial, build_result, run_id=run_id, step="image_prompts"):
            if event.get("type") == "usage":
                _persist_usage(db, run_id, "image_prompts", event)
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/images")
async def step_images(request: ImagesStepRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    container = http_request.app.state.container
    graph = http_request.app.state.steps["images"]
    topic = request.topic.strip()
    cfg = _merge_script_config(container, request.script_config, db, request.run_id or "")
    run_id = request.run_id or ""
    active_config.set_current_run(run_id)

    def build_result(st: dict) -> dict:
        sb = st.get("storyboard_obj")
        result = {"topic": topic, "storyboard": sb.model_dump() if sb is not None else st.get("storyboard")}
        if sb is not None and run_id:
            try:
                _persist_bulk_storyboard(db, run_id, sb.model_dump(), step="images")
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist storyboard (images) for run {run_id}")
        return result

    async def stream():
        try:
            sb = StoryboardDocument.model_validate(request.storyboard)
        except ValidationError as e:
            yield sse({"type": "error", "message": f"Invalid storyboard JSON: {e}"})
            return
        initial = {"topic": topic, "storyboard_obj": sb, "script_config": cfg, "regenerate_all": request.regenerate_all}
        async for event in run_step(graph, container, initial, build_result, run_id=run_id, step="images"):
            if event.get("type") == "usage":
                _persist_usage(db, run_id, "images", event)
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/video_prompts")
async def step_video_prompts(request: VideoPromptsStepRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Step 5: storyboard → a Veo motion prompt (prompt_video) per shot (batch)."""
    container = http_request.app.state.container
    graph = http_request.app.state.steps["video_prompts"]
    topic = request.topic.strip()
    cfg = _merge_script_config(container, request.script_config, db, request.run_id or "")   # carries voice_desc/category
    voice = _merge_voice_config(container, request, db, request.run_id or "")   # the run's character
    run_id = request.run_id or ""
    active_config.set_current_run(run_id)

    def build_result(st: dict) -> dict:
        sb = st.get("storyboard_obj")
        result = {
            "topic": topic,
            "storyboard": sb.model_dump() if sb is not None else st.get("storyboard"),
            # carry the chosen frame mode forward so Step 6 renders to match the prompts
            "video_mode": {"use_image_seed": request.use_image_seed, "use_last_frame": request.use_last_frame},
        }
        if sb is not None and run_id:
            try:
                dumped = sb.model_dump()
                StoryboardRepo(db).save_new_version(
                    run_id, title=dumped.get("title", ""), ingredients=dumped.get("ingredients", []),
                    equipment=dumped.get("equipment", []), production=dumped.get("production", {}),
                    scenes=dumped.get("scenes", []),
                    # the image still matches its prompt_img; only the clip was made from the
                    # prompt_video this run is replacing
                    media="drop_video",
                )
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist storyboard (video_prompts) for run {run_id}")
        return result

    async def stream():
        try:
            sb = StoryboardDocument.model_validate(request.storyboard)
        except ValidationError as e:
            yield sse({"type": "error", "message": f"Invalid storyboard JSON: {e}"})
            return
        # Per-shot Omni image manifest so the author weaves POC-style tags into prompt_video. Resolved
        # from the REQUEST storyboard's shot_id (works for both bulk and per-shot regen, which sends
        # run_id=null) via the SAME DB source as render → author & render tags/order stay in sync.
        omni_manifests: dict[str, dict] = {}
        try:
            for sc in sb.scenes:
                prev_shot = None
                for shot in sc.shots:
                    if shot.shot_id is not None:
                        _join = shot.join_with_prev or ""
                        joins_prev = _join in ("continuous", "match_cut")
                        prev_id = prev_shot.shot_id if (prev_shot is not None and joins_prev) else None
                        if prev_id is None and joins_prev:
                            # per-shot regen posts a ONE-shot storyboard, so the payload walk above has
                            # no predecessor to offer — resolve it from the DB or the prompt would be
                            # authored without the prev-frame ref the render will actually send.
                            _p = StoryboardRepo(db).get_prev_shot(shot.shot_id)
                            prev_id = _p.id if _p else None
                        items = _omni_ref_manifest_preview(shot.shot_id, db, prev_shot_id=prev_id,
                                                           match_cut=(_join == "match_cut"))
                        header = _omni_source_header(items)
                        omni_manifests[f"{sc.scene_id}:{shot.no}"] = {
                            "header": header,
                            "listing": _omni_manifest_listing(items),
                            "usage": _omni_lock_block(items),
                        }
                        # what the RUNNING process actually authored — the 🖼 strip is the same
                        # `items`, so a mismatch in the log means a stale process, not stale data
                        logger.info(f"omni manifest shot={shot.shot_id} {sc.scene_id}.{shot.no} header={header}")
                    prev_shot = shot
        except Exception:
            logger.exception("omni manifest resolve failed (video_prompts)")

        initial = {
            "topic": topic, "storyboard_obj": sb, "script_config": cfg,
            "use_image_seed": request.use_image_seed, "use_last_frame": request.use_last_frame,
            # Omni narration voice → read by generate_video_prompts node. Comes from the global
            # Character config unless this request explicitly overrode a field.
            "vo_language": voice.language, "vo_pace": voice.vo_pace,
            "vo_tone": voice.tone, "vo_style": voice.style, "vo_gender": voice.gender,
            "omni_manifests": omni_manifests,   # {"<scene_id>:<no>": tag→role listing}
        }
        async for event in run_step(graph, container, initial, build_result, run_id=run_id, step="video_prompts"):
            if event.get("type") == "usage":
                _persist_usage(db, run_id, "video_prompts", event)
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/video/ref_manifest")
def step_video_ref_manifest(request: RefManifestRequest, db: Session = Depends(get_db)) -> dict:
    """Per-shot Omni image-ref manifest (tags + thumbnail URLs, in send order) for the Step5 check
    panel — so the user can verify which image each <FIRST_FRAME>/<IMAGE_REF_k> binds to. A continuous
    shot's FIRST_FRAME is `pending` (no url) until the previous shot's clip is rendered, then fills in.

    `stale` flags a prompt_video authored against a DIFFERENT ref set than the one above (typically
    join_with_prev was changed afterwards, which inserts/removes the prev-frame ref and shifts every
    <IMAGE_REF_k> down). The render rebuilds the header and lock block, but the LLM-written body still
    names the OLD tags, so its "keep the setting as <IMAGE_REF_1>" would now point at the person.

    Plain `def` on purpose: everything below is blocking IO (SQL per shot; the legacy path even
    downloads the clip and runs ffmpeg). As `async def` all of that ran ON the event loop, so a
    burst of these queued behind each other and starved every other request — over a tunnel that
    surfaced as cloudflared "context canceled" on unrelated endpoints."""
    out: dict[str, dict] = {}
    repo = StoryboardRepo(db)
    for it in request.shots:
        try:
            # The manifest resolves a shot's scene ref through `scene_store`, which reads the brand
            # off active_config's contextvar — unbound, a brand's own scene silently degrades to the
            # global scene_ref, so the panel showed a kitchen the render never sends. The request
            # carries no run_id; the shot knows its own. Per shot, since a batch may span runs.
            _shot = repo.get_shot(it.shot_id)
            active_config.set_current_run(
                _shot.storyboard_scene.storyboard_document.run_id if _shot else "")
            items = _omni_ref_manifest_preview(it.shot_id, db, prev_shot_id=it.prev_shot_id,
                                               match_cut=it.match_cut)
            out[str(it.shot_id)] = {"items": items, "stale": _omni_prompt_is_stale(it.shot_id, db, items)}
        except Exception as e:  # noqa: BLE001 — one bad shot shouldn't blank the whole panel
            logger.warning(f"ref_manifest failed for shot {it.shot_id}: {e}")
            out[str(it.shot_id)] = {"items": [], "stale": False}
    return out


@router.post("/steps/image")
async def step_image(request: ImageStepRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Generate ONE storyboard shot's image (per-shot, on demand) — like /steps/video
    but for images. Saves the .png under /api/media and returns its URL."""
    container = http_request.app.state.container
    topic = request.topic.strip()
    # Bind the run BEFORE any graph node runs: active_config resolves brand_id/menu_id from this
    # contextvar, and it is per-request (never inherited). Without it active_config.get() returns {},
    # so the shot silently renders against the DEFAULT scene_ref instead of the run's brand scene,
    # loses every menu subject ref, and drops the brand's theme/mood — /steps/images (bulk) always
    # bound it, which is why "Generate All" looked right and this per-shot button did not.
    active_config.set_current_run(request.run_id or "")

    async def stream():
        if not request.prompt_img.strip():
            yield sse({"type": "error", "message": "prompt_img must not be empty."})
            return
        yield sse({"type": "status", "message": "Generating image..."})
        cfg_gen = container.settings.image_gen
        plan: dict | None = None
        tracker = UsageTracker()
        set_tracker(tracker)
        _t0 = time.perf_counter()
        try:
            if cfg_gen.dynamic_image and request.shot:
                data, plan, _ = await generate_shot_dynamic(
                    container, request.shot, request.prev_shot, topic,
                    all_ingredients=request.all_ingredients, ingredient_images=request.ingredient_images,
                    prev_state_url=request.prev_state_url, used_ingredients=request.used_ingredients,
                    all_equipment=request.all_equipment, equipment_images=request.equipment_images,
                    extra_refs=request.extra_refs, ref_notes=request.ref_notes, removed_refs=request.removed_refs,
                    ref_person=request.ref_person, ref_scene=request.ref_scene, ref_state=request.ref_state,
                    category=_image_category(db, request.run_id or "", request.category, container),
                    prompt_override=request.prompt_override,
                    qc_feedback=request.qc_feedback)
            else:
                base_refs = await load_base_image_refs(container)
                data = await render_shot_image(
                    container, request.prompt_img, request.on_screen_text, request.image_results, base_refs, request.shot_kind
                )
        except Exception as e:  # noqa: BLE001 — surface failure to the client
            logger.exception("Image generation failed")
            yield sse({"type": "error", "message": f"Image generation failed: {e}"})
            return
        if not data:
            yield sse({"type": "error", "message": "Image generation returned nothing (check image_gen config / model)."})
            return

        # unique filename per regenerate → URL changes → React re-renders + browser fetches fresh
        # (overwriting the same name kept the old cached image on screen).
        # ponytail: leaves an orphan png per manual regen — negligible for a per-topic dir
        fname = f"{_safe_name_part(request.scene_id)}_{int(request.no)}_{int(time.time())}.png"
        s3_key = f"{_slug(topic)}_images/{fname}"
        url = _persist_media(s3_key, data, "image/png")
        obs.attach_media(url, "image")

        # v2: track the render as a MediaAsset, and if the request carries a real shot_id, point
        # the storyboard shot at it (replaces v1's plain string overwrite).
        if request.shot_id is not None:
            try:
                asset = MediaAssetRepo(db).create(kind="image", s3_key=s3_key, s3_url=url, run_id=request.run_id)
                board = StoryboardRepo(db)
                board.set_shot_image(request.shot_id, asset.id)
                # Persist the resolved ref plan too — otherwise it only lives in this SSE response
                # and the "Refs" thumbnails go blank again the moment the storyboard is reloaded
                # (GET /runs/{id}/storyboard had nothing to serialize back).
                if plan is not None:
                    # The whole plan, not a hand-picked pair — upsert's `allowed` set is the filter.
                    # Cherry-picking here is why classifier/QC verdicts sat at column defaults for
                    # every per-shot render (only 2 of 21 fields were ever passed). Keys absent from
                    # the dict are left untouched rather than reset.
                    board.upsert_image_plan(
                        request.shot_id,
                        refs_used=plan.get("refs_used"),
                        **{k: v for k, v in plan.items() if k != "refs_used"},
                    )
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist generated image for shot {request.shot_id}")

        ev = {"type": "image_generated", "scene_id": request.scene_id, "no": request.no, "url": url}
        if plan is not None:
            ev["image_plan"] = plan
        yield sse(ev)
        usage_summary = tracker.summary()
        _persist_usage(db, request.run_id, "image", usage_summary)
        yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
        yield sse({"type": "done"})

    return StreamingResponse(obs.traced_stream(stream(), "step.image", run_id=request.run_id, scene_id=request.scene_id, no=request.no, shot_id=request.shot_id), media_type="text/event-stream", headers=_SSE_HEADERS)


class EditRefLoadError(Exception):
    """One of the attached refs couldn't be read — see _load_edit_refs for why this is fatal."""


async def _load_edit_refs(container, urls: list[str], notes: list[str]) -> list[dict]:
    """Extra reference images attached to ONE image edit, in send order — they land after the image
    being edited, so the model sees them as image 2, 3, … Loaded (and downscaled) through the same
    `load_reference` the generate path uses, so an edit doesn't ship full-size originals.

    RAISES when a ref can't be read instead of skipping it: the UI labels each thumbnail with the
    number the model will see, and the user may have typed that number into the instruction ("replace
    the bowl with the one in image 3"). Dropping a ref silently shifts every later number down, so
    the instruction would point at the wrong picture. Better to stop and say which one is broken."""
    cfg_gen = container.settings.image_gen
    out: list[dict] = []
    for i, url in enumerate(urls or []):
        r = await load_reference(_ref_src(url), cfg=container.settings.image_search, max_px=cfg_gen.ref_resize_px)
        if not r:
            logger.warning(f"image edit: could not load extra ref {url}")
            raise EditRefLoadError(f"โหลด ref ที่ image {i + 2} ไม่สำเร็จ — เอาออกหรืออัปใหม่ก่อนสั่งแก้")
        note = (notes[i].strip() if i < len(notes or []) else "")
        label = "Reference the user attached for this edit — take the new object's appearance from it"
        out.append({"label": f"{label}: {note}" if note else label, **r})
    return out


async def _edit_anchor_refs(container, db: Session, shot) -> list[dict]:
    """Character/scene ANTI-DRIFT anchors for an image EDIT, mirroring what the prev-frame generate
    path attaches. An edit used to send only [current image, user refs] — but the provider wrapper
    already tells the model "where the template disagrees with an anchor, THE ANCHOR WINS" naming
    the Character and Background references, so it promised anchors the payload never carried. With
    nothing real to hold onto, each edit re-painted the previous render's softness and drift.

    Person gate: the persisted plan's has_human, else the same heuristic _omni_shot_ref_urls uses.
    Scene gate: the plan's shows_kitchen, else attach unless the shot is a flat-lay style insert —
    an ingredient intro on bare marble must not get a kitchen room forced into it."""
    plan = StoryboardRepo(db).get_image_plan(shot.id)
    if plan is not None:
        wants_person = bool(plan.has_human)
        wants_scene = bool(plan.shows_kitchen)
        scene_match = plan.scene_match or ""
    else:
        wants_person = shot.shot_kind == "person" or _prompt_has_person(f"{shot.prompt_img} {shot.motion_description}")
        wants_scene = shot.shot_kind != "insert"
        scene_match = ""
    if not (wants_person or wants_scene):
        return []
    base = await load_base_image_refs(container, scene_match=scene_match)
    keep = []
    for r in base:
        lbl = str(r.get("label", ""))
        if lbl.startswith("Character reference") and wants_person:
            keep.append(r)
        elif lbl.startswith("Background reference") and wants_scene:
            keep.append(r)
    return keep


@router.post("/steps/image/edit_prompt")
async def step_image_edit_prompt(request: ImageEditPromptRequest, http_request: Request) -> dict:
    """Polish a rough image-edit note into one clear prompt, with the attached refs described so it
    can cite them by number. Best-effort — never fails the caller."""
    container = http_request.app.state.container
    instruction = request.instruction.strip()
    if not instruction:
        return {"instruction": ""}
    polished = await container.gemini.generate_image_edit_prompt(
        instruction, original_desc=request.original_desc, ref_notes=request.ref_notes, mode=request.mode)
    return {"instruction": polished}


@router.post("/steps/image/revise_prompt")
async def step_image_revise_prompt(request: ImagePromptReviseRequest, http_request: Request) -> dict:
    """Rewrite a shot's FULL image prompt so it fixes what the user says was wrong with the render.

    Nothing is generated or saved here — the caller puts the result in the shot's `prompt_full` and
    the user reads it before pressing Generate, which then sends it verbatim as `prompt_override`.
    Deliberately NOT the sibling of /steps/image/edit: that one repaints the existing picture and
    leaves the prompt as wrong as it was, so the next generate reproduces the same fault."""
    container = http_request.app.state.container
    full_prompt, feedback = request.full_prompt.strip(), request.feedback.strip()
    if not (full_prompt and feedback):
        return {"full_prompt": full_prompt}
    return {"full_prompt": await container.gemini.revise_image_prompt(full_prompt, feedback)}


@router.post("/steps/image/edit")
async def step_image_edit(request: ImageEditRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Edit a shot's EXISTING image with a short instruction, instead of re-rendering from
    prompt_img — reuses the SAME provider path as normal generation (_provider_generate: gpt-image-2
    edit endpoint or Gemini image edit, whichever image_gen.provider is configured), just with the
    current image as the sole reference and the instruction as the prompt. Saves as a new MediaAsset
    (history preserved via set_shot_image), same as a normal regenerate."""
    container = http_request.app.state.container
    topic = request.topic.strip()
    active_config.set_current_run(request.run_id or "")   # see /steps/image

    async def stream():
        instruction = request.instruction.strip()
        if not instruction:
            yield sse({"type": "error", "message": "instruction ว่าง"})
            return
        board = StoryboardRepo(db)
        shot = board.get_shot(request.shot_id)
        if shot is None:
            yield sse({"type": "error", "message": "ไม่พบช็อตนี้"})
            return
        img_url = _resolve_asset_url(db, shot.current_image_asset_id)
        if not img_url:
            yield sse({"type": "error", "message": "ช็อตนี้ยังไม่มีรูป — generate ก่อน"})
            return
        base_image = _media_bytes(img_url)
        if not base_image:
            yield sse({"type": "error", "message": "ไม่พบไฟล์รูป (S3/local) — generate ใหม่ก่อน"})
            return

        tracker = UsageTracker()
        set_tracker(tracker)
        _t0 = time.perf_counter()
        yield sse({"type": "status", "message": "กำลังแก้ไขรูป..."})
        # image 1 is always the clip being edited — the provider wrappers key their strict
        # "reproduce it, change only the delta" instruction off this label starting with "Current image".
        refs = [{**base_image, "label": "Current image — apply ONLY the instruction below; otherwise keep "
                                        "the composition, subjects, background, lighting and colours EXACTLY identical"}]
        try:
            refs += await _load_edit_refs(container, request.extra_refs, request.ref_notes)
        except EditRefLoadError as e:
            yield sse({"type": "error", "message": str(e)})
            return
        # Anchors go AFTER the user's refs: the UI numbers those from image 2, and an instruction
        # may reference that number — inserting anchors in between would shift every one of them.
        refs += await _edit_anchor_refs(container, db, shot)
        try:
            data = await _provider_generate(container, instruction, refs)
        except Exception as e:  # noqa: BLE001
            logger.exception("Image edit failed")
            yield sse({"type": "error", "message": f"แก้ไขรูปไม่สำเร็จ: {e}"})
            return
        if not data:
            yield sse({"type": "error", "message": "แก้ไขรูปไม่สำเร็จ (provider ไม่คืนรูป)"})
            return

        fname = f"{_safe_name_part(request.scene_id)}_{int(request.no)}_{int(time.time())}.png"
        s3_key = f"{_slug(topic)}_images/{fname}"
        url = _persist_media(s3_key, data, "image/png")
        obs.attach_media(url, "image")
        try:
            asset = MediaAssetRepo(db).create(kind="image", s3_key=s3_key, s3_url=url, run_id=request.run_id)
            board.set_shot_image(request.shot_id, asset.id)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(f"failed to persist edited image for shot {request.shot_id}")

        yield sse({"type": "image_edited", "scene_id": request.scene_id, "no": request.no, "url": url})
        usage_summary = tracker.summary()
        _persist_usage(db, request.run_id, "image_edit", usage_summary)
        yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
        yield sse({"type": "done"})

    return StreamingResponse(obs.traced_stream(stream(), "step.image_edit", run_id=request.run_id, scene_id=request.scene_id, no=request.no, shot_id=request.shot_id), media_type="text/event-stream", headers=_SSE_HEADERS)


def _outpaint_canvas(image_bytes: bytes, target_w: int, target_h: int) -> bytes:
    """Pad `image_bytes` onto a `target_w`x`target_h` RGBA canvas, original centered — the new
    border area is fully TRANSPARENT (alpha=0), which gpt-image's edit endpoint reads as "regenerate
    this region" directly from the image's own alpha channel (no separate mask needed). Raises
    ValueError if the target is smaller than the source in either dimension (nothing to outpaint)."""
    from PIL import Image

    src = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    if target_w < src.width or target_h < src.height:
        raise ValueError(f"outpaint target {target_w}x{target_h} is smaller than the source {src.width}x{src.height}")
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    canvas.paste(src, ((target_w - src.width) // 2, (target_h - src.height) // 2), src)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _crop_to_same_size(image_bytes: bytes, x: float, y: float, w: float, h: float) -> bytes:
    """Keep the rectangle (fractions of the image, 0-1) and scale it back to the ORIGINAL pixel size.

    LANCZOS, not a model: the caller asked for a reframe, so the pixels that survive must be the ones
    they selected. Sending the crop through gpt-image would add detail but redraw the frame — this
    repo's own sharpen POC measured the host's face drifting when it enlarged (face PSNR 27.2
    enlarging vs 30.9 not). The cost of the honest version is softness, which is why the UI shows the
    magnification before you commit.

    Reads the size from the bytes rather than config: MediaAsset.width/height are null on every row,
    and the Gemini provider config carries an aspect ratio but no pixel size."""
    from PIL import Image

    src = Image.open(io.BytesIO(image_bytes))
    W, H = src.size
    left, top = round(x * W), round(y * H)
    right, bottom = round((x + w) * W), round((y + h) * H)
    if not (0 <= left < right <= W and 0 <= top < bottom <= H):
        raise ValueError(f"crop box ({x:.3f},{y:.3f},{w:.3f},{h:.3f}) falls outside the {W}x{H} image")
    out = src.crop((left, top, right, bottom)).resize((W, H), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


@router.post("/steps/image/crop")
async def step_image_crop(request: ImageCropRequest, db: Session = Depends(get_db)) -> dict:
    """Reframe a shot's image: keep the selected rectangle, scaled back to the same pixel size.

    A plain POST rather than an SSE stream because nothing here is slow or streamed — no model is
    called at all (see _crop_to_same_size), which is also why this is not a third mode of
    /steps/image/edit_region: that route is gated to the openai provider, and a crop has no reason
    to be. Persists exactly like every other path that replaces a shot's image, so the previous
    render becomes a superseded asset rather than being lost."""
    # Nothing here reads the brand/menu today, but every /steps route that carries a run_id binds it
    # — the guard in test_run_context_bound exists because "this one doesn't need it yet" is exactly
    # how a route ends up silently resolving the wrong run's refs later.
    active_config.set_current_run(request.run_id or "")
    board = StoryboardRepo(db)
    shot = board.get_shot(request.shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")
    img_url = _resolve_asset_url(db, shot.current_image_asset_id)
    if not img_url:
        raise HTTPException(status_code=400, detail="ช็อตนี้ยังไม่มีรูป — generate ก่อน")
    base_image = _media_bytes(img_url)
    if not base_image:
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์รูป (S3/local) — generate ใหม่ก่อน")
    try:
        data = await asyncio.to_thread(
            _crop_to_same_size, base_image["data"], request.x, request.y, request.w, request.h)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    fname = f"{_safe_name_part(request.scene_id)}_{int(request.no)}_{int(time.time())}.png"
    s3_key = f"{_slug(request.topic.strip())}_images/{fname}"
    url = _persist_media(s3_key, data, "image/png")
    try:
        asset = MediaAssetRepo(db).create(kind="image", s3_key=s3_key, s3_url=url, run_id=request.run_id)
        board.set_shot_image(request.shot_id, asset.id)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(f"failed to persist cropped image for shot {request.shot_id}")
        raise HTTPException(status_code=500, detail="บันทึกรูปที่ครอปไม่สำเร็จ")
    logger.info(f"Cropped shot image {request.scene_id}/{request.no} → {url} ({len(data):,} bytes)")
    return {"scene_id": request.scene_id, "no": request.no, "url": url}


@router.post("/steps/image/edit_region")
async def step_image_edit_region(request: ImageRegionEditRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Region edit (outpaint or freehand mask inpaint) of a shot's EXISTING image — gpt-image-2 only
    (see edit_image_region's docstring). "outpaint" pads the current image server-side (Pillow) with
    a transparent border sized to (target_width, target_height) and lets gpt-image fill it in from
    its own alpha channel; "mask" uses the client-drawn mask verbatim. Saves as a new MediaAsset,
    same history semantics as /steps/image/edit."""
    container = http_request.app.state.container
    topic = request.topic.strip()
    active_config.set_current_run(request.run_id or "")   # see /steps/image
    cfg_gen = container.settings.image_gen

    async def stream():
        instruction = request.instruction.strip()
        if not instruction:
            yield sse({"type": "error", "message": "instruction ว่าง"})
            return
        if cfg_gen.provider != "openai":
            yield sse({"type": "error", "message": "Outpaint/Mask edit ใช้ได้เฉพาะตอน image_gen.provider = openai (gpt-image-2) เท่านั้น"})
            return
        board = StoryboardRepo(db)
        shot = board.get_shot(request.shot_id)
        if shot is None:
            yield sse({"type": "error", "message": "ไม่พบช็อตนี้"})
            return
        img_url = _resolve_asset_url(db, shot.current_image_asset_id)
        if not img_url:
            yield sse({"type": "error", "message": "ช็อตนี้ยังไม่มีรูป — generate ก่อน"})
            return
        base_image = _media_bytes(img_url)
        if not base_image:
            yield sse({"type": "error", "message": "ไม่พบไฟล์รูป (S3/local) — generate ใหม่ก่อน"})
            return

        mask_bytes: bytes | None = None
        image_bytes = base_image["data"]
        if request.mode == "outpaint":
            if not (request.target_width and request.target_height):
                yield sse({"type": "error", "message": "outpaint ต้องระบุ target_width/target_height"})
                return
            try:
                image_bytes = _outpaint_canvas(image_bytes, request.target_width, request.target_height)
            except ValueError as e:
                yield sse({"type": "error", "message": str(e)})
                return
        elif request.mode == "mask":
            if not request.mask_data_url:
                yield sse({"type": "error", "message": "mask ว่าง — วาด mask ก่อน"})
                return
            try:
                b64 = request.mask_data_url.split(",", 1)[-1]
                mask_bytes = base64.b64decode(b64)
            except Exception:
                yield sse({"type": "error", "message": "mask_data_url ไม่ใช่ base64 PNG ที่ถูกต้อง"})
                return
        else:
            yield sse({"type": "error", "message": f"mode ไม่รู้จัก: {request.mode!r} (ต้องเป็น outpaint หรือ mask)"})
            return

        tracker = UsageTracker()
        set_tracker(tracker)
        _t0 = time.perf_counter()
        yield sse({"type": "status", "message": "กำลัง outpaint..." if request.mode == "outpaint" else "กำลังแก้ไขบริเวณที่เลือก..."})
        try:
            edit_refs = await _load_edit_refs(container, request.extra_refs, request.ref_notes)
        except EditRefLoadError as e:
            yield sse({"type": "error", "message": str(e)})
            return
        try:
            data = await openai_image.edit_image_region(
                instruction, image_bytes, mask_bytes,
                api_key=container.openai_api_key, model=cfg_gen.openai.model,
                size=cfg_gen.openai.size, quality=cfg_gen.openai.quality, refs=edit_refs)
        except Exception as e:  # noqa: BLE001
            logger.exception("Image region edit failed")
            yield sse({"type": "error", "message": f"แก้ไขรูปไม่สำเร็จ: {e}"})
            return
        if not data:
            yield sse({"type": "error", "message": "แก้ไขรูปไม่สำเร็จ (provider ไม่คืนรูป)"})
            return

        fname = f"{_safe_name_part(request.scene_id)}_{int(request.no)}_{int(time.time())}.png"
        s3_key = f"{_slug(topic)}_images/{fname}"
        url = _persist_media(s3_key, data, "image/png")
        obs.attach_media(url, "image")
        try:
            asset = MediaAssetRepo(db).create(kind="image", s3_key=s3_key, s3_url=url, run_id=request.run_id)
            board.set_shot_image(request.shot_id, asset.id)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(f"failed to persist region-edited image for shot {request.shot_id}")

        yield sse({"type": "image_edited", "scene_id": request.scene_id, "no": request.no, "url": url})
        usage_summary = tracker.summary()
        _persist_usage(db, request.run_id, "image_edit_region", usage_summary)
        yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
        yield sse({"type": "done"})

    return StreamingResponse(obs.traced_stream(stream(), "step.image_edit_region", run_id=request.run_id, scene_id=request.scene_id, no=request.no, shot_id=request.shot_id), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/image_plan")
async def step_image_plan(request: ImageStepRequest, http_request: Request, db: Session = Depends(get_db)) -> dict:
    """Dry-run: resolve which reference images a shot WILL use (classify → plan → resolve) WITHOUT
    generating the image. Returns the image_plan (refs_used) so the UI can preview refs before Generate."""
    container = http_request.app.state.container
    topic = request.topic.strip()
    # Must bind the SAME run as /steps/image — a preview resolved without the brand/menu context
    # would advertise refs the real Generate never uses, which is worse than showing nothing.
    active_config.set_current_run(request.run_id or "")
    cfg_gen = container.settings.image_gen
    tracker = UsageTracker()
    set_tracker(tracker)
    try:
        if cfg_gen.dynamic_image and request.shot:
            _, plan, _ = await generate_shot_dynamic(
                container, request.shot, request.prev_shot, topic,
                all_ingredients=request.all_ingredients, ingredient_images=request.ingredient_images,
                prev_state_url=request.prev_state_url, used_ingredients=request.used_ingredients,
                all_equipment=request.all_equipment, equipment_images=request.equipment_images,
                extra_refs=request.extra_refs, ref_notes=request.ref_notes, removed_refs=request.removed_refs,
                ref_person=request.ref_person, ref_scene=request.ref_scene, ref_state=request.ref_state,
                category=_image_category(db, request.run_id or "", request.category, container), plan_only=True)
        else:
            # non-dynamic: the fixed base refs (character + scene) apply to every shot
            base = await load_base_image_refs(container)
            used: list[dict] = []
            if any(str(r["label"]).startswith("Character") for r in base):
                used.append({"kind": "person", "label": "person", "url": "", "source": "config"})
            if any(str(r["label"]).startswith("Background") for r in base):
                ds = scene_store.active_default_scene() or {}
                used.append({"kind": "kitchen", "label": ds.get("name") or "kitchen",
                             "url": scene_store.scene_preview_url(ds.get("id", "")), "source": "config"})
            plan = {"image_generate_type": "new_generate", "refs_used": used}
    except Exception as e:  # noqa: BLE001 — surface failure to the client
        logger.exception("Image plan (dry-run) failed")
        raise HTTPException(status_code=500, detail=f"Image plan failed: {e}")
    usage_summary = tracker.summary()
    _persist_usage(db, request.run_id, "image_plan", usage_summary)
    return {"scene_id": request.scene_id, "no": request.no, "image_plan": plan, "usage": usage_summary}


@router.post("/steps/image/upload")
async def upload_shot_image(
    http_request: Request,
    topic: str,
    scene_id: str,
    no: int,
    shot_id: int | None = None,
    run_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    """Use a user-uploaded image as a shot's image (instead of generating). Raw image bytes in the
    body; saved into the same media dir as generated shot images and returned as a /api/media URL, so
    it displays + is reusable as a ref for later shots + as the video first-frame seed."""
    data = await http_request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    fname = f"{_safe_name_part(scene_id)}_{int(no)}_{int(time.time())}.{_img_ext(http_request)}"
    _ct = "image/jpeg" if fname.endswith((".jpg", ".jpeg")) else "image/png"
    s3_key = f"{_slug(topic)}_images/{fname}"
    url = _persist_media(s3_key, data, _ct)
    logger.info(f"Uploaded shot image {scene_id}/{no} → {url} ({len(data):,} bytes)")

    if shot_id is not None:
        try:
            asset = MediaAssetRepo(db).create(kind="image", s3_key=s3_key, s3_url=url, run_id=run_id)
            StoryboardRepo(db).set_shot_image(shot_id, asset.id)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(f"failed to persist uploaded image for shot {shot_id}")

    return {"scene_id": scene_id, "no": no, "url": url}


_THAI_FONT = ROOT_DIR / "assets" / "fonts" / "NotoSansThai-Regular.ttf"


def _inject_thai_font(html: str) -> str:
    """Embed the bundled Thai font as an @font-face so WeasyPrint renders Thai (the export HTML lists
    'Noto Sans Thai' in its font stack but WeasyPrint has no system copy)."""
    if not _THAI_FONT.exists():
        return html
    face = ("<style>"
            f"@font-face{{font-family:'Noto Sans Thai';src:url('{_THAI_FONT.resolve().as_uri()}')}}"
            "body{font-family:'Noto Sans Thai',sans-serif}"
            "</style>")
    return html.replace("</head>", face + "</head>", 1) if "</head>" in html else face + html


def _pdf_url_fetcher(url: str):
    """WeasyPrint renders client-supplied HTML — without a guard, an <img src="file:///etc/passwd">
    or url(http://169.254.169.254/...) would be fetched and embedded into the returned PDF (SSRF / local
    file read). Allow ONLY the bundled Thai font (a file:// we injected) and block every other file:// plus
    any http(s) that resolves to a private / loopback / link-local address; otherwise use the default."""
    import ipaddress
    import socket
    from urllib.parse import urlparse
    from weasyprint import default_url_fetcher

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        if _THAI_FONT.exists() and url == _THAI_FONT.resolve().as_uri():
            return default_url_fetcher(url)
        raise ValueError(f"blocked file:// resource in export HTML: {url[:80]}")
    if scheme in ("http", "https"):
        host = parsed.hostname or ""
        try:
            for res in socket.getaddrinfo(host, None):
                ip = ipaddress.ip_address(res[4][0])
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    raise ValueError(f"blocked internal address in export HTML: {host}")
        except socket.gaierror:
            raise ValueError(f"could not resolve host in export HTML: {host}")
        return default_url_fetcher(url)
    if scheme == "data":
        return default_url_fetcher(url)
    raise ValueError(f"blocked URL scheme {scheme!r} in export HTML")


def _render_pdf(html: str) -> bytes:
    """Render export HTML → PDF bytes (SSRF-safe fetcher). Blocking/CPU — call via asyncio.to_thread."""
    from weasyprint import HTML  # lazy: needs native libs (pango/glib) — see app/main.py dyld setup
    return HTML(string=_inject_thai_font(html)).write_pdf(url_fetcher=_pdf_url_fetcher)


@router.post("/export/pdf")
async def export_pdf(req: PdfExportRequest) -> Response:
    """Render a self-contained export HTML doc (from the frontend's exportHtml) to a downloadable PDF."""
    try:
        # to_thread: WeasyPrint is CPU-heavy + blocking — off the event loop so other SSE streams keep flowing
        pdf = await asyncio.to_thread(_render_pdf, req.html)
    except Exception as e:  # noqa: BLE001 — surface a clear message (e.g. missing native libs)
        logger.exception("PDF export failed")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")
    # RFC 5987: HTTP headers are latin-1, so a non-ASCII (e.g. Thai) filename needs filename*=UTF-8''…
    from urllib.parse import quote
    fn = (req.filename or "export").replace('"', "").strip() or "export"
    ascii_fn = (fn.encode("ascii", "ignore").decode().strip() or "export")
    cd = f"attachment; filename=\"{ascii_fn}.pdf\"; filename*=UTF-8''{quote(fn)}.pdf"
    return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": cd})


@router.post("/revoice")
async def revoice(request: RevoiceRequest, http_request: Request) -> dict:
    """Re-voice a rendered clip: keep its ambience, replace the Veo voice with TTS. A single shot
    passes `voice_over`; a GROUP clip passes `voice_overs` (each shot's line, in order) → each is
    placed in its own detected speech window. Overwrites the clip in place so merge/UI pick it up."""
    from app.services.revoice import add_narration, add_narration_multi, revoice_clip, revoice_clip_multi

    vos = [v for v in (request.voice_overs or []) if (v or "").strip()]
    if not vos and not request.voice_over.strip():
        raise HTTPException(status_code=400, detail="ไม่มีบทพูด (voice_over) ให้พากย์")
    cfg = http_request.app.state.container.settings.tts
    add_mode = request.mode == "add"   # "add" = overlay narration, keep music+ambience (no demucs)
    _multi, _single = (add_narration_multi, add_narration) if add_mode else (revoice_clip_multi, revoice_clip)
    # A local working file for the clip: the local copy if present, else the S3 copy downloaded
    # into a temp dir (cleaned up in `finally`), so re-voicing works once media lives only on S3.
    work = Path(tempfile.mkdtemp(prefix="revoice_"))
    clip = await asyncio.to_thread(_local_media_file, request.video_url, work)
    if clip is None or not clip.exists():
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์คลิป (generate video ของช็อตนี้ก่อน)")
    try:
        _t0 = time.perf_counter()
        out_path = clip.with_name(f"{clip.stem}_voiced.mp4")
        if len(vos) > 1:
            voiced, spans = await asyncio.to_thread(
                _multi, clip, vos,
                voice=request.voice or cfg.voice, style=request.style or cfg.style, out=out_path)
        else:
            voiced, spans = await asyncio.to_thread(
                _single, clip, (vos[0] if vos else request.voice_over),
                voice=request.voice or cfg.voice, style=request.style or cfg.style, out=out_path)
        # replace the clip in place → same URL (UI cache-busts with ?t=)
        await asyncio.to_thread(lambda: Path(voiced).replace(clip))
        if _storage.is_configured():
            from urllib.parse import unquote
            _key = None
            if "/api/media/" in request.video_url:
                _key = request.video_url.split("/api/media/", 1)[-1]
            else:   # S3 URL: strip AWS_URL prefix + decode → raw object key (same object → same URL)
                _aws = os.getenv("AWS_URL", "").rstrip("/")
                if _aws and request.video_url.startswith(_aws + "/"):
                    _key = unquote(request.video_url[len(_aws) + 1:])
            if _key:
                _voiced_data = await asyncio.to_thread(clip.read_bytes)
                _storage.upload_bytes(_key, _voiced_data, "video/mp4")
        seconds = round(time.perf_counter() - _t0, 1)
    except Exception as e:  # noqa: BLE001
        logger.exception("revoice failed")
        raise HTTPException(status_code=502, detail=f"ใส่เสียงพากย์ไม่สำเร็จ: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"url": request.video_url, "seconds": seconds, "spans": spans}


@router.post("/steps/video/subtitle")
async def step_video_subtitle(request: SubtitleRequest, http_request: Request) -> dict:
    """Burn ONE clip's caption, timed to the window where speech actually happens (Silero VAD, or the
    exact start/end from /revoice spans) — so it appears on speech-onset and clears when the line ends.
    Overwrites the clip in place (same URL; UI cache-busts with ?t=), mirroring /revoice."""
    from app.services.revoice import detect_speech_window

    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="ไม่มีข้อความ subtitle")
    cfg = http_request.app.state.container.settings.video_gen
    ok, reason = _subs_ready(cfg)
    if not ok:
        raise HTTPException(status_code=400, detail=f"เบิร์นซับไม่ได้: {reason}")
    await asyncio.to_thread(_resolve_ffmpeg, cfg)

    work = Path(tempfile.mkdtemp(prefix="subtitle_"))
    clip = await asyncio.to_thread(_local_media_file, request.video_url, work)
    if clip is None or not clip.exists():
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=400, detail="ไม่พบไฟล์คลิป (generate video ของช็อตนี้ก่อน)")
    try:
        _t0 = time.perf_counter()
        clip_sec = await _probe_duration(clip)
        if request.start is not None and request.end is not None and request.end > request.start:
            start, end, source = max(0.0, request.start), min(clip_sec, request.end), "revoice"
        elif cfg.subtitle_sync_to_speech:
            win = await asyncio.to_thread(detect_speech_window, clip)
            start, end = win if win else (0.0, clip_sec)
            source = "detect" if win else "whole_clip"
        else:
            # speech-sync disabled (subtitle_sync_to_speech=False) — caption spans the whole clip
            start, end, source = 0.0, clip_sec, "whole_clip"
        ass = Path(tempfile.mkdtemp()) / "subs.ass"   # no-space path → clean filter syntax
        if not _write_ass([(start, end)], [text], ass, cfg):
            raise HTTPException(status_code=400, detail="สร้างไฟล์ซับไม่สำเร็จ")
        fp = _subtitle_font_path(cfg)
        fontdir = f":fontsdir={fp.parent}" if fp.exists() else ""
        subbed = work / f"{clip.stem}_sub.mp4"
        rc, err = await _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(clip), "-vf", f"subtitles={ass}{fontdir}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
            "-movflags", "+faststart", str(subbed)])
        if rc != 0:
            raise HTTPException(status_code=502, detail=f"ffmpeg เบิร์นซับล้มเหลว: {err[-300:]}")
        await asyncio.to_thread(lambda: subbed.replace(clip))
        if _storage.is_configured():   # same object key → same URL (mirrors /revoice)
            from urllib.parse import unquote
            _key = None
            if "/api/media/" in request.video_url:
                _key = request.video_url.split("/api/media/", 1)[-1]
            else:
                _aws = os.getenv("AWS_URL", "").rstrip("/")
                if _aws and request.video_url.startswith(_aws + "/"):
                    _key = unquote(request.video_url[len(_aws) + 1:])
            if _key:
                _data = await asyncio.to_thread(clip.read_bytes)
                _storage.upload_bytes(_key, _data, "video/mp4")
        return {"url": request.video_url, "start": round(start, 2), "end": round(end, 2),
                "source": source, "seconds": round(time.perf_counter() - _t0, 1)}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("subtitle burn failed")
        raise HTTPException(status_code=502, detail=f"เบิร์นซับไม่สำเร็จ: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


@router.post("/steps/video_upload")
async def upload_video_file(
    file: UploadFile = File(...),
    old_url: str = Form(""),
    scene_id: str = Form(...),
    no: int = Form(...),
    topic: str = Form(""),
    shot_id: int | None = Form(None),
    run_id: str | None = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    """Replace a shot's generated video with an uploaded file.
    Deletes the old S3 object when old_url points to one. Returns {url} of the stored clip."""
    data = await file.read()
    if old_url:
        old_key = _storage.key_from_url(old_url)
        if old_key:
            await asyncio.to_thread(_storage.delete_bytes, old_key)
    slug = _slug(topic) if topic else "upload"
    # Timestamped like generated-clip keys: with a fixed key, uploading twice to one shot silently
    # overwrote the first upload's bytes, so every older MediaAsset row in the shot's clip history
    # pointed at the replaced file — the history said "three versions", S3 held one.
    s3_key = f"{slug}_videos/{scene_id}_{no}_{int(time.time())}.mp4"
    url = await asyncio.to_thread(_persist_media, s3_key, data, "video/mp4")

    if shot_id is not None:
        try:
            asset = MediaAssetRepo(db).create(kind="video", s3_key=s3_key, s3_url=url, run_id=run_id,
                                              shot_id=shot_id)
            StoryboardRepo(db).set_shot_video(shot_id, asset.id)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(f"failed to persist uploaded video for shot {shot_id}")

    return {"url": url}


@router.get("/steps/video_config")
async def get_video_config(http_request: Request) -> dict:
    """Current video_gen settings (config.yaml) — the UI loads these as defaults."""
    return http_request.app.state.container.settings.video_gen.model_dump()


@router.post("/voice_desc")
async def make_voice_desc(http_request: Request, character_desc: str = Body("", embed=True)) -> dict:
    """Auto-synthesize a narrator voice descriptor (voice_desc) from character_desc + the active brand's
    vo_tone — the UI's '✨ auto' button on the เสียงพากย์ field."""
    container = http_request.app.state.container
    vo_tone = (scene_store.active_brand() or {}).get("vo_tone", "")
    voice_desc = await container.gemini.generate_voice_desc(character_desc, vo_tone)
    return {"voice_desc": voice_desc}


# ─── Kitchen fixtures (extracted from scene_ref, editable, used at generation) ──

@router.get("/steps/kitchen_fixtures")
async def get_kitchen_fixtures_ep(http_request: Request) -> dict:
    """Current configured fixture list (used to keep generation from inventing a new sink/stove)."""
    return {"fixtures": load_kitchen_fixtures(http_request.app.state.container)}


@router.post("/steps/kitchen_fixtures/extract")
async def extract_kitchen_fixtures_ep(http_request: Request) -> dict:
    """Vision-extract fixtures from the current scene_ref image and store them."""
    return {"fixtures": await extract_kitchen_fixtures_from_scene(http_request.app.state.container)}


@router.put("/steps/kitchen_fixtures")
async def set_kitchen_fixtures_ep(request: KitchenFixturesRequest, http_request: Request) -> dict:
    """Save the user-edited fixture list."""
    save_kitchen_fixtures(request.fixtures)
    return {"fixtures": load_kitchen_fixtures(http_request.app.state.container)}


# ─── Generation reference images (character_ref / scene_ref) ────────────────────

_REF_ATTR = {"character": "character_ref", "scene": "scene_ref"}


def _upsert_config_ref(db: Session, kind: str, asset_id: str | None, s3_url: str) -> None:
    """Durably store the global character/scene ref (one row per kind) so it survives restart."""
    from app.db.models import ConfigRef
    row = db.get(ConfigRef, kind)
    if row is None:
        db.add(ConfigRef(kind=kind, image_asset_id=asset_id, s3_url=s3_url))
    else:
        row.image_asset_id = asset_id
        row.s3_url = s3_url


def _fold_voice(stored: dict, voice) -> None:
    """Apply a stored voice config over `voice` in place, skipping anything blank.

    Same rule as the character's name/picture: an unset field means "use config.yaml", never
    "use nothing". Mutates the existing object rather than rebinding so every holder of
    `settings.voice` sees the change."""
    for k in _VOICE_FIELDS:
        v = stored.get(k)
        if isinstance(v, str) and v.strip():
            setattr(voice, k, v.strip())
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v:
            setattr(voice, k, v)


def load_config_refs_into_settings(db: Session, image_gen, script=None, voice=None) -> None:
    """Startup: override image_gen.character_ref/scene_ref from durable config_refs (falls back to the
    config.yaml default when a kind has no persisted row yet).

    `script` and `voice` (optional) get the same treatment for the character's NAME/DESCRIPTION and
    for the narration voice, all stored on the same row as the picture. Only a non-empty value
    overrides, so a config that was never filled in keeps whatever config.yaml says — the same rule
    the image follows.

    Skipping the `voice` half would not fail loudly: the value would work until the next restart and
    then quietly revert, which is the exact bug config_refs exists to prevent."""
    from app.db.models import ConfigRef
    for kind, attr in _REF_ATTR.items():
        row = db.get(ConfigRef, kind)
        if row and row.s3_url:
            setattr(image_gen, attr, row.s3_url)
    if script is not None or voice is not None:
        row = db.get(ConfigRef, "character")
        if script is not None:
            if row and (row.name or "").strip():
                script.character_name = row.name.strip()
            if row and (row.description or "").strip():
                script.character_desc = row.description.strip()
        if voice is not None:
            _fold_voice((row.voice if row else None) or {}, voice)


@router.get("/refs/character/meta")
async def get_character_meta(http_request: Request) -> dict:
    """The host's name, description and narration voice — the text half of the character config,
    stored beside the reference photo.

    Returns what generation will ACTUALLY use, not the raw row: startup folds `config_refs` over
    config.yaml into `settings`, so reading it here means a never-configured install shows the real
    fallback host instead of empty boxes that look like nothing is set."""
    settings = http_request.app.state.container.settings
    return {"name": settings.script.character_name, "description": settings.script.character_desc,
            "voice": settings.voice.model_dump()}


@router.put("/refs/character/meta")
async def set_character_meta(request: CharacterMetaRequest, http_request: Request,
                             db: Session = Depends(get_db)) -> dict:
    """Save the host's name/description/voice and apply them to the RUNNING config immediately, the
    same way uploading the reference photo repoints `image_gen.character_ref` without a restart.

    Every field is optional and `None` means "leave it alone" — the panel saves the name and the
    voice from two separate cards, and a voice-only save must not blank the host's name."""
    from app.db.models import ConfigRef
    try:
        row = db.get(ConfigRef, "character")
        if row is None:
            row = ConfigRef(kind="character")
            db.add(row)
        if request.name is not None:
            row.name = request.name.strip()
        if request.description is not None:
            row.description = request.description.strip()
        if request.voice is not None:
            row.voice = request.voice.model_dump()
        db.commit()
    except Exception:
        db.rollback()
        raise
    # Clearing a field has to fall back to config.yaml, not leave the old value stranded in memory
    # until the next restart — so re-read the file rather than assigning row values blindly.
    from app.core.config import Settings
    yaml = Settings.load()
    settings = http_request.app.state.container.settings
    settings.script.character_name = row.name or yaml.script.character_name
    settings.script.character_desc = row.description or yaml.script.character_desc
    for k in _VOICE_FIELDS:                       # reset to the yaml floor, then fold the row over it
        setattr(settings.voice, k, getattr(yaml.voice, k))
    _fold_voice(row.voice or {}, settings.voice)
    return {"name": settings.script.character_name, "description": settings.script.character_desc,
            "voice": settings.voice.model_dump()}


@router.post("/refs/{kind}")
async def upload_ref(kind: str, http_request: Request, db: Session = Depends(get_db)) -> dict:
    """Upload the character/scene reference image (raw image bytes as the body). Uploads to S3 and
    persists it durably in `config_refs` (so it survives restart — previously this only set the value
    in-memory and reverted to the config.yaml default on restart) + points the running config at it."""
    if kind not in _REF_ATTR:
        raise HTTPException(status_code=400, detail="kind must be 'character' or 'scene'")
    data = await http_request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    ct = http_request.headers.get("content-type", "")
    ext = "jpg" if ("jpeg" in ct or "jpg" in ct) else "png"
    _ct = "image/jpeg" if ext == "jpg" else "image/png"
    s3_key = f"refs/{kind}.{ext}"
    # No local copy when S3 is on (_persist_media); the config field now holds whatever
    # URL generation/preview should read from — never a bare local filesystem path.
    ref_url = _persist_media(s3_key, data, _ct)
    asset = MediaAssetRepo(db).create(kind="image", s3_key=s3_key, s3_url=ref_url)
    _upsert_config_ref(db, kind, asset.id, ref_url)
    db.commit()
    setattr(http_request.app.state.container.settings.image_gen, _REF_ATTR[kind], ref_url)
    logger.info(f"Updated {kind}_ref → {ref_url} ({len(data):,} bytes) [persisted to config_refs]")
    return {"kind": kind, "url": ref_url, "bytes": len(data)}


@router.get("/refs/{kind}/preview")
async def ref_preview(kind: str, http_request: Request, db: Session = Depends(get_db)):
    """Stream the reference image behind a `/api/refs/{kind}/preview` URL.

    For "character" this now means THE RUN'S character (falling back to the default one), not a
    single global picture — the URL string is unchanged because it is already persisted inside old
    shots' `refs_used`, and its meaning ("whichever host this shot belongs to") still holds."""
    if kind not in _REF_ATTR:
        raise HTTPException(status_code=404, detail="unknown kind")
    if kind == "character":
        src = character_store.active_character_ref()
    else:
        src = getattr(http_request.app.state.container.settings.image_gen, _REF_ATTR[kind], "") or ""
    return _ref_url_response(src)


def _ref_url_response(src: str):
    """Serve a stored image URL: redirect to S3, or stream the local file it points at.

    One implementation for every ref preview route (character / scene / brand scene) — they had
    three near-identical copies that already disagreed about the legacy config.yaml path case."""
    src = src or ""
    if not src:
        raise HTTPException(status_code=404, detail="no ref configured")
    if src.startswith(("http://", "https://")):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(src, status_code=302)
    if "/api/media/" in src:   # S3-off _persist_media fallback: "/api/media/refs/{kind}.ext"
        p = _safe_media_path(src)
    else:                      # legacy: config.yaml default (e.g. "docs/teacher.png") or an old absolute path
        p = Path(src)
        if not p.is_absolute():
            p = ROOT_DIR / src
        p = p if p.exists() else None
    if p is None:
        raise HTTPException(status_code=404, detail="ref file not found")
    return FileResponse(p, headers=_NO_STORE)


# ─── Bumpers: global intro / end-credit clips the merge step wraps around the episode ────────────
# Stored as config_refs rows (kind "intro"/"end_credit") exactly like the character/scene refs —
# same table, same upsert — the pointed-at MediaAsset is just kind="video". No settings.image_gen
# coupling: the only consumer is step_merge, which reads the rows directly at merge time.

_BUMPER_KINDS = ("intro", "end_credit")


def _bumper_urls(db: Session) -> dict[str, str]:
    """kind → stored clip url for the configured bumpers (missing kinds omitted)."""
    from app.db.models import ConfigRef
    out: dict[str, str] = {}
    for kind in _BUMPER_KINDS:
        row = db.get(ConfigRef, kind)
        if row and row.s3_url:
            out[kind] = row.s3_url
    return out


@router.get("/bumpers")
async def bumpers_status(db: Session = Depends(get_db)) -> dict:
    urls = _bumper_urls(db)
    return {kind: urls.get(kind) for kind in _BUMPER_KINDS}


@router.post("/bumpers/{kind}")
async def upload_bumper(kind: str, http_request: Request, db: Session = Depends(get_db)) -> dict:
    """Upload the intro/end-credit clip (raw video bytes as the body, same convention as /refs).

    Normalized HERE, once, to the app's merge standard (1280x720 pad, 24fps CFR, h264 yuv420p,
    aac 48k stereo — the exact _normalized_concat recipe): the merge's raw mode stream-copies the
    Veo clips losslessly, and a bumper with foreign params would force the whole episode through
    the re-encode fallback. A silent upload gets a silent audio track for the same reason — a
    concat with one audio-less input fails outright."""
    if kind not in _BUMPER_KINDS:
        raise HTTPException(status_code=400, detail="kind must be 'intro' or 'end_credit'")
    data = await http_request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    with tempfile.TemporaryDirectory(prefix="bumper_") as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "in.mp4"
        src.write_bytes(data)
        dur = await _probe_duration(src)
        if not dur:
            raise HTTPException(status_code=400, detail="อ่านไฟล์วิดีโอไม่ได้ — ต้องเป็นไฟล์วิดีโอ (mp4/mov)")
        proc = await asyncio.create_subprocess_exec(
            _FFPROBE_BIN, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index",
            "-of", "csv=p=0", str(src),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        probe_out, _ = await proc.communicate()
        has_audio = bool(probe_out.decode().strip())
        out = tmp / "out.mp4"
        vf = ("scale=1280:720:force_original_aspect_ratio=decrease,"
              "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24,format=yuv420p")
        args = ["ffmpeg", "-y", "-i", str(src)]
        if not has_audio:
            args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
                     "-map", "0:v:0", "-map", "1:a:0"]
        args += ["-vf", vf, "-r", "24", "-fps_mode", "cfr", "-c:v", "libx264", "-crf", "18",
                 "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
                 "-ac", "2", "-movflags", "+faststart", str(out)]
        rc, err = await _run_ffmpeg(args)
        if rc != 0 or not out.exists():
            logger.error(f"bumper normalize failed for {kind}: {err[-500:]}")
            raise HTTPException(status_code=400, detail="แปลงไฟล์วิดีโอไม่สำเร็จ — เช็คว่าเป็นไฟล์วิดีโอจริง")
        norm = out.read_bytes()
        norm_dur = await _probe_duration(out)
    # timestamped key: replacing a bumper must change the URL, or every <video> that already
    # loaded the old file keeps showing cached bytes (same trap revoiceTick exists for)
    s3_key = f"refs/bumpers/{kind}_{int(time.time())}.mp4"
    url = _persist_media(s3_key, norm, "video/mp4")
    from app.db.models import ConfigRef
    old = db.get(ConfigRef, kind)
    if old and old.image_asset_id:
        MediaAssetRepo(db).supersede(old.image_asset_id)
    asset = MediaAssetRepo(db).create(kind="video", s3_key=s3_key, s3_url=url,
                                      duration_seconds=norm_dur or None)
    _upsert_config_ref(db, kind, asset.id, url)
    db.commit()
    logger.info(f"Bumper {kind} → {url} ({norm_dur:.2f}s, {len(norm):,} bytes)")
    return {"kind": kind, "url": url, "duration": round(norm_dur, 2)}


@router.delete("/bumpers/{kind}")
async def delete_bumper(kind: str, db: Session = Depends(get_db)) -> dict:
    if kind not in _BUMPER_KINDS:
        raise HTTPException(status_code=400, detail="kind must be 'intro' or 'end_credit'")
    from app.db.models import ConfigRef
    row = db.get(ConfigRef, kind)
    if row is None:
        raise HTTPException(status_code=404, detail="ยังไม่ได้ตั้งค่าคลิปนี้")
    if row.image_asset_id:
        MediaAssetRepo(db).supersede(row.image_asset_id)
    db.delete(row)
    db.commit()
    return {"kind": kind, "deleted": True}


@router.get("/bumpers/{kind}/preview")
async def bumper_preview(kind: str, db: Session = Depends(get_db)):
    if kind not in _BUMPER_KINDS:
        raise HTTPException(status_code=404, detail="unknown kind")
    return _ref_url_response(_bumper_urls(db).get(kind, ""))


# ─── Active config pointer (which brand + menu generation uses), per-run ─────────

@router.get("/active_config")
async def get_active_config_ep(run_id: str = "") -> dict:
    active_config.set_current_run(run_id)
    return await asyncio.to_thread(active_config.get)


@router.put("/active_config")
async def set_active_config_ep(
    run_id: str = "", brand_id: str | None = None, menu_id: str | None = None,
    character_id: str | None = None, db: Session = Depends(get_db),
) -> dict:
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id required")
    try:
        fields = {}
        if brand_id is not None:
            fields["brand_id"] = brand_id
        if menu_id is not None:
            fields["menu_id"] = menu_id
        if character_id is not None:
            fields["character_id"] = character_id
        run = RunRepo(db).set_active_pointers(run_id, **fields)
        db.commit()
    except NotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        db.rollback()
        raise
    return {"brand_id": run.brand_id or "", "menu_id": run.menu_id or "",
            "director_id": run.director_prompt_id or "", "character_id": run.character_id or ""}


# ─── Characters = the hosts a run can be made with (one per run, with a default) ──
#
# Deliberately the same shape as /brands below — list summary, detail dict, PUT of a fields dict,
# raw-body image upload, a default flag — because a run picking a host is the same problem as a run
# picking a brand and there is no reason for a second vocabulary.

def _character_dict(c) -> dict:
    return {"id": c.id, "name": c.name or "", "description": c.description or "",
            "image": c.s3_url or "", "voice": c.voice or {}, "default": bool(c.is_default)}


@router.get("/characters")
async def list_characters_ep(db: Session = Depends(get_db)) -> dict:
    return {"characters": [_character_dict(c) for c in CharacterRepo(db).list_characters()]}


@router.post("/characters")
async def create_character_ep(name: str, db: Session = Depends(get_db)) -> dict:
    if not name.strip():
        raise HTTPException(status_code=400, detail="name required")
    try:
        c = CharacterRepo(db).create_character(name.strip())
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"character": _character_dict(c)}


@router.get("/characters/{character_id}")
async def get_character_ep(character_id: str, db: Session = Depends(get_db)) -> dict:
    c = CharacterRepo(db).get_character(character_id)
    if not c:
        raise HTTPException(status_code=404, detail="character not found")
    return {"character": _character_dict(c)}


@router.put("/characters/{character_id}")
async def update_character_ep(character_id: str, fields: dict = Body(...),
                              db: Session = Depends(get_db)) -> dict:
    """Edit name/description and/or the whole `voice` object. Only the keys sent are touched."""
    try:
        c = CharacterRepo(db).update_character(character_id, fields)
        db.commit()
    except Exception:
        db.rollback()
        raise
    if c is None:
        raise HTTPException(status_code=404, detail="character not found")
    return {"character": _character_dict(c)}


@router.delete("/characters/{character_id}")
async def delete_character_ep(character_id: str, db: Session = Depends(get_db)) -> dict:
    """Runs that were using it fall back to the default (runs.character_id is ON DELETE SET NULL),
    and if the deleted one WAS the default the next one takes over — so there is always an answer."""
    try:
        ok = CharacterRepo(db).delete_character(character_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"deleted": ok}


@router.post("/characters/{character_id}/image")
async def upload_character_image_ep(character_id: str, http_request: Request,
                                    db: Session = Depends(get_db)) -> dict:
    """Raw image bytes as the body — same contract as a brand scene's image."""
    data = await http_request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    try:
        c = CharacterRepo(db).require_character(character_id)
        ext = _img_ext(http_request)
        key = f"refs/characters/{_safe_name_part(character_id)}.{ext}"
        ct = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        url = await asyncio.to_thread(_persist_media, key, data, ct)
        asset = MediaAssetRepo(db).create(kind="image", s3_key=key, s3_url=url)
        c.image_asset_id, c.s3_url = asset.id, url
        db.commit()
    except NotFoundError:
        db.rollback()
        raise HTTPException(status_code=404, detail="character not found")
    except Exception:
        db.rollback()
        raise
    return {"character": _character_dict(c)}


@router.post("/characters/{character_id}/default")
async def set_default_character_ep(character_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        ok = CharacterRepo(db).set_default(character_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    if not ok:
        raise HTTPException(status_code=404, detail="character not found")
    return {"ok": True}


@router.get("/characters/{character_id}/preview")
async def character_preview_ep(character_id: str, db: Session = Depends(get_db)):
    c = CharacterRepo(db).get_character(character_id)
    if not c:
        raise HTTPException(status_code=404, detail="character not found")
    return _ref_url_response(c.s3_url or "")


# ─── Brands = named groups of scene reference images (picked per shot) ───────────

def _brand_dict(brand, db: Session) -> dict:
    return {
        "id": brand.id, "name": brand.name, "tagline": brand.tagline or "",
        "platform_style": brand.platform_style or "", "theme": brand.theme or "",
        "mood": brand.mood or "", "material_palette": brand.material_palette or "",
        "lighting": brand.lighting or "", "editing_style": brand.editing_style or "",
        "vo_tone": brand.vo_tone or "", "music": brand.music or "",
        "camera_movement": brand.camera_movement or "", "words_per_second": brand.words_per_second or 0.0,
        "scenes": [_scene_out(s, db) for s in brand.scenes],
    }


@router.get("/brands")
async def list_brands_ep(db: Session = Depends(get_db)) -> dict:
    brands = BrandRepo(db).list_brands()
    return {"brands": [{"id": b.id, "name": b.name, "n_scenes": len(b.scenes)} for b in brands]}


@router.post("/brands")
async def create_brand_ep(name: str, db: Session = Depends(get_db)) -> dict:
    if not name.strip():
        raise HTTPException(status_code=400, detail="name required")
    try:
        brand = BrandRepo(db).create_brand(name)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"brand": _brand_dict(brand, db)}


@router.get("/brands/{brand_id}")
async def get_brand_ep(brand_id: str, db: Session = Depends(get_db)) -> dict:
    b = BrandRepo(db).get_brand(brand_id)
    if not b:
        raise HTTPException(status_code=404, detail="brand not found")
    return {"brand": _brand_dict(b, db)}


@router.put("/brands/{brand_id}")
async def update_brand_ep(brand_id: str, fields: dict = Body(...), db: Session = Depends(get_db)) -> dict:
    try:
        b = BrandRepo(db).update_brand(brand_id, fields or {})
        db.commit()
    except Exception:
        db.rollback()
        raise
    if not b:
        raise HTTPException(status_code=404, detail="brand not found")
    return {"brand": _brand_dict(b, db)}


@router.delete("/brands/{brand_id}")
async def delete_brand_ep(brand_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        deleted = BrandRepo(db).delete_brand(brand_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"deleted": deleted}


def _scene_out(scene, db: Session) -> dict:
    return {"id": scene.id, "name": scene.name, "desc": scene.desc or "", "layout": scene.layout or "",
            "image": _resolve_asset_url(db, scene.image_asset_id),
            "image_asset_id": scene.image_asset_id, "default": scene.is_default}


@router.post("/brands/{brand_id}/scenes")
async def add_scene_ep(
    brand_id: str, http_request: Request, name: str, desc: str = "", layout: str = "", default: bool = False,
    db: Session = Depends(get_db),
) -> dict:
    if not name.strip():
        raise HTTPException(status_code=400, detail="name required")
    data = await http_request.body()
    try:
        image_asset_id = None
        if data:
            ext = _img_ext(http_request)
            key = f"refs/scenes/{brand_id}/{_slug(name)}.{ext}"
            ct = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            url = _persist_media(key, data, ct)
            asset = MediaAssetRepo(db).create(kind="image", s3_key=key, s3_url=url)
            image_asset_id = asset.id
        scene = BrandRepo(db).add_scene(brand_id, name=name, desc=desc, layout=layout,
                                        image_asset_id=image_asset_id, is_default=default)
        db.commit()
    except NotFoundError:
        db.rollback()
        raise HTTPException(status_code=404, detail="brand not found")
    except Exception:
        db.rollback()
        raise
    return {"scene": _scene_out(scene, db)}


@router.put("/brands/{brand_id}/scenes/{scene_id}")
async def update_scene_ep(brand_id: str, scene_id: int, fields: dict = Body(...),
                          db: Session = Depends(get_db)) -> dict:
    """Edit a scene's text. `desc` says WHEN to pick this scene (the classifier reads it); `layout`
    says HOW it is composed — camera angle, what sits in the fore/background, where the host belongs
    — and is what the image-prompt author was missing. Image and default flag have their own routes."""
    try:
        scene = BrandRepo(db).update_scene(brand_id, scene_id, fields)
        db.commit()
    except Exception:
        db.rollback()
        raise
    if scene is None:
        raise HTTPException(status_code=404, detail="scene not found")
    return {"scene": _scene_out(scene, db)}


@router.post("/brands/{brand_id}/scenes/{scene_id}/describe")
async def describe_scene_ep(brand_id: str, scene_id: int, http_request: Request,
                            db: Session = Depends(get_db)) -> dict:
    """Draft this scene's `layout` from its reference photo. Returns the text for the operator to
    review and save — deliberately does NOT write it, so a bad draft can't overwrite a good
    hand-written one."""
    scene = next((s for s in BrandRepo(db).list_scenes(brand_id) if s.id == scene_id), None)
    if scene is None:
        raise HTTPException(status_code=404, detail="scene not found")
    if not scene.image_asset_id:
        raise HTTPException(status_code=400, detail="ฉากนี้ยังไม่มีรูป — อัปโหลดรูปก่อนจึงให้ AI บรรยายได้")
    asset = MediaAssetRepo(db).get(scene.image_asset_id)
    img = _media_bytes((asset.s3_url if asset else "") or "")
    if not img or not img.get("data"):
        raise HTTPException(status_code=502, detail="โหลดรูปฉากไม่สำเร็จ")
    try:
        layout = await http_request.app.state.container.gemini.describe_scene_layout(img)
    except Exception as e:  # noqa: BLE001 — surface the failure, the operator just clicked a button
        logger.exception("describe scene layout failed")
        raise HTTPException(status_code=502, detail=f"ให้ AI บรรยายไม่สำเร็จ: {e}")
    return {"scene_id": scene_id, "layout": layout}


@router.post("/brands/{brand_id}/scenes/{scene_id}/default")
async def set_default_scene_ep(brand_id: str, scene_id: int, db: Session = Depends(get_db)) -> dict:
    try:
        ok = BrandRepo(db).set_default_scene(brand_id, scene_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"ok": ok}


@router.delete("/brands/{brand_id}/scenes/{scene_id}")
async def delete_scene_ep(brand_id: str, scene_id: int, db: Session = Depends(get_db)) -> dict:
    try:
        deleted = BrandRepo(db).delete_scene(brand_id, scene_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"deleted": deleted}


@router.get("/brands/{brand_id}/scenes/{scene_id}/preview")
async def scene_preview(brand_id: str, scene_id: int, db: Session = Depends(get_db)):
    scenes = BrandRepo(db).list_scenes(brand_id)
    scene = next((s for s in scenes if s.id == scene_id), None)
    if not scene or not scene.image_asset_id:
        raise HTTPException(status_code=404, detail="scene image not found")
    asset = MediaAssetRepo(db).get(scene.image_asset_id)
    ref = (asset.s3_url if asset else "") or ""
    if not ref:
        raise HTTPException(status_code=404, detail="scene image not found")
    if ref.startswith(("http://", "https://")):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(ref, status_code=302)
    p = _safe_media_path(ref) or (Path(ref) if Path(ref).is_absolute() else ROOT_DIR / ref)
    if not p.exists():
        raise HTTPException(status_code=404, detail="scene image not found")
    return FileResponse(p, headers=_NO_STORE)


# ─── Menus = named groups of ingredient / equipment reference images ─────────────

def _menu_dict(menu, db: Session) -> dict:
    return {
        "id": menu.id, "name": menu.name,
        "subject_refs": [
            {
                "id": r.id, "name": r.name, "kind": r.kind,
                "image": _resolve_asset_url(db, r.image_asset_id),  # resolved URL, same shape v1 returned
                "image_asset_id": r.image_asset_id,
            }
            for r in menu.subject_refs
        ],
    }


@router.get("/menus")
async def list_menus_ep(db: Session = Depends(get_db)) -> dict:
    menus = MenuRepo(db).list_menus()
    return {"menus": [{"id": m.id, "name": m.name, "n_refs": len(m.subject_refs)} for m in menus]}


@router.post("/menus")
async def create_menu_ep(name: str, db: Session = Depends(get_db)) -> dict:
    if not name.strip():
        raise HTTPException(status_code=400, detail="name required")
    try:
        menu = MenuRepo(db).create_menu(name)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"menu": _menu_dict(menu, db)}


@router.get("/menus/{menu_id}")
async def get_menu_ep(menu_id: str, db: Session = Depends(get_db)) -> dict:
    m = MenuRepo(db).get_menu(menu_id)
    if not m:
        raise HTTPException(status_code=404, detail="menu not found")
    return {"menu": _menu_dict(m, db)}


@router.put("/menus/{menu_id}")
async def update_menu_ep(menu_id: str, fields: dict = Body(...), db: Session = Depends(get_db)) -> dict:
    try:
        m = MenuRepo(db).update_menu(menu_id, str(fields.get("name", "")))
        db.commit()
    except Exception:
        db.rollback()
        raise
    if not m:
        raise HTTPException(status_code=404, detail="menu not found")
    return {"menu": _menu_dict(m, db)}


@router.delete("/menus/{menu_id}")
async def delete_menu_ep(menu_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        deleted = MenuRepo(db).delete_menu(menu_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"deleted": deleted}


@router.post("/menus/{menu_id}/subjects")
async def add_subject_ref_ep(
    menu_id: str, http_request: Request, name: str, kind: str = "ingredient", db: Session = Depends(get_db)
) -> dict:
    if not name.strip():
        raise HTTPException(status_code=400, detail="name required")
    data = await http_request.body()
    try:
        image_asset_id = None
        if data:
            ext = _img_ext(http_request)
            key = f"refs/subjects/{menu_id}/{_slug(name)}-{kind}.{ext}"
            ct = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            url = _persist_media(key, data, ct)
            asset = MediaAssetRepo(db).create(kind="image", s3_key=key, s3_url=url)
            image_asset_id = asset.id
        ref = MenuRepo(db).add_subject_ref(menu_id, name=name, kind=kind, image_asset_id=image_asset_id)
        db.commit()
    except NotFoundError:
        db.rollback()
        raise HTTPException(status_code=404, detail="menu not found")
    except Exception:
        db.rollback()
        raise
    return {"subject_ref": _subject_ref_dict(db, ref)}


def _subject_ref_dict(db: Session, ref) -> dict:
    return {"id": ref.id, "name": ref.name, "kind": ref.kind,
            "image": _resolve_asset_url(db, ref.image_asset_id), "image_asset_id": ref.image_asset_id}


@router.post("/menus/{menu_id}/subjects/{ref_id}/image")
async def set_subject_ref_image_ep(menu_id: str, ref_id: int, http_request: Request,
                                   db: Session = Depends(get_db)) -> dict:
    """Attach or replace the photo on an existing subject — raw image bytes as the body, same
    contract as a character's image. Subjects seeded from research arrive without one, and the only
    way to give them a photo used to be delete-and-re-add, which changes the row id that every
    stored ref url is built from."""
    data = await http_request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    repo = MenuRepo(db)
    ref = next((r for r in repo.list_subject_refs(menu_id) if r.id == ref_id), None)
    if ref is None:
        raise HTTPException(status_code=404, detail="subject ref not found")
    try:
        ext = _img_ext(http_request)
        # Same key as the create path, so replacing overwrites rather than accumulating files. The
        # url therefore does not change either — the frontend cache-busts its <img> for that reason.
        key = f"refs/subjects/{menu_id}/{_slug(ref.name)}-{ref.kind}.{ext}"
        ct = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        url = await asyncio.to_thread(_persist_media, key, data, ct)
        asset = MediaAssetRepo(db).create(kind="image", s3_key=key, s3_url=url)
        ref = repo.set_subject_ref_image(menu_id, ref_id, asset.id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"subject_ref": _subject_ref_dict(db, ref)}


@router.delete("/menus/{menu_id}/subjects/{ref_id}/image")
async def clear_subject_ref_image_ep(menu_id: str, ref_id: int, db: Session = Depends(get_db)) -> dict:
    """Drop the photo, keep the subject. The MediaAsset row and the stored file are left alone —
    nothing in this codebase deletes assets, and the same file may be referenced elsewhere."""
    try:
        ref = MenuRepo(db).set_subject_ref_image(menu_id, ref_id, None)
        db.commit()
    except Exception:
        db.rollback()
        raise
    if ref is None:
        raise HTTPException(status_code=404, detail="subject ref not found")
    return {"subject_ref": _subject_ref_dict(db, ref)}


@router.delete("/menus/{menu_id}/subjects/{ref_id}")
async def delete_subject_ref_ep(menu_id: str, ref_id: int, db: Session = Depends(get_db)) -> dict:
    try:
        deleted = MenuRepo(db).delete_subject_ref(menu_id, ref_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"deleted": deleted}


@router.get("/menus/{menu_id}/subjects/{ref_id}/preview")
async def subject_ref_preview(menu_id: str, ref_id: int, db: Session = Depends(get_db)):
    refs = MenuRepo(db).list_subject_refs(menu_id)
    ref = next((r for r in refs if r.id == ref_id), None)
    if not ref or not ref.image_asset_id:
        raise HTTPException(status_code=404, detail="subject ref image not found")
    asset = MediaAssetRepo(db).get(ref.image_asset_id)
    url = (asset.s3_url if asset else "") or ""
    if not url:
        raise HTTPException(status_code=404, detail="subject ref image not found")
    if url.startswith(("http://", "https://")):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url, status_code=302)
    p = _safe_media_path(url) or (Path(url) if Path(url).is_absolute() else ROOT_DIR / url)
    if not p.exists():
        raise HTTPException(status_code=404, detail="subject ref image not found")
    return FileResponse(p, headers=_NO_STORE)


def _sync_menu_subjects(db: Session, rid: str, name: str, ingredients: list[str], equipment: list[str],
                        prune: bool = False) -> dict:
    """Build/refresh the config menu {rid} from explicit lists: one subject per ingredient/equipment
    (blank image; an existing ref and its uploaded image are preserved on re-sync).

    Dedupe is per (name, kind): the same name may legitimately exist as both an ingredient and a
    tool (the old name-only key silently dropped the second one).

    `prune=True` also DELETES subjects the incoming lists no longer mention — for re-running RESEARCH,
    where the new lists replace the old recipe rather than extend it, and leftovers from the previous
    research would otherwise sit in the menu forever. Rows whose (name, kind) still match are not
    touched at all, so their id and uploaded photo survive. Off everywhere else: importing a script
    or seeding a legacy run must never remove what the operator curated."""
    ingredients = [x for x in (ingredients or []) if isinstance(x, str) and x.strip()]
    equipment = [x for x in (equipment or []) if isinstance(x, str) and x.strip()]
    repo = MenuRepo(db)
    repo.ensure_menu(rid, name or rid)
    existing = {(r.name.strip().lower(), r.kind) for r in repo.list_subject_refs(rid)}
    wanted: set[tuple[str, str]] = set()
    for kind, names in (("ingredient", ingredients), ("equipment", equipment)):
        for item in names:
            key = (item.strip().lower(), kind)
            wanted.add(key)
            if key not in existing:
                repo.add_subject_ref(rid, name=item, kind=kind)
                existing.add(key)   # a duplicate within the incoming list is added once
    removed = 0
    if prune:
        for r in repo.list_subject_refs(rid):
            if (r.name.strip().lower(), r.kind) not in wanted:
                repo.delete_subject_ref(rid, r.id)
                removed += 1
    return {"menu": _menu_dict(repo.get_menu(rid), db), "n_ingredients": len(ingredients),
            "n_equipment": len(equipment), "n_removed": removed}


def _menu_from_script_doc(db: Session, rid: str, doc: dict) -> dict:
    """Script-doc wrapper over _sync_menu_subjects (kept for /menus/from_script_upload and the
    legacy no-menu fallback of the script step)."""
    script = doc.get("script") or doc
    prod = script.get("production") or {}
    return _sync_menu_subjects(db, rid, str(script.get("title") or rid),
                               prod.get("ingredients") or [], prod.get("equipment") or [])


@router.post("/menus/from_script_upload")
async def menu_from_script_upload(http_request: Request, name: str, db: Session = Depends(get_db)) -> dict:
    """Build a config menu from an UPLOADED script JSON (body). The menu NAME is the uploaded
    filename; its id is auto-generated/deduped (filenames can collide as ids)."""
    import json as _json

    if not name.strip():
        raise HTTPException(status_code=400, detail="name required")
    try:
        doc = _json.loads(await http_request.body())
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="body must be a script JSON document")
    if not isinstance(doc, dict):
        raise HTTPException(status_code=400, detail="script JSON must be an object")

    try:
        menu = MenuRepo(db).create_menu(name)
        result = _menu_from_script_doc(db, menu.id, doc)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return result


@router.post("/menus/{menu_id}/pull_s3")
async def menu_pull_s3(menu_id: str, http_request: Request, db: Session = Depends(get_db)) -> dict:
    """For each subject WITHOUT an image, try {base_url}/{subject_id}.{ext} for each configured ext and
    attach the first that is a real image (fetch_image validates content-type + size + decode). Subjects
    that already have an image are left untouched (don't clobber manual uploads)."""
    import httpx

    from app.services.image_fetch import fetch_image

    settings = http_request.app.state.container.settings
    s3 = settings.s3_refs
    if not (s3.enabled and s3.base_url):
        raise HTTPException(status_code=400, detail="s3_refs is not configured (base_url empty)")
    repo = MenuRepo(db)
    m = repo.get_menu(menu_id)
    if not m:
        raise HTTPException(status_code=404, detail="menu not found")
    base = s3.base_url.rstrip("/")
    todo = [r for r in repo.list_subject_refs(menu_id) if not r.image_asset_id]

    async with httpx.AsyncClient() as client:
        async def fill(ref) -> int | None:
            for ext in s3.extensions:
                r = await fetch_image(f"{base}/{ref.id}.{ext}", client=client, cfg=settings.image_search)
                if r:
                    key = f"refs/subjects/{menu_id}/{ref.id}.jpg"
                    url = _persist_media(key, r["data"], "image/jpeg")
                    asset = MediaAssetRepo(db).create(kind="image", s3_key=key, s3_url=url)
                    # Fill the EXISTING row. This used to add_subject_ref, which left the blank
                    # original behind and added a duplicate carrying the photo — one press of
                    # "ดึงรูปจาก S3" doubled every subject it managed to fill.
                    repo.set_subject_ref_image(menu_id, ref.id, asset.id)
                    return ref.id
            return None
        results = await asyncio.gather(*(fill(r) for r in todo))

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    filled = [rid for rid in results if rid]
    blank = [r.id for r in todo if r.id not in set(filled)]
    return {"filled": len(filled), "blank": blank, "menu": _menu_dict(repo.get_menu(menu_id), db)}


@router.post("/steps/video")
async def step_video(request: VideoStepRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Step 6: render ONE shot's video with Veo (slow long-running op, streamed with
    a heartbeat). First frame = this shot's image; optional last frame = the next
    shot's image (seamless join). Saves the .mp4 under /api/media and returns its URL."""
    container = http_request.app.state.container
    topic = request.topic.strip()
    active_config.set_current_run(request.run_id or "")   # see /steps/image — _omni_images_for_shot
    cfg = _merge_video_config(container, request)         # falls back to the default scene without it

    async def stream():
        if not request.prompt_video.strip():
            yield sse({"type": "error", "message": "prompt_video must not be empty."})
            return
        # step-7 accounting: Veo call counts + QC tokens land here; without a tracker they are silently dropped
        tracker = UsageTracker()
        set_tracker(tracker)
        _t0 = time.perf_counter()

        image = _media_bytes(request.image_url) if request.use_image_seed else None
        if image:
            image["url"] = request.image_url   # so the still/FIRST_FRAME ref logs its S3 link in Langfuse
        last_image = _media_bytes(request.last_image_url) if request.use_last_frame else None
        if last_image:
            last_image["url"] = request.last_image_url

        # image-to-video was requested but there's no first-frame image → fail loudly instead of
        # silently rendering text-to-video (which ignores the storyboard frame).
        if request.use_image_seed and not image:
            yield sse({"type": "error", "message": "โหมด image-to-video แต่ไม่มีรูปเฟรมแรก — generate รูปของช็อตนี้ก่อน"})
            return

        # OMNI ENGINE — this shot forced onto Gemini Omni Flash (multi-ref Interactions API)
        # regardless of config.yaml video_gen.model, so Veo and Omni shots can coexist in one
        # storyboard. Same vision-QC regen loop as the Veo path below (qc_generated_video is
        # provider-agnostic — it just watches the resulting clip against the shot's intent).
        if True:   # Omni-only (POC parity): always render on Gemini Omni Flash; the Veo branch below is dormant
            # Real Omni continuity: the previous shot's clip's actual LAST FRAME (not its own still
            # image) becomes this shot's FIRST_FRAME — see _omni_prev_frame_ref/_omni_images_for_shot.
            prev_frame = await _omni_prev_frame_ref(request.prev_video_url)
            omni_images = _omni_images_for_shot(image, last_image, request.shot_id, db,
                                                prev_frame=prev_frame, match_cut=request.match_cut)
            omni_cfg = cfg.model_copy(update={"model": cfg.omni_model})
            qc_cfg = getattr(cfg, "output_qc", None)
            qc: dict = {"pass": True, "issues": []}
            # LOCKED blocks: rebuild [# Sources …] + the per-ref usage block from the images actually
            # being sent, overriding whatever the stored prompt froze at authoring time.
            render_prompt = _omni_apply_manifest(request.prompt_video, omni_images)
            attempt = 0
            while True:
                yield sse({"type": "status", "message": "Submitting to Gemini Omni Flash..." if attempt == 0
                           else f"QC ไม่ผ่าน → render ใหม่ ({attempt}/{qc_cfg.max_regens})…"})
                # Step5 now authors prompt_video DIRECTLY for Omni (POC v2 template + verbatim
                # voice/duration/on-screen blocks), so there is no Veo→Omni adapt round-trip — send as-is.
                try:
                    data = await container.gemini.generate_video_omni_multi(render_prompt, omni_images, cfg=omni_cfg)
                except Exception as e:  # noqa: BLE001
                    logger.exception("Omni video generation failed")
                    yield sse({"type": "error", "message": f"Omni failed: {e}"})
                    return
                if not data:
                    yield sse({"type": "error", "message": "Omni returned no video."})
                    return
                if not (qc_cfg and qc_cfg.enabled):
                    break
                yield sse({"type": "status", "message": "กำลัง QC คลิป (Gemini ดูวิดีโอ)…"})
                qc = await container.gemini.qc_generated_video(
                    data, {"shot_kind": request.shot_kind, "prompt_video": request.prompt_video}, topic)
                if not qc.get("pass"):
                    issues = "; ".join(qc.get("issues") or []) or "the QC rules listed"
                    hint = (qc.get("fix_hint") or "").strip()
                    qc["suggested_prompt"] = (
                        request.prompt_video + "\n\nQC FEEDBACK — the previous clip FAILED review because: "
                        f"{issues}. {hint} Re-render this SAME shot correcting ONLY these problems; keep the "
                        "framing, subjects, dialogue and sounds otherwise identical."
                    )
                if qc.get("pass") or attempt >= max(0, qc_cfg.max_regens):
                    break
                attempt += 1
                render_prompt = qc["suggested_prompt"]
                yield sse({"type": "qc_retry", "scene_id": request.scene_id, "no": request.no,
                           "attempt": attempt, "max_regens": qc_cfg.max_regens,
                           "issues": qc.get("issues") or [], "fix_hint": (qc.get("fix_hint") or ""), "prompt": render_prompt})
                logger.info(f"omni video QC fail {request.scene_id}/{request.no}: {qc.get('issues')} → retry {attempt}/{qc_cfg.max_regens}")

            fname = f"{_safe_name_part(request.scene_id)}_{request.no}_{int(time.time())}.mp4"
            s3_key = f"{_slug(topic)}_videos/{fname}"
            url = _persist_media(s3_key, data, "video/mp4")
            obs.attach_media(url, "video")
            # Omni has no duration KNOB (unlike Veo's config.yaml video_gen.duration_seconds) — it
            # renders "roughly 4-10s" on its own, so probe what it actually produced instead of
            # recording Veo's configured value against an Omni clip.
            real_duration = await asyncio.to_thread(_probe_duration_bytes, data)
            lf_url: str | None = None
            lf_bytes = await asyncio.to_thread(_extract_last_frame_bytes, data)
            if lf_bytes:
                lf_key = s3_key.rsplit(".", 1)[0] + "_lastframe.png"   # rides the clip's key → never overwritten either
                lf_url = _persist_media(lf_key, lf_bytes, "image/png")
            if request.shot_id is not None:
                try:
                    asset = MediaAssetRepo(db).create(
                        kind="video", s3_key=s3_key, s3_url=url, duration_seconds=real_duration,
                        run_id=request.run_id, last_frame_s3_url=lf_url,
                        shot_id=request.shot_id, refs_used=_refs_used_of(omni_images))
                    StoryboardRepo(db).set_shot_video(request.shot_id, asset.id)
                    db.commit()
                except Exception:
                    db.rollback()
                    logger.exception(f"failed to persist generated video for shot {request.shot_id}")
            yield sse({
                "type": "video_generated", "scene_id": request.scene_id, "no": request.no, "url": url,
                "duration_seconds": real_duration, "resolution": cfg.resolution,
                "generate_audio": cfg.generate_audio, "last_frame": bool(prev_frame),
                "qc_passed": bool(qc.get("pass", True)), "qc_issues": qc.get("issues") or [],
                "qc_suggested_prompt": qc.get("suggested_prompt") or "", "engine": "omni",
            })
            usage_summary = tracker.summary()
            _persist_usage(db, request.run_id, "video", usage_summary)
            yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
            yield sse({"type": "done"})
            return

        # Render + vision-QC loop: a clip that fails QC (state drift / morph / containment / tail scene)
        # is re-rendered with the QC feedback APPENDED to the prompt (mirror of the image QC loop), up
        # to output_qc.max_regens times. The last render is always kept; its verdict + the retry prompt
        # ride on the SSE events so the UI can show exactly what was sent to the model.
        qc_cfg = getattr(cfg, "output_qc", None)
        qc: dict = {"pass": True, "issues": []}
        render_prompt = request.prompt_video
        attempt = 0
        while True:
            yield sse({"type": "status", "message": "Submitting to Veo..." if attempt == 0 else f"QC ไม่ผ่าน → render ใหม่ ({attempt}/{qc_cfg.max_regens})…"})
            task = asyncio.create_task(
                container.gemini.generate_video(render_prompt, image, last_image, cfg=cfg)
            )
            elapsed = 0
            try:
                while not task.done():
                    # Client hit Stop / closed the tab → cancel the Veo task so the render stops billing.
                    if await http_request.is_disconnected():
                        logger.info(f"client disconnected — cancelling Veo render {request.scene_id}/{request.no}")
                        return
                    yield sse({"type": "status", "message": f"Rendering video with Veo… ({elapsed}s, ~1-3 min)"})
                    await asyncio.sleep(10)
                    elapsed += 10
                data = task.result()
            except asyncio.CancelledError:
                raise  # generator itself cancelled (disconnect mid-yield) — finally below cleans the task up
            except Exception as e:  # noqa: BLE001 — surface Veo failure to the client
                logger.exception("Video generation failed")
                yield sse({"type": "error", "message": f"Veo failed: {e}"})
                return
            finally:
                if not task.done():
                    task.cancel()
            if not data:
                yield sse({"type": "error", "message": "Veo returned no video."})
                return
            if not (qc_cfg and qc_cfg.enabled):
                break
            yield sse({"type": "status", "message": "กำลัง QC คลิป (Gemini ดูวิดีโอ)…"})
            qc = await container.gemini.qc_generated_video(
                data, {"shot_kind": request.shot_kind, "prompt_video": request.prompt_video}, topic)
            if not qc.get("pass"):
                # Compose the feedback-augmented prompt on EVERY fail — used for the auto-retry when
                # max_regens > 0, and surfaced to the UI (qc_suggested_prompt) so the user can choose
                # to attach it when THEY press regenerate (judge-only mode, max_regens = 0).
                issues = "; ".join(qc.get("issues") or []) or "the QC rules listed"
                hint = (qc.get("fix_hint") or "").strip()
                qc_suggested = (
                    request.prompt_video + "\n\nQC FEEDBACK — the previous clip FAILED review because: "
                    f"{issues}. {hint} Re-render this SAME shot correcting ONLY these problems; keep the "
                    "framing, subjects, dialogue and sounds otherwise identical."
                )
                qc["suggested_prompt"] = qc_suggested
            if qc.get("pass") or attempt >= max(0, qc_cfg.max_regens):
                break
            attempt += 1
            render_prompt = qc["suggested_prompt"]
            yield sse({"type": "qc_retry", "scene_id": request.scene_id, "no": request.no,
                       "attempt": attempt, "max_regens": qc_cfg.max_regens,
                       "issues": qc.get("issues") or [], "fix_hint": (qc.get("fix_hint") or ""), "prompt": render_prompt})
            logger.info(f"video QC fail {request.scene_id}/{request.no}: {qc.get('issues')} → retry {attempt}/{qc_cfg.max_regens} with feedback")

        fname = f"{_safe_name_part(request.scene_id)}_{request.no}_{int(time.time())}.mp4"
        s3_key = f"{_slug(topic)}_videos/{fname}"
        url = _persist_media(s3_key, data, "video/mp4")

        if request.shot_id is not None:
            try:
                asset = MediaAssetRepo(db).create(
                    kind="video", s3_key=s3_key, s3_url=url, duration_seconds=cfg.duration_seconds,
                    run_id=request.run_id, shot_id=request.shot_id)
                StoryboardRepo(db).set_shot_video(request.shot_id, asset.id)
                db.commit()
            except Exception:
                db.rollback()
                logger.exception(f"failed to persist generated video for shot {request.shot_id}")

        yield sse({
            "type": "video_generated", "scene_id": request.scene_id, "no": request.no, "url": url,
            "duration_seconds": cfg.duration_seconds, "resolution": cfg.resolution,
            "generate_audio": cfg.generate_audio, "last_frame": bool(last_image and image),
            "qc_passed": bool(qc.get("pass", True)), "qc_issues": qc.get("issues") or [],
            "qc_suggested_prompt": qc.get("suggested_prompt") or "",
        })
        usage_summary = tracker.summary()
        _persist_usage(db, request.run_id, "video", usage_summary)
        yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
        yield sse({"type": "done"})

    return StreamingResponse(obs.traced_stream(stream(), "step.video", run_id=request.run_id, scene_id=request.scene_id, no=request.no, shot_id=request.shot_id), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/image/undo")
async def step_image_undo(request: ImageUndoRequest, db: Session = Depends(get_db)) -> dict:
    """Put back the image this shot's current one replaced (render / edit / crop / upload).
    Symmetric — calling it again redoes. Exactly one step, which is all the single
    `prev_image_asset_id` pointer can honestly promise; older renders remain in the asset table."""
    board = StoryboardRepo(db)
    shot = board.get_shot(request.shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="ไม่พบช็อตนี้")
    if not shot.prev_image_asset_id:
        raise HTTPException(status_code=400, detail="ไม่มีรูปก่อนหน้าให้ย้อนกลับ")
    try:
        shot = board.undo_shot_image(request.shot_id)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(f"undo image failed for shot {request.shot_id}")
        raise HTTPException(status_code=500, detail="ย้อนรูปไม่สำเร็จ")
    return {"url": _resolve_asset_url(db, shot.current_image_asset_id) or "",
            "has_prev": bool(shot.prev_image_asset_id)}


@router.post("/steps/video/undo")
async def step_video_undo(request: VideoUndoRequest, db: Session = Depends(get_db)) -> dict:
    """Put back the clip this shot's current one replaced (render / edit / upload). Symmetric —
    calling it again redoes. Not a general history browser: exactly one step, which is what the
    single `prev_video_asset_id` pointer can honestly promise."""
    board = StoryboardRepo(db)
    shot = board.get_shot(request.shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="ไม่พบช็อตนี้")
    if not shot.prev_video_asset_id:
        raise HTTPException(status_code=400, detail="ไม่มีคลิปก่อนหน้าให้ย้อนกลับ")
    try:
        shot = board.undo_shot_video(request.shot_id)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(f"undo video failed for shot {request.shot_id}")
        raise HTTPException(status_code=500, detail="ย้อนคลิปไม่สำเร็จ")
    return {"url": _resolve_asset_url(db, shot.current_video_asset_id) or "",
            "has_prev": bool(shot.prev_video_asset_id)}


async def _load_shot_clip(db: Session, shot_id: int, tmp: Path) -> Path:
    """The shot's current clip as a temp file, or the HTTPException that explains why not."""
    shot = StoryboardRepo(db).get_shot(shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="ไม่พบช็อตนี้")
    url = _resolve_asset_url(db, shot.current_video_asset_id)
    if not url:
        raise HTTPException(status_code=400, detail="ช็อตนี้ยังไม่มีคลิป — เรนเดอร์ก่อน")
    r = await asyncio.to_thread(_media_bytes, url)
    if not r:
        raise HTTPException(status_code=400, detail="โหลดไฟล์คลิปไม่สำเร็จ (S3/local)")
    src = tmp / "in.mp4"
    src.write_bytes(r["data"])
    return src


# Same near-lossless settings for every local clip rewrite: CRF 18 (the merge path measured the
# default 23 losing ~7% edge energy), yuv420p+faststart for player compatibility.
_REENCODE = ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def _persist_shot_clip(db: Session, request, data: bytes, *, duration: float | None = None) -> str:
    """Store a locally rewritten clip exactly like the render/edit/upload paths do, so
    prev_video_asset_id records what it replaced and the existing undo button covers it.
    No keep_omni_chain: the clip's frames/timeline changed, so continuing the old Omni
    conversation against it would drift — same reasoning as upload."""
    fname = f"{_safe_name_part(request.scene_id)}_{request.no}_{int(time.time())}.mp4"
    s3_key = f"{_slug(request.topic.strip())}_videos/{fname}"
    url = _persist_media(s3_key, data, "video/mp4")
    asset = MediaAssetRepo(db).create(kind="video", s3_key=s3_key, s3_url=url,
                                      run_id=request.run_id, shot_id=request.shot_id,
                                      duration_seconds=duration)
    StoryboardRepo(db).set_shot_video(request.shot_id, asset.id)
    db.commit()
    return url


@router.post("/steps/video/crop")
async def step_video_crop(request: VideoCropRequest, db: Session = Depends(get_db)) -> dict:
    """Reframe a shot's clip: keep the selected rectangle on every frame, scaled back to the
    original pixel size. Plain POST like /steps/image/crop — pure ffmpeg, nothing streams."""
    active_config.set_current_run(request.run_id or "")
    with tempfile.TemporaryDirectory(prefix="vidcrop_") as tmpdir:
        tmp = Path(tmpdir)
        src = await _load_shot_clip(db, request.shot_id, tmp)
        W, H = await _probe_wh(src)
        left, top = round(request.x * W), round(request.y * H)
        right, bottom = round((request.x + request.w) * W), round((request.y + request.h) * H)
        if not (0 <= left < right <= W and 0 <= top < bottom <= H):
            raise HTTPException(status_code=400, detail=(
                f"crop box ({request.x:.3f},{request.y:.3f},{request.w:.3f},{request.h:.3f}) "
                f"falls outside the {W}x{H} frame"))
        out = tmp / "out.mp4"
        rc, err = await _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"crop={right - left}:{bottom - top}:{left}:{top},scale={W}:{H}",
            *_REENCODE, "-c:a", "copy", str(out)])
        if rc != 0 or not out.exists():
            logger.error(f"video crop ffmpeg failed for shot {request.shot_id}: {err[-500:]}")
            raise HTTPException(status_code=502, detail="ครอปคลิปไม่สำเร็จ (ffmpeg)")
        data = out.read_bytes()
    try:
        url = _persist_shot_clip(db, request, data)
    except Exception:
        db.rollback()
        logger.exception(f"failed to persist cropped video for shot {request.shot_id}")
        raise HTTPException(status_code=500, detail="บันทึกคลิปที่ครอปไม่สำเร็จ")
    logger.info(f"Cropped shot clip {request.scene_id}/{request.no} → {url} ({len(data):,} bytes)")
    return {"scene_id": request.scene_id, "no": request.no, "url": url}


@router.post("/steps/video/trim")
async def step_video_trim(request: VideoTrimRequest, db: Session = Depends(get_db)) -> dict:
    """Cut a shot's clip down to the KEEP segments, concatenated in order. One segment is the
    plain -ss/-to case; several go through trim/atrim+concat so a deleted middle stays
    frame-accurate. Audio follows the video."""
    active_config.set_current_run(request.run_id or "")
    segs = [(float(s.start), float(s.end)) for s in request.segments]
    if not segs:
        raise HTTPException(status_code=400, detail="ไม่มีช่วงที่เหลือให้เก็บ — ตัดทิ้งหมดไม่ได้")
    for i, (a, b) in enumerate(segs):
        if a < 0 or b <= a:
            raise HTTPException(status_code=400, detail=f"ช่วงที่ {i + 1} ไม่ถูกต้อง ({a:.2f}–{b:.2f})")
        if i and a < segs[i - 1][1]:
            raise HTTPException(status_code=400, detail="ช่วงเวลาซ้อนกัน — เรียงใหม่ก่อน")
    with tempfile.TemporaryDirectory(prefix="vidtrim_") as tmpdir:
        tmp = Path(tmpdir)
        src = await _load_shot_clip(db, request.shot_id, tmp)
        dur = await _probe_duration(src)
        if dur and segs[-1][1] > dur + 0.25:
            raise HTTPException(status_code=400,
                                detail=f"ช่วงตัดเกินความยาวคลิป ({segs[-1][1]:.2f}s > {dur:.2f}s)")
        out = tmp / "out.mp4"
        # An uploaded clip may carry no audio stream at all — a filter graph naming [0:a] (or a
        # bare -c:a aac map) hard-fails on it, so build the audio legs only when one exists.
        proc = await asyncio.create_subprocess_exec(
            _FFPROBE_BIN, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index",
            "-of", "csv=p=0", str(src),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        probe_out, _ = await proc.communicate()
        has_audio = bool(probe_out.decode().strip())
        if len(segs) == 1:
            a, b = segs[0]
            args = ["ffmpeg", "-y", "-i", str(src), "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
                    *_REENCODE, *(["-c:a", "aac"] if has_audio else ["-an"]), str(out)]
        else:
            fc = "".join(
                f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{i}];"
                + (f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[a{i}];" if has_audio else "")
                for i, (a, b) in enumerate(segs))
            fc += "".join(f"[v{i}][a{i}]" if has_audio else f"[v{i}]" for i in range(len(segs)))
            fc += f"concat=n={len(segs)}:v=1:a={int(has_audio)}[v]" + ("[a]" if has_audio else "")
            args = (["ffmpeg", "-y", "-i", str(src), "-filter_complex", fc, "-map", "[v]"]
                    + (["-map", "[a]"] if has_audio else [])
                    + [*_REENCODE, *(["-c:a", "aac"] if has_audio else []), str(out)])
        rc, err = await _run_ffmpeg(args)
        if rc != 0 or not out.exists():
            logger.error(f"video trim ffmpeg failed for shot {request.shot_id}: {err[-500:]}")
            raise HTTPException(status_code=502, detail="ตัดคลิปไม่สำเร็จ (ffmpeg)")
        data = out.read_bytes()
        new_dur = await _probe_duration(out)
    try:
        url = _persist_shot_clip(db, request, data, duration=new_dur or None)
    except Exception:
        db.rollback()
        logger.exception(f"failed to persist trimmed video for shot {request.shot_id}")
        raise HTTPException(status_code=500, detail="บันทึกคลิปที่ตัดไม่สำเร็จ")
    logger.info(f"Trimmed shot clip {request.scene_id}/{request.no} → {url} "
                f"({new_dur:.2f}s, {len(data):,} bytes)")
    return {"scene_id": request.scene_id, "no": request.no, "url": url,
            "duration": round(new_dur, 2)}


@router.post("/steps/video/edit_prompt")
async def step_video_edit_prompt(request: VideoEditPromptRequest, http_request: Request) -> dict:
    """Polish a rough edit note into a short Omni edit instruction (golden rule: short beats long).
    Best-effort — never fails the caller; falls back to the original instruction on any LLM error."""
    container = http_request.app.state.container
    instruction = request.instruction.strip()
    if not instruction:
        return {"instruction": ""}
    polished = await container.gemini.generate_omni_edit_prompt(instruction, request.refs)
    return {"instruction": polished}


@router.post("/steps/video/edit")
async def step_video_edit(request: VideoEditRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Conversational video edit (Omni): apply a short instruction to a shot's EXISTING
    generated_video, keeping the Interactions API conversation open across edits (see
    StoryboardShot.omni_interaction_id) so the model has full context of what it already changed —
    no video re-upload needed after the first turn. Overwrites generated_video in place, same as
    /revoice, so merge/UI pick up the new clip at the same shot."""
    container = http_request.app.state.container
    topic = request.topic.strip()
    active_config.set_current_run(request.run_id or "")   # see /steps/image
    cfg = container.settings.video_gen

    async def stream():
        instruction = request.instruction.strip()
        if not instruction:
            yield sse({"type": "error", "message": "instruction ว่าง"})
            return
        tracker = UsageTracker()
        set_tracker(tracker)
        _t0 = time.perf_counter()

        board = StoryboardRepo(db)
        shot = board.get_shot(request.shot_id)
        if shot is None:
            yield sse({"type": "error", "message": "ไม่พบช็อตนี้"})
            return
        previous_interaction_id = shot.omni_interaction_id
        video = None
        if not previous_interaction_id:
            video_url = _resolve_asset_url(db, shot.current_video_asset_id)
            if not video_url:
                yield sse({"type": "error", "message": "ช็อตนี้ยังไม่มีวิดีโอ — generate ก่อน"})
                return
            video = _media_bytes(video_url)
            if not video:
                yield sse({"type": "error", "message": "ไม่พบไฟล์วิดีโอ (S3/local) — generate ใหม่ก่อน"})
                return
        # Loaded with _media_bytes (no downscale) to match the video GENERATE path. A ref that fails
        # to load aborts the edit instead of being skipped: the UI tags each thumbnail <IMAGE_REF_k>
        # and the instruction may cite that tag, so dropping one silently shifts every later tag and
        # points the instruction at the wrong picture.
        refs = []
        for i, u in enumerate(request.extra_refs):
            r = _media_bytes(u)
            if not r:
                logger.warning(f"video edit: could not load extra ref {u}")
                yield sse({"type": "error",
                           "message": f"โหลด ref <IMAGE_REF_{i}> ไม่สำเร็จ — เอาออกหรืออัปใหม่ก่อนสั่งแก้"})
                return
            refs.append(r)

        yield sse({"type": "status", "message": "แก้ไขคลิปด้วย Gemini Omni Flash..."})
        try:
            data, interaction_id = await container.gemini.edit_omni_video(
                instruction, video=video, refs=refs, previous_interaction_id=previous_interaction_id, cfg=cfg)
        except Exception as e:  # noqa: BLE001
            logger.exception("Omni video edit failed")
            yield sse({"type": "error", "message": f"แก้ไขคลิปไม่สำเร็จ: {e}"})
            return

        fname = f"{_safe_name_part(request.scene_id)}_{request.no}_{int(time.time())}.mp4"
        s3_key = f"{_slug(topic)}_videos/{fname}"
        url = _persist_media(s3_key, data, "video/mp4")
        obs.attach_media(url, "video")
        try:
            asset = MediaAssetRepo(db).create(kind="video", s3_key=s3_key, s3_url=url,
                                              run_id=request.run_id, shot_id=request.shot_id)
            board.set_shot_video(request.shot_id, asset.id, keep_omni_chain=True)
            board.set_omni_interaction_id(request.shot_id, interaction_id)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(f"failed to persist edited video for shot {request.shot_id}")

        yield sse({"type": "video_edited", "scene_id": request.scene_id, "no": request.no, "url": url})
        usage_summary = tracker.summary()
        _persist_usage(db, request.run_id, "video_edit", usage_summary)
        yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
        yield sse({"type": "done"})

    return StreamingResponse(obs.traced_stream(stream(), "step.video_edit", run_id=request.run_id, scene_id=request.scene_id, no=request.no, shot_id=request.shot_id), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/steps/video_group")
async def step_video_group(request: VideoGroupRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Render a CONTINUOUS group of shots as ONE Veo take (extension chain). First frame =
    the group's first shot image; each later shot is appended by prompt only (Veo extension
    can't take a frame). Saves one combined .mp4 covering the whole group."""
    container = http_request.app.state.container
    topic = request.topic.strip()
    active_config.set_current_run(request.run_id or "")   # see /steps/image
    cfg = container.settings.video_gen

    async def stream():
        prompts = [p for p in request.prompts if (p or "").strip()]
        if not prompts:
            yield sse({"type": "error", "message": "prompts must not be empty."})
            return
        tracker = UsageTracker()
        set_tracker(tracker)
        _t0 = time.perf_counter()
        image = _media_bytes(request.image_url)
        if image:
            image["url"] = request.image_url   # so the group's own-frame ref logs its S3 link in Langfuse
        if not image:
            yield sse({"type": "error", "message": "first-frame image not found on disk."})
            return

        # OMNI ENGINE — no extension-chain API. Instead of one Omni call per shot (which broke
        # continuity the moment a shot fell outside the group and rendered separately on Veo),
        # bin-pack consecutive shots into batches that fit Omni's real ~10s-per-clip cap: a batch of
        # >=2 short shots renders as ONE merged clip (their prompts joined into a single continuous
        # beat), a batch of 1 (a shot too long to share a clip) renders alone same as before.
        # Continuity BETWEEN batches still uses the real last-frame-of-previous-clip chain.
        if request.engine == "omni":
            n = len(prompts)
            nos = request.nos or list(range(request.no, request.no + n))
            voice_overs = request.voice_overs or [""] * n
            omni_cfg = cfg.model_copy(update={"model": cfg.omni_model})

            # Prefer each shot's own authored estimate over the word-count heuristic: the prompt
            # author sized the clip against the motion as well as the line, and that same number is
            # what the render prompt asks Omni for — packing against a different figure would
            # contradict it. Falls back per shot, so a board authored before target_seconds existed
            # packs exactly as before.
            authored: dict[int, float] = {}
            if request.shot_ids:
                for _sid in request.shot_ids:
                    _row = db.get(DBStoryboardShot, _sid) if _sid else None
                    if _row is not None and _row.target_seconds:
                        authored[_sid] = float(_row.target_seconds)

            from app.services.gemini_client import _clamp_clip_seconds   # local, like every other use here

            batches: list[list[int]] = []
            batch_secs: list[float] = []   # each batch's own total, reused as the joined clip's target length
            cur: list[int] = []
            cur_secs = 0.0
            for j in range(n):
                _sid = request.shot_ids[j] if j < len(request.shot_ids) else None
                secs = authored.get(_sid) or _estimate_shot_seconds(voice_overs[j] if j < len(voice_overs) else "")
                if cur and cur_secs + secs > OMNI_SAFE_CLIP_SECONDS:
                    batches.append(cur)
                    batch_secs.append(cur_secs)
                    cur, cur_secs = [], 0.0
                cur.append(j)
                cur_secs += secs
            if cur:
                batches.append(cur)
                batch_secs.append(cur_secs)

            # Real Omni continuity WITHIN and ACROSS batches — chain each clip's actual last frame
            # into the next batch's FIRST_FRAME directly from the in-memory bytes just rendered.
            prev_frame: dict | None = None
            for batch_i, batch in enumerate(batches):
                if await http_request.is_disconnected():
                    return
                first_j, last_j = batch[0], batch[-1]
                batch_nos = [nos[j] for j in batch]
                label = f"{batch_nos[0]}-{batch_nos[-1]}" if len(batch) > 1 else str(batch_nos[0])
                yield sse({"type": "status", "message": f"Omni render {request.scene_id}.{label} (batch {batch_i + 1}/{len(batches)})…"})

                img_first = _media_bytes(request.images[first_j]) if first_j < len(request.images) else (image if first_j == 0 else None)
                if img_first and first_j < len(request.images):
                    img_first["url"] = request.images[first_j]
                shot_id_first = request.shot_ids[first_j] if first_j < len(request.shot_ids) else None
                omni_images = _omni_images_for_shot(img_first, None, shot_id_first, db, prev_frame=prev_frame)
                if len(batch) > 1:
                    img_last = _media_bytes(request.images[last_j]) if last_j < len(request.images) else None
                    if img_last and img_last.get("data"):
                        omni_images.append({
                            **img_last, "role": "IMAGE_REF",
                            "label": "this batch's LAST shot's intended look — the clip should arrive here by the end",
                        })

                # prompts are already authored directly for Omni (POC v2) — no Veo→Omni adapt needed.
                # This batch's shots are joined as consecutive beats of one continuous clip.
                if len(batch) > 1:
                    # One clip, one length: each beat's own "Target length" line is dropped and the
                    # batch total put in front, or the renderer gets several contradicting numbers.
                    combined_prompt = "\n\n---NEXT BEAT (same continuous clip, no cut)---\n\n".join(
                        _OMNI_TARGET_LEN_RE.sub("", prompts[j]) for j in batch)
                    _total = _clamp_clip_seconds(batch_secs[batch_i])
                    if _total:
                        combined_prompt = f"Target length: about {_total:g} seconds.\n\n{combined_prompt}"
                else:
                    combined_prompt = prompts[batch[0]]
                # LOCKED blocks rebuilt from the images this batch actually sends (each beat's frozen
                # header is stripped first, so one authoritative header fronts the combined prompt).
                combined_prompt = _omni_apply_manifest(combined_prompt, omni_images)
                try:
                    clip = await container.gemini.generate_video_omni_multi(combined_prompt, omni_images, cfg=omni_cfg)
                except Exception as e:  # noqa: BLE001 — one bad batch must not sink the others
                    logger.warning(f"omni group batch {request.scene_id}.{label} failed: {e}")
                    yield sse({"type": "error", "scene_id": request.scene_id, "no": batch_nos[0], "message": f"ช็อต {label}: {e}"})
                    prev_frame = None   # a failed/missing clip breaks the continuity chain
                    continue
                if not clip:
                    yield sse({"type": "error", "scene_id": request.scene_id, "no": batch_nos[0], "message": f"ช็อต {label}: Omni ไม่คืนวิดีโอ"})
                    prev_frame = None
                    continue

                last_frame_bytes = await asyncio.to_thread(_extract_last_frame_bytes, clip)
                prev_frame = {"mime": "image/png", "data": last_frame_bytes} if last_frame_bytes else None
                fname_j = f"{_safe_name_part(request.scene_id)}_{batch_nos[0]}_{int(time.time())}.mp4"
                s3_key_j = f"{_slug(topic)}_videos/{fname_j}"
                clip_url = _persist_media(s3_key_j, clip, "video/mp4")
                real_duration = await asyncio.to_thread(_probe_duration_bytes, clip)
                lf_url_j: str | None = None
                if last_frame_bytes:
                    lf_key_j = s3_key_j.rsplit(".", 1)[0] + "_lastframe.png"
                    lf_url_j = _persist_media(lf_key_j, last_frame_bytes, "image/png")

                if request.run_id:
                    try:
                        # One clip covers several shots; the row can name only one, so it names
                        # the batch's FIRST — every covered shot still points at this asset id.
                        _first_sid = request.shot_ids[batch[0]] if batch[0] < len(request.shot_ids) else None
                        asset = MediaAssetRepo(db).create(
                            kind="video", s3_key=s3_key_j, s3_url=clip_url,
                            duration_seconds=real_duration, run_id=request.run_id,
                            last_frame_s3_url=lf_url_j,
                            shot_id=_first_sid, refs_used=_refs_used_of(omni_images))
                        board = StoryboardRepo(db)
                        for j in batch:
                            shot_id_j = request.shot_ids[j] if j < len(request.shot_ids) else None
                            if shot_id_j is not None:
                                board.set_shot_video(shot_id_j, asset.id)
                        db.commit()
                    except Exception:
                        db.rollback()
                        logger.exception(f"failed to persist omni group video for shots {batch_nos}")

                yield sse({
                    "type": "video_generated", "scene_id": request.scene_id, "no": batch_nos[0], "url": clip_url,
                    "duration_seconds": real_duration,
                    **({"group": True, "shots": len(batch), "covered_nos": batch_nos} if len(batch) > 1 else {}),
                })
            usage_summary = tracker.summary()
            _persist_usage(db, request.run_id, "video_group", usage_summary)
            yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
            yield sse({"type": "done"})
            return

        yield sse({"type": "status", "message": f"Rendering a {len(prompts)}-shot continuous take with Veo..."})
        # chain runs in a thread; it reports the clip it's on via on_progress (plain dict write → GIL-safe)
        # so the heartbeat below can show "clip i/n" live in the UI.
        progress = {"i": 0, "n": len(prompts)}
        def _on_progress(i: int, n: int) -> None:
            progress["i"], progress["n"] = i, n
        task = asyncio.create_task(container.gemini.generate_video_chain(prompts, image, cfg=cfg, on_progress=_on_progress))
        elapsed = 0
        try:
            while not task.done():
                # Client hit Stop / closed the tab → cancel the chain so remaining clips don't render.
                if await http_request.is_disconnected():
                    logger.info(f"client disconnected — cancelling Veo chain {request.scene_id}/{request.no}")
                    return
                cur = f"clip {progress['i']}/{progress['n']}" if progress["i"] else "starting"
                yield sse({"type": "status", "message": f"Chaining take with Veo… {cur} ({elapsed}s)"})
                await asyncio.sleep(10)
                elapsed += 10
            data, covered = task.result()   # covered = shots the cumulative clip actually holds (partial-salvage)
        except asyncio.CancelledError:
            raise  # generator itself cancelled (disconnect mid-yield) — finally below cleans the task up
        except Exception as e:  # noqa: BLE001 — surface Veo failure to the client
            logger.exception("Video chain failed")
            yield sse({"type": "error", "message": f"Veo chain failed: {e}"})
            return
        finally:
            if not task.done():
                task.cancel()

        n = len(prompts)
        nos = request.nos or list(range(request.no, request.no + n))

        # the salvaged prefix (if any) → one continuous group clip on the anchor shot
        if data and covered >= 1:
            fname = f"{_safe_name_part(request.scene_id)}_{request.no}_group_{int(time.time())}.mp4"
            s3_key = f"{_slug(topic)}_videos/{fname}"
            group_url = _persist_media(s3_key, data, "video/mp4")

            # v2: the group clip covers `covered` shots at once — every one of them points its
            # current_video_asset_id at the SAME MediaAsset row (one clip, several shots), same
            # as v1's plain generated_video string being shared across the group's shots.
            if request.run_id and request.shot_ids:
                try:
                    asset = MediaAssetRepo(db).create(
                        kind="video", s3_key=s3_key, s3_url=group_url,
                        duration_seconds=cfg.duration_seconds, run_id=request.run_id,
                        shot_id=request.shot_ids[0] if request.shot_ids else None)
                    board = StoryboardRepo(db)
                    for shot_id in request.shot_ids[:covered]:
                        board.set_shot_video(shot_id, asset.id)
                    db.commit()
                except Exception:
                    db.rollback()
                    logger.exception(f"failed to persist group video for shots {request.shot_ids[:covered]}")

            yield sse({"type": "video_generated", "scene_id": request.scene_id, "no": request.no,
                       "url": group_url, "group": True,
                       "shots": covered, "covered_nos": nos[:covered]})

        # anything the chain couldn't reach → render each shot individually (image-to-video, own frame +
        # rai_retries/backoff) so one bad clip no longer sinks the whole group.
        if covered < n:
            yield sse({"type": "status", "message": f"chain เก็บได้ {covered}/{n} ช็อต · กำลัง render เดี่ยวช็อตที่เหลือ…"})
        for j in range(covered, n):
            if await http_request.is_disconnected():
                return
            shot_no = nos[j]
            img_j = _media_bytes(request.images[j]) if j < len(request.images) else None
            yield sse({"type": "status", "message": f"render เดี่ยว shot {request.scene_id}.{shot_no} ({j - covered + 1}/{n - covered})…"})
            try:
                clip = await container.gemini.generate_video(prompts[j], img_j, None, cfg=cfg)
            except Exception as e:  # noqa: BLE001 — one bad fallback shot must not sink the others
                logger.warning(f"fallback shot {request.scene_id}.{shot_no} failed: {e}")
                yield sse({"type": "error", "scene_id": request.scene_id, "no": shot_no,
                           "message": f"ช็อต {shot_no} เดี่ยวก็ไม่ผ่าน: {e}"})
                continue
            if not clip:
                yield sse({"type": "error", "scene_id": request.scene_id, "no": shot_no, "message": f"ช็อต {shot_no}: Veo ไม่คืนวิดีโอ"})
                continue
            fname_j = f"{_safe_name_part(request.scene_id)}_{shot_no}_{int(time.time())}.mp4"
            s3_key_j = f"{_slug(topic)}_videos/{fname_j}"
            clip_url = _persist_media(s3_key_j, clip, "video/mp4")

            if request.run_id and j < len(request.shot_ids):
                try:
                    asset = MediaAssetRepo(db).create(
                        kind="video", s3_key=s3_key_j, s3_url=clip_url,
                        duration_seconds=cfg.duration_seconds, run_id=request.run_id,
                        shot_id=request.shot_ids[j])
                    StoryboardRepo(db).set_shot_video(request.shot_ids[j], asset.id)
                    db.commit()
                except Exception:
                    db.rollback()
                    logger.exception(f"failed to persist fallback video for shot {request.shot_ids[j]}")

            yield sse({"type": "video_generated", "scene_id": request.scene_id, "no": shot_no,
                       "url": clip_url})

        if not data and covered == 0 and n == 0:
            yield sse({"type": "error", "message": "Veo returned no video."})
        usage_summary = tracker.summary()
        _persist_usage(db, request.run_id, "video_group", usage_summary)
        yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
        yield sse({"type": "done"})

    return StreamingResponse(obs.traced_stream(stream(), "step.video_group", run_id=request.run_id, scene_id=request.scene_id, no=request.no), media_type="text/event-stream", headers=_SSE_HEADERS)


# ─── ffmpeg / font resolution (subtitle burn-in must never produce tofu) ────────

_FFMPEG_BIN = "ffmpeg"
_FFPROBE_BIN = "ffprobe"
_FFMPEG_RESOLVED = False


@lru_cache(maxsize=8)
def _has_subtitles_filter(binpath: str) -> bool:
    """True if this ffmpeg build has the libass `subtitles` filter."""
    try:
        out = subprocess.run([binpath, "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=15).stdout
        return re.search(r"^\s*\S+\s+subtitles\s", out, re.M) is not None
    except Exception:
        return False


def _resolve_ffmpeg(cfg) -> None:
    """Pick the ffmpeg/ffprobe binaries once. Honors cfg.ffmpeg_bin; if a bare `ffmpeg` lacks
    the subtitles (libass) filter, fall back to a known libass build (e.g. ffmpeg-full)."""
    global _FFMPEG_BIN, _FFPROBE_BIN, _FFMPEG_RESOLVED
    if _FFMPEG_RESOLVED:
        return
    binp = getattr(cfg, "ffmpeg_bin", "") or "ffmpeg"
    if binp == "ffmpeg" and not _has_subtitles_filter("ffmpeg"):
        for cand in ("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg", "/usr/local/opt/ffmpeg-full/bin/ffmpeg"):
            if Path(cand).exists() and _has_subtitles_filter(cand):
                binp = cand
                logger.info(f"ffmpeg: using libass build {cand} (PATH ffmpeg lacks subtitles filter)")
                break
    _FFMPEG_BIN = binp
    probe = Path(binp).with_name("ffprobe")
    _FFPROBE_BIN = str(probe) if probe.exists() else "ffprobe"
    _FFMPEG_RESOLVED = True


@lru_cache(maxsize=8)
def _font_meta(path: str) -> tuple[str | None, bool]:
    """(family_name, covers_thai) read straight from the .ttf via fontTools — so the .ass
    Fontname matches the font's real family (no tofu) and we know it has Thai glyphs."""
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(path, fontNumber=0, lazy=True)
        nm = f["name"]
        family = nm.getDebugName(16) or nm.getDebugName(1)
        cmap = f.getBestCmap()
        thai = all(c in cmap for c in (0x0E01, 0x0E33, 0x0E48, 0x0E51))  # ก ำ ่ ๑
        f.close()
        return (family, bool(thai))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"font meta read failed ({path}): {e}")
        return (None, False)


def _subtitle_font_path(cfg) -> Path:
    fp = Path(cfg.subtitle_font)
    return fp if fp.is_absolute() else ROOT_DIR / cfg.subtitle_font


def _subs_ready(cfg) -> tuple[bool, str]:
    """Pre-flight: only burn subtitles when it is PROVABLY safe (no tofu). Returns (ok, reason)."""
    _resolve_ffmpeg(cfg)
    if not _has_subtitles_filter(_FFMPEG_BIN):
        return False, "ffmpeg ไม่มี libass (subtitles filter) — ติดตั้ง ffmpeg ที่มี libass"
    if not cfg.subtitle_font:
        return False, "ไม่ได้ตั้ง subtitle_font"
    fp = _subtitle_font_path(cfg)
    if not fp.exists():
        return False, f"ไม่พบไฟล์ฟอนต์: {cfg.subtitle_font}"
    _, thai = _font_meta(str(fp))
    if not thai:
        return False, f"ฟอนต์ไม่มี glyph ไทย: {cfg.subtitle_font}"
    return True, ""


async def _run_ffmpeg(args: list[str]) -> tuple[int, str]:
    if args and args[0] == "ffmpeg":            # use the resolved (libass-capable) binary
        args = [_FFMPEG_BIN, *args[1:]]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    return proc.returncode, (stderr or b"").decode("utf-8", "replace")


async def _copy_concat(paths: list[Path], out: Path, run_dir: Path) -> tuple[bool, str]:
    """Concat clips with the video STREAM-COPIED — the only assembly that costs nothing in quality.
    Measured against a real Omni clip: SSIM 1.000000, identical edge energy; the re-encode path
    below (libx264 at its default CRF 23) loses ~7% edge energy and half the bitrate.

    Audio is still re-encoded to one uniform rate: clips can differ (re-voiced clips are 44.1kHz
    vs Veo's 48kHz), and a full `-c copy` concat adopts the FIRST clip's audio params → every
    later clip goes silent.

    Fails (returns False) when the clips' video params don't match — a mixed-resolution or
    mixed-codec set makes the concat demuxer bail — and the caller falls back to
    `_normalized_concat`, which scales everything to a common shape at re-encode cost.

    Known trade-off, measured: Omni/Veo clips carry ~5-9ms more audio than video, and stream-copy
    lets it accumulate (27ms over 3 clips, so ~90ms over 10 — around where lip-sync slip starts to
    show). Some players also reset their decoder at a stream-copy seam and stutter there. Both are
    why `_normalized_concat` exists; if they bite, merge through it instead."""
    listfile = run_dir / "concat.txt"
    listfile.write_text(
        "".join(f"file '{str(p.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for p in paths),
        encoding="utf-8",
    )
    rc, err = await _run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
                                 "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2", str(out)])
    return rc == 0, err


async def _normalized_concat(paths: list[Path], out: Path, *, fps: int = 24, width: int = 1280, height: int = 720) -> tuple[bool, str]:
    """Concat clips in order into ONE continuous re-encoded stream (concat FILTER, not copy-concat).
    Re-encode is required, not optional, because (1) Veo clips' audio runs ~5ms longer than video per
    clip — copy-concat across many clips accumulates drift + the browser resets its decoder at every
    stream-copy seam (stutter); (2) clips can differ in resolution (720p/1080p) and codec (h264/hevc)
    — a plain concat demuxer errors (-22) on mismatched params. Every input is scaled/padded to a
    common WxH/SAR/fps/pixel-format and its audio resampled to a common format BEFORE the concat, so
    the filter accepts them. Returns (ok, err_tail) — caller decides the fallback on failure (e.g. a
    clip with no audio stream can't be mapped as `[i:a]`)."""
    n = len(paths)
    if n == 0:
        return False, "no clips"
    args = ["ffmpeg", "-y"]
    for p in paths:
        args += ["-i", str(p)]
    parts = []
    for i in range(n):
        parts.append(
            f"[{i}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v{i}];"
            f"[{i}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
    concat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
    fc = ";".join(parts) + f";{concat_in}concat=n={n}:v=1:a=1[v][a]"
    rc, err = await _run_ffmpeg(args + [
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-r", str(fps), "-fps_mode", "cfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(out)])
    return rc == 0, err


async def _probe_duration(path: Path) -> float:
    proc = await asyncio.create_subprocess_exec(
        _FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    try:
        return float(out.decode().strip())
    except ValueError:
        return 0.0


async def _probe_wh(path: Path) -> tuple[int, int]:
    proc = await asyncio.create_subprocess_exec(
        _FFPROBE_BIN, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    try:
        w, h = out.decode().strip().split("x")
        return int(w), int(h)
    except (ValueError, IndexError):
        return 1280, 720


async def _probe_fps(path: Path) -> float:
    """Real frame rate of the first video stream (r_frame_rate, e.g. '24/1'). 0.0 on failure."""
    proc = await asyncio.create_subprocess_exec(
        _FFPROBE_BIN, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    try:
        num, _, den = out.decode().strip().partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


# ─── C2: post color pipeline (color match → assemble → grade) ────────────────────

def _ff_filter_path(p: Path) -> str:
    """Escape a filesystem path for use INSIDE an ffmpeg filtergraph option value (lut3d=...):
    forward slashes + escaped drive colon (the caller wraps it in single quotes)."""
    return p.as_posix().replace(":", r"\:")


async def _clip_color_stats(path: Path) -> dict | None:
    """Mean luma (YAVG, 0–255) + mean chroma saturation (SATAVG) of a clip, via the ffmpeg
    signalstats filter (metadata printed to stdout, one line per frame). None on failure."""
    proc = await asyncio.create_subprocess_exec(
        _FFMPEG_BIN, "-hide_banner", "-i", str(path), "-map", "0:v:0",
        "-vf", "signalstats,metadata=print:file=-", "-f", "null", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    y: list[float] = []
    sat: list[float] = []
    for line in out.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if "signalstats.YAVG=" in line:
            try:
                y.append(float(line.rsplit("=", 1)[1]))
            except ValueError:
                pass
        elif "signalstats.SATAVG=" in line:
            try:
                sat.append(float(line.rsplit("=", 1)[1]))
            except ValueError:
                pass
    if not y:
        return None
    return {"y": sum(y) / len(y), "sat": (sum(sat) / len(sat)) if sat else 0.0}


def _color_match_corrections(stats: list[dict | None], cfg) -> list[str | None]:
    """Per-clip `eq` correction string pulling each clip's mean luma/saturation toward the BATCH
    MEDIAN (clamped so a stylistic outlier is nudged, never crushed). None = clip is fine as-is
    (within tolerance, or its stats couldn't be measured). Pure function — unit-testable."""
    vals = [s for s in stats if s]
    if len(vals) < 2:
        return [None] * len(stats)
    med_y = statistics.median(s["y"] for s in vals)
    sats = [s["sat"] for s in vals if s["sat"] > 1e-3]
    med_sat = statistics.median(sats) if sats else 0.0
    out: list[str | None] = []
    for s in stats:
        if not s:
            out.append(None)
            continue
        db = (med_y - s["y"]) / 255.0
        db = max(-cfg.max_brightness_shift, min(cfg.max_brightness_shift, db))
        if med_sat > 0 and s["sat"] > 1e-3:
            ratio = med_sat / s["sat"]
            lo, hi = 1.0 / cfg.max_saturation_ratio, cfg.max_saturation_ratio
            ratio = max(lo, min(hi, ratio))
        else:
            ratio = 1.0
        terms = []
        if abs(db) >= cfg.min_correction:
            terms.append(f"brightness={db:.4f}")
        if abs(ratio - 1.0) >= cfg.min_correction:
            terms.append(f"saturation={ratio:.3f}")
        out.append(("eq=" + ":".join(terms)) if terms else None)
    return out


def _resolve_lut(cfg) -> Path | None:
    """The LUT to grade final.mp4 with: the ACTIVE brand's `lut` field wins over
    video_gen.post_color.lut (both may be project-relative). Missing file → None + warn."""
    raw = str((scene_store.active_brand() or {}).get("lut") or "").strip() or (cfg.lut or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT_DIR / raw
    if not p.exists():
        logger.warning(f"post_color: LUT not found: {raw} — grading without a LUT")
        return None
    return p


def _grade_filter(cfg, lut_path: Path | None) -> str:
    """Final-grade -vf chain: brand LUT (lut3d) → saturation/contrast trim (eq) → film grain
    (noise). "" = nothing configured → the grade pass (a full re-encode) is skipped entirely."""
    parts: list[str] = []
    if lut_path:
        parts.append(f"lut3d='{_ff_filter_path(lut_path)}'")
    eq_terms = []
    if abs(cfg.saturation - 1.0) >= 0.01:
        eq_terms.append(f"saturation={cfg.saturation:.3f}")
    if abs(cfg.contrast - 1.0) >= 0.01:
        eq_terms.append(f"contrast={cfg.contrast:.3f}")
    if eq_terms:
        parts.append("eq=" + ":".join(eq_terms))
    if cfg.grain and cfg.grain > 0:
        parts.append(f"noise=alls={cfg.grain:g}:allf=t+u")   # temporal+uniform → film-like grain
    return ",".join(parts)


# ─── C3: episode gate (measure the REAL final.mp4) ───────────────────────────────

def _gate_checks(gcfg, dur: float, w: int, h: int, fps: float) -> list[dict]:
    """TOR checks on the measured final.mp4. Each: {name, value, expected, ok}. Pure — testable."""
    checks: list[dict] = []
    if gcfg.min_seconds or gcfg.max_seconds:
        lo = gcfg.min_seconds or 0
        hi = gcfg.max_seconds or float("inf")
        exp = (f"{lo:g}–{gcfg.max_seconds:g}s" if gcfg.max_seconds else f">= {lo:g}s")
        checks.append({"name": "duration", "value": round(dur, 2), "expected": exp,
                       "ok": lo <= dur <= hi})
    if gcfg.required_width or gcfg.required_height:
        ok = ((not gcfg.required_width or w == gcfg.required_width)
              and (not gcfg.required_height or h == gcfg.required_height))
        checks.append({"name": "resolution", "value": f"{w}x{h}",
                       "expected": f"{gcfg.required_width or '*'}x{gcfg.required_height or '*'}", "ok": ok})
    if gcfg.required_fps:
        checks.append({"name": "fps", "value": round(fps, 3), "expected": f"{gcfg.required_fps:g} (±0.5)",
                       "ok": abs(fps - gcfg.required_fps) <= 0.5})
    return checks


async def _effect_clip(src: Path, eff: dict, fps: int, run_dir: Path) -> Path:
    """Apply per-clip post effects (speed ramp + Ken Burns push-in) → a new file, or return
    src unchanged when the effect is a no-op. Done before assembly so the effected duration
    flows naturally into the transition/subtitle timing."""
    speed = float(eff.get("speed", 1.0) or 1.0)
    ken = bool(eff.get("ken_burns"))
    if abs(speed - 1.0) < 0.01 and not ken:
        return src

    dur = await _probe_duration(src)
    frames = max(1, int(round(dur * fps)))
    # Normalise fps FIRST so zoompan (d=1, one out per in frame) preserves the clip length —
    # otherwise a 30fps clip driven at a different fps changes duration.
    vchain = [f"fps={fps}", "format=yuv420p"]
    if ken:
        w, h = await _probe_wh(src)
        # smooth push-in 1.0 → 1.12 over the clip; d=1 = one output frame per input frame.
        vchain.append(f"zoompan=z='min(1.0+0.12*on/{frames},1.12)':d=1:"
                      f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}")
    if abs(speed - 1.0) >= 0.01:
        vchain.append(f"setpts=PTS/{speed}")
    vstr = ",".join(vchain) if vchain else "null"
    astr = f"atempo={min(2.0, max(0.5, speed))}" if abs(speed - 1.0) >= 0.01 else "anull"

    out = run_dir / f"fx_{src.stem}.mp4"
    fc = f"[0:v]{vstr}[v];[0:a]{astr}[a]"
    rc, err = await _run_ffmpeg(["ffmpeg", "-y", "-i", str(src), "-filter_complex", fc,
                                 "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                 "-c:a", "aac", str(out)])
    if rc != 0:
        logger.warning(f"clip effect failed, using original: {err[-200:]}")
        return src
    return out


def _build_transition_filter(durations: list[float], transitions: list[str], cfg) -> tuple[str, str, str]:
    """Compose all clips into one stream with per-boundary transitions.
    Returns (filter_complex, video_label, audio_label). Boundary i (1-based) uses
    transitions[i]: 'dissolve' → crossfade, 'fade' → fade-through-black,
    'j_cut'/'l_cut' → hard video cut with the audio bridged across it, else hard cut.

    Video is a left-fold (xfade for soft transitions, concat for hard ones). Audio is
    built on an absolute timeline — each clip's audio is delayed to its own start time and
    everything is amix'd — so a J/L cut just shifts one clip's audio earlier (it overlaps
    the cut) without ever drifting out of sync with the video.
    """
    fps = cfg.reencode_fps or 24
    n = len(durations)
    parts: list[str] = []
    for i in range(n):
        parts.append(f"[{i}:v]fps={fps},format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[{i}:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a{i}]")

    # ── video fold + per-clip start times ──
    cur_v, cum = "[v0]", durations[0]
    audio_start = [0.0] * n              # where each clip's audio sits on the timeline
    for i in range(1, n):
        t = transitions[i] if i < len(transitions) else "cut"
        nv = f"[v{i}o]"
        if t in ("dissolve", "fade"):
            d = max(0.1, min(cfg.scene_fade_seconds if t == "fade" else cfg.dissolve_seconds,
                             durations[i] - 0.05, cum - 0.05))
            xf = "fadeblack" if t == "fade" else "fade"
            off = max(0.0, cum - d)
            parts.append(f"{cur_v}[v{i}]xfade=transition={xf}:duration={d:.3f}:offset={off:.3f}{nv}")
            audio_start[i] = off
            cum = cum + durations[i] - d
        else:  # cut / match_cut / j_cut / l_cut → hard video concat
            parts.append(f"{cur_v}[v{i}]concat=n=2:v=1:a=0{nv}")
            start = cum
            if t in ("j_cut", "l_cut"):     # bridge the audio over the hard cut
                start = max(0.0, cum - cfg.jl_cut_seconds)
            audio_start[i] = start
            cum = cum + durations[i]
        cur_v = nv

    # ── audio timeline: delay each clip to its start, then mix ──
    a_labels = []
    for i in range(n):
        ms = int(round(audio_start[i] * 1000))
        lbl = f"[da{i}]"
        parts.append(f"[a{i}]adelay={ms}:all=1{lbl}")
        a_labels.append(lbl)
    if n == 1:
        cur_a = a_labels[0]
    else:
        cur_a = "[amixed]"
        parts.append(f"{''.join(a_labels)}amix=inputs={n}:normalize=0:dropout_transition=0{cur_a}")

    # NOTE: the bookend fade is applied as a SEPARATE pass (see step_merge), NOT here. Putting the
    # video fade-out AND audio afade-out in this same graph (alongside concat + amix) triggers an
    # ffmpeg quirk where amix EOFs early → audio goes silent after the first clip.
    return ";".join(parts), cur_v.strip("[]"), cur_a.strip("[]")


def _clip_spans(durations: list[float], transitions: list[str], cfg) -> list[tuple[float, float]]:
    """Each clip's [start, end) on the FINAL assembled timeline (soft transitions overlap,
    so the timeline is shorter than the raw sum). Used to time burned-in subtitles."""
    spans: list[tuple[float, float]] = []
    cum = 0.0
    for i, d in enumerate(durations):
        if i == 0:
            start = 0.0
        else:
            t = transitions[i] if i < len(transitions) else "cut"
            if t in ("dissolve", "fade"):
                ov = max(0.1, min(cfg.scene_fade_seconds if t == "fade" else cfg.dissolve_seconds, d - 0.05, cum - 0.05))
                start = cum - ov
            else:
                start = cum
        spans.append((start, start + d))
        cum = start + d
    return spans


async def _speech_timed_spans(paths: list[Path], spans: list[tuple[float, float]], captions: list[str]) -> list[tuple[float, float]]:
    """Narrow each clip's subtitle span to the window where speech ACTUALLY occurs in that clip
    (Silero VAD via revoice.detect_speech_window), so captions don't sit on-screen through silent
    lead-in/lead-out. Best-effort per clip — a clip with no caption, or where detection fails/finds
    nothing, keeps its original full-clip span (never worse than before this ran)."""
    from app.services.revoice import detect_speech_window

    out: list[tuple[float, float]] = []
    for p, (start, end), cap in zip(paths, spans, captions):
        if not (cap or "").strip():
            out.append((start, end))
            continue
        window = await asyncio.to_thread(detect_speech_window, p)
        if window is None:
            out.append((start, end))
            continue
        w_start, w_end = window
        out.append((start + w_start, min(end, start + w_end)))
    return out


def _ass_ts(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _write_ass(spans: list[tuple[float, float]], captions: list[str], path: Path, cfg) -> bool:
    """Write a libass (.ass) subtitle file with the style baked into the header — so the
    `subtitles` filter needs no force_style (which is a pain to escape in a filtergraph).
    Returns False if nothing to write. A Thai-capable font is set via subtitle_font."""
    # Fontname MUST equal the font's real family (libass matches by family) — detect it, never the stem.
    family = _font_meta(str(_subtitle_font_path(cfg)))[0] if cfg.subtitle_font else None
    font = family or (Path(cfg.subtitle_font).stem if cfg.subtitle_font else "Arial")
    size = cfg.subtitle_font_size
    # ScaleX/Y in % — bump font for a 720p canvas. Alignment 2 = bottom-centre.
    head = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,"
        "100,100,0,0,1,2,1,2,40,40,40,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for (start, end), text in zip(spans, captions):
        text = (text or "").strip().replace("\n", "\\N")
        if not text:
            continue
        lines.append(f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(max(start + 0.5, end - 0.05))},Default,,0,0,0,,{text}")
    if not lines:
        return False
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    return True


@router.post("/steps/merge")
async def step_merge(request: MergeRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Assemble the per-shot clips (in order) into one final.mp4 — with the right transition
    at each boundary (cut / crossfade / fade-through-black) plus a fade in/out, all driven by
    config.yaml video_gen. Plain all-cut sequences keep the fast stream-copy concat path."""
    topic = request.topic.strip()
    active_config.set_current_run(request.run_id or "")   # see /steps/image — the brand's LUT
    cfg = http_request.app.state.container.settings.video_gen
    await asyncio.to_thread(_resolve_ffmpeg, cfg)   # subprocess probe (ffmpeg -filters) off the event loop

    async def stream():
        tracker = UsageTracker()
        set_tracker(tracker)
        _t0 = time.perf_counter()
        # Assemble in a scratch dir: when S3 is on that's a TEMP dir wiped at the end (all the
        # downloaded clips + ffmpeg intermediates + final.mp4 live there and never touch outputs/);
        # when S3 is off it's the served media dir (final.mp4 is served from /api/media there).
        _s3 = _storage.is_configured()
        run_dir = Path(tempfile.mkdtemp(prefix="merge_")) if _s3 else _MEDIA_DIR / f"{_slug(topic)}_videos"
        run_dir.mkdir(parents=True, exist_ok=True)

        def _cleanup() -> None:
            if _s3:
                shutil.rmtree(run_dir, ignore_errors=True)

        # Build the per-clip lists in ONE loop so transitions/captions/effects stay aligned to the
        # SURVIVING clips — a missing file must not shift speed/zoom onto the wrong clip.
        paths, trans, caps, effs, skipped = [], [], [], [], []
        reqeff = request.effects or []
        for i, url in enumerate(request.video_urls):
            p = _safe_media_path(url)
            # S3/HTTP URL with no local copy → download to run_dir for ffmpeg
            if p is None and (url or "").startswith(("http://", "https://")):
                fname = url.rsplit("/", 1)[-1].split("?")[0] or f"dl_{i}.mp4"
                dest = run_dir / f"_dl_{fname}"
                if not dest.exists():
                    data = await asyncio.to_thread(_fetch_bytes, url)
                    if data:
                        dest.write_bytes(data)
                p = dest if dest.exists() else None
            if p is not None:
                paths.append(p)
                trans.append(request.transitions[i] if i < len(request.transitions) else "cut")
                caps.append(request.captions[i] if i < len(request.captions) else "")
                effs.append(reqeff[i] if i < len(reqeff) else {})
            else:
                skipped.append(url.rsplit("/", 1)[-1] or url)
        if not paths:
            _cleanup()
            yield sse({"type": "error", "message": "ไม่พบไฟล์วิดีโอที่จะรวม (generate ก่อน)."})
            return

        # Configured intro/end-credit clips wrap the episode: intro first, credit last. Inserted
        # into ALL FOUR lists to keep the per-clip alignment above intact (trans "cut" — index 0's
        # value is ignored anyway, and the credit joins hard; caption/effect empty so the subtitle
        # and speed passes leave them alone). A bumper that fails to load is skipped with a
        # warning, never a failed merge. Bumpers are normalized to the app standard at upload
        # (see upload_bumper), so raw mode's lossless copy-concat still has its chance.
        if request.include_bumpers:
            for kind, url in _bumper_urls(db).items():
                p = _safe_media_path(url)
                if p is None and url.startswith(("http://", "https://")):
                    dest = run_dir / f"_bumper_{kind}.mp4"
                    data = await asyncio.to_thread(_fetch_bytes, url)
                    if data:
                        dest.write_bytes(data)
                    p = dest if dest.exists() else None
                if p is None:
                    logger.warning(f"merge: could not load {kind} bumper from {url} — skipping it")
                    skipped.append(f"{kind} (bumper)")
                elif kind == "intro":
                    paths.insert(0, p); trans.insert(0, "cut"); caps.insert(0, ""); effs.insert(0, {})
                    yield sse({"type": "status", "message": "ต่อคลิป Intro นำหน้า"})
                else:
                    paths.append(p); trans.append("cut"); caps.append(""); effs.append({})
                    yield sse({"type": "status", "message": "ต่อคลิป End credit ปิดท้าย"})

        out = run_dir / "final.mp4"

        # Per-clip post effects (speed ramp / Ken Burns) — applied first so the effected
        # durations flow into the transition + subtitle timing below (effs aligned to paths).
        if not request.raw and any(abs(float((e or {}).get("speed", 1.0) or 1.0) - 1.0) >= 0.01 or (e or {}).get("ken_burns")
               for e in effs):
            yield sse({"type": "status", "message": "Applying clip effects (speed / zoom)..."})
            fps = cfg.reencode_fps or 24
            paths = [await _effect_clip(p, effs[i], fps, run_dir) for i, p in enumerate(paths)]

        # C2 stage 1 — COLOR MATCH: pull per-clip luma/saturation outliers toward the batch median
        # BEFORE assembly, so the look doesn't jump across cuts. Only deviating clips re-encode.
        pc = cfg.post_color
        if not request.raw and pc.enabled and pc.match_colors and len(paths) > 1:
            yield sse({"type": "status", "message": f"Matching colors across {len(paths)} clips..."})
            stats = [await _clip_color_stats(p) for p in paths]
            fixed, n_fix = [], 0
            for p, corr in zip(paths, _color_match_corrections(stats, pc)):
                if not corr:
                    fixed.append(p)
                    continue
                outp = run_dir / f"cm_{p.stem}.mp4"
                rc, err = await _run_ffmpeg(["ffmpeg", "-y", "-i", str(p), "-vf", corr,
                                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(outp)])
                if rc == 0:
                    fixed.append(outp)
                    n_fix += 1
                else:   # correction is an enhancement — never lose the clip over it
                    logger.warning(f"color match failed for {p.name} (using original): {err[-200:]}")
                    fixed.append(p)
            paths = fixed
            if n_fix:
                yield sse({"type": "progress", "message": f"ปรับสี {n_fix} คลิปเข้าหาค่ากลางของชุด (กันสีกระโดดข้ามคัต)"})

        # Soft transitions, J/L cuts, or a bookend fade need a re-encode filtergraph;
        # otherwise the cheap stream-copy concat is enough.
        has_soft = any(t in ("dissolve", "fade", "j_cut", "l_cut") for t in trans[1:])
        wants_fade = bool(cfg.bookend_fade_seconds and cfg.bookend_fade_seconds > 0)
        # Subtitles: only burn when PROVABLY safe (libass + Thai font) — else skip + warn (never tofu).
        subs_requested = bool((request.burn_subtitles or cfg.subtitles_in_video) and any((c or "").strip() for c in caps))
        subs_ok, subs_reason = _subs_ready(cfg) if subs_requested else (False, "")
        want_subs = subs_requested and subs_ok
        sub_burned = False
        sub_warning = subs_reason if (subs_requested and not subs_ok) else ""
        durations: list[float] = []
        built = False

        if not request.raw and cfg.transitions_enabled and (has_soft or wants_fade):
            yield sse({"type": "status", "message": f"Assembling {len(paths)} clips with transitions..."})
            durations = [await _probe_duration(p) for p in paths]
            if all(d > 0 for d in durations):
                fc, vlabel, alabel = _build_transition_filter(durations, trans, cfg)
                args = ["ffmpeg", "-y"]
                for p in paths:
                    args += ["-i", str(p)]
                args += ["-filter_complex", fc, "-map", f"[{vlabel}]", "-map", f"[{alabel}]",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                         "-movflags", "+faststart", str(out)]
                rc, err = await _run_ffmpeg(args)
                built = rc == 0
                if not built:
                    logger.warning(f"transition assembly failed, falling back to concat: {err[-300:]}")

        # RAW MODE — clips ต่อกันตามลำดับดิบๆ video stream-copy = เสียคุณภาพ 0% (ดู _copy_concat).
        # ถ้าคลิปพารามิเตอร์ไม่ตรงกัน (คนละ resolution/codec) ค่อยตกไป _normalized_concat ข้างล่าง
        if not built and request.raw:
            yield sse({"type": "status", "message": f"Merging {len(paths)} clips (lossless)..."})
            built, err = await _copy_concat(paths, out, run_dir)
            if not built:
                logger.warning(f"copy-concat failed (clips differ?), re-encoding instead: {err[-300:]}")
                yield sse({"type": "status", "message": "Clips differ — re-encoding to merge..."})
                built, err = await _normalized_concat(paths, out, fps=cfg.reencode_fps or 24)
                if not built:
                    logger.warning(f"raw concat-filter failed, falling back to copy-concat: {err[-300:]}")

        if not built:
            # Fast path / fallback — plain concat (see _copy_concat for why audio still re-encodes).
            yield sse({"type": "status", "message": f"Merging {len(paths)} clips with ffmpeg..."})
            built, err = await _copy_concat(paths, out, run_dir)
            if not built:
                yield sse({"type": "status", "message": "Re-encoding (clips differ)..."})
                rc, err = await _run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                                             "-i", str(run_dir / "concat.txt"),
                                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)])
                built = rc == 0
            if not built:
                yield sse({"type": "error", "message": f"ffmpeg failed: {err[-400:]}"})
                return

        # BOOKEND FADE — a SEPARATE pass over the assembled file (fade in at the head, out at the tail,
        # video + audio). Must NOT live in the assembly filtergraph: video-fade + audio-afade next to
        # concat/amix makes amix EOF early → audio silent after clip 1. On a plain file they're fine.
        bf = cfg.bookend_fade_seconds
        total = await _probe_duration(out)
        if not request.raw and bf and bf > 0 and total > 2 * bf:
            yield sse({"type": "status", "message": "Fade in/out (หัว-ท้าย)..."})
            faded = run_dir / "final_fade.mp4"
            rc, err = await _run_ffmpeg([
                "ffmpeg", "-y", "-i", str(out),
                "-vf", f"fade=t=in:st=0:d={bf},fade=t=out:st={total - bf:.3f}:d={bf}",
                "-af", f"afade=t=in:st=0:d={bf},afade=t=out:st={total - bf:.3f}:d={bf}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(faded)])
            if rc == 0:
                faded.replace(out)
            else:   # a fade is cosmetic — never fail the merge over it
                logger.warning(f"bookend fade pass failed (keeping unfaded): {err[-300:]}")

        # C2 stage 2 — GRADE: one re-encode over the assembled file (brand LUT → sat/contrast trim →
        # film grain), BEFORE the subtitle burn so captions stay clean on top of the graded picture.
        grade_applied = ""
        if not request.raw and pc.enabled:
            lut = _resolve_lut(pc)
            vf = _grade_filter(pc, lut)
            if vf:
                yield sse({"type": "status", "message": "Grading (LUT / grain)..."})
                graded = run_dir / "final_grade.mp4"
                rc, err = await _run_ffmpeg(["ffmpeg", "-y", "-i", str(out), "-vf", vf,
                                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
                                             "-movflags", "+faststart", str(graded)])
                if rc == 0:
                    graded.replace(out)
                    bits = ([f"LUT {lut.name}"] if lut else []) \
                        + (["eq trim"] if (abs(pc.saturation - 1.0) >= 0.01 or abs(pc.contrast - 1.0) >= 0.01) else []) \
                        + ([f"grain {pc.grain:g}"] if pc.grain and pc.grain > 0 else [])
                    grade_applied = " + ".join(bits)
                else:   # grading is an enhancement — keep the ungraded film rather than fail the merge
                    logger.warning(f"grade pass failed (keeping ungraded video): {err[-300:]}")
                    yield sse({"type": "progress", "message": "⚠️ Grade (LUT/grain) ล้มเหลว — ใช้วิดีโอแบบไม่ grade"})

        # Burn in subtitles (on_screen_text per clip) as a second pass over final.mp4.
        if not request.raw and want_subs:
            if not durations:
                durations = [await _probe_duration(p) for p in paths]
            spans = _clip_spans(durations, trans, cfg)
            if cfg.subtitle_sync_to_speech:
                yield sse({"type": "status", "message": "Syncing subtitle timing to speech..."})
                spans = await _speech_timed_spans(paths, spans, caps)
            ass = Path(tempfile.mkdtemp()) / "subs.ass"   # no-space path → clean filter syntax
            if _write_ass(spans, caps, ass, cfg):
                yield sse({"type": "status", "message": "Burning in subtitles..."})
                fp = _subtitle_font_path(cfg)
                fontdir = f":fontsdir={fp.parent}" if fp.exists() else ""
                vf = f"subtitles={ass}{fontdir}"
                subbed = run_dir / "final_sub.mp4"
                rc, err = await _run_ffmpeg(["ffmpeg", "-y", "-i", str(out), "-vf", vf,
                                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
                                            "-movflags", "+faststart", str(subbed)])
                if rc == 0:
                    subbed.replace(out)
                    sub_burned = True
                else:
                    logger.warning(f"subtitle burn-in failed (keeping clean video): {err[-300:]}")
                    sub_warning = "เบิร์นซับไม่สำเร็จ (ffmpeg error) — วิดีโอนี้ไม่มีซับ"

        # C3 — EPISODE GATE: measure the REAL delivered file (never trust estimates) and report
        # pass/fail per TOR check. Non-blocking: a failing episode still saves, but nobody finds
        # out at delivery time that it rendered 720p or ran past 6:30.
        gate = None
        gcfg = cfg.episode_gate
        if gcfg.enabled:
            dur = await _probe_duration(out)
            w, h = await _probe_wh(out)
            fps_real = await _probe_fps(out)
            checks = _gate_checks(gcfg, dur, w, h, fps_real)
            gate = {"ok": all(c["ok"] for c in checks), "duration": round(dur, 2),
                    "width": w, "height": h, "fps": round(fps_real, 3), "checks": checks}
            yield sse({"type": "episode_gate", **gate})
            if not gate["ok"]:
                fails = "; ".join(f"{c['name']}={c['value']} (ต้องการ {c['expected']})"
                                  for c in checks if not c["ok"])
                yield sse({"type": "progress", "message": f"⚠️ Episode gate ไม่ผ่าน: {fails}"})

        _final_key = f"{_slug(topic)}_videos/final.mp4"
        if _storage.is_configured():
            _final_data = await asyncio.to_thread(out.read_bytes)
            url = _storage.upload_bytes(_final_key, _final_data, "video/mp4")
        else:
            url = f"/api/media/{_final_key}"
        obs.attach_media(url, "video")
        yield sse({"type": "merged", "url": url, "count": len(paths),
                   "subtitles_burned": sub_burned, "subtitles_warning": sub_warning,
                   "skipped": skipped, "grade": grade_applied, "gate": gate})
        usage_summary = tracker.summary()
        _persist_usage(db, request.run_id, "merge", usage_summary)
        yield sse({"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage_summary})
        _cleanup()   # wipe the S3 scratch dir (no-op when serving locally)
        yield sse({"type": "done"})

    return StreamingResponse(obs.traced_stream(stream(), "step.merge", run_id=request.run_id), media_type="text/event-stream", headers=_SSE_HEADERS)


# ─── Director prompts ─────────────────────────────────────────────────────────

@router.post("/directors")
async def create_director(request: DirectorCreateRequest, http_request: Request, db: Session = Depends(get_db)) -> StreamingResponse:
    """Watch a reference video and extract a directorial style guide (streamed)."""
    url = (request.youtube_url or "").strip()
    container = http_request.app.state.container

    async def stream():
        if not url:
            yield sse({"type": "error", "message": "YouTube URL must not be empty."})
            return
        try:
            yield sse({"type": "status", "message": "Fetching video info..."})
            meta = await fetch_video_meta(url)
            src_title = meta.get("title", "")

            yield sse({"type": "status", "message": f"Watching reference video — {src_title[:50] or url}..."})
            guide = await container.gemini.generate_director_prompt(meta.get("url", url), src_title)
            if not guide:
                yield sse({"type": "error", "message": "Could not extract a style guide from this video."})
                return

            saved = DirectorRepo(db).save(
                title=(request.title or "").strip() or src_title,
                source_url=meta.get("url", url),
                source_title=src_title,
                summary=str(guide.get("summary", "")),
                sections={k: guide.get(k, "") for k in SECTION_KEYS},
            )
            db.commit()
            yield sse({"type": "director", "director": {
                "id": saved.id, "title": saved.title, "source_url": saved.source_url,
                "source_title": saved.source_title, "created_at": iso_utc(saved.created_at),
                "summary": saved.summary, "sections": saved.sections,
            }})
        except Exception as e:  # noqa: BLE001 — surface failure to the client
            db.rollback()
            logger.exception("Director creation failed")
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.get("/directors")
async def list_directors(db: Session = Depends(get_db)) -> list[dict]:
    return [
        {"id": d.id, "title": d.title, "source_url": d.source_url, "source_title": d.source_title,
         "created_at": iso_utc(d.created_at), "summary": d.summary}
        for d in DirectorRepo(db).list_recent()
    ]


@router.get("/directors/{director_id}")
async def get_director(director_id: str, db: Session = Depends(get_db)) -> dict:
    d = DirectorRepo(db).get(director_id)
    if not d:
        raise HTTPException(status_code=404, detail="Director prompt not found")
    return {"id": d.id, "title": d.title, "source_url": d.source_url, "source_title": d.source_title,
            "created_at": iso_utc(d.created_at), "summary": d.summary, "sections": d.sections}


@router.delete("/directors/{director_id}")
async def delete_director(director_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        deleted = DirectorRepo(db).delete(director_id)
        db.commit()
    except Exception:
        db.rollback()
        raise
    if not deleted:
        raise HTTPException(status_code=404, detail="Director prompt not found")
    return {"status": "deleted", "id": director_id}


# ─── Prompt templates (editable on the web; override store in prompts.py) ───────

@router.get("/prompts")
async def list_prompt_templates() -> list[dict]:
    """All editable prompt templates with active text, placeholders and override state."""
    return prompts.list_prompts()


@router.post("/prompts/{name}")
async def set_prompt_template(name: str, text: str = Body(..., embed=True)) -> dict:
    err = prompts.set_override(name, text)
    if err:
        raise HTTPException(status_code=422, detail=err)
    return {"name": name, "text": prompts.get_prompt(name), "is_overridden": True}


@router.delete("/prompts/{name}")
async def reset_prompt_template(name: str) -> dict:
    if name not in prompts.PROMPT_NAMES:
        raise HTTPException(status_code=404, detail=f"unknown prompt '{name}'")
    prompts.reset_override(name)
    return {"name": name, "text": prompts.get_prompt(name), "is_overridden": False}
