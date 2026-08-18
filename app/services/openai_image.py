"""OpenAI image generation (gpt-image-*) — drop-in alternative to the Gemini
image path.

Same contract as ``GeminiClient.generate_image``: takes a prompt plus optional
labelled reference images (``[{"label", "mime", "data": bytes}]``) and returns
PNG/JPEG bytes, or None on any failure (best-effort — never breaks the pipeline).

With references we call the Images *edit* endpoint (gpt-image supports multiple
input images for composition); without references, the *generate* endpoint. The
``openai`` SDK is imported lazily so the Gemini path keeps working when it is not
installed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time

from loguru import logger

from app.core import observability as obs
from app.core.usage import record_usage


def _ref_instruction(refs: list[dict], prompt: str) -> str:
    """Fold the reference roles into the prompt (OpenAI edit has no per-image labels)."""
    if not refs:
        return prompt
    roles = "\n".join(f"image {i}: {r.get('label', 'reference')}" for i, r in enumerate(refs, 1))
    # When image 1 is the PREVIOUS frame this is a true EDIT: reproduce it and change only the delta.
    # "Current image" (/steps/image/edit's base) is the same situation and wants the same wrapper —
    # without it that route fell through to the generic branch, which explicitly allows re-framing.
    if str(refs[0].get("label", "")).startswith(("Previous frame", "Current image")):
        return (
            "Use the provided reference images by role:\n" + roles + "\n"
            "image 1 is the TEMPLATE: reproduce its camera angle, crop, framing, surface, background, "
            "bowls/props, lighting and EVERY object that is not being changed. Do NOT redesign the scene, "
            "do NOT rearrange or add objects, do NOT switch to a different layout. Apply ONLY the single "
            "change described below.\n"
            "The later images are ANTI-DRIFT ANCHORS, not new scenes — where image 1 disagrees with an "
            "anchor, THE ANCHOR WINS: a Character reference fixes the host's face, hair, skin tone and "
            "outfit; a Background reference fixes the kitchen's fixtures and materials; a 'Reference photo "
            "of <subject>' shows how the new or changed object AND its vessel should look.\n\n"
            + prompt
        )
    return (
        "Use the provided reference images by role:\n" + roles + "\n"
        "Keep the person and the background IDENTICAL to their references "
        "(same face, outfit, place); render any named subject objects — AND the vessel each one sits "
        "in (same bowl/cup/plate, unless the description moves it) — to match their reference photos. "
        "Only pose, action and framing change.\n\n" + prompt
    )


async def generate_image(
    prompt: str,
    references: list[dict] | None = None,
    *,
    api_key: str,
    model: str,
    size: str = "1024x1024",
    quality: str = "low",
    prebuilt: str | None = None,
) -> bytes | None:
    refs = references or []

    def _run() -> bytes | None:
        if not (api_key and (prebuilt if prebuilt is not None else prompt).strip()):
            return None
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai SDK not installed — cannot use the openai image provider")
            return None

        client = OpenAI(api_key=api_key)
        # prebuilt = the caller's already-assembled full prompt (user-edited "ครบ") → send verbatim
        full_prompt = prebuilt if prebuilt is not None else _ref_instruction(refs, prompt)
        with obs.generation(
            "openai.generate_image", model=model,
            # refs carry `_url`/`_kind` from resolve_shot_refs — log them (not just the label) so a trace
            # says WHICH stored image went into this render, same shape as the video path's `refs`.
            input={"prompt": full_prompt, "size": size,
                   "refs": [{"kind": r.get("_kind", ""), "label": r.get("label", ""),
                             "url": r.get("_url", "")} for r in refs]},
        ) as gen:
            _t0 = time.perf_counter()
            try:
                if refs:
                    files = []
                    for i, r in enumerate(refs):
                        bio = io.BytesIO(r["data"])
                        bio.name = f"ref{i}.jpg"  # SDK infers mime from the name
                        files.append(bio)
                    resp = client.images.edit(model=model, image=files, prompt=full_prompt, size=size, quality=quality)
                else:
                    resp = client.images.generate(model=model, prompt=full_prompt, size=size, quality=quality)
                duration_ms = (time.perf_counter() - _t0) * 1000
                b64 = resp.data[0].b64_json if resp.data else None
                data = base64.b64decode(b64) if b64 else None
                gen.update(output={"bytes": len(data) if data else 0})
                # gpt-image-1 returns resp.usage with input/output/total_tokens
                raw_usage = getattr(resp, "usage", None)
                usage_dict = None
                if raw_usage:
                    usage_dict = {
                        "input": getattr(raw_usage, "input_tokens", 0) or 0,
                        "output": getattr(raw_usage, "output_tokens", 0) or 0,
                        "total": getattr(raw_usage, "total_tokens", 0) or 0,
                    }
                record_usage("openai.generate_image", usage_dict, duration_ms)
                return data
            except Exception as e:  # noqa: BLE001 — best-effort
                logger.warning(f"OpenAI image generation failed: {e}")
                gen.update(level="ERROR", status_message=str(e))
                record_usage("openai.generate_image", None, (time.perf_counter() - _t0) * 1000)
                return None

    return await asyncio.to_thread(_run)


_MAX_EDIT_IMAGES = 16   # "For GPT image models, you can provide up to 16 images." — OpenAI images/edits reference


async def edit_image_region(
    prompt: str,
    image: bytes,
    mask: bytes | None = None,
    *,
    api_key: str,
    model: str,
    size: str = "1024x1024",
    quality: str = "low",
    refs: list[dict] | None = None,
) -> bytes | None:
    """Region edit (mask-based inpaint, or alpha-channel outpaint when `mask` is None and `image`
    already carries transparent padding) — gpt-image's edit endpoint regenerates only the
    TRANSPARENT (alpha=0) pixels of `mask` (or of `image` itself when no `mask` is given) and
    preserves every opaque pixel exactly.

    `refs` are extra images showing how the new content should look. They are sent AFTER `image`
    because the OpenAI guide states "If you provide multiple input images, the mask will be applied
    to the first image" — so the thing being edited has to stay at index 0. With refs present the
    prompt is wrapped by `_ref_instruction` so each image's role is named; without them the prompt
    goes through verbatim, exactly as before."""

    def _run() -> bytes | None:
        if not (api_key and prompt.strip()):
            return None
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai SDK not installed — cannot use the openai image provider")
            return None

        extra = list(refs or [])
        if 1 + len(extra) > _MAX_EDIT_IMAGES:
            dropped = 1 + len(extra) - _MAX_EDIT_IMAGES
            logger.warning(f"edit_image_region: {dropped} ref(s) dropped — the endpoint takes at most "
                           f"{_MAX_EDIT_IMAGES} images including the one being edited")
            extra = extra[: _MAX_EDIT_IMAGES - 1]

        client = OpenAI(api_key=api_key)
        with obs.generation(
            "openai.edit_image_region", model=model,
            input={"prompt": prompt, "size": size, "has_mask": bool(mask), "n_refs": len(extra)},
        ) as gen:
            _t0 = time.perf_counter()
            try:
                img_file = io.BytesIO(image)
                img_file.name = "image.png"   # PNG (not .jpg like generate_image's refs) — alpha must survive
                images: list = [img_file]
                for i, r in enumerate(extra):
                    bio = io.BytesIO(r["data"])
                    bio.name = f"ref{i}.jpg"   # SDK infers mime from the name
                    images.append(bio)
                full_prompt = prompt
                if extra:
                    base_role = {"label": "Current image — the one being edited; the mask applies to THIS image"}
                    full_prompt = _ref_instruction([base_role, *extra], prompt)
                kwargs: dict = {"model": model, "image": images if extra else img_file,
                                "prompt": full_prompt, "size": size, "quality": quality}
                if mask:
                    mask_file = io.BytesIO(mask)
                    mask_file.name = "mask.png"
                    kwargs["mask"] = mask_file
                resp = client.images.edit(**kwargs)
                duration_ms = (time.perf_counter() - _t0) * 1000
                b64 = resp.data[0].b64_json if resp.data else None
                data = base64.b64decode(b64) if b64 else None
                gen.update(output={"bytes": len(data) if data else 0})
                raw_usage = getattr(resp, "usage", None)
                usage_dict = None
                if raw_usage:
                    usage_dict = {
                        "input": getattr(raw_usage, "input_tokens", 0) or 0,
                        "output": getattr(raw_usage, "output_tokens", 0) or 0,
                        "total": getattr(raw_usage, "total_tokens", 0) or 0,
                    }
                record_usage("openai.edit_image_region", usage_dict, duration_ms)
                return data
            except Exception as e:  # noqa: BLE001 — best-effort
                logger.warning(f"OpenAI region edit failed: {e}")
                gen.update(level="ERROR", status_message=str(e))
                record_usage("openai.edit_image_region", None, (time.perf_counter() - _t0) * 1000)
                return None

    return await asyncio.to_thread(_run)
