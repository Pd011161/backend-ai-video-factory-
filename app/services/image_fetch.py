"""Download + downscale candidate images so Gemini can actually look at them.

Vertex AI's ``Part.from_uri`` only fetches GCS/YouTube URIs — arbitrary web image
URLs (from Tavily) must be downloaded as bytes and passed inline via
``Part.from_data``. To keep token cost and bandwidth low we resize every image so
its longest edge is at most ``resize_px`` and re-encode to JPEG before sending.

Everything is best-effort: a failed/oversized/unsupported download returns None
and the candidate is simply dropped — image search must never break the pipeline.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import httpx
from loguru import logger

from app.core import observability as obs
from app.core.config import ROOT_DIR, ImageSearchConfig

# Gemini accepts JPEG/PNG/WEBP/HEIC/HEIF. We re-encode everything to JPEG, so the
# only constraint is that Pillow can open it (SVG cannot be — those are skipped by
# the content-type / decode guard below).
_OK_PREFIX = "image/"


def _resize_to_jpeg(raw: bytes, max_px: int) -> bytes | None:
    """Decode, downscale longest edge to ``max_px``, re-encode JPEG. None on failure."""
    try:
        from PIL import Image
    except ImportError:  # Pillow not installed — fall back to raw bytes
        logger.warning("Pillow not installed; sending image at original resolution")
        return raw

    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")  # flatten alpha / palette / CMYK → JPEG-safe
        longest = max(img.width, img.height)
        if longest > max_px:
            scale = max_px / longest
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.debug(f"Image decode/resize failed: {e}")
        return None


async def fetch_image(
    url: str,
    *,
    client: httpx.AsyncClient,
    cfg: ImageSearchConfig,
    max_px: int | None = None,
) -> dict | None:
    """Download one image URL and return {"mime", "data"} of a downscaled JPEG, or None.

    ``max_px`` overrides the longest-edge resize (defaults to ``cfg.resize_px``).
    """
    if not url:
        return None
    try:
        resp = await client.get(url, timeout=cfg.download_timeout, follow_redirects=True)
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype and not ctype.startswith(_OK_PREFIX):
            logger.debug(f"Skip non-image content-type {ctype!r}: {url[:80]}")
            return None
        raw = resp.content
        if len(raw) > cfg.max_image_bytes:
            logger.debug(f"Skip oversized image ({len(raw)} bytes): {url[:80]}")
            return None
        data = _resize_to_jpeg(raw, max_px or cfg.resize_px)
        if not data:
            return None
        return {"mime": "image/jpeg", "data": data}
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.debug(f"Image download failed for {url[:80]!r}: {e}")
        return None


async def load_reference(src: str, *, cfg: ImageSearchConfig, max_px: int | None = None) -> dict | None:
    """Load a reference image from a local path (under project root) or a URL.

    Returns {"mime", "data"} of a downscaled JPEG, or None on any failure.
    ``max_px`` overrides the longest-edge resize (defaults to ``cfg.resize_px``).
    """
    src = (src or "").strip()
    if not src:
        return None
    if src.startswith(("http://", "https://")):
        async with httpx.AsyncClient() as client:
            return await fetch_image(src, client=client, cfg=cfg, max_px=max_px)
    # local file (relative paths resolve under the project root)
    path = Path(src)
    if not path.is_absolute():
        path = ROOT_DIR / src
    if not path.exists():
        logger.warning(f"Reference image not found: {path}")
        return None
    data = _resize_to_jpeg(path.read_bytes(), max_px or cfg.resize_px)
    return {"mime": "image/jpeg", "data": data} if data else None


async def fetch_many(
    urls: list[str],
    *,
    api_key: str,  # reserved for providers needing auth; Tavily image URLs are public
    cfg: ImageSearchConfig,
) -> dict[str, dict | None]:
    """Download a de-duplicated set of URLs concurrently. Returns {url: {mime,data} | None}."""
    unique = [u for u in dict.fromkeys(urls) if u]
    if not unique:
        return {}
    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    out: dict[str, dict | None] = {}

    with obs.span("image.download", input={"urls": len(unique)}) as span:
        async with httpx.AsyncClient() as client:
            async def _one(u: str) -> None:
                async with sem:
                    out[u] = await fetch_image(u, client=client, cfg=cfg)

            await asyncio.gather(*(_one(u) for u in unique))
        span.update(output={"ok": sum(1 for v in out.values() if v)})
    return out
