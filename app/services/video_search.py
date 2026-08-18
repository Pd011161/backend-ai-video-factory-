import asyncio

import yt_dlp
from loguru import logger

from app.core.config import Settings


def _search_single(query: str, per_query: int) -> list[dict]:
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f"ytsearch{per_query}:{query}", download=False)
        entries = result.get("entries") or []

    videos = []
    for e in entries:
        if not e:
            continue
        vid_id = e.get("id")
        url = f"https://www.youtube.com/watch?v={vid_id}" if vid_id else e.get("url", "")
        if not url:
            continue
        videos.append({
            "id": vid_id,
            "url": url,
            "title": e.get("title", ""),
            "duration": e.get("duration") or 0,
        })

    return videos


def _fetch_meta_single(url: str) -> dict:
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id", ""),
        "title": info.get("title", ""),
        "duration": info.get("duration") or 0,
        "url": info.get("webpage_url") or url,
    }


async def fetch_video_meta(url: str) -> dict:
    """Resolve a single YouTube URL's metadata (title, canonical url). Best-effort."""
    try:
        return await asyncio.to_thread(_fetch_meta_single, url)
    except Exception as e:  # noqa: BLE001 — caller still has the raw url
        logger.warning(f"Failed to fetch video meta for {url!r}: {e}")
        return {"id": "", "title": "", "duration": 0, "url": url}


async def search_videos(queries: list[str], settings: Settings) -> list[dict]:
    n = settings.video.max_results
    max_duration = settings.video.max_duration_seconds
    per_query = n * 3

    logger.info(f"Searching YouTube with {len(queries)} queries, {per_query} results each")

    tasks = [asyncio.to_thread(_search_single, q, per_query) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_ids: set[str] = set()
    combined: list[dict] = []
    for i, batch in enumerate(results):
        if isinstance(batch, Exception):
            logger.warning(f"Query {i + 1} failed: {batch}")
            continue
        for v in batch:
            if not v["id"] or v["id"] in seen_ids:
                continue
            if max_duration and v["duration"] > max_duration:
                logger.debug(f"Skipped '{v['title'][:50]}' (duration={v['duration']}s > {max_duration}s)")
                continue
            seen_ids.add(v["id"])
            combined.append(v)

    logger.info(f"Combined pool: {len(combined)} unique videos across {len(queries)} queries")
    return combined
