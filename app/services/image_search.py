"""Tavily image search.

Returns up to `limit` candidate images for a query — each a {"url", "description"}
dict (descriptions feed candidate validation). Returns [] on any failure: image
search is best-effort enrichment and must never break the pipeline.
"""

from __future__ import annotations

import httpx
from loguru import logger

from app.core import observability as obs
from app.core.config import ImageSearchConfig

_ENDPOINT = "https://api.tavily.com/search"


async def search_images(
    query: str,
    *,
    api_key: str,
    cfg: ImageSearchConfig,
    limit: int,
) -> list[dict]:
    if not (api_key and query.strip()):
        return []

    payload = {
        "query": query,
        "include_images": True,
        "include_image_descriptions": True,
        "search_depth": cfg.search_depth,
        "max_results": 1,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    with obs.generation("tavily.image_search", input={"query": query}) as gen:
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                resp = await client.post(_ENDPOINT, json=payload, headers=headers)
                resp.raise_for_status()
                images = resp.json().get("images") or []
                out: list[dict] = []
                for it in images:
                    if isinstance(it, dict):
                        url, desc = it.get("url"), it.get("description", "")
                    else:
                        url, desc = it, ""
                    if url:
                        out.append({"url": url, "description": desc})
                out = out[: max(1, limit)]
                gen.update(output={"found": len(out)})
                return out
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning(f"Image search failed for {query[:60]!r}: {e}")
            gen.update(level="ERROR", status_message=str(e))
            return []
