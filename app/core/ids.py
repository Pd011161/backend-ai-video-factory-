"""Id helpers shared by every repository — ported from v1's store_io.py / scene_store.py.

Ids here end up as SQL primary keys AND as path components in S3 keys
(`refs/scenes/{brand_id}/{scene_id}.png`), so the same path-traversal guard from v1 still applies.
"""
from __future__ import annotations

import re
import unicodedata

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, fallback: str = "item") -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug or fallback


def safe_id(sid: str, fallback: str) -> str:
    """A store id must be a single path component — never a separator or traversal sequence."""
    sid = (sid or "").strip()
    if not sid or "/" in sid or "\\" in sid or ".." in sid:
        return fallback
    return sid


def dedupe_id(candidate: str, exists: "callable[[str], bool]") -> str:
    """Append -2, -3, ... until `exists(id)` is False, matching v1's brand/menu id dedup behavior."""
    if not exists(candidate):
        return candidate
    n = 2
    while exists(f"{candidate}-{n}"):
        n += 1
    return f"{candidate}-{n}"
