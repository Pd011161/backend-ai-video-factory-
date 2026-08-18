"""Generic atomic-file-write helper (ported unchanged from v1's app/services/store_io.py).

Only `atomic_write_text` is still used in v2 (by app/services/prompts.py, for the local
prompt-overrides JSON file) — the brand/menu/director JSON-store functions this module used to
back have been replaced by the SQL repositories in app/repositories/.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def safe_id(sid: str, fallback: str) -> str:
    sid = (sid or "").strip()
    if not sid or "/" in sid or "\\" in sid or ".." in sid:
        return fallback
    return sid


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text to `path` atomically (temp file in the same directory + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # atomic on the same filesystem
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
