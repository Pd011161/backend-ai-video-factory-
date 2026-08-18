"""Replaces v1's app/services/active_config.py (a JSON pointer file `{jobs: {id: {brand_id, menu_id}}}`).

v2 promotes "the active job" to a real `Run` row — brand_id/menu_id/director_prompt_id now live as
columns on `RUNS`. Pipeline graph nodes (app/graph/nodes.py) were ported unchanged from v1 and call
`active_config.get()` / `.set()` with no run context of their own, so this shim keeps a per-request
"current run id" in a contextvar (set by the route before invoking the graph) and resolves it against
`RunRepo` with a short-lived session. This is a pragmatic compatibility layer, not a persistence
pattern to copy elsewhere — new code should call `RunRepo` directly with the request's `db` session.
"""
from __future__ import annotations

from contextvars import ContextVar

from app.db.base import SessionLocal
from app.repositories import RunRepo

_current_run_id: ContextVar[str] = ContextVar("_active_config_current_run_id", default="")

_FIELD_MAP = {"brand_id": "brand_id", "menu_id": "menu_id", "director_id": "director_prompt_id",
              "character_id": "character_id"}


def set_current_run(run_id: str) -> None:
    """Call at the top of each pipeline route/step before invoking graph nodes."""
    _current_run_id.set(run_id or "")


def get_current_run() -> str:
    return _current_run_id.get()


def get() -> dict:
    run_id = _current_run_id.get()
    if not run_id:
        return {}
    db = SessionLocal()
    try:
        run = RunRepo(db).get(run_id)
        if run is None:
            return {}
        return {
            "brand_id": run.brand_id or "",
            "menu_id": run.menu_id or "",
            "director_id": run.director_prompt_id or "",
            "character_id": run.character_id or "",
        }
    finally:
        db.close()


def set(**kwargs) -> None:
    run_id = _current_run_id.get()
    if not run_id:
        return
    fields = {_FIELD_MAP[k]: v for k, v in kwargs.items() if k in _FIELD_MAP}
    if not fields:
        return
    db = SessionLocal()
    try:
        RunRepo(db).set_active_pointers(run_id, **fields)
        db.commit()
    finally:
        db.close()
