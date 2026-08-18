"""Version retention for the per-run document tables (research/script/storyboard).

Every save — the generate steps and the manual save buttons alike — goes through
`save_new_version`, and each call inserts a whole new document tree. Nothing ever deleted, so a
long-lived run accumulated versions without bound. Retention is enforced HERE, at the repo layer,
precisely so both write paths get it; pruning only in the save endpoints would have left the
generate path growing forever.

Policy (operator's call): keep the newest 3 versions per run, delete the rest. Deleting a
StoryboardDocument cascades scenes → shots → image plans → refs (all delete-orphan), but never
touches `media_assets` — shots only point at assets by id, and the surviving current version holds
the same links via carry-forward, so no generated image or clip is lost.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

KEEP_VERSIONS = 3


def prune_old_versions(db: Session, model, run_id: str, *, keep: int = KEEP_VERSIONS, **scope) -> int:
    """Delete `model` rows for `run_id` beyond the newest `keep` versions. Returns rows deleted.

    `scope` narrows what "the newest `keep`" is counted over — storyboards pass
    `script_version=n` so each script version keeps its own 3. Counting across the whole run
    instead would let one script's boards evict another's entirely, which is the opposite of the
    point of grouping them.

    ORM-level delete (not a bulk DELETE statement) so the relationship cascades run — a bulk
    statement would skip them and strand child rows on SQLite, which has FK enforcement off."""
    stmt = select(model).where(model.run_id == run_id)
    for column, value in scope.items():
        stmt = stmt.where(getattr(model, column) == value)
    stmt = stmt.order_by(model.version.desc()).offset(keep)
    stale = list(db.scalars(stmt))
    for doc in stale:
        db.delete(doc)
    if stale:
        db.flush()
    return len(stale)
