"""Replaces v1's active_config.py (the {"jobs": {id: {brand_id, menu_id}}} pointer file).

A run is now a real row: creating one replaces `active_config.set(job_id, ...)`, and its
brand_id/menu_id/director_prompt_id columns replace the old per-job pointer dict.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Run
from app.repositories.errors import NotFoundError


class RunRepo:
    def __init__(self, db: Session):
        self.db = db

    def create(self, *, id: str, topic: str, title: str = "") -> Run:
        run = Run(id=id, topic=topic, title=title)
        self.db.add(run)
        self.db.flush()
        return run

    def get(self, run_id: str) -> Run | None:
        return self.db.get(Run, run_id)

    def require(self, run_id: str) -> Run:
        run = self.get(run_id)
        if run is None:
            raise NotFoundError("Run", run_id)
        return run

    def list(self, *, limit: int = 100, offset: int = 0) -> list[Run]:
        stmt = select(Run).order_by(Run.updated_at.desc()).limit(limit).offset(offset)
        return list(self.db.scalars(stmt))

    def set_status(self, run_id: str, status: str) -> Run:
        run = self.require(run_id)
        run.status = status
        run.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        return run

    def set_active_pointers(
        self,
        run_id: str,
        *,
        brand_id: str | None = ...,
        menu_id: str | None = ...,
        director_prompt_id: str | None = ...,
        character_id: str | None = ...,
    ) -> Run:
        """Set the active brand/menu/director for a run. Pass only the fields you want to change —
        omitted kwargs (default `...`) leave the existing value untouched, matching v1's
        active_config.set(job_id, brand_id=..., menu_id=...) partial-update behavior."""
        run = self.require(run_id)
        if brand_id is not ...:
            run.brand_id = brand_id
        if menu_id is not ...:
            run.menu_id = menu_id
        if director_prompt_id is not ...:
            run.director_prompt_id = director_prompt_id
        if character_id is not ...:
            run.character_id = character_id
        run.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        return run

    def update(
        self,
        run_id: str,
        *,
        topic: str = ...,
        title: str = ...,
        type: str = ...,      # noqa: A002 — mirrors the column name
        avatar: str = ...,
    ) -> Run:
        """Edit a run's own descriptive fields. Same partial-update contract as
        `set_active_pointers`: omitted kwargs (default `...`) are left alone, so a caller that only
        renames a topic cannot blank the rest by accident.

        `id` is intentionally not editable — six tables carry it as a foreign key and menus are
        keyed on it (`menu.id == run.id`), so changing it would mean rewriting all of them."""
        run = self.require(run_id)
        if topic is not ...:
            run.topic = topic
        if title is not ...:
            run.title = title
        if type is not ...:
            run.type = type
        if avatar is not ...:
            run.avatar = avatar
        run.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        return run

    # Everything that hangs off a run, in the order it must be removed. Nothing here declares
    # `ondelete` (see the note on Run.brand_id), so the database will not clear these for us —
    # and the two backends fail differently: PostgreSQL refuses the delete outright, while SQLite
    # runs without `PRAGMA foreign_keys=ON` and quietly leaves the children orphaned, which looks
    # like success. Deleting them here explicitly behaves identically on both.
    def _child_models(self):
        from app.db.models import (MediaAsset, RegenEvent, ResearchDocument, ScriptDocument,
                                    StoryboardDocument, UsageRecord)
        return [RegenEvent, UsageRecord, MediaAsset, StoryboardDocument, ScriptDocument,
                ResearchDocument]

    def count_children(self, run_id: str) -> dict[str, int]:
        """How many rows a delete would take with it, per table — so the operator can be shown
        what they are about to lose instead of a bare "are you sure?"."""
        out: dict[str, int] = {}
        for model in self._child_models():
            n = len(list(self.db.scalars(select(model).where(model.run_id == run_id))))
            if n:
                out[model.__tablename__] = n
        return out

    def delete(self, run_id: str) -> bool:
        """Delete a run and everything under it. ORM-level deletes (not bulk) so the
        relationship cascades already declared on the documents — scene → shot → image plan —
        still run."""
        run = self.get(run_id)
        if run is None:
            return False
        for model in self._child_models():
            for row in self.db.scalars(select(model).where(model.run_id == run_id)):
                self.db.delete(row)
            self.db.flush()
        self.db.delete(run)
        self.db.flush()
        return True
