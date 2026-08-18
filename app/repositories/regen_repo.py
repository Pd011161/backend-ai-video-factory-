"""Persists user-initiated "this AI result wasn't good enough" events against `regen_events`.

Distinct from UsageRepo (usage_records): usage counts every provider CALL including the
successful first try; this counts only regenerate/edit/upload-replace actions the user chose to
take on an ALREADY-produced result — a durable signal of which steps need the most quality work.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import RegenEvent


class RegenRepo:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        *,
        run_id: str,
        step: str,
        action: str,
        scene_id: str = "",
        shot_no: int | None = None,
        note: str = "",
    ) -> RegenEvent:
        event = RegenEvent(run_id=run_id, step=step, action=action, scene_id=scene_id, shot_no=shot_no, note=note)
        self.db.add(event)
        self.db.flush()
        return event

    def list_for_run(self, run_id: str) -> list[RegenEvent]:
        stmt = select(RegenEvent).where(RegenEvent.run_id == run_id).order_by(RegenEvent.created_at)
        return list(self.db.scalars(stmt))
