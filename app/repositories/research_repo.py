"""ResearchDocument was modeled in app/db/models.py from day one but never got a repo or route
(the v1→v2 port never exposed a research-history endpoint). Added now for the Drive migration —
research.json is the third Drive category and deserves the same versioned-JSON treatment as
Script/Storyboard rather than being dropped on the floor."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ResearchDocument
from app.repositories.retention import prune_old_versions


class ResearchRepo:
    def __init__(self, db: Session):
        self.db = db

    def save_new_version(self, run_id: str, *, payload: dict) -> ResearchDocument:
        current = self.get_current(run_id)
        if current is not None:
            current.is_current = False
        next_version = (current.version + 1) if current is not None else 1

        doc = ResearchDocument(run_id=run_id, version=next_version, is_current=True, payload=payload)
        self.db.add(doc)
        self.db.flush()
        prune_old_versions(self.db, ResearchDocument, run_id)
        return doc

    def get_current(self, run_id: str) -> ResearchDocument | None:
        stmt = select(ResearchDocument).where(ResearchDocument.run_id == run_id, ResearchDocument.is_current.is_(True))
        return self.db.scalars(stmt).first()

    def list_versions(self, run_id: str) -> list[ResearchDocument]:
        stmt = select(ResearchDocument).where(ResearchDocument.run_id == run_id).order_by(ResearchDocument.version.desc())
        return list(self.db.scalars(stmt))
