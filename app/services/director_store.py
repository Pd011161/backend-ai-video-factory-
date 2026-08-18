"""Replaces v1's app/services/director_store.py (file-backed director prompt store).

Ported graph nodes only call `get(id)`; save/list/delete are used by routes, which should call
`DirectorRepo` directly with the request's session instead of this shim.
"""
from __future__ import annotations

from app.db.base import SessionLocal
from app.db.models import iso_utc
from app.models.director import DirectorPrompt as DirectorPromptModel
from app.repositories import DirectorRepo


def get(id: str) -> DirectorPromptModel | None:
    if not id:
        return None
    db = SessionLocal()
    try:
        row = DirectorRepo(db).get(id)
        if row is None:
            return None
        return DirectorPromptModel(
            id=row.id,
            title=row.title,
            source_url=row.source_url or "",
            source_title=row.source_title or "",
            created_at=iso_utc(row.created_at),
            summary=row.summary or "",
            sections=row.sections or {},
        )
    finally:
        db.close()
