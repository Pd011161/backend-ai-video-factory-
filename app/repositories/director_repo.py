"""Replaces v1's app/services/director_store.py.

Append-only, same as v1: every `save()` is a new immutable row (no update-in-place),
`list_meta()`/`list_recent` is the history view, sorted newest first.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ids import slugify
from app.db.models import DirectorPrompt


class DirectorRepo:
    def __init__(self, db: Session):
        self.db = db

    def save(
        self,
        *,
        title: str,
        source_url: str = "",
        source_title: str = "",
        summary: str = "",
        sections: dict | None = None,
    ) -> DirectorPrompt:
        created_at = datetime.now(timezone.utc)
        prompt_id = f"{created_at:%Y%m%d_%H%M%S}_{slugify(title or source_title, fallback='director')}"
        prompt = DirectorPrompt(
            id=prompt_id,
            title=title,
            source_url=source_url,
            source_title=source_title,
            summary=summary,
            sections=sections or {},
            created_at=created_at,
        )
        self.db.add(prompt)
        self.db.flush()
        return prompt

    def get(self, prompt_id: str) -> DirectorPrompt | None:
        return self.db.get(DirectorPrompt, prompt_id)

    def list_recent(self, *, limit: int = 50) -> list[DirectorPrompt]:
        stmt = select(DirectorPrompt).order_by(DirectorPrompt.created_at.desc()).limit(limit)
        return list(self.db.scalars(stmt))

    def delete(self, prompt_id: str) -> bool:
        prompt = self.get(prompt_id)
        if prompt is None:
            return False
        self.db.delete(prompt)
        self.db.flush()
        return True
