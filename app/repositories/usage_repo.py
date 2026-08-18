"""Persists per-step Gemini/OpenAI/Veo usage against the `usage_records` table.

New in v2: v1 only emitted usage over the pipeline SSE stream (app/core/usage.py, unchanged) and
optionally wrote a Drive xlsx report; v2 additionally persists one row per step call so usage can be
queried per run without Drive.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import UsageRecord


class UsageRepo:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        *,
        run_id: str,
        step: str,
        agent: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        duration_ms: int = 0,
    ) -> UsageRecord:
        record = UsageRecord(
            run_id=run_id,
            step=step,
            agent=agent,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        )
        self.db.add(record)
        self.db.flush()
        return record

    def list_for_run(self, run_id: str) -> list[UsageRecord]:
        stmt = select(UsageRecord).where(UsageRecord.run_id == run_id).order_by(UsageRecord.created_at)
        return list(self.db.scalars(stmt))
