"""Persists editable $/unit cost-estimate rates (see app/services/cost_estimate.py). Global table,
seeded from DEFAULT_RATES the first time it's read empty; DB values win over the snapshot after
that — editing a rate here is the whole point of not hardcoding these as Python constants."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CostRate
from app.services.cost_estimate import DEFAULT_RATES


class CostRateRepo:
    def __init__(self, db: Session):
        self.db = db

    def ensure_seeded(self) -> None:
        if self.db.scalars(select(CostRate.key)).first() is not None:
            return
        for key, (label, unit, value) in DEFAULT_RATES.items():
            self.db.add(CostRate(key=key, label=label, unit=unit, value=value))
        self.db.flush()

    def list_all(self) -> list[CostRate]:
        self.ensure_seeded()
        stmt = select(CostRate).order_by(CostRate.key)
        return list(self.db.scalars(stmt))

    def as_dict(self) -> dict[str, float]:
        return {r.key: r.value for r in self.list_all()}

    def update(self, key: str, value: float) -> CostRate:
        self.ensure_seeded()
        rate = self.db.get(CostRate, key)
        if rate is None:
            raise KeyError(key)
        rate.value = value
        self.db.flush()
        return rate
