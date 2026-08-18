"""Copy an existing SQLite database into PostgreSQL.

    uv run python scripts/sqlite_to_postgres.py \
        --source sqlite:///./app.db \
        --target 'postgresql+psycopg://user:pass@localhost:5432/videofactory'

Run `alembic upgrade head` against the TARGET first — this only moves rows, it never creates
tables. Reads table order from SQLAlchemy metadata, which is already sorted so parents come before
children, so foreign keys are satisfied without hand-maintaining a list.

The sequence reset at the end is not optional. Rows are copied with their original integer ids, but
PostgreSQL's SERIAL sequences stay at 1 — so the first row the app inserts afterwards collides with
an existing id, and keeps colliding until the sequence catches up. That surfaces hours later as an
intermittent 500 with no obvious link to the migration.
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import create_engine, insert, inspect, select, text

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from app.db.base import Base  # noqa: E402
from app.db import models  # noqa: E402,F401  (registers every table on Base.metadata)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sqlite:///./app.db")
    ap.add_argument("--target", required=True)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    if not args.target.startswith("postgresql"):
        print(f"target ต้องเป็น postgresql://… (ได้ {args.target!r})")
        return 2

    src = create_engine(args.source)
    dst = create_engine(args.target)

    dst_tables = set(inspect(dst).get_table_names())
    missing = [t.name for t in Base.metadata.sorted_tables if t.name not in dst_tables]
    if missing:
        print(f"ปลายทางยังไม่มีตาราง: {missing}\nรัน `alembic upgrade head` ที่ target ก่อน")
        return 2

    copied: dict[str, int] = {}
    with src.connect() as s, dst.begin() as d:
        for table in Base.metadata.sorted_tables:            # parents first — FK-safe
            rows = [dict(r) for r in s.execute(select(table)).mappings()]
            copied[table.name] = len(rows)
            for i in range(0, len(rows), args.batch):
                d.execute(insert(table), rows[i:i + args.batch])
            print(f"  {table.name:<24} {len(rows):>6}")

        # Integer PKs were copied verbatim; move each sequence past the highest id in use.
        for table in Base.metadata.sorted_tables:
            pk = list(table.primary_key.columns)
            if len(pk) != 1 or not isinstance(pk[0].type.python_type, type) or pk[0].type.python_type is not int:
                continue
            col = pk[0].name
            d.execute(text(
                f"SELECT setval(pg_get_serial_sequence('{table.name}', '{col}'), "
                f"COALESCE((SELECT MAX({col}) FROM {table.name}), 1))"
            ))
            print(f"  setval {table.name}.{col}")

    # Independent read-back, not a count of what we just sent.
    print("\nตรวจสอบ:")
    bad = 0
    with src.connect() as s, dst.connect() as d:
        for table in Base.metadata.sorted_tables:
            a = len(s.execute(select(table)).fetchall())
            b = len(d.execute(select(table)).fetchall())
            flag = "" if a == b else "  ← ไม่ตรง!"
            if a != b:
                bad += 1
            print(f"  {table.name:<24} sqlite={a:<6} pg={b:<6}{flag}")
    print("\n" + ("✅ ครบทุกตาราง" if not bad else f"❌ {bad} ตารางไม่ตรง"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
