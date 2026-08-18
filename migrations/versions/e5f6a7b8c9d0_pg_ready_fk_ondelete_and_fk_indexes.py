"""PostgreSQL-ready: ON DELETE SET NULL on run refs + indexes on FK/sort columns

SQLite never enforced foreign keys (no `PRAGMA foreign_keys=ON` anywhere), so deleting a brand,
menu or director silently left dangling ids on `runs` and the DELETE routes appeared to work.
PostgreSQL does enforce them — without a rule those three endpoints would start returning 500.

Also adds the indexes PostgreSQL does not create for foreign keys automatically. On SQLite with a
few thousand rows nobody noticed; on a server these are the columns every per-run lookup filters by.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-07-30 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'e5f6a7b8c9d0'
down_revision = 'd4e5f6a7b8c9'
branch_labels = None
depends_on = None

# (constraint, table, column, referred table)
_RUN_FKS = [
    ('fk_runs_brand_id_brands', 'runs', 'brand_id', 'brands'),
    ('fk_runs_menu_id_menus', 'runs', 'menu_id', 'menus'),
    ('fk_runs_director_prompt_id_director_prompts', 'runs', 'director_prompt_id', 'director_prompts'),
]

_INDEXES = [
    ('ix_media_assets_run_id', 'media_assets', ['run_id']),
    ('ix_usage_records_run_id', 'usage_records', ['run_id']),
    ('ix_regen_events_run_id', 'regen_events', ['run_id']),
    ('ix_runs_updated_at', 'runs', ['updated_at']),   # RunRepo.list() orders by this on every page load
]


def upgrade() -> None:
    bind = op.get_bind()
    # SQLite can't ALTER a constraint — batch mode rebuilds the table. It also never enforced these
    # constraints, so the rebuild is the only way the new rule lands there at all.
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table('runs') as batch:
            for name, _t, col, ref in _RUN_FKS:
                batch.drop_constraint(name, type_='foreignkey')
                batch.create_foreign_key(name, ref, [col], ['id'], ondelete='SET NULL')
    else:
        for name, table, col, ref in _RUN_FKS:
            op.drop_constraint(name, table, type_='foreignkey')
            op.create_foreign_key(name, table, ref, [col], ['id'], ondelete='SET NULL')

    for name, table, cols in _INDEXES:
        op.create_index(name, table, cols)


def downgrade() -> None:
    for name, table, _cols in _INDEXES:
        op.drop_index(name, table_name=table)

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table('runs') as batch:
            for name, _t, col, ref in _RUN_FKS:
                batch.drop_constraint(name, type_='foreignkey')
                batch.create_foreign_key(name, ref, [col], ['id'])
    else:
        for name, table, col, ref in _RUN_FKS:
            op.drop_constraint(name, table, type_='foreignkey')
            op.create_foreign_key(name, table, ref, [col], ['id'])
