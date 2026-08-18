"""add full_prompt to shot_image_plans

The exact assembled prompt of a shot's last render was computed every time (nodes.py
assemble_full_prompt) but never stored, so the UI's "ครบ" editor was empty after any reload and
each look cost a paid dry-run.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-08-01
"""
from alembic import op
import sqlalchemy as sa

revision = 'a7b8c9d0e1f2'
down_revision = 'f6a7b8c9d0e1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch_alter_table: SQLite is still a supported backend (config.yaml `database.backend`)
    with op.batch_alter_table('shot_image_plans', schema=None) as batch_op:
        batch_op.add_column(sa.Column('full_prompt', sa.String(), nullable=False, server_default=''))


def downgrade() -> None:
    with op.batch_alter_table('shot_image_plans', schema=None) as batch_op:
        batch_op.drop_column('full_prompt')
