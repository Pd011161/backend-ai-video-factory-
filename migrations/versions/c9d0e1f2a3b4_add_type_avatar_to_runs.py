"""add type + avatar to runs

Two editable free-text fields on a run, blank until something needs them: `type` for classifying
a run (recipe / review / …) and `avatar` for whichever presenter it belongs to. Nothing in the
pipeline reads either yet, so they carry no constraint or FK — the point is to have the column in
place before the meaning is settled, so filling it in later costs no migration.

`brand` is deliberately NOT added: runs already have `brand_id`, a real FK to `brands`.

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-08-03

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c9d0e1f2a3b4'
down_revision = 'b8c9d0e1f2a3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch_alter_table: SQLite is still a supported backend (config.yaml `database.backend`, and
    # it is the DEFAULT). server_default='' so the existing rows fill in without a rewrite.
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('type', sa.String(), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('avatar', sa.String(), nullable=False, server_default=''))


def downgrade() -> None:
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.drop_column('avatar')
        batch_op.drop_column('type')
