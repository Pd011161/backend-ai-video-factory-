"""add name + description to config_refs

The character's name and description used to live in Step 2's Script Settings form — per-run React
state, re-typed for every run, even though the person is the same one whose photo is already
configured globally. These two columns let the text sit with the picture, so it is set once.

Named generically (not `character_name`) because the `scene` row shares this table and leaves them
blank.

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-08-04

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd0e1f2a3b4c5'
down_revision = 'c9d0e1f2a3b4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch_alter_table: SQLite is still a supported backend and is the config default.
    # server_default='' so existing rows fill in without a rewrite.
    with op.batch_alter_table('config_refs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('name', sa.String(), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('description', sa.String(), nullable=False, server_default=''))


def downgrade() -> None:
    with op.batch_alter_table('config_refs', schema=None) as batch_op:
        batch_op.drop_column('description')
        batch_op.drop_column('name')
