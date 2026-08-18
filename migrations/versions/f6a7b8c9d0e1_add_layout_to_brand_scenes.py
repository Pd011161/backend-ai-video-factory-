"""add layout to brand_scenes

`desc` answers WHEN to use a scene (the classifier's selection criteria). `layout` answers HOW it
is composed — camera angle, what is in the fore/background, where the host belongs. They are
separate columns because merging geometry into the selection text made the classifier pick the
wrong scene. Nullable/empty default so existing scenes keep working untouched.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-07-31

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f6a7b8c9d0e1'
down_revision = 'e5f6a7b8c9d0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch_alter_table: SQLite is still a supported backend (config.yaml `database.backend`)
    with op.batch_alter_table('brand_scenes', schema=None) as batch_op:
        batch_op.add_column(sa.Column('layout', sa.String(), nullable=False, server_default=''))


def downgrade() -> None:
    with op.batch_alter_table('brand_scenes', schema=None) as batch_op:
        batch_op.drop_column('layout')
