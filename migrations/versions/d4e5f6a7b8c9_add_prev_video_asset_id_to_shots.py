"""add prev_video_asset_id to storyboard_shots (one-step clip undo)

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-07-30 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd4e5f6a7b8c9'
down_revision = 'c3d4e5f6a7b8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # No FK constraint: SQLite can't ADD one without a table rebuild, and the column is only ever a
    # pointer at a media_assets row this app itself wrote.
    op.add_column('storyboard_shots', sa.Column('prev_video_asset_id', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('storyboard_shots', 'prev_video_asset_id')
