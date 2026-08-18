"""add last_frame_s3_url to media_assets

Revision ID: c3d4e5f6a7b8
Revises: b1c2d3e4f5a6
Create Date: 2026-07-27 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3d4e5f6a7b8'
down_revision = 'b1c2d3e4f5a6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('media_assets', sa.Column('last_frame_s3_url', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('media_assets', 'last_frame_s3_url')
