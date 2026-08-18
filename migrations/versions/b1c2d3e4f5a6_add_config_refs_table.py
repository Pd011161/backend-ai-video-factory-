"""add config_refs table (durable global character/scene ref images)

Revision ID: b1c2d3e4f5a6
Revises: 34d233af8b69
Create Date: 2026-07-26 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b1c2d3e4f5a6'
down_revision = '34d233af8b69'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'config_refs',
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('image_asset_id', sa.String(), nullable=True),
        sa.Column('s3_url', sa.String(), nullable=False, server_default=''),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['image_asset_id'], ['media_assets.id'], name=op.f('fk_config_refs_image_asset_id_media_assets')),
        sa.PrimaryKeyConstraint('kind', name=op.f('pk_config_refs')),
    )


def downgrade() -> None:
    op.drop_table('config_refs')
