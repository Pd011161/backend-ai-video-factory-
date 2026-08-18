"""one step of undo for a shot's IMAGE, mirroring prev_video_asset_id

Replacing a shot's image — a regenerate, a quick edit, an outpaint/mask edit, a crop, an upload —
superseded the old asset and forgot which one it was. The clip side has had `prev_video_asset_id`
and an undo button since the beginning; the image side had no way back from the UI at all, and
recovering a render meant reading the asset table by hand.

Same shape as the video column, and safe for the same reason: image keys carry a timestamp
(`{scene}_{no}_{unix}.png`), so the superseded asset still points at bytes that exist.

Revision ID: c4d5e6f7a8b9
Revises: b2c3d4e5f6a7
Create Date: 2026-08-16

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c4d5e6f7a8b9'
down_revision = 'b2c3d4e5f6a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NULL on every existing row: nothing to go back to until the next replacement, which is the
    # honest answer — the previous asset of an old render was never recorded.
    # Not a ForeignKey, matching prev_video_asset_id (SQLite cannot add one without a table rebuild,
    # and declaring one only in the model made autogenerate fight every real database).
    with op.batch_alter_table('storyboard_shots', schema=None) as batch_op:
        batch_op.add_column(sa.Column('prev_image_asset_id', sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('storyboard_shots', schema=None) as batch_op:
        batch_op.drop_column('prev_image_asset_id')
