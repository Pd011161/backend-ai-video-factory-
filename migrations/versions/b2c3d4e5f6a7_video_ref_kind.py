"""what a shot_video_refs row is: an attached ref, or the user's answer for an auto slot

`shot_video_refs` held one thing — a reference photo the user attached. The video side still had no
equivalent of the image side's `removed_refs` / `ref_person` / `ref_scene`: whatever
_omni_shot_ref_urls decided to send for person and kitchen, the user had to live with.

Rather than a second table (and a second CRUD path, a second carry-forward, a second staleness
input), the existing rows gain a `kind`:

    kind='user'                 → a ref the user attached — every existing row, unchanged
    kind='person'|'kitchen'     → the user's answer for that AUTO slot:
                                    url set   → replaces it
                                    url empty → removes it
    (no row for a slot)         → the automatic rule stands

An empty url is only meaningful on a slot row; on a 'user' row it stays a blank the writer drops.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-08-10

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6a7'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # server_default='user' backfills every existing row as an attached ref, which is exactly what
    # they are — so the read paths behave identically until the user touches a slot.
    with op.batch_alter_table('shot_video_refs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('kind', sa.String(), nullable=False, server_default='user'))


def downgrade() -> None:
    # Slot rows (kind != 'user') have no meaning without the column — an empty-url row would read as
    # an attached ref pointing nowhere — so they go with it.
    op.execute("DELETE FROM shot_video_refs WHERE kind <> 'user'")
    with op.batch_alter_table('shot_video_refs', schema=None) as batch_op:
        batch_op.drop_column('kind')
