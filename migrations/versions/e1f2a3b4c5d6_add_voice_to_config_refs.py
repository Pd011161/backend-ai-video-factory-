"""add voice config to config_refs

The narration voice (language/gender/pace/tone/style) lived in Step 5's local React state, so it
reset to the hardcoded defaults on every page load and the same values were re-typed for every run.
This column parks it beside the character's photo and name, where it is set once.

One JSON blob rather than five columns: only the "character" row uses it, the field set is still
growing (gender was added the same day), and nothing queries it — the typing lives in VoiceConfig.

nullable with no server_default on purpose: a JSON server_default renders differently on SQLite
(TEXT) than on PostgreSQL, and NULL-as-empty costs one `or {}` in code and behaves the same on both.

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-08-04

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e1f2a3b4c5d6'
down_revision = 'd0e1f2a3b4c5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch_alter_table: SQLite is still a supported backend.
    with op.batch_alter_table('config_refs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('voice', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('config_refs', schema=None) as batch_op:
        batch_op.drop_column('voice')
