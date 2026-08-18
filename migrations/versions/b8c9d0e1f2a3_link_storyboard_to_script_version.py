"""link storyboard_documents to the script version they were built from

Storyboards and scripts versioned independently with no link at all, so nothing could answer "which
boards belong to this script?" — regenerating a script left its now-orphaned storyboard looking
perfectly current.

Existing rows are backfilled to their run's CURRENT script version. That is a guess (the real answer
was never recorded), but it is the only one that keeps every run's history reachable: leaving NULL
would hide every pre-existing board the moment grouping starts filtering by script version.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-08-02
"""
from alembic import op
import sqlalchemy as sa

revision = 'b8c9d0e1f2a3'
down_revision = 'a7b8c9d0e1f2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch_alter_table: SQLite is still a supported backend (config.yaml `database.backend`)
    with op.batch_alter_table('storyboard_documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('script_version', sa.Integer(), nullable=True))

    op.execute(sa.text("""
        UPDATE storyboard_documents AS sb
           SET script_version = (
               SELECT sd.version FROM script_documents AS sd
                WHERE sd.run_id = sb.run_id AND sd.is_current = TRUE
                LIMIT 1)
         WHERE sb.script_version IS NULL
    """))


def downgrade() -> None:
    with op.batch_alter_table('storyboard_documents', schema=None) as batch_op:
        batch_op.drop_column('script_version')
