"""per-shot video duration, user-attached video refs, and video history

Four storage gaps on the video side, all additive:

1. `storyboard_shots.target_seconds` — Omni's API has NO duration parameter (see
   gemini_client.generate_video_omni_multi), so clip length is purely a prompt-text lever. Until now
   every shot got the same "about 4 to 10 seconds" sentence and the model drifted long. The prompt
   author now estimates seconds from the shot's own script and this is where that lands, so it can
   be shown, hand-corrected, and reused by the merge bin-packer.

2. `shot_video_refs` — the image side lets a user attach their own reference photos; video had no
   equivalent on the render route at all, and refs attached at the image step are filtered out
   before video (_OMNI_REF_KINDS). Persisted rather than session-local because the authored
   prompt_video TEXT references them by tag — a ref that vanished on reload would leave the prompt
   citing an image that is no longer sent.

3. `media_assets.shot_id` — clips were only reachable through the shot's two pointers
   (current/prev), so the third-most-recent render was unrecoverable even though its row and its S3
   bytes both still existed. Deliberately NOT a ForeignKey, matching prev_video_asset_id's own
   comment in models.py: SQLite cannot add one in place, and declaring it would make the model
   disagree with every existing database.

4. `media_assets.refs_used` + `shot_prompt_video_versions` — the image side records refs_used and
   full_prompt per render; video recorded neither, so after the fact you could not tell which
   reference images produced a clip, and a per-shot prompt regenerate overwrote the previous text
   with no copy anywhere.

Revision ID: a1b2c3d4e5f6
Revises: f2a3b4c5d6e7
Create Date: 2026-08-07

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'f2a3b4c5d6e7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1 — per-shot target clip length. 0.0 = never estimated (every existing row), which the code
    # reads as "fall back to the generic duration wording", so old shots behave exactly as before.
    with op.batch_alter_table('storyboard_shots', schema=None) as batch_op:
        batch_op.add_column(sa.Column('target_seconds', sa.Float(), nullable=False,
                                      server_default='0'))

    # 3 + 4 — attribute a clip to its shot, and record what it was rendered with.
    with op.batch_alter_table('media_assets', schema=None) as batch_op:
        batch_op.add_column(sa.Column('shot_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('refs_used', sa.JSON(), nullable=True))
    op.create_index(op.f('ix_media_assets_shot_id'), 'media_assets', ['shot_id'])

    # 2 — reference images the user attached to a shot for VIDEO generation.
    op.create_table(
        'shot_video_refs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('shot_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        # What the ref is, in the user's words. Reaches BOTH the prompt author (which never sees the
        # image) and the ref's own rule line in the rendered prompt — the same two readers ref_notes
        # serves on the image side.
        sa.Column('note', sa.String(), nullable=False, server_default=''),
        sa.Column('position', sa.Integer(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['shot_id'], ['storyboard_shots.id'],
                                name=op.f('fk_shot_video_refs_shot_id_storyboard_shots'),
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_shot_video_refs')),
    )
    op.create_index(op.f('ix_shot_video_refs_shot_id'), 'shot_video_refs', ['shot_id'])

    # 4 — prompt_video history. Text is cheap; keeping a handful per shot costs nothing next to the
    # clips, and it is the only way back from a regenerate that lands worse than what it replaced.
    op.create_table(
        'shot_prompt_video_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('shot_id', sa.Integer(), nullable=False),
        sa.Column('prompt_video', sa.String(), nullable=False, server_default=''),
        sa.Column('source', sa.String(), nullable=False, server_default=''),  # regen | manual | restore
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['shot_id'], ['storyboard_shots.id'],
                                name=op.f('fk_shot_prompt_video_versions_shot_id_storyboard_shots'),
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_shot_prompt_video_versions')),
    )
    op.create_index(op.f('ix_shot_prompt_video_versions_shot_id'),
                    'shot_prompt_video_versions', ['shot_id'])


def downgrade() -> None:
    op.drop_index(op.f('ix_shot_prompt_video_versions_shot_id'),
                  table_name='shot_prompt_video_versions')
    op.drop_table('shot_prompt_video_versions')
    op.drop_index(op.f('ix_shot_video_refs_shot_id'), table_name='shot_video_refs')
    op.drop_table('shot_video_refs')
    op.drop_index(op.f('ix_media_assets_shot_id'), table_name='media_assets')
    with op.batch_alter_table('media_assets', schema=None) as batch_op:
        batch_op.drop_column('refs_used')
        batch_op.drop_column('shot_id')
    with op.batch_alter_table('storyboard_shots', schema=None) as batch_op:
        batch_op.drop_column('target_seconds')
