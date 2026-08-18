"""characters table + runs.character_id, seeded from the single config_refs character row

The host used to be ONE row in `config_refs` keyed on kind="character" — one per installation by
construction. This promotes it to a table shaped like `brands`, with `runs.character_id` pointing at
one, so a run can pick its own host.

The data migration is the part that matters: the existing character (name, description, voice, and
the reference photo already uploaded to S3) becomes the FIRST character with is_default=True. Runs
keep character_id NULL and resolve to that default, so every existing run renders exactly as before
— without it they would fall back to config.yaml's `docs/teacher.png` and lose the uploaded photo.

The old config_refs row is left in place: the "scene" row still lives there and is still read at
startup, and leaving the character row costs nothing while making this migration reversible.

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-08-04

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = 'f2a3b4c5d6e7'
down_revision = 'e1f2a3b4c5d6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'characters',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('image_asset_id', sa.String(), nullable=True),
        sa.Column('s3_url', sa.String(), nullable=True),
        sa.Column('voice', sa.JSON(), nullable=True),
        sa.Column('is_default', sa.Boolean(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['image_asset_id'], ['media_assets.id'],
                                name=op.f('fk_characters_image_asset_id_media_assets')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_characters')),
    )

    # batch_alter_table: SQLite cannot ALTER a table to add a foreign key in place.
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('character_id', sa.String(), nullable=True))
        batch_op.create_foreign_key(op.f('fk_runs_character_id_characters'), 'characters',
                                    ['character_id'], ['id'], ondelete='SET NULL')
    op.create_index(op.f('ix_runs_character_id'), 'runs', ['character_id'])

    _seed_from_config_ref()


def _seed_from_config_ref() -> None:
    """Carry the one existing character over, so nothing changes for runs that already exist.

    Seeds the EFFECTIVE values, not the raw row. Today a blank field in `config_refs` falls through
    to config.yaml at startup, so the host the operator actually sees is the two layers folded
    together — and in a fresh install every text field is blank with only the photo set. Copying the
    raw row would produce a nameless character whose prompts only worked by accident, so the fold is
    done once here and the table becomes the real source of truth."""
    import json

    from app.core.config import Settings

    bind = op.get_bind()
    if 'config_refs' not in inspect(bind).get_table_names():
        return
    row = bind.execute(sa.text(
        "SELECT name, description, s3_url, image_asset_id, voice FROM config_refs WHERE kind = 'character'"
    )).mappings().first()

    yaml = Settings.load()
    stored = row['voice'] if row else None
    if isinstance(stored, str):          # SQLite hands JSON back as raw text
        try:
            stored = json.loads(stored)
        except ValueError:
            stored = None
    voice = yaml.voice.model_dump()
    for k, v in (stored or {}).items():
        if k in voice and v not in ("", None):
            voice[k] = v

    bind.execute(
        sa.text("""INSERT INTO characters (id, name, description, image_asset_id, s3_url, voice,
                                           is_default, updated_at)
                   VALUES (:id, :name, :description, :image_asset_id, :s3_url, :voice, :is_default, :updated_at)"""),
        {"id": "default",
         "name": ((row['name'] if row else "") or "").strip() or yaml.script.character_name or "ตัวละครหลัก",
         "description": ((row['description'] if row else "") or "").strip() or yaml.script.character_desc,
         "image_asset_id": row['image_asset_id'] if row else None,
         "s3_url": (row['s3_url'] if row else "") or yaml.image_gen.character_ref or "",
         "voice": json.dumps(voice, ensure_ascii=False),
         "is_default": True,
         "updated_at": None if bind.dialect.name == 'sqlite' else sa.func.now()},
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_runs_character_id'), table_name='runs')
    with op.batch_alter_table('runs', schema=None) as batch_op:
        batch_op.drop_constraint(op.f('fk_runs_character_id_characters'), type_='foreignkey')
        batch_op.drop_column('character_id')
    op.drop_table('characters')
