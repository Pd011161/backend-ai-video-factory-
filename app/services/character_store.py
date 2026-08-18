"""Read path for "which host is this run using", for callers that have no db session.

Same shim as scene_store.py and for the same reason: graph nodes were ported from v1 and take
neither a session nor a run id, so this resolves the contextvar `active_config` keeps and opens a
short-lived session per call. Real CRUD lives on the /characters routes with the request's session.

Note what this closes: the SCENE already worked this way (the run's brand wins, the global config
is only a fallback), while the character was read straight off the process-global settings in every
one of ~14 places. That asymmetry is what made a second host impossible.
"""
from __future__ import annotations

from app.db.base import SessionLocal
from app.repositories import CharacterRepo
from app.services import active_config


def _character_dict(character) -> dict:
    return {
        "id": character.id,
        "name": character.name or "",
        "description": character.description or "",
        "image": character.s3_url or "",
        "voice": character.voice or {},
        "default": bool(character.is_default),
    }


def active_character() -> dict | None:
    """The run's chosen character, or the default one. None only when the table is empty."""
    db = SessionLocal()
    try:
        found = CharacterRepo(db).resolve(active_config.get_current_run())
        return _character_dict(found) if found else None
    finally:
        db.close()


def active_character_ref() -> str:
    """Just the reference photo URL — what the image-generation refs need."""
    c = active_character()
    return (c or {}).get("image", "")
