"""The hosts a run can be made with — one selected per run, with a default for runs that never chose.

Shaped after BrandRepo down to the method names: same slug-id + dedupe, same allowlisted update,
and the single-default rule BrandRepo applies to a brand's scenes. A run picking a character is the
same problem as a run picking a brand, so it is deliberately not a new pattern.

`resolve()` is the one method that is not in BrandRepo, and it is the whole point: everything
downstream asks "which host is this run using" and must get an answer even for the runs that
predate this table.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ids import dedupe_id, slugify
from app.db.models import Character, Run
from app.repositories.errors import NotFoundError

# Text fields the API may set. `voice` is handled separately (it is a whole sub-object), and
# id/is_default/s3_url have their own endpoints — letting them through here would bypass the
# single-default rule and the image upload's asset bookkeeping.
_UPDATABLE = {"name", "description"}


class CharacterRepo:
    def __init__(self, db: Session):
        self.db = db

    def list_characters(self) -> list[Character]:
        # Default first, for the same reason BrandRepo orders scenes that way: it is the one every
        # unassigned run lands on, so it is the one worth seeing first.
        return list(self.db.scalars(select(Character).order_by(Character.is_default.desc(), Character.name)))

    def create_character(self, name: str) -> Character:
        base = slugify(name, fallback="character")
        cid = dedupe_id(base, lambda c: self.db.get(Character, c) is not None)
        # The first character ever created is the default — otherwise nothing would answer
        # `resolve()` for a run that has not chosen, exactly as BrandRepo does for a brand's scenes.
        character = Character(id=cid, name=name, is_default=not self._has_any())
        self.db.add(character)
        self.db.flush()
        return character

    def get_character(self, character_id: str) -> Character | None:
        return self.db.get(Character, character_id)

    def require_character(self, character_id: str) -> Character:
        character = self.get_character(character_id)
        if character is None:
            raise NotFoundError("Character", character_id)
        return character

    def update_character(self, character_id: str, fields: dict) -> Character | None:
        character = self.get_character(character_id)
        if character is None:
            return None
        for key, value in fields.items():
            if key in _UPDATABLE:
                setattr(character, key, value)
        if "voice" in fields and isinstance(fields["voice"], dict):
            character.voice = fields["voice"]
        self.db.flush()
        return character

    def delete_character(self, character_id: str) -> bool:
        character = self.get_character(character_id)
        if character is None:
            return False
        was_default = bool(character.is_default)
        # Clear the pointers ourselves rather than trusting ON DELETE SET NULL. SQLite is a
        # supported backend and runs without `PRAGMA foreign_keys=ON`, so there the rule is
        # inert and the runs would keep a dangling id — the same trap RunRepo.delete documents
        # for its children. Doing it here behaves identically on both backends.
        for run in self.db.scalars(select(Run).where(Run.character_id == character_id)):
            run.character_id = None
        self.db.flush()
        self.db.delete(character)
        self.db.flush()
        if was_default:
            # Never leave the set without a default; the next one by name takes over.
            remaining = self.list_characters()
            if remaining:
                remaining[0].is_default = True
                self.db.flush()
        return True

    def set_default(self, character_id: str) -> bool:
        character = self.get_character(character_id)
        if character is None:
            return False
        self._clear_default()
        character.is_default = True
        self.db.flush()
        return True

    def default_character(self) -> Character | None:
        found = self.db.scalars(select(Character).where(Character.is_default.is_(True))).first()
        # A table that somehow lost its default still has to answer, or every unassigned run breaks.
        return found or self.db.scalars(select(Character).order_by(Character.name)).first()

    def resolve(self, run_id: str) -> Character | None:
        """The character a run is actually made with: its own, else the default.

        A NULL `run.character_id` means "not chosen", not "no host" — every run created before this
        table exists is in that state, and they must keep rendering with the same face and voice
        they always had."""
        if run_id:
            run = self.db.get(Run, run_id)
            if run is not None and run.character_id:
                found = self.get_character(run.character_id)
                if found is not None:
                    return found
        return self.default_character()

    def _has_any(self) -> bool:
        return self.db.scalars(select(Character).limit(1)).first() is not None

    def _clear_default(self) -> None:
        for c in self.db.scalars(select(Character).where(Character.is_default.is_(True))):
            c.is_default = False
