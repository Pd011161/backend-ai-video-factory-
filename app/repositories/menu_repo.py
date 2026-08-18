"""Replaces v1's app/services/subject_ref_store.py (JSON-file-per-menu + Drive fallback)."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ids import dedupe_id, slugify
from app.db.models import Menu, SubjectRef
from app.repositories.errors import NotFoundError


class MenuRepo:
    def __init__(self, db: Session):
        self.db = db

    def list_menus(self) -> list[Menu]:
        return list(self.db.scalars(select(Menu).order_by(Menu.name)))

    def create_menu(self, name: str) -> Menu:
        base = slugify(name, fallback="menu")
        menu_id = dedupe_id(base, lambda cid: self.db.get(Menu, cid) is not None)
        menu = Menu(id=menu_id, name=name)
        self.db.add(menu)
        self.db.flush()
        return menu

    def ensure_menu(self, menu_id: str, name: str) -> Menu:
        """Key a menu directly on an external id (e.g. a run's topic id) instead of a slugged
        name — used to auto-create a per-run config menu, same as v1's ensure_menu()."""
        menu = self.get_menu(menu_id)
        if menu is not None:
            return menu
        menu = Menu(id=menu_id, name=name)
        self.db.add(menu)
        self.db.flush()
        return menu

    def get_menu(self, menu_id: str) -> Menu | None:
        return self.db.get(Menu, menu_id)

    def require_menu(self, menu_id: str) -> Menu:
        menu = self.get_menu(menu_id)
        if menu is None:
            raise NotFoundError("Menu", menu_id)
        return menu

    def update_menu(self, menu_id: str, name: str) -> Menu | None:
        menu = self.get_menu(menu_id)
        if menu is None:
            return None
        menu.name = name
        self.db.flush()
        return menu

    def delete_menu(self, menu_id: str) -> bool:
        menu = self.get_menu(menu_id)
        if menu is None:
            return False
        self.db.delete(menu)  # cascades to subject_refs
        self.db.flush()
        return True

    # -- subject refs (ingredient/equipment reference photos) ------------

    def add_subject_ref(
        self,
        menu_id: str,
        *,
        name: str,
        kind: str = "ingredient",
        image_asset_id: str | None = None,
    ) -> SubjectRef:
        if kind not in ("ingredient", "equipment"):
            raise ValueError(f"invalid subject ref kind: {kind!r}")
        self.require_menu(menu_id)
        ref = SubjectRef(menu_id=menu_id, name=name, kind=kind, image_asset_id=image_asset_id)
        self.db.add(ref)
        self.db.flush()
        return ref

    def delete_subject_ref(self, menu_id: str, ref_id: int) -> bool:
        ref = self.db.get(SubjectRef, ref_id)
        if ref is None or ref.menu_id != menu_id:
            return False
        self.db.delete(ref)
        self.db.flush()
        return True

    def set_subject_ref_image(self, menu_id: str, ref_id: int, image_asset_id: str | None) -> SubjectRef | None:
        """Attach, replace or (with None) clear the photo on an existing subject — the row itself,
        its id and its name are untouched.

        Without this the only way to give a subject a photo was to delete it and add it back, which
        changes the row id that `subject_preview_url` bakes into every stored ref url. `pull_s3` used
        to call add_subject_ref here too, which is why it left a duplicate blank row behind every
        time it filled one in."""
        ref = self.db.get(SubjectRef, ref_id)
        if ref is None or ref.menu_id != menu_id:
            return None
        ref.image_asset_id = image_asset_id
        self.db.flush()
        return ref

    def list_subject_refs(self, menu_id: str) -> list[SubjectRef]:
        stmt = select(SubjectRef).where(SubjectRef.menu_id == menu_id).order_by(SubjectRef.id)
        return list(self.db.scalars(stmt))

    def match_subject(self, menu_id: str, query: str, kind: str | None = None) -> SubjectRef | None:
        """Fuzzy (case-insensitive substring) match, same rule as v1's match_subject()."""
        query = (query or "").strip().lower()
        if not query:
            return None
        candidates = self.list_subject_refs(menu_id)
        if kind is not None:
            candidates = [c for c in candidates if c.kind == kind]
        for ref in candidates:
            if ref.name.strip().lower() == query:
                return ref
        for ref in candidates:
            name = ref.name.strip().lower()
            if query in name or name in query:
                return ref
        return None
