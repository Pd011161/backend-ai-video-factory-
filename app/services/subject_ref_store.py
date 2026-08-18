"""Replaces v1's app/services/subject_ref_store.py (file/Drive-backed menu+subject-ref store).

Same shim pattern as scene_store.py: only the read-only subset the ported graph nodes call
(`match_subject`, `subject_image_ref`, `subject_preview_url`) is implemented here, backed by
`MenuRepo` via a short-lived session. Route-level CRUD calls `MenuRepo` directly.
"""
from __future__ import annotations

from urllib.parse import quote

from app.db.base import SessionLocal
from app.repositories import MediaAssetRepo, MenuRepo
from app.services import active_config


def _ref_dict(ref, db) -> dict:
    image = ""
    if ref.image_asset_id:
        asset = MediaAssetRepo(db).get(ref.image_asset_id)
        if asset:
            image = asset.s3_url or ""
    return {"id": str(ref.id), "name": ref.name, "kind": ref.kind, "image": image}


def active_subject_refs() -> list[dict]:
    menu_id = active_config.get().get("menu_id", "")
    if not menu_id:
        return []
    db = SessionLocal()
    try:
        menu = MenuRepo(db).get_menu(menu_id)
        return [_ref_dict(r, db) for r in menu.subject_refs] if menu else []
    finally:
        db.close()


def match_subject(query: str, kind: str | None = None) -> dict | None:
    menu_id = active_config.get().get("menu_id", "")
    if not menu_id or not (query or "").strip():
        return None
    db = SessionLocal()
    try:
        ref = MenuRepo(db).match_subject(menu_id, query, kind)
        return _ref_dict(ref, db) if ref else None
    finally:
        db.close()


def match_image(query: str, kind: str | None = None) -> str:
    s = match_subject(query, kind)
    return s["image"] if s else ""


def subject_image_ref(menu_id: str, ref_id: str) -> str:
    db = SessionLocal()
    try:
        menu = MenuRepo(db).get_menu(menu_id)
        for r in (menu.subject_refs if menu else []):
            if str(r.id) == str(ref_id):
                return _ref_dict(r, db).get("image", "")
        return ""
    finally:
        db.close()


def subject_preview_url(subject_id: str) -> str:
    mid = (active_config.get().get("menu_id") or "").strip()
    if not mid or not subject_id:
        return ""
    return f"/api/menus/{quote(mid)}/subjects/{quote(str(subject_id))}/preview"
