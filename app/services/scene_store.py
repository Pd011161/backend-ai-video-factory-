"""Replaces v1's app/services/scene_store.py (file/Drive-backed brand+scene store).

Ported graph nodes (app/graph/nodes.py) call the small read-only subset of this module's API
(`active_brand`, `active_scenes`, `active_default_scene`, `scene_image_ref`, `scene_preview_url`)
with no db session of their own, so this shim opens a short-lived session per call against
`BrandRepo` — the real CRUD (create/update/delete brand & scene) now lives on routes that call
`BrandRepo` directly with the request's session; this module only backs the pipeline read path.
"""
from __future__ import annotations

from urllib.parse import quote

from app.db.base import SessionLocal
from app.repositories import BrandRepo, MediaAssetRepo
from app.services import active_config


def _scene_dict(scene, db) -> dict:
    image = ""
    if scene.image_asset_id:
        asset = MediaAssetRepo(db).get(scene.image_asset_id)
        if asset:
            image = asset.s3_url or ""
    # `desc` = WHEN to pick this scene (read by the shot classifier). `layout` = HOW it is composed
    # (read by the image-prompt author). Both ride along; each consumer takes only the one it needs.
    return {"id": str(scene.id), "name": scene.name, "desc": scene.desc or "",
            "layout": scene.layout or "", "image": image, "default": bool(scene.is_default)}


def _brand_dict(brand, db) -> dict:
    return {
        "id": brand.id,
        "name": brand.name,
        "scenes": [_scene_dict(s, db) for s in brand.scenes],
        "tagline": brand.tagline or "",
        "platform_style": brand.platform_style or "",
        "theme": brand.theme or "",
        "mood": brand.mood or "",
        "material_palette": brand.material_palette or "",
        "lighting": brand.lighting or "",
        "editing_style": brand.editing_style or "",
        "vo_tone": brand.vo_tone or "",
        "music": brand.music or "",
        "camera_movement": brand.camera_movement or "",
        "words_per_second": brand.words_per_second or 0.0,
    }


def active_brand() -> dict | None:
    brand_id = active_config.get().get("brand_id", "")
    if not brand_id:
        return None
    db = SessionLocal()
    try:
        brand = BrandRepo(db).get_brand(brand_id)
        return _brand_dict(brand, db) if brand else None
    finally:
        db.close()


def active_scenes() -> list[dict]:
    b = active_brand()
    return b.get("scenes", []) if b else []


def active_default_scene() -> dict | None:
    scenes = active_scenes()
    return next((s for s in scenes if s.get("default")), scenes[0] if scenes else None)


def scene_image_ref(brand_id: str, scene_id: str) -> str:
    db = SessionLocal()
    try:
        brand = BrandRepo(db).get_brand(brand_id)
        for s in (brand.scenes if brand else []):
            if str(s.id) == str(scene_id):
                return _scene_dict(s, db).get("image", "")
        return ""
    finally:
        db.close()


def scene_preview_url(scene_id: str) -> str:
    b = active_brand()
    if not b or not scene_id or not any(str(s.get("id")) == str(scene_id) for s in b.get("scenes", [])):
        return ""
    return f"/api/brands/{quote(b['id'])}/scenes/{quote(str(scene_id))}/preview"
