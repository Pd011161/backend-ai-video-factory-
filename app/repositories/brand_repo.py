"""Replaces v1's app/services/scene_store.py (JSON-file-per-brand + Drive fallback).

Method names mirror the v1 module's free functions so callers moving from
`scene_store.create_brand(...)` to `BrandRepo(db).create_brand(...)` stay a near-mechanical rename.
Image bytes still go through the unchanged S3 storage.py helper — this repo only ever stores the
resulting MediaAsset id, never bytes.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ids import dedupe_id, slugify
from app.db.models import Brand, BrandScene
from app.repositories.errors import NotFoundError


class BrandRepo:
    def __init__(self, db: Session):
        self.db = db

    def list_brands(self) -> list[Brand]:
        return list(self.db.scalars(select(Brand).order_by(Brand.name)))

    def create_brand(self, name: str) -> Brand:
        base = slugify(name, fallback="brand")
        brand_id = dedupe_id(base, lambda cid: self.db.get(Brand, cid) is not None)
        brand = Brand(id=brand_id, name=name)
        self.db.add(brand)
        self.db.flush()
        return brand

    def get_brand(self, brand_id: str) -> Brand | None:
        return self.db.get(Brand, brand_id)

    def require_brand(self, brand_id: str) -> Brand:
        brand = self.get_brand(brand_id)
        if brand is None:
            raise NotFoundError("Brand", brand_id)
        return brand

    def update_brand(self, brand_id: str, fields: dict) -> Brand | None:
        brand = self.get_brand(brand_id)
        if brand is None:
            return None
        allowed = {
            "name", "tagline", "platform_style", "theme", "mood", "material_palette",
            "lighting", "editing_style", "vo_tone", "music", "camera_movement", "words_per_second",
        }
        for key, value in fields.items():
            if key in allowed:
                setattr(brand, key, value)
        self.db.flush()
        return brand

    def delete_brand(self, brand_id: str) -> bool:
        brand = self.get_brand(brand_id)
        if brand is None:
            return False
        self.db.delete(brand)  # cascades to brand_scenes
        self.db.flush()
        return True

    # -- scenes ----------------------------------------------------------

    def add_scene(
        self,
        brand_id: str,
        *,
        name: str,
        desc: str = "",
        layout: str = "",
        image_asset_id: str | None = None,
        is_default: bool = False,
    ) -> BrandScene:
        self.require_brand(brand_id)
        if is_default:
            self._clear_default(brand_id)
        elif not self._has_any_scene(brand_id):
            is_default = True  # first scene on a brand is always the default, as in v1
        scene = BrandScene(brand_id=brand_id, name=name, desc=desc, layout=layout,
                           image_asset_id=image_asset_id, is_default=is_default)
        self.db.add(scene)
        self.db.flush()
        return scene

    def update_scene(self, brand_id: str, scene_id: int, fields: dict) -> BrandScene | None:
        """Edit an existing scene's text. Only name/desc/layout — the image and the default flag
        have their own endpoints, and letting them through here would silently bypass
        _clear_default's single-default invariant."""
        scene = self.db.get(BrandScene, scene_id)
        if scene is None or scene.brand_id != brand_id:
            return None
        for key in ("name", "desc", "layout"):
            if key in fields and fields[key] is not None:
                setattr(scene, key, str(fields[key]))
        self.db.flush()
        return scene

    def delete_scene(self, brand_id: str, scene_id: int) -> bool:
        scene = self.db.get(BrandScene, scene_id)
        if scene is None or scene.brand_id != brand_id:
            return False
        was_default = scene.is_default
        self.db.delete(scene)
        self.db.flush()
        if was_default:
            remaining = self.list_scenes(brand_id)
            if remaining:
                self.set_default_scene(brand_id, remaining[0].id)
        return True

    def set_default_scene(self, brand_id: str, scene_id: int) -> bool:
        scene = self.db.get(BrandScene, scene_id)
        if scene is None or scene.brand_id != brand_id:
            return False
        self._clear_default(brand_id)
        scene.is_default = True
        self.db.flush()
        return True

    def list_scenes(self, brand_id: str) -> list[BrandScene]:
        # Default first: it is the fallback every shot lands on when the classifier matches nothing,
        # so it is the one worth reading first — buried mid-list (plain id order) it was easy to miss
        # which scene that even was. `id` still breaks the tie, so the rest keeps its creation order.
        stmt = (select(BrandScene).where(BrandScene.brand_id == brand_id)
                .order_by(BrandScene.is_default.desc(), BrandScene.id))
        return list(self.db.scalars(stmt))

    def default_scene(self, brand_id: str) -> BrandScene | None:
        stmt = select(BrandScene).where(BrandScene.brand_id == brand_id, BrandScene.is_default.is_(True))
        return self.db.scalars(stmt).first()

    def _clear_default(self, brand_id: str) -> None:
        for scene in self.list_scenes(brand_id):
            if scene.is_default:
                scene.is_default = False

    def _has_any_scene(self, brand_id: str) -> bool:
        return bool(self.list_scenes(brand_id))
