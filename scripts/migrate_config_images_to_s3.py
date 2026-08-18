"""One-time migration: push every CONFIG reference image to S3 and rewrite stored URLs to the direct
S3 link, so the Omni render (and everything else) fetches config refs straight from S3 — no local
files, no internal preview-endpoint indirection.

Covers:
  1. character / scene config defaults (docs/teacher.png, docs/kitchen.jpeg or current settings) →
     upload to S3 + create media_assets + upsert config_refs.
  2. brand_scenes + subject_refs media_assets whose s3_url is still local (/api/media/...) → re-upload
     bytes to S3 and rewrite media_assets.s3_url.
  3. shot_refs_used person/kitchen/scene rows that stored a preview-endpoint URL → rewrite to the
     direct S3 URL (person→character config; brand-scene→the scene's asset.s3_url).

Idempotent (skips rows already on S3). Requires S3 to be configured.

Usage:
    uv run python scripts/migrate_config_images_to_s3.py --dry-run   # preview
    uv run python scripts/migrate_config_images_to_s3.py             # apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # AWS_* / secrets live in .env, same as the app

from app.core.config import ROOT_DIR, settings
from app.db.base import SessionLocal
from app.db.models import BrandScene, ConfigRef, MediaAsset, ShotRefUsed, SubjectRef
from app.repositories import BrandRepo, MediaAssetRepo
from app.services import storage

_CT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_DEFAULTS = {"character": ("character_ref", "docs/teacher.png"), "scene": ("scene_ref", "docs/kitchen.jpeg")}
_OMNI_KINDS = {"person", "kitchen", "scene"}


def _ct(key: str) -> str:
    return _CT.get(Path(key).suffix.lower(), "image/png")


def _local_bytes(url: str) -> bytes | None:
    """Read the local bytes behind a /api/media/... or bare-path URL, else None."""
    from app.api.routes import _safe_media_path
    p = _safe_media_path(url)
    if p is None:
        p = Path(url) if Path(url).is_absolute() else (ROOT_DIR / url)
    return p.read_bytes() if p.exists() else None


def _is_s3(url: str) -> bool:
    return bool(url) and storage.key_from_url(url) is not None


def migrate(dry: bool) -> None:
    if not storage.is_configured():
        raise SystemExit("S3 not configured (AWS_* env) — cannot migrate.")
    db = SessionLocal()
    n_assets = n_refs = n_cfg = skipped = 0
    try:
        # 1. character/scene config defaults → S3 + config_refs
        for kind, (attr, default) in _DEFAULTS.items():
            row = db.get(ConfigRef, kind)
            if row and _is_s3(row.s3_url):
                skipped += 1
                continue
            src = getattr(settings.image_gen, attr, "") or default
            if _is_s3(src):
                # config already points to S3 — just record the config_ref row
                asset = MediaAssetRepo(db).create(kind="image", s3_key=(storage.key_from_url(src) or f"refs/{kind}.png"), s3_url=src)
                _upsert(db, kind, asset.id, src, dry)
                n_cfg += 1
                continue
            data = _local_bytes(src)
            if not data:
                print(f"  [skip] {kind}: source not found ({src})")
                continue
            key = f"refs/{kind}{Path(src).suffix or '.png'}"
            url = src if dry else storage.upload_bytes(key, data, _ct(key))
            print(f"  [config] {kind}: {src} → {url}")
            if not dry:
                asset = MediaAssetRepo(db).create(kind="image", s3_key=key, s3_url=url)
                _upsert(db, kind, asset.id, url, dry)
                setattr(settings.image_gen, attr, url)
            n_cfg += 1

        # 2. brand_scenes + subject_refs media_assets: local → S3
        asset_ids = {s.image_asset_id for s in db.query(BrandScene) if s.image_asset_id} | \
                    {r.image_asset_id for r in db.query(SubjectRef) if r.image_asset_id}
        for aid in asset_ids:
            a = db.get(MediaAsset, aid)
            if not a or _is_s3(a.s3_url):
                skipped += 1
                continue
            data = _local_bytes(a.s3_url)
            if not data:
                print(f"  [skip] asset {aid}: local bytes not found ({a.s3_url})")
                continue
            url = a.s3_url if dry else storage.upload_bytes(a.s3_key, data, _ct(a.s3_key))
            print(f"  [asset] {aid}: {a.s3_url} → {url}")
            if not dry:
                a.s3_url = url
            n_assets += 1

        # 3. shot_refs_used: preview URLs → direct S3
        for r in db.query(ShotRefUsed):
            if (r.kind or "").lower() not in _OMNI_KINDS or _is_s3(r.url):
                continue
            new = _resolve_direct(db, r.url)
            if new and new != r.url:
                print(f"  [refs_used {r.id}] {r.kind}: {r.url} → {new}")
                if not dry:
                    r.url = new
                n_refs += 1

        if dry:
            print("\nDRY RUN — no changes written.")
        else:
            db.commit()
            print("\nCommitted.")
        print(f"config refs: {n_cfg} | assets migrated: {n_assets} | refs_used backfilled: {n_refs} | skipped(already S3): {skipped}")
    finally:
        db.close()


def _upsert(db, kind: str, asset_id: str, url: str, dry: bool) -> None:
    if dry:
        return
    from app.api.routes import _upsert_config_ref
    _upsert_config_ref(db, kind, asset_id, url)


def _resolve_direct(db, url: str) -> str | None:
    """preview-endpoint URL → the direct S3 URL it points at (for refs_used backfill)."""
    u = (url or "").split("?")[0]
    if u == "/api/refs/character/preview":
        v = settings.image_gen.character_ref or ""
        return v if _is_s3(v) else None
    if u == "/api/refs/scene/preview":
        v = settings.image_gen.scene_ref or ""
        return v if _is_s3(v) else None
    if u.startswith("/api/brands/") and u.endswith("/preview") and "/scenes/" in u:
        parts = u.strip("/").split("/")   # api brands {bid} scenes {sid} preview
        try:
            brand_id, scene_id = parts[2], int(parts[4])
        except (IndexError, ValueError):
            return None
        scene = next((s for s in BrandRepo(db).list_scenes(brand_id) if s.id == scene_id), None)
        asset = MediaAssetRepo(db).get(scene.image_asset_id) if (scene and scene.image_asset_id) else None
        return asset.s3_url if (asset and _is_s3(asset.s3_url)) else None
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="preview changes without writing")
    args = ap.parse_args()
    migrate(args.dry_run)
