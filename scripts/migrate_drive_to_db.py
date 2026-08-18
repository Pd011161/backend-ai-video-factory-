#!/usr/bin/env python3
"""One-off Drive -> SQLite migration (v1 Google Drive store -> v2 DB).

Read-only against Drive (drive.readonly scope) — never writes back, never deletes. Idempotent:
a run whose id already exists in the target DB is skipped entirely (safe to re-run after fixing
an error on one run without re-importing everything).

Usage:
    python scripts/migrate_drive_to_db.py                  # migrate everything found
    python scripts/migrate_drive_to_db.py --dry-run         # list what WOULD be migrated, no writes
    python scripts/migrate_drive_to_db.py --only 22 3       # migrate only these rids

Reads the same research_drive config v1 used (config.yaml at the v1 project root) so folder ids
and the service-account path never need to be duplicated/hardcoded here.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.base import SessionLocal  # noqa: E402
from app.db.models import Brand, BrandScene, DirectorPrompt  # noqa: E402
from app.repositories import (  # noqa: E402
    MediaAssetRepo,
    MenuRepo,
    ResearchRepo,
    RunRepo,
    ScriptRepo,
    StoryboardRepo,
)

V1_ROOT = Path("/Volumes/DevSSD/Development/InnovateAI/DEMO/ai-video-factory")


def load_drive_config() -> dict:
    cfg = yaml.safe_load((V1_ROOT / "config.yaml").read_text())
    return cfg["research_drive"]


def drive_service(cfg: dict):
    sa_path = V1_ROOT / cfg["service_account"]
    creds = service_account.Credentials.from_service_account_file(
        str(sa_path), scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_ids(service, folder_id: str) -> list[str]:
    out: list[str] = []
    page_token = None
    while True:
        params: dict = dict(
            q=f"'{folder_id}' in parents and trashed=false and mimeType='application/json'",
            fields="nextPageToken,files(name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000,
        )
        if page_token:
            params["pageToken"] = page_token
        res = service.files().list(**params).execute()
        for f in res.get("files") or []:
            name = f.get("name", "")
            if name.endswith(".json"):
                out.append(name[:-5])
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return sorted(out)


def read_json(service, folder_id: str, rid: str) -> dict | None:
    safe_rid = rid.replace("'", "")
    res = service.files().list(
        q=f"name='{safe_rid}.json' and '{folder_id}' in parents and trashed=false",
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files") or []
    if not files:
        return None
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, service.files().get_media(fileId=files[0]["id"], supportsAllDrives=True))
    done = False
    while not done:
        _, done = dl.next_chunk()
    try:
        return json.loads(buf.getvalue())
    except Exception:
        return None


def media_key_from_url(url: str) -> str:
    """Best-effort object key for a shot's generated_img/generated_video URL — mirrors
    app/api/routes.py's _media_key_from_url (kept standalone; this script doesn't import routes)."""
    if "/api/media/" in url:
        return url.split("/api/media/", 1)[-1]
    return url


def migrate_storyboard_stage(db, run_id: str, dumped: dict, *, step: str, verbose: bool):
    """Save one storyboard-category doc as the next version, wiring generated_img/generated_video
    through MediaAssetRepo the same way a bulk /steps/images call does (see
    app/api/routes.py::_persist_bulk_storyboard) — every shot with real media gets its own asset row."""
    doc = StoryboardRepo(db).save_new_version(
        run_id, title=dumped.get("title", ""), ingredients=dumped.get("ingredients", []),
        equipment=dumped.get("equipment", []), production=dumped.get("production", {}),
        scenes=dumped.get("scenes", []),
    )
    shots_by_key = {(sc.scene_id, sh.no): sh for sc in doc.scenes for sh in sc.shots}
    media = MediaAssetRepo(db)
    board = StoryboardRepo(db)
    n_images = n_videos = 0
    for scene in dumped.get("scenes", []):
        for shot_data in scene.get("shots", []):
            shot_row = shots_by_key.get((scene.get("scene_id"), shot_data.get("no")))
            if shot_row is None:
                continue
            img_url = (shot_data.get("generated_img") or "").strip()
            if img_url:
                asset = media.create(kind="image", s3_key=media_key_from_url(img_url), s3_url=img_url, run_id=run_id)
                board.set_shot_image(shot_row.id, asset.id)
                n_images += 1
            vid_url = (shot_data.get("generated_video") or "").strip()
            if vid_url:
                asset = media.create(kind="video", s3_key=media_key_from_url(vid_url), s3_url=vid_url, run_id=run_id)
                board.set_shot_video(shot_row.id, asset.id)
                n_videos += 1
    if verbose:
        print(f"    storyboard[{step}] -> v{doc.version} ({n_images} images, {n_videos} videos linked)")
    return doc


def migrate_run(db, service, cfg: dict, rid: str, *, dry_run: bool, verbose: bool, research_folder_id: str | None = None) -> bool:
    """Returns True if migrated (or would be, in dry-run), False if skipped.

    `research_folder_id` overrides cfg["research"]["folder_id"] — used for the full 77-course
    production research set, which lives in a different Drive folder than the small hand-tested
    script/storyboard/video folders (only a handful of the 77 topics ever got that far)."""
    if RunRepo(db).get(rid) is not None:
        print(f"[{rid}] already exists in the target DB — skipping")
        return False

    research = read_json(service, research_folder_id or cfg["research"]["folder_id"], rid)
    script = read_json(service, cfg["script"]["folder_id"], rid)
    storyboard_text = read_json(service, cfg["storyboard_text"]["folder_id"], rid)
    storyboard_image = read_json(service, cfg["storyboard_image"]["folder_id"], rid)
    video_prompt = read_json(service, cfg["video_prompt"]["folder_id"], rid)
    video_result = read_json(service, cfg["video_result"]["folder_id"], rid)

    topic = (
        (research or {}).get("topic")
        or (script or {}).get("topic")
        or (storyboard_text or {}).get("topic")
        or rid
    )

    present = [
        name for name, doc in [
            ("research", research), ("script", script), ("storyboard_text", storyboard_text),
            ("storyboard_image", storyboard_image), ("video_prompt", video_prompt), ("video_result", video_result),
        ] if doc
    ]
    print(f"[{rid}] topic={topic!r} — found: {', '.join(present) or '(nothing)'}")

    if dry_run:
        return True

    run = RunRepo(db).create(id=rid, topic=topic, title=topic)
    db.flush()

    if research:
        ResearchRepo(db).save_new_version(rid, payload=research)
        if verbose:
            print(f"    research -> v1 ({len(research.get('summaries', []))} summaries)")

    if script and script.get("script"):
        s = script["script"]
        ScriptRepo(db).save_new_version(
            rid, title=s.get("title", ""), production=s.get("production", {}),
            overview=s.get("overview", []), parts=s.get("parts", []),
        )
        if verbose:
            print(f"    script -> v1 ({len(s.get('parts', []))} parts)")

    # Import every storyboard-category doc as a successive version, in real pipeline order —
    # these categories ARE the successive edits of the same document (text -> +images -> +prompts
    # -> +rendered video), so this gives the migrated run real, meaningful version history instead
    # of collapsing everything into one "current" blob.
    for label, doc in [
        ("storyboard_text", storyboard_text), ("storyboard_image", storyboard_image),
        ("video_prompt", video_prompt), ("video_result", video_result),
    ]:
        if doc and doc.get("storyboard"):
            migrate_storyboard_stage(db, rid, doc["storyboard"], step=label, verbose=verbose)

    db.commit()
    print(f"[{rid}] migrated ✓ (run_id={run.id})")
    return True


def migrate_menu(db, service, cfg: dict, rid: str, *, dry_run: bool, verbose: bool):
    doc = read_json(service, cfg["menu"]["folder_id"], rid)
    if not doc:
        return
    n_refs = len(doc.get("subject_refs", []))
    print(f"[menu {rid}] {doc.get('name', '')!r} — {n_refs} subject refs")
    if dry_run:
        return
    menu_repo = MenuRepo(db)
    menu = menu_repo.ensure_menu(doc["id"], doc.get("name", doc["id"]))
    media = MediaAssetRepo(db)
    for ref in doc.get("subject_refs", []):
        image_asset_id = None
        url = (ref.get("image") or "").strip()
        if url:
            asset = media.create(kind="image", s3_key=media_key_from_url(url), s3_url=url)
            image_asset_id = asset.id
        menu_repo.add_subject_ref(menu.id, name=ref.get("name", ""), kind=ref.get("kind", "ingredient"), image_asset_id=image_asset_id)
    db.commit()


def migrate_brand(db, service, cfg: dict, brand_id: str, *, dry_run: bool, verbose: bool):
    doc = read_json(service, cfg["brand"]["folder_id"], brand_id)
    if not doc:
        return
    print(f"[brand {brand_id}] {doc.get('name', '')!r} — {len(doc.get('scenes', []))} scenes")
    if dry_run:
        return
    if db.get(Brand, doc["id"]) is not None:
        print(f"    brand {doc['id']!r} already exists — skipping")
        return
    brand = Brand(
        id=doc["id"], name=doc.get("name", ""), tagline=doc.get("tagline", ""),
        platform_style=doc.get("platform_style", ""), theme=doc.get("theme", ""), mood=doc.get("mood", ""),
        material_palette=doc.get("material_palette", ""), lighting=doc.get("lighting", ""),
        editing_style=doc.get("editing_style", ""), vo_tone=doc.get("vo_tone", ""), music=doc.get("music", ""),
        camera_movement=doc.get("camera_movement", ""), words_per_second=doc.get("words_per_second", 0.0),
    )
    db.add(brand)
    db.flush()
    media = MediaAssetRepo(db)
    for sc in doc.get("scenes", []):
        image_asset_id = None
        url = (sc.get("image") or "").strip()
        if url:
            asset = media.create(kind="image", s3_key=media_key_from_url(url), s3_url=url)
            image_asset_id = asset.id
        db.add(BrandScene(
            brand_id=brand.id, name=sc.get("name", ""), desc=sc.get("desc", ""),
            image_asset_id=image_asset_id, is_default=bool(sc.get("default", False)),
        ))
    db.commit()


def migrate_director(db, service, cfg: dict, director_id: str, *, dry_run: bool, verbose: bool):
    doc = read_json(service, cfg["director"]["folder_id"], director_id)
    if not doc:
        return
    print(f"[director {director_id}] {doc.get('title', '')!r}")
    if dry_run:
        return
    if db.get(DirectorPrompt, doc["id"]) is not None:
        print(f"    director {doc['id']!r} already exists — skipping")
        return
    created_at = datetime.now(timezone.utc)
    raw_created = doc.get("created_at", "")
    if raw_created:
        try:
            created_at = datetime.fromisoformat(raw_created).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    db.add(DirectorPrompt(
        id=doc["id"], title=doc.get("title", ""), source_url=doc.get("source_url", ""),
        source_title=doc.get("source_title", ""), summary=doc.get("summary", ""),
        sections=doc.get("sections", {}), created_at=created_at,
    ))
    db.commit()


def migrate_active_config(db, service, cfg: dict, *, dry_run: bool, verbose: bool):
    doc = read_json(service, cfg["app_config"]["folder_id"], "active_config")
    if not doc:
        return
    jobs = doc.get("jobs", {})
    print(f"[active_config] {len(jobs)} job pointer(s)")
    if dry_run:
        return
    repo = RunRepo(db)
    for job_id, pointers in jobs.items():
        if not job_id or repo.get(job_id) is None:
            continue  # "" is v1's no-job-selected placeholder; skip pointers for runs we didn't migrate
        repo.set_active_pointers(job_id, brand_id=pointers.get("brand_id") or None, menu_id=pointers.get("menu_id") or None)
        if verbose:
            print(f"    run {job_id} -> brand={pointers.get('brand_id')!r} menu={pointers.get('menu_id')!r}")
    db.commit()


#: The full 77-course production research set — a separate Drive folder from the small
#: hand-tested one in config.yaml's research_drive.research.folder_id (that one only has 3
#: sample topics with script/storyboard/video attached; this one has research for all 77 real
#: courses but nothing past that stage for most of them). Found by following config.yaml's
#: commented-out "# USE" alternate for research + confirming against the real course-list Sheet.
PRODUCTION_RESEARCH_FOLDER = "14M2gYjxg86zcOSIbM4_WdlgEDqYfsiUK"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="List what would be migrated; write nothing.")
    ap.add_argument("--only", nargs="*", help="Only migrate these run ids (default: everything found).")
    ap.add_argument(
        "--no-production-research", action="store_true",
        help="Skip the full 77-course production research folder; only migrate the small test set.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_drive_config()
    service = drive_service(cfg)
    db = SessionLocal()

    try:
        test_rids = set(list_ids(service, cfg["research"]["folder_id"])) | set(list_ids(service, cfg["script"]["folder_id"]))
        prod_rids: set[str] = set()
        if not args.no_production_research:
            prod_rids = set(list_ids(service, PRODUCTION_RESEARCH_FOLDER))

        rids = sorted(test_rids | prod_rids)
        if args.only:
            rids = [r for r in rids if r in args.only]
        print(
            f"Found {len(test_rids)} run(s) in the test set, {len(prod_rids)} in the production "
            f"research set — {len(rids)} total after dedup: {rids}\n"
        )

        for rid in rids:
            # A rid present in BOTH sets keeps its test-set script/storyboard/video (richer data);
            # a production-only rid gets research pulled from the production folder instead.
            override = PRODUCTION_RESEARCH_FOLDER if (rid in prod_rids and rid not in test_rids) else None
            migrate_run(db, service, cfg, rid, dry_run=args.dry_run, verbose=args.verbose, research_folder_id=override)
            print()

        for rid in rids:
            migrate_menu(db, service, cfg, rid, dry_run=args.dry_run, verbose=args.verbose)

        for brand_id in list_ids(service, cfg["brand"]["folder_id"]):
            migrate_brand(db, service, cfg, brand_id, dry_run=args.dry_run, verbose=args.verbose)

        for director_id in list_ids(service, cfg["director"]["folder_id"]):
            migrate_director(db, service, cfg, director_id, dry_run=args.dry_run, verbose=args.verbose)

        migrate_active_config(db, service, cfg, dry_run=args.dry_run, verbose=args.verbose)

        print("\nDone." if not args.dry_run else "\nDry run complete — nothing was written.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
