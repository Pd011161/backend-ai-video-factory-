"""Every generated image/video becomes its own row instead of overwriting a string field
(v1's `StoryboardShot.generated_img` / `generated_video`). Superseding an asset keeps the old
row queryable instead of leaving an orphaned S3 file with no pointer, as v1 did for images,
or silently overwriting in place, as v1 did for videos.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import MediaAsset
from app.repositories.errors import NotFoundError


class MediaAssetRepo:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        *,
        kind: str,
        s3_key: str,
        s3_url: str = "",
        run_id: str | None = None,
        status: str = "generated",
        width: int | None = None,
        height: int | None = None,
        duration_seconds: float | None = None,
        last_frame_s3_url: str | None = None,
        shot_id: int | None = None,
        refs_used: list[dict] | None = None,
    ) -> MediaAsset:
        if kind not in ("image", "video"):
            raise ValueError(f"invalid media kind: {kind!r}")
        asset = MediaAsset(
            kind=kind,
            s3_key=s3_key,
            s3_url=s3_url,
            run_id=run_id,
            status=status,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            last_frame_s3_url=last_frame_s3_url,
            # Which shot this belongs to, and what it was rendered with. Both optional: a ref/brand/
            # menu image belongs to no shot, and only the video render paths know their ref set.
            shot_id=shot_id,
            refs_used=refs_used,
        )
        self.db.add(asset)
        self.db.flush()
        return asset

    def get(self, asset_id: str) -> MediaAsset | None:
        return self.db.get(MediaAsset, asset_id)

    def require(self, asset_id: str) -> MediaAsset:
        asset = self.get(asset_id)
        if asset is None:
            raise NotFoundError("MediaAsset", asset_id)
        return asset

    def set_status(self, asset_id: str, status: str) -> MediaAsset:
        asset = self.require(asset_id)
        asset.status = status
        self.db.flush()
        return asset

    def supersede(self, asset_id: str) -> MediaAsset:
        return self.set_status(asset_id, "superseded")

    def history_for_shot(self, shot_id: int, *, kind: str = "video") -> list[MediaAsset]:
        """Every asset ever rendered for ONE shot, newest first.

        The shot only ever points at two of these (current + one step of undo), so before assets
        carried a shot_id the third-most-recent clip was unreachable even though its row and its S3
        bytes both still existed — clip keys carry a timestamp, so nothing was overwritten.
        """
        stmt = (select(MediaAsset)
                .where(MediaAsset.shot_id == shot_id, MediaAsset.kind == kind)
                .order_by(MediaAsset.created_at.desc(), MediaAsset.id.desc()))
        return list(self.db.scalars(stmt))

    def history_for_run(self, run_id: str, *, kind: str | None = None) -> list[MediaAsset]:
        stmt = select(MediaAsset).where(MediaAsset.run_id == run_id).order_by(MediaAsset.created_at.desc())
        if kind is not None:
            stmt = stmt.where(MediaAsset.kind == kind)
        return list(self.db.scalars(stmt))
