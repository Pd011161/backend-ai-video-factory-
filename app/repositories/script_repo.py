"""Replaces the `script` Drive-JSON category (app/api/routes.py's generic /drive/script/* routes).

A ScriptDocument is now a real versioned row: `regenerate_part` (or any full-document save)
inserts a new version instead of overwriting the one Drive JSON file per run, so a bad
regenerate is one query away from being reverted via get_version()/list_versions().
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ScriptDocument, ScriptPart, ScriptScene
from app.repositories.errors import NotFoundError
from app.repositories.retention import KEEP_VERSIONS, prune_old_versions
from app.repositories.storyboard_repo import StoryboardRepo


class ScriptRepo:
    def __init__(self, db: Session):
        self.db = db

    def save_new_version(
        self,
        run_id: str,
        *,
        title: str,
        production: dict,
        overview: list[dict],
        parts: list[dict],
        script_config: dict | None = None,
    ) -> ScriptDocument:
        """`parts` = [{number, title, description, duration, scenes: [{scene_id, name, ...}]}].
        `script_config` is the full merged ScriptConfig used for this generation (None for callers
        that don't have it, e.g. a future part-level edit save — defaults to {})."""
        current = self.get_current(run_id)
        if current is not None:
            current.is_current = False
        # Highest version ever issued for this run, not current.version + 1 — `activate_version`
        # can make an OLDER version current again, and `current.version + 1` would then re-issue a
        # number that already exists and trip uq_script_documents_run_version.
        highest = self.db.scalar(
            select(ScriptDocument.version)
            .where(ScriptDocument.run_id == run_id)
            .order_by(ScriptDocument.version.desc())
            .limit(1)
        )
        next_version = (highest + 1) if highest is not None else 1

        doc = ScriptDocument(
            run_id=run_id,
            version=next_version,
            is_current=True,
            title=title,
            production=production,
            overview=overview,
            script_config=script_config or {},
        )
        for part_data in parts:
            scenes_data = part_data.get("scenes", [])
            part = ScriptPart(
                number=part_data["number"],
                title=part_data.get("title", ""),
                description=part_data.get("description", ""),
                duration=part_data.get("duration", ""),
            )
            part.scenes = [
                ScriptScene(
                    scene_id=s["scene_id"],
                    name=s.get("name", ""),
                    transition=s.get("transition", ""),
                    timecode_start=s.get("timecode_start", ""),
                    timecode_end=s.get("timecode_end", ""),
                    shot_type=s.get("shot_type", ""),
                    visual_direction=s.get("visual_direction", ""),
                    voice_over=s.get("voice_over", ""),
                    on_screen_text=s.get("on_screen_text", ""),
                    key_message=s.get("key_message", ""),
                    music=s.get("music", ""),
                )
                for s in scenes_data
            ]
            doc.parts.append(part)

        self.db.add(doc)
        self.db.flush()
        # A brand-new script version has no storyboard yet, so this deactivates every board: the
        # run correctly has nothing current until one is generated. Doing it here rather than in
        # the routes covers the generate, the per-part regenerate and the manual save alike — all
        # three mint a version, and a board built from the previous text is not a board for this
        # one. The old board is not deleted; activating its script version brings it back.
        StoryboardRepo(self.db).activate_for_script(run_id, doc.version)
        self._prune_with_storyboards(run_id)
        return doc

    def _prune_with_storyboards(self, run_id: str) -> None:
        """Prune old script versions AND the storyboards that belonged to them.

        Storyboards record `script_version`, so a pruned script would otherwise leave boards
        grouped under a version that no longer exists — reachable by nothing, counting against
        nothing. Operator's call: they go with their script. Their shot rows cascade; the
        media_assets they referenced are not deleted (nothing ever deletes those), but the last
        link to those files goes with the boards.
        """
        surviving = {d.version for d in self.list_versions(run_id)[:KEEP_VERSIONS]}
        doomed = [d.version for d in self.list_versions(run_id) if d.version not in surviving]
        prune_old_versions(self.db, ScriptDocument, run_id)
        for version in doomed:
            for board in StoryboardRepo(self.db).list_versions(run_id, script_version=version):
                self.db.delete(board)   # ORM delete so scenes → shots → plans cascade
        if doomed:
            self.db.flush()

    def get_current(self, run_id: str) -> ScriptDocument | None:
        stmt = select(ScriptDocument).where(ScriptDocument.run_id == run_id, ScriptDocument.is_current.is_(True))
        return self.db.scalars(stmt).first()

    def require_current(self, run_id: str) -> ScriptDocument:
        doc = self.get_current(run_id)
        if doc is None:
            raise NotFoundError("ScriptDocument", run_id)
        return doc

    def get_version(self, run_id: str, version: int) -> ScriptDocument | None:
        stmt = select(ScriptDocument).where(ScriptDocument.run_id == run_id, ScriptDocument.version == version)
        return self.db.scalars(stmt).first()

    def activate_version(self, run_id: str, version: int) -> ScriptDocument | None:
        """Make an existing script version current again.

        Storyboards belong to a script version, so the caller must re-point the storyboard too —
        see StoryboardRepo.activate_for_script, which the route calls right after this.
        """
        target = self.get_version(run_id, version)
        if target is None:
            return None
        for doc in self.list_versions(run_id):
            doc.is_current = doc.id == target.id
        self.db.flush()
        return target

    def list_versions(self, run_id: str) -> list[ScriptDocument]:
        stmt = select(ScriptDocument).where(ScriptDocument.run_id == run_id).order_by(ScriptDocument.version.desc())
        return list(self.db.scalars(stmt))

    @staticmethod
    def to_dict(doc: ScriptDocument) -> dict:
        """Round-trips back to the shape the frontend's ScriptDocument model expects."""
        return {
            "title": doc.title,
            "production": doc.production,
            "overview": doc.overview,
            "parts": [
                {
                    "number": part.number,
                    "title": part.title,
                    "description": part.description,
                    "duration": part.duration,
                    "scenes": [
                        {
                            "scene_id": scene.scene_id,
                            "name": scene.name,
                            "transition": scene.transition,
                            "timecode_start": scene.timecode_start,
                            "timecode_end": scene.timecode_end,
                            "shot_type": scene.shot_type,
                            "visual_direction": scene.visual_direction,
                            "voice_over": scene.voice_over,
                            "on_screen_text": scene.on_screen_text,
                            "key_message": scene.key_message,
                            "music": scene.music,
                        }
                        for scene in part.scenes
                    ],
                }
                for part in doc.parts
            ],
        }
