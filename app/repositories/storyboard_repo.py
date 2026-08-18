"""Replaces the `storyboard_text`/`storyboard_image`/`video_prompt`/`video_result` Drive-JSON
categories — in v1 these were four separate Drive files re-saved at each pipeline stage; here
they're one versioned StoryboardDocument per run (save_new_version()), plus MediaAssetRepo
tracks each shot's generated image/video as its own row instead of a plain string field.
"""
from __future__ import annotations

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    ShotImagePlan,
    ShotPromptVideoVersion,
    ShotRefUsed,
    ShotVideoRef,
    StoryboardDocument,
    StoryboardScene,
    StoryboardShot,
)
from app.repositories.errors import NotFoundError
from app.repositories.media_asset_repo import MediaAssetRepo
from app.repositories.retention import prune_old_versions


class StoryboardRepo:
    def __init__(self, db: Session):
        self.db = db
        self._media = MediaAssetRepo(db)

    def save_new_version(
        self,
        run_id: str,
        *,
        title: str,
        ingredients: list[str],
        equipment: list[str],
        production: dict,
        scenes: list[dict],
        media: str = "keep",
    ) -> StoryboardDocument:
        """`scenes` = [{scene_id, name, transition_in, music, shots: [{no, ...}]}].

        `media` — what happens to the previous version's per-shot renders:
          "keep"       carry them onto the matching (scene_id, no) — for a text EDIT, where the
                       shot is still the same shot. The default, so every existing caller and test
                       keeps its behaviour.
          "drop_all"   start clean — for a storyboard/image-prompt REGENERATE, where the shot at a
                       given (scene_id, no) is often a different shot entirely. Carrying renders
                       forward there left images that contradicted their own prompt (measured on
                       one real regenerate: 43 of 68 shots).
          "drop_video" keep the image, drop the clip — for a video-prompt regenerate, where the
                       image is still valid but the clip came from the prompt being replaced.

        Nothing is deleted either way: the previous version keeps its own copies of the asset ids
        and `activate_version` can bring them back.
        """
        keep_image = media in ("keep", "drop_video")
        keep_video = media == "keep"
        # Read here rather than taken as an argument: activating a script version already makes
        # "the current script" the server's own truth, so no caller (5 of them) has to thread a
        # version number through — and none of them could, since the client never learns one.
        from app.repositories.script_repo import ScriptRepo   # local: ScriptRepo imports nothing here
        script = ScriptRepo(self.db).get_current(run_id)
        script_version = script.version if script is not None else None
        current = self.get_current(run_id)
        # index the about-to-be-superseded version's shots by (scene_id, no) so newly-built shots
        # below can carry forward media/plan links a plain scenes-dict rebuild would otherwise drop
        prev_shots: dict[tuple[str, int], StoryboardShot] = {}
        if current is not None:
            for prev_scene in current.scenes:
                for prev_shot in prev_scene.shots:
                    prev_shots[(prev_scene.scene_id, prev_shot.no)] = prev_shot
            current.is_current = False
        # Highest version ever issued for this run, not current.version + 1 — `activate_version`
        # can make an OLDER version current again, and `current.version + 1` would then re-issue a
        # number that already exists and trip uq_storyboard_documents_run_version.
        highest = self.db.scalar(
            select(StoryboardDocument.version)
            .where(StoryboardDocument.run_id == run_id)
            .order_by(StoryboardDocument.version.desc())
            .limit(1)
        )
        next_version = (highest + 1) if highest is not None else 1

        doc = StoryboardDocument(
            run_id=run_id,
            version=next_version,
            script_version=script_version,
            is_current=True,
            title=title,
            ingredients=ingredients,
            equipment=equipment,
            production=production,
        )
        for scene_data in scenes:
            scene = StoryboardScene(
                scene_id=scene_data["scene_id"],
                name=scene_data.get("name", ""),
                transition_in=scene_data.get("transition_in", "cut"),
                music=scene_data.get("music", ""),
            )
            new_shots = []
            for shot_data in scene_data.get("shots", []):
                shot = self._build_shot(shot_data)
                prev_shot = prev_shots.get((scene.scene_id, shot.no))
                if prev_shot is not None:
                    if keep_image:
                        shot.current_image_asset_id = prev_shot.current_image_asset_id
                        shot.prev_image_asset_id = prev_shot.prev_image_asset_id   # undo survives a save
                        # The user attached these by hand and the authored prompt_video cites them
                        # by tag — dropping them on a video-prompt regenerate would leave the new
                        # prompt referencing images that are no longer sent.
                        shot.video_refs = [
                            ShotVideoRef(url=r.url, note=r.note, kind=r.kind, position=r.position)
                            for r in prev_shot.video_refs
                        ]
                    if keep_video:
                        shot.current_video_asset_id = prev_shot.current_video_asset_id
                        # the undo target and the open Omni edit conversation both belong to the
                        # clip above — carrying them without it would point at nothing
                        shot.prev_video_asset_id = prev_shot.prev_video_asset_id
                        shot.omni_interaction_id = prev_shot.omni_interaction_id
                new_shots.append(shot)
            scene.shots = new_shots
            doc.scenes.append(scene)

        self.db.add(doc)
        self.db.flush()

        # image plans are keyed by shot_id (FK), so they need re-pointing at the new shot rows too.
        # Skipped entirely when the image is dropped: the plan describes the refs, the QC verdict
        # and the exact prompt of a render this version no longer has.
        for scene_data, scene in zip(scenes, doc.scenes if keep_image else []):
            for shot_data, shot in zip(scene_data.get("shots", []), scene.shots):
                prev_shot = prev_shots.get((scene.scene_id, shot.no))
                if prev_shot is None:
                    continue
                prev_plan = self.get_image_plan(prev_shot.id)
                if prev_plan is None:
                    continue
                self.upsert_image_plan(
                    shot.id,
                    refs_used=[
                        {"kind": r.kind, "label": r.label, "url": r.url, "source": r.source}
                        for r in prev_plan.refs_used
                    ],
                    classified=prev_plan.classified,
                    has_human=prev_plan.has_human,
                    has_ingredients=prev_plan.has_ingredients,
                    is_process=prev_plan.is_process,
                    shows_dish=prev_plan.shows_dish,
                    shows_kitchen=prev_plan.shows_kitchen,
                    shot_scale=prev_plan.shot_scale,
                    same_framing_as_prev=prev_plan.same_framing_as_prev,
                    scene_match=prev_plan.scene_match,
                    reuse_prev=prev_plan.reuse_prev,
                    image_generate_type=prev_plan.image_generate_type,
                    ingredient_changed=prev_plan.ingredient_changed,
                    ingredient_change=prev_plan.ingredient_change,
                    equipment_changed=prev_plan.equipment_changed,
                    equipment_change=prev_plan.equipment_change,
                    process_changed=prev_plan.process_changed,
                    process_change=prev_plan.process_change,
                    ref_keywords=prev_plan.ref_keywords,
                    qc_passed=prev_plan.qc_passed,
                    qc_attempts=prev_plan.qc_attempts,
                    qc_issues=prev_plan.qc_issues,
                    full_prompt=prev_plan.full_prompt,
                )
        self.db.flush()
        # Media survives pruning: assets live in media_assets (separate table, referenced by id) and
        # this new current version just re-linked them via the carry-forward above.
        # Per script version, not per run: each script keeps its own 3 boards, so switching back to
        # an earlier script still finds its history instead of it having been evicted by boards
        # belonging to a different script.
        prune_old_versions(self.db, StoryboardDocument, run_id, script_version=script_version)
        return doc

    @staticmethod
    def _build_shot(shot_data: dict) -> StoryboardShot:
        # `shot_data` is LLM-generated JSON, so a numeric field can arrive as "3" or 3.0 and a
        # boolean as 1. SQLite is dynamically typed and stored whatever it was given; PostgreSQL
        # rejects it with DataError mid-save. Coerce here rather than trusting the model's output.
        def _int(v, default=0):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return default

        def _float(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        return StoryboardShot(
            no=_int(shot_data["no"]),
            time=shot_data.get("time", ""),
            motion_description=shot_data.get("motion_description", ""),
            voice_over=shot_data.get("voice_over", ""),
            on_screen_text=shot_data.get("on_screen_text", ""),
            key_message=shot_data.get("key_message", ""),
            shot_kind=shot_data.get("shot_kind", "person"),
            join_with_prev=shot_data.get("join_with_prev", "cut"),
            screen_direction=shot_data.get("screen_direction", ""),
            prompt_img=shot_data.get("prompt_img", ""),
            prompt_full=shot_data.get("prompt_full", ""),
            dish_state=shot_data.get("dish_state", ""),
            prompt_video=shot_data.get("prompt_video", ""),
            target_seconds=_float(shot_data.get("target_seconds", 0.0), 0.0),
            speed=_float(shot_data.get("speed", 1.0), 1.0),
            ken_burns=bool(shot_data.get("ken_burns", False)),
            ingredient_refs=shot_data.get("ingredient_refs", []),
            equipment_refs=shot_data.get("equipment_refs", []),
            image_subjects=shot_data.get("image_subjects", []),
        )

    def get_current(self, run_id: str) -> StoryboardDocument | None:
        stmt = select(StoryboardDocument).where(
            StoryboardDocument.run_id == run_id, StoryboardDocument.is_current.is_(True)
        )
        return self.db.scalars(stmt).first()

    def require_current(self, run_id: str) -> StoryboardDocument:
        doc = self.get_current(run_id)
        if doc is None:
            raise NotFoundError("StoryboardDocument", run_id)
        return doc

    def get_version(self, run_id: str, version: int) -> StoryboardDocument | None:
        stmt = select(StoryboardDocument).where(
            StoryboardDocument.run_id == run_id, StoryboardDocument.version == version
        )
        return self.db.scalars(stmt).first()

    def activate_version(self, run_id: str, version: int) -> StoryboardDocument | None:
        """Make an existing version current again — text AND its own generated images/videos.

        Each version owns its shot rows, and those rows hold their own copies of
        current_image_asset_id / current_video_asset_id, so a version is already a complete
        snapshot: switching back to it restores the renders that were made while it was current.
        Deliberately does NOT create a new version — switching would otherwise burn a slot of the
        3-version retention window, which is the very history this is meant to reach into.
        """
        target = self.get_version(run_id, version)
        if target is None:
            return None
        for doc in self.list_versions(run_id):
            doc.is_current = doc.id == target.id
        self.db.flush()
        return target

    def activate_for_script(self, run_id: str, script_version: int | None) -> StoryboardDocument | None:
        """Point the run at the newest storyboard belonging to `script_version` — or at none.

        Called whenever the current script changes. A script version that has never been broken
        down has no board, and then EVERY board is deactivated: `require_current` raises, the API
        answers 404, and Step 3 correctly shows nothing to build on. That is what makes
        "regenerate the script and the storyboard is gone" a fact about the data rather than a
        piece of page state that a reload would undo.
        """
        boards = self.list_versions(run_id)          # newest first
        target = next((d for d in boards if d.script_version == script_version), None)
        for doc in boards:
            doc.is_current = target is not None and doc.id == target.id
        self.db.flush()
        return target

    def list_versions(self, run_id: str, script_version: int | None = None) -> list[StoryboardDocument]:
        """Newest first. Pass `script_version` to see only that script's boards — the history bar
        does, so it never offers a board built from a different script's text."""
        stmt = select(StoryboardDocument).where(StoryboardDocument.run_id == run_id)
        if script_version is not None:
            stmt = stmt.where(StoryboardDocument.script_version == script_version)
        return list(self.db.scalars(stmt.order_by(StoryboardDocument.version.desc())))

    # -- shot-level media (replaces overwriting generated_img/generated_video strings) ---------

    def get_shot(self, shot_id: int) -> StoryboardShot | None:
        return self.db.get(StoryboardShot, shot_id)

    def require_shot(self, shot_id: int) -> StoryboardShot:
        shot = self.get_shot(shot_id)
        if shot is None:
            raise NotFoundError("StoryboardShot", shot_id)
        return shot

    def get_prev_shot(self, shot_id: int) -> StoryboardShot | None:
        """The shot right before this one in the SAME scene. Callers that walk a whole storyboard
        already have the predecessor in hand; this is for the ones handed a single shot (per-shot
        prompt regen sends a one-shot payload) and would otherwise see no previous shot at all.
        ponytail: in-scene only — StoryboardScene has no ordering column, so a join that crosses a
        scene boundary still resolves to None, exactly as the payload walk does."""
        shot = self.get_shot(shot_id)
        if shot is None:
            return None
        stmt = (
            select(StoryboardShot)
            .where(
                StoryboardShot.storyboard_scene_id == shot.storyboard_scene_id,
                StoryboardShot.no < shot.no,
            )
            .order_by(StoryboardShot.no.desc())
            .limit(1)
        )
        return self.db.scalars(stmt).first()

    def _redirect_to_current(self, shot: StoryboardShot) -> StoryboardShot:
        """The CURRENT version's row for the same (scene_id, no) — or the given row when it already
        is current, or when no current counterpart exists.

        The frontend holds shot ids of one specific version's rows, every save creates brand-new
        rows, and a finished render reports back with whatever id the page was loaded with — so the
        id is stale the moment any save has happened in between. Writing the pointer onto a
        superseded version's row loses it silently: the next save carries pointers forward from the
        CURRENT version only. That is exactly how a tunnel drop plus one save cost run V.4.2 seven
        rendered images. The (scene_id, no) match is the same key save_new_version's carry-forward
        uses, so the two agree about which shot is "the same shot".

        A shot with no counterpart in the current version (deleted, renumbered) keeps the stale row,
        with a warning: the render is already paid for, and parked on the old version it is still
        recoverable — dropped it is not."""
        doc = shot.storyboard_scene.storyboard_document
        if doc.is_current:
            return shot
        current = self.get_current(doc.run_id)
        if current is not None:
            for scene in current.scenes:
                if scene.scene_id == shot.storyboard_scene.scene_id:
                    for cur in scene.shots:
                        if cur.no == shot.no:
                            return cur
        logger.warning(f"shot {shot.id} ({shot.storyboard_scene.scene_id}.{shot.no}) belongs to "
                       f"superseded board v{doc.version} of run {doc.run_id!r} and has no current "
                       f"counterpart — writing to the stale row")
        return shot

    def set_shot_image(self, shot_id: int, media_asset_id: str) -> StoryboardShot:
        """Point the shot at a new rendered image, keeping the previous one as history
        (status='superseded') instead of orphaning it the way v1's timestamp-suffixed
        S3 keys did."""
        shot = self._redirect_to_current(self.require_shot(shot_id))
        if shot.current_image_asset_id and shot.current_image_asset_id != media_asset_id:
            self._media.supersede(shot.current_image_asset_id)
            # One step back, same as the clip side: a regenerate, a quick edit, an outpaint/mask
            # edit, a crop and an upload all land here, so all of them become undoable.
            shot.prev_image_asset_id = shot.current_image_asset_id
        shot.current_image_asset_id = media_asset_id
        self.db.flush()
        return shot

    def undo_shot_image(self, shot_id: int) -> StoryboardShot | None:
        """Swap the shot's image with the one it replaced. Symmetric, so calling it again redoes.
        None when there is nothing to go back to — a first render, or any shot last replaced before
        this pointer existed."""
        shot = self._redirect_to_current(self.require_shot(shot_id))
        prev = shot.prev_image_asset_id
        if not prev:
            return None
        shot.prev_image_asset_id = shot.current_image_asset_id
        shot.current_image_asset_id = prev
        self._media.set_status(prev, "generated")
        if shot.prev_image_asset_id:
            self._media.supersede(shot.prev_image_asset_id)
        self.db.flush()
        return shot

    def set_shot_video(self, shot_id: int, media_asset_id: str, *, keep_omni_chain: bool = False) -> StoryboardShot:
        """Same as set_shot_image, but for video — v1 overwrote the video at a fixed S3 key with
        no history at all; here the previous render is kept as a superseded row.

        `keep_omni_chain`: a from-scratch render (Veo, Omni generate, upload) makes this a NEW base
        clip → any Omni edit conversation on the OLD clip no longer applies, so omni_interaction_id
        is cleared. An Omni EDIT call also lands here (the edited clip becomes current) but wants to
        KEEP the conversation going — pass True there (see /steps/video/edit)."""
        shot = self._redirect_to_current(self.require_shot(shot_id))
        if shot.current_video_asset_id and shot.current_video_asset_id != media_asset_id:
            self._media.supersede(shot.current_video_asset_id)
            shot.prev_video_asset_id = shot.current_video_asset_id   # one step of undo
        shot.current_video_asset_id = media_asset_id
        if not keep_omni_chain:
            shot.omni_interaction_id = None
        self.db.flush()
        return shot

    def undo_shot_video(self, shot_id: int) -> StoryboardShot | None:
        """Swap the shot's clip with the one it replaced. Symmetric, so calling it again redoes.
        Clears omni_interaction_id: that conversation produced the clip we just stepped away from,
        so continuing it would branch off the version the user discarded — the next edit should
        start fresh from whatever clip is now current. None when there is nothing to go back to."""
        shot = self.require_shot(shot_id)
        prev = shot.prev_video_asset_id
        if not prev:
            return None
        shot.prev_video_asset_id = shot.current_video_asset_id
        shot.current_video_asset_id = prev
        self._media.set_status(prev, "generated")
        if shot.prev_video_asset_id:
            self._media.supersede(shot.prev_video_asset_id)
        shot.omni_interaction_id = None
        self.db.flush()
        return shot

    def set_omni_interaction_id(self, shot_id: int, interaction_id: str | None) -> StoryboardShot:
        shot = self.require_shot(shot_id)
        shot.omni_interaction_id = interaction_id
        self.db.flush()
        return shot

    # -- image plan / QC (1:1 per shot) ---------------------------------------------------------

    def upsert_image_plan(self, shot_id: int, *, refs_used: list[dict] | None = None, **fields) -> ShotImagePlan:
        # Redirected like set_shot_image: the plan must land beside the image it explains.
        shot_id = self._redirect_to_current(self.require_shot(shot_id)).id
        stmt = select(ShotImagePlan).where(ShotImagePlan.shot_id == shot_id)
        plan = self.db.scalars(stmt).first()
        if plan is None:
            plan = ShotImagePlan(shot_id=shot_id)
            self.db.add(plan)

        allowed = {
            "classified", "has_human", "has_ingredients", "is_process", "shows_dish", "shows_kitchen",
            "shot_scale", "same_framing_as_prev", "scene_match", "reuse_prev", "image_generate_type",
            "ingredient_changed", "ingredient_change", "equipment_changed", "equipment_change",
            "process_changed", "process_change", "ref_keywords", "qc_passed", "qc_attempts", "qc_issues",
            "full_prompt",
        }
        for key, value in fields.items():
            if key in allowed:
                setattr(plan, key, value)

        if refs_used is not None:
            plan.refs_used = [
                ShotRefUsed(kind=r.get("kind", ""), label=r.get("label", ""), url=r.get("url", ""), source=r.get("source", ""))
                for r in refs_used
            ]

        self.db.flush()
        return plan

    def get_image_plan(self, shot_id: int) -> ShotImagePlan | None:
        stmt = select(ShotImagePlan).where(ShotImagePlan.shot_id == shot_id)
        return self.db.scalars(stmt).first()

    # ── video: user-attached refs + prompt history ───────────────────────────────────────────
    # Both are per-SHOT rather than per-document: the image side proved that a user's own reference
    # and a hand-tuned prompt are things they expect to survive a regenerate of the step, and the
    # per-shot save path (PUT /shots/{id}/prompt_video) deliberately writes no new document version.

    KEEP_PROMPT_VIDEO_VERSIONS = 10

    def list_video_refs(self, shot_id: int) -> list[ShotVideoRef]:
        stmt = select(ShotVideoRef).where(ShotVideoRef.shot_id == shot_id).order_by(ShotVideoRef.position)
        return list(self.db.scalars(stmt))

    def set_video_refs(self, shot_id: int, refs: list[dict]) -> list[ShotVideoRef]:
        """Replace the shot's video refs wholesale (same shape as upsert_image_plan's refs_used).
        Order is meaningful: it decides each attached ref's @ImageN tag at render.

        Rows are either an attached ref (kind "user") or the user's answer for an auto slot
        (kind "person"/"kitchen") — see ShotVideoRef."""
        self.require_shot(shot_id)
        for row in self.list_video_refs(shot_id):
            self.db.delete(row)
        self.db.flush()
        # Numbered after dropping the blanks, so positions stay contiguous — they become @ImageN
        # offsets at render and a gap would be confusing to read in the DB. An empty url survives on
        # a SLOT row, where it is the instruction "remove this auto ref"; on a 'user' row it is just
        # a blank the editor left behind.
        kept = [r for r in refs
                if (r.get("url") or "").strip() or (r.get("kind") or "user") != "user"]
        rows = [
            ShotVideoRef(shot_id=shot_id, url=(r.get("url") or "").strip(),
                         note=(r.get("note") or "").strip(),
                         kind=(r.get("kind") or "user").strip() or "user", position=i)
            for i, r in enumerate(kept)
        ]
        for row in rows:
            self.db.add(row)
        self.db.flush()
        return rows

    def log_prompt_video(self, shot_id: int, prompt_video: str, *, source: str = "manual") -> None:
        """Record the prompt_video being REPLACED, so an overwrite is recoverable. No-op for an empty
        string (nothing to lose) or when it matches the newest entry (a blur that changed nothing)."""
        text = (prompt_video or "").strip()
        if not text:
            return
        newest = self.db.scalars(
            select(ShotPromptVideoVersion)
            .where(ShotPromptVideoVersion.shot_id == shot_id)
            .order_by(ShotPromptVideoVersion.id.desc())
            .limit(1)
        ).first()
        if newest is not None and newest.prompt_video == text:
            return
        self.db.add(ShotPromptVideoVersion(shot_id=shot_id, prompt_video=text, source=source))
        self.db.flush()
        rows = self.list_prompt_video_versions(shot_id)
        for old in rows[self.KEEP_PROMPT_VIDEO_VERSIONS:]:
            self.db.delete(old)
        self.db.flush()

    def list_prompt_video_versions(self, shot_id: int) -> list[ShotPromptVideoVersion]:
        """Newest first."""
        stmt = (select(ShotPromptVideoVersion)
                .where(ShotPromptVideoVersion.shot_id == shot_id)
                .order_by(ShotPromptVideoVersion.id.desc()))
        return list(self.db.scalars(stmt))
