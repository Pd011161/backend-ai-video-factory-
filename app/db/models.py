"""SQLAlchemy models for the SQLite persistence layer (replaces Drive+JSON stores from v1).

Mirrors the ER diagram in the v2 migration plan. Fields that are always read/written
whole and never filtered on (production specs, QC issue lists, director sections) are
kept as JSON columns; fields used in joins/filters (scene_id, status, kind, is_current,
brand/menu FKs) are real columns.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    """Serialize a stored timestamp identically whichever database is behind us.

    Everything here is written as UTC by `_now`, but SQLite's DATETIME drops the offset and hands
    back a NAIVE datetime while PostgreSQL returns an aware one — so a plain `.isoformat()` would
    change the API's output the day someone switches DATABASE_URL. Reattaching the UTC that a naive
    value already represents keeps both backends on `...+00:00`, which is also what lets the client
    render local time instead of showing UTC digits with no label."""
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Runs — the promoted first-class job/project (replaces Sheet-row-id + active_config.json)
# ---------------------------------------------------------------------------
class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="research")

    # Editable free text, blank until something needs them. `type` is meant for classifying a run
    # (recipe / review / …) and `avatar` for whichever presenter it belongs to; neither is read by
    # the pipeline yet, so they carry no constraint or FK until their meaning is settled.
    type: Mapped[str] = mapped_column(String, default="", server_default="")
    avatar: Mapped[str] = mapped_column(String, default="", server_default="")

    # ON DELETE SET NULL: deleting a brand/menu/director must not be blocked by old runs that used
    # it, and the run itself stays valid without one. SQLite never enforced these at all, so the
    # delete routes appeared to work while quietly leaving dangling ids; PostgreSQL does enforce
    # them, and without a rule here those three DELETE endpoints would start returning 500.
    brand_id: Mapped[str | None] = mapped_column(ForeignKey("brands.id", ondelete="SET NULL"), nullable=True)
    menu_id: Mapped[str | None] = mapped_column(ForeignKey("menus.id", ondelete="SET NULL"), nullable=True)
    director_prompt_id: Mapped[str | None] = mapped_column(
        ForeignKey("director_prompts.id", ondelete="SET NULL"), nullable=True)
    # NULL = "use whichever character is the default", not "no host" — see CharacterRepo.resolve.
    character_id: Mapped[str | None] = mapped_column(
        ForeignKey("characters.id", ondelete="SET NULL"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    brand: Mapped["Brand | None"] = relationship()
    menu: Mapped["Menu | None"] = relationship()
    director_prompt: Mapped["DirectorPrompt | None"] = relationship()
    character: Mapped["Character | None"] = relationship()

    research_documents: Mapped[list["ResearchDocument"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    script_documents: Mapped[list["ScriptDocument"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    storyboard_documents: Mapped[list["StoryboardDocument"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    media_assets: Mapped[list["MediaAsset"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    usage_records: Mapped[list["UsageRecord"]] = relationship(back_populates="run", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Research documents (versioned JSON payload — not heavily normalized, rarely queried by sub-field)
# ---------------------------------------------------------------------------
class ResearchDocument(Base):
    __tablename__ = "research_documents"
    __table_args__ = (UniqueConstraint("run_id", "version", name="uq_research_documents_run_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped["Run"] = relationship(back_populates="research_documents")


# ---------------------------------------------------------------------------
# Script documents / parts / scenes
# ---------------------------------------------------------------------------
class ScriptDocument(Base):
    __tablename__ = "script_documents"
    __table_args__ = (UniqueConstraint("run_id", "version", name="uq_script_documents_run_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    title: Mapped[str] = mapped_column(String, default="")
    production: Mapped[dict] = mapped_column(JSON, default=dict)  # ProductionSpec
    overview: Mapped[list] = mapped_column(JSON, default=list)  # list[PartOverview]
    # The full ScriptConfig actually used for this generation (merged config.yaml + request
    # overrides) — so version history can show exactly what settings (duration_mode, director,
    # word targets, ...) produced each version, not just the LLM-echoed subset in `production`.
    script_config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped["Run"] = relationship(back_populates="script_documents")
    parts: Mapped[list["ScriptPart"]] = relationship(back_populates="script_document", cascade="all, delete-orphan", order_by="ScriptPart.number")


class ScriptPart(Base):
    __tablename__ = "script_parts"
    __table_args__ = (UniqueConstraint("script_document_id", "number", name="uq_script_parts_document_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    script_document_id: Mapped[int] = mapped_column(ForeignKey("script_documents.id"), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String, default="")
    description: Mapped[str] = mapped_column(String, default="")
    duration: Mapped[str] = mapped_column(String, default="")

    script_document: Mapped["ScriptDocument"] = relationship(back_populates="parts")
    # order_by id, NOT scene_id: scene_id is a String, so ordering by it is a character sort that
    # puts "1.10" before "1.2" — and everything downstream (storyboard generation walks these
    # scenes) inherited that order. id is insertion order = the story order the LLM wrote.
    scenes: Mapped[list["ScriptScene"]] = relationship(back_populates="script_part", cascade="all, delete-orphan", order_by="ScriptScene.id")


class ScriptScene(Base):
    __tablename__ = "script_scenes"
    __table_args__ = (UniqueConstraint("script_part_id", "scene_id", name="uq_script_scenes_part_scene_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    script_part_id: Mapped[int] = mapped_column(ForeignKey("script_parts.id"), nullable=False)
    scene_id: Mapped[str] = mapped_column(String, nullable=False)  # natural key, e.g. "1.1"
    name: Mapped[str] = mapped_column(String, default="")
    transition: Mapped[str] = mapped_column(String, default="")
    timecode_start: Mapped[str] = mapped_column(String, default="")
    timecode_end: Mapped[str] = mapped_column(String, default="")
    shot_type: Mapped[str] = mapped_column(String, default="")
    visual_direction: Mapped[str] = mapped_column(String, default="")
    voice_over: Mapped[str] = mapped_column(String, default="")
    on_screen_text: Mapped[str] = mapped_column(String, default="")
    key_message: Mapped[str] = mapped_column(String, default="")
    music: Mapped[str] = mapped_column(String, default="")

    script_part: Mapped["ScriptPart"] = relationship(back_populates="scenes")


# ---------------------------------------------------------------------------
# Storyboard documents / scenes / shots / image plans
# ---------------------------------------------------------------------------
class StoryboardDocument(Base):
    __tablename__ = "storyboard_documents"
    __table_args__ = (UniqueConstraint("run_id", "version", name="uq_storyboard_documents_run_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Which ScriptDocument.version this board was broken down from. Version numbers stay global per
    # run (v1..vN in creation order) but each belongs to one script version, so going back to an
    # earlier script brings back its own storyboards: script v1 → boards v2, v3, v6.
    # A plain int, not a FK: it matches how the operator reads the history, and the delete side is
    # one explicit step in ScriptRepo rather than a cascade reaching across two repositories.
    # NULL only for a board saved on a run that has no script at all.
    script_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    title: Mapped[str] = mapped_column(String, default="")
    ingredients: Mapped[list] = mapped_column(JSON, default=list)
    equipment: Mapped[list] = mapped_column(JSON, default=list)
    production: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped["Run"] = relationship(back_populates="storyboard_documents")
    scenes: Mapped[list["StoryboardScene"]] = relationship(back_populates="storyboard_document", cascade="all, delete-orphan")


class StoryboardScene(Base):
    __tablename__ = "storyboard_scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    storyboard_document_id: Mapped[int] = mapped_column(ForeignKey("storyboard_documents.id"), nullable=False)
    # Soft reference to ScriptScene.scene_id by value (not a DB FK): script and storyboard
    # documents version independently, same as v1's string-matched scene_id linkage.
    scene_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, default="")
    transition_in: Mapped[str] = mapped_column(String, default="cut")
    music: Mapped[str] = mapped_column(String, default="")

    storyboard_document: Mapped["StoryboardDocument"] = relationship(back_populates="scenes")
    shots: Mapped[list["StoryboardShot"]] = relationship(back_populates="storyboard_scene", cascade="all, delete-orphan", order_by="StoryboardShot.no")


class StoryboardShot(Base):
    __tablename__ = "storyboard_shots"
    __table_args__ = (UniqueConstraint("storyboard_scene_id", "no", name="uq_storyboard_shots_scene_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    storyboard_scene_id: Mapped[int] = mapped_column(ForeignKey("storyboard_scenes.id"), nullable=False)
    no: Mapped[int] = mapped_column(Integer, nullable=False)

    time: Mapped[str] = mapped_column(String, default="")
    motion_description: Mapped[str] = mapped_column(String, default="")
    voice_over: Mapped[str] = mapped_column(String, default="")
    on_screen_text: Mapped[str] = mapped_column(String, default="")
    key_message: Mapped[str] = mapped_column(String, default="")
    shot_kind: Mapped[str] = mapped_column(String, default="person")  # "person" | "insert"
    join_with_prev: Mapped[str] = mapped_column(String, default="cut")
    screen_direction: Mapped[str] = mapped_column(String, default="")  # "left" | "right" | "neutral" | ""

    prompt_img: Mapped[str] = mapped_column(String, default="")
    prompt_full: Mapped[str] = mapped_column(String, default="")
    dish_state: Mapped[str] = mapped_column(String, default="")
    prompt_video: Mapped[str] = mapped_column(String, default="")
    # How long this shot's clip should run, in seconds, as estimated by the prompt author from the
    # shot's own script. Omni has no duration parameter — this reaches the model only as a sentence
    # in prompt_video (see OMNI_DURATION_BLOCK). 0.0 = never estimated, which keeps the old generic
    # "about 4 to 10 seconds" wording.
    target_seconds: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    speed: Mapped[float] = mapped_column(Float, default=1.0)
    ken_burns: Mapped[bool] = mapped_column(Boolean, default=False)

    ingredient_refs: Mapped[list] = mapped_column(JSON, default=list)
    equipment_refs: Mapped[list] = mapped_column(JSON, default=list)
    image_subjects: Mapped[list] = mapped_column(JSON, default=list)

    current_image_asset_id: Mapped[str | None] = mapped_column(ForeignKey("media_assets.id"), nullable=True)
    current_video_asset_id: Mapped[str | None] = mapped_column(ForeignKey("media_assets.id"), nullable=True)
    # The clip the current one replaced — one step of undo (see StoryboardRepo.undo_shot_video).
    # Only meaningful because clip keys carry a timestamp: superseding used to overwrite the same S3
    # object, so an older asset row pointed at bytes that no longer existed.
    # Deliberately NOT a ForeignKey: the migration that added the column didn't create one (SQLite
    # couldn't without a table rebuild), so declaring it here made the model disagree with every
    # real database and every autogenerate wanted to "add" it. Matching reality instead.
    prev_video_asset_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # The image the current one replaced — one step of undo, mirroring the clip column above (see
    # StoryboardRepo.undo_shot_image). Safe for the same reason: image keys carry a timestamp, so a
    # superseded asset still points at bytes that exist. Not a ForeignKey, same as its sibling.
    prev_image_asset_id: Mapped[str | None] = mapped_column(String, nullable=True)

    # Gemini Omni Flash Interactions API conversation handle for THIS shot's clip — set after the
    # first /steps/video/edit call, sent as `previous_interaction_id` on the next one so the model
    # edits the SAME conversation (no video re-upload) instead of starting fresh. Cleared whenever
    # the shot's video is re-generated from scratch (set_shot_video) — a new base clip means the
    # old edit conversation no longer applies.
    omni_interaction_id: Mapped[str | None] = mapped_column(String, nullable=True)

    storyboard_scene: Mapped["StoryboardScene"] = relationship(back_populates="shots")
    current_image_asset: Mapped["MediaAsset | None"] = relationship(foreign_keys=[current_image_asset_id])
    current_video_asset: Mapped["MediaAsset | None"] = relationship(foreign_keys=[current_video_asset_id])
    image_plan: Mapped["ShotImagePlan | None"] = relationship(back_populates="shot", uselist=False, cascade="all, delete-orphan")
    video_refs: Mapped[list["ShotVideoRef"]] = relationship(
        back_populates="shot", cascade="all, delete-orphan", order_by="ShotVideoRef.position")
    prompt_video_versions: Mapped[list["ShotPromptVideoVersion"]] = relationship(
        back_populates="shot", cascade="all, delete-orphan",
        order_by="ShotPromptVideoVersion.id.desc()")


class ShotImagePlan(Base):
    __tablename__ = "shot_image_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shot_id: Mapped[int] = mapped_column(ForeignKey("storyboard_shots.id"), nullable=False, unique=True)

    classified: Mapped[bool] = mapped_column(Boolean, default=False)
    has_human: Mapped[bool] = mapped_column(Boolean, default=False)
    has_ingredients: Mapped[bool] = mapped_column(Boolean, default=False)
    is_process: Mapped[bool] = mapped_column(Boolean, default=False)
    shows_dish: Mapped[bool] = mapped_column(Boolean, default=False)
    shows_kitchen: Mapped[bool] = mapped_column(Boolean, default=False)

    shot_scale: Mapped[str] = mapped_column(String, default="")
    same_framing_as_prev: Mapped[bool] = mapped_column(Boolean, default=False)
    scene_match: Mapped[str] = mapped_column(String, default="")
    reuse_prev: Mapped[bool] = mapped_column(Boolean, default=False)
    image_generate_type: Mapped[str] = mapped_column(String, default="new_generate")

    ingredient_changed: Mapped[bool] = mapped_column(Boolean, default=False)
    ingredient_change: Mapped[str] = mapped_column(String, default="")
    equipment_changed: Mapped[bool] = mapped_column(Boolean, default=False)
    equipment_change: Mapped[str] = mapped_column(String, default="")
    process_changed: Mapped[bool] = mapped_column(Boolean, default=False)
    process_change: Mapped[str] = mapped_column(String, default="")

    ref_keywords: Mapped[list] = mapped_column(JSON, default=list)

    qc_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    qc_attempts: Mapped[int] = mapped_column(Integer, default=0)
    qc_issues: Mapped[list] = mapped_column(JSON, default=list)

    # The EXACT assembled text the image model received on this shot's last render (prompt_img +
    # text rules + provider ref wrapper). Persisted so the "ครบ" editor still has something to show
    # after a reload instead of forcing a paid /steps/image_plan dry-run per shot.
    full_prompt: Mapped[str] = mapped_column(String, default="")

    shot: Mapped["StoryboardShot"] = relationship(back_populates="image_plan")
    refs_used: Mapped[list["ShotRefUsed"]] = relationship(back_populates="image_plan", cascade="all, delete-orphan")


class ShotRefUsed(Base):
    __tablename__ = "shot_refs_used"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shot_image_plan_id: Mapped[int] = mapped_column(ForeignKey("shot_image_plans.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String, default="")
    label: Mapped[str] = mapped_column(String, default="")
    url: Mapped[str] = mapped_column(String, default="")
    source: Mapped[str] = mapped_column(String, default="")

    image_plan: Mapped["ShotImagePlan"] = relationship(back_populates="refs_used")


class ShotVideoRef(Base):
    """A reference image the USER attached to this shot for video generation.

    The video side had no equivalent of the image side's `extra_refs`: the render route accepted
    none, and refs attached during the image step are filtered out before video (only person /
    kitchen / scene kinds survive `_OMNI_REF_KINDS`). Persisted rather than kept in the card's
    state because the authored `prompt_video` cites each ref by position tag — a ref that vanished
    on reload would leave the stored prompt pointing at an image no longer sent.
    """
    __tablename__ = "shot_video_refs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shot_id: Mapped[int] = mapped_column(ForeignKey("storyboard_shots.id", ondelete="CASCADE"), nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    # The user's own words for what this is. Reaches the prompt author (which never sees the image)
    # AND becomes this ref's rule line in the rendered prompt — the two readers `ref_notes` serves
    # on the image side.
    note: Mapped[str] = mapped_column(String, default="", server_default="")
    position: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # 'user' = a ref the user attached — the original meaning of this table.
    # 'person' / 'kitchen' = their answer for that AUTO slot in _omni_shot_ref_urls: a url REPLACES
    # that ref, an empty url REMOVES it, no row at all leaves the rule alone. Same table because all
    # three answer one question — what does this shot actually send — and splitting them would mean a
    # second CRUD path, a second carry-forward in save_new_version, and a second staleness input.
    kind: Mapped[str] = mapped_column(String, default="user", server_default="user")

    shot: Mapped["StoryboardShot"] = relationship(back_populates="video_refs")


class ShotPromptVideoVersion(Base):
    """One previous value of a shot's `prompt_video`.

    `PUT /shots/{id}/prompt_video` overwrites in place on purpose (a version per keystroke would be
    absurd), but that left a per-shot regenerate or a hand edit with no way back short of activating
    an older storyboard version — which reverts every other shot too. Text is cheap next to clips,
    so a shallow per-shot log buys the undo without touching the document-version machinery.
    """
    __tablename__ = "shot_prompt_video_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shot_id: Mapped[int] = mapped_column(ForeignKey("storyboard_shots.id", ondelete="CASCADE"), nullable=False)
    prompt_video: Mapped[str] = mapped_column(String, default="", server_default="")
    source: Mapped[str] = mapped_column(String, default="", server_default="")  # regen | manual | restore
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    shot: Mapped["StoryboardShot"] = relationship(back_populates="prompt_video_versions")


# ---------------------------------------------------------------------------
# Media assets — normalizes every S3-backed image/video reference and gives real version history
# ---------------------------------------------------------------------------
class MediaAsset(Base):
    __tablename__ = "media_assets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)  # "image" | "video"
    s3_key: Mapped[str] = mapped_column(String, nullable=False)
    s3_url: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="generated")  # generated|qc_passed|qc_failed|superseded
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_frame_s3_url: Mapped[str | None] = mapped_column(String, nullable=True)
    # Which shot this asset was rendered for. Without it a clip was reachable only through the
    # shot's two pointers (current + one step of undo), so the third-most-recent render was
    # unrecoverable even though this row and its S3 bytes both still existed.
    # Deliberately NOT a ForeignKey — same reason as StoryboardShot.prev_video_asset_id: SQLite
    # cannot add one in place, so declaring it would make the model disagree with real databases.
    shot_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # The reference images this render actually used: [{kind, label, url, tag}]. The video analogue
    # of ShotImagePlan.refs_used — before this, nothing recorded whether a clip opened on the
    # previous shot's last frame or its own still, or which character/scene ref was live.
    refs_used: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped["Run | None"] = relationship(back_populates="media_assets")


class ConfigRef(Base):
    """Durable global reference images (character / scene). Before this, upload_ref only set the ref
    in-memory on settings.image_gen — so it reverted to the config.yaml default on restart. One row
    per kind; loaded into settings at startup and used by generation + Omni refs."""
    __tablename__ = "config_refs"

    kind: Mapped[str] = mapped_column(String, primary_key=True)  # "character" | "scene"
    image_asset_id: Mapped[str | None] = mapped_column(ForeignKey("media_assets.id"), nullable=True)
    s3_url: Mapped[str] = mapped_column(String, default="")
    # Text that belongs with the picture. For "character" these are the host's name and
    # description, which used to be typed into Step 2's per-run form and therefore had to be
    # re-entered for every run even though the person never changes. Named generically because the
    # "scene" row shares this table and leaves them blank.
    name: Mapped[str] = mapped_column(String, default="", server_default="")
    description: Mapped[str] = mapped_column(String, default="", server_default="")
    # The narration voice for Omni's Voice direction block (language/gender/pace/tone/style). One
    # JSON blob rather than five columns: only the "character" row uses it, the set is still
    # growing, and nothing queries it — the typing lives in VoiceConfig. NULL on rows written
    # before this existed, so always read it as `row.voice or {}`.
    voice: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    image_asset: Mapped["MediaAsset | None"] = relationship()


# ---------------------------------------------------------------------------
# Brands / brand scenes
# ---------------------------------------------------------------------------
class Character(Base):
    """A host: their reference photo, who they are in words, and how they sound.

    Replaces the single `config_refs` row keyed on kind="character", which could not hold two by
    construction. Shaped like Brand on purpose — slug id, a run points at one, and the same
    `is_default` single-winner rule BrandScene uses — because a run picking a character is the same
    problem as a run picking a brand, and copying that shape means one mechanism to understand.

    `Run.character_id` NULL means "use the default", not "no host": every existing run predates this
    table, and the fallback is what keeps their output identical."""
    __tablename__ = "characters"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")
    image_asset_id: Mapped[str | None] = mapped_column(ForeignKey("media_assets.id"), nullable=True)
    s3_url: Mapped[str] = mapped_column(String, default="")
    # The narration voice (language/gender/vo_pace/tone/style) — one JSON blob for the same reason
    # config_refs.voice was one: the field set is still moving and nothing queries it.
    voice: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    image_asset: Mapped["MediaAsset | None"] = relationship()


class Brand(Base):
    __tablename__ = "brands"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    tagline: Mapped[str] = mapped_column(String, default="")
    platform_style: Mapped[str] = mapped_column(String, default="")
    theme: Mapped[str] = mapped_column(String, default="")
    mood: Mapped[str] = mapped_column(String, default="")
    material_palette: Mapped[str] = mapped_column(String, default="")
    lighting: Mapped[str] = mapped_column(String, default="")
    editing_style: Mapped[str] = mapped_column(String, default="")
    vo_tone: Mapped[str] = mapped_column(String, default="")
    music: Mapped[str] = mapped_column(String, default="")
    camera_movement: Mapped[str] = mapped_column(String, default="")
    words_per_second: Mapped[float] = mapped_column(Float, default=0.0)

    # Default first — it is the fallback every shot lands on when the classifier matches no scene,
    # so it is the one worth seeing first. Ordered on the relationship, not just in BrandRepo, since
    # the API and scene_store both read `brand.scenes` directly and would otherwise disagree.
    scenes: Mapped[list["BrandScene"]] = relationship(
        back_populates="brand", cascade="all, delete-orphan",
        order_by="(BrandScene.is_default.desc(), BrandScene.id)")


class BrandScene(Base):
    __tablename__ = "brand_scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brand_id: Mapped[str] = mapped_column(ForeignKey("brands.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, default="")
    # WHEN to pick this scene ("ขั้นตอนที่ต้องใช้เตาแก๊ส เช่น การทอด") — read by the shot classifier.
    desc: Mapped[str] = mapped_column(String, default="")
    # HOW the scene is laid out: camera angle, what sits in the fore/background, where the host
    # belongs, where props can rest. Kept apart from `desc` on purpose — mixing geometry into the
    # selection criteria made the classifier pick the wrong scene. Feeds the image-prompt author,
    # which had no scene geometry at all and let the model put the host beside the stove instead of
    # behind it. Written once per scene (optionally drafted by AI from the image).
    layout: Mapped[str] = mapped_column(String, default="")
    image_asset_id: Mapped[str | None] = mapped_column(ForeignKey("media_assets.id"), nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    brand: Mapped["Brand"] = relationship(back_populates="scenes")
    image_asset: Mapped["MediaAsset | None"] = relationship()


# ---------------------------------------------------------------------------
# Menus / subject refs
# ---------------------------------------------------------------------------
class Menu(Base):
    __tablename__ = "menus"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)

    subject_refs: Mapped[list["SubjectRef"]] = relationship(back_populates="menu", cascade="all, delete-orphan")


class SubjectRef(Base):
    __tablename__ = "subject_refs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    menu_id: Mapped[str] = mapped_column(ForeignKey("menus.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)  # "ingredient" | "equipment"
    image_asset_id: Mapped[str | None] = mapped_column(ForeignKey("media_assets.id"), nullable=True)

    menu: Mapped["Menu"] = relationship(back_populates="subject_refs")
    image_asset: Mapped["MediaAsset | None"] = relationship()


# ---------------------------------------------------------------------------
# Director prompts (append-only history, same semantics as v1)
# ---------------------------------------------------------------------------
class DirectorPrompt(Base):
    __tablename__ = "director_prompts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, default="")
    source_url: Mapped[str] = mapped_column(String, default="")
    source_title: Mapped[str] = mapped_column(String, default="")
    summary: Mapped[str] = mapped_column(String, default="")
    sections: Mapped[dict] = mapped_column(JSON, default=dict)  # 7 fixed section keys -> Thai text
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------------------
# Cost rates — editable $/unit figures the Usage Report's cost estimate is computed from (see
# app/services/cost_estimate.py). Global (not per-run): provider list prices don't vary by run.
# Seeded once with a snapshot of public pricing (see cost_estimate.DEFAULT_RATES's docstring for the
# date/source) and editable from the Usage Report UI from then on — no code deploy needed when a
# provider changes its price, which was the whole problem with hardcoding these as constants.
# ---------------------------------------------------------------------------
class CostRate(Base):
    __tablename__ = "cost_rates"

    key: Mapped[str] = mapped_column(String, primary_key=True)   # e.g. "gemini_text_input_per_1m"
    label: Mapped[str] = mapped_column(String, nullable=False)   # human-readable, shown in the editor table
    unit: Mapped[str] = mapped_column(String, nullable=False)    # "USD / 1M tokens" | "USD / clip" | "USD / image" | "THB per USD"
    value: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


# ---------------------------------------------------------------------------
# Usage records (promoted from ephemeral SSE-only telemetry)
# ---------------------------------------------------------------------------
class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    step: Mapped[str] = mapped_column(String, nullable=False)
    agent: Mapped[str] = mapped_column(String, default="")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped["Run"] = relationship(back_populates="usage_records")


# ---------------------------------------------------------------------------
# Regen/edit events — how often the USER had to regenerate or manually fix an AI-produced result.
# Separate from usage_records (which counts every LLM/provider CALL, including the successful
# first try): this counts only user-initiated "that wasn't good enough" actions — a rough, durable
# signal of which steps need the most quality work, independent of raw API cost.
# ---------------------------------------------------------------------------
class RegenEvent(Base):
    __tablename__ = "regen_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    step: Mapped[str] = mapped_column(String, nullable=False)      # "script" | "storyboard" | "image" | "video_prompt" | "video" | "revoice" | ...
    action: Mapped[str] = mapped_column(String, nullable=False)    # "regenerate" | "regenerate_all" | "edit" | "edit_region" | "upload_replace" | ...
    scene_id: Mapped[str] = mapped_column(String, default="")
    shot_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    note: Mapped[str] = mapped_column(String, default="")          # free-form context, e.g. "12 shots"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
