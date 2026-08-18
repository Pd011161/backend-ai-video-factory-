from typing import Literal
import json

from pydantic import BaseModel

from app.core.config import ScriptConfig


class RegenEventRequest(BaseModel):
    """One user-initiated 'this AI result wasn't good enough' action — fire-and-forget from the
    frontend right when the user clicks Regenerate/Edit/re-Upload on an ALREADY-produced result.
    Not sent for a first-time generate (nothing to regenerate yet)."""
    step: str                      # "script" | "storyboard" | "video_prompt" | "image" | "video" | "revoice" | ...
    action: str                    # "regenerate" | "regenerate_all" | "edit" | "edit_region" | "upload_replace" | ...
    scene_id: str = ""
    no: int | None = None
    note: str = ""


class ShotPromptVideoUpdateRequest(BaseModel):
    """Lightweight per-shot save — mutates the ONE field on the shot's existing row directly,
    unlike /steps/video_prompts' bulk call which persists a whole new StoryboardDocument VERSION.
    Used by Step 5's per-shot '↻ Prompt รายช็อต' regenerate (and manual inline edits) so a single
    shot's prompt_video is actually saved instead of only living in the browser tab's memory."""
    prompt_video: str
    # Hand-corrected clip length. None = leave whatever the prompt author estimated; the two travel
    # together because editing the prompt and re-timing the clip are the same act for the user.
    target_seconds: float | None = None
    # What caused this save, recorded on the version log entry: "regen" (↻ per-shot) or "manual"
    # (typed edit). Blank is treated as manual.
    source: str = ""


class VideoRefIn(BaseModel):
    url: str
    # The user's own words for what this image is. Reaches the prompt author (which never sees the
    # picture) AND becomes the ref's rule line in the rendered prompt — same two readers ref_notes
    # serves on the image side.
    note: str = ""
    # "user" = a ref the user attached. "person" / "kitchen" = their answer for that AUTO slot —
    # a url replaces it, an empty url removes it. Defaulted so a payload written before slots
    # existed still means what it used to.
    kind: str = "user"


class ShotVideoRefsUpdateRequest(BaseModel):
    """Whole-list replace, like upsert_image_plan's refs_used. Order decides each image's @ImageN
    tag at render, so it is part of the payload rather than something the server sorts."""
    refs: list[VideoRefIn] = []


# ── Per-step manual save (PUT /runs/{id}/research|script|storyboard) ─────────────────────────────
# The step endpoints persist only what they GENERATE; edits typed into the UI lived exclusively in
# React state and evaporated on refresh. These carry a whole edited document back. Each save is a
# new version through the same save_new_version the generate path uses — history stays browsable in
# DocVersionBar, and an unchanged payload is a no-op rather than a version bump.


class SaveResearchRequest(BaseModel):
    payload: dict                      # the whole research doc, same shape GET returns


class SaveScriptRequest(BaseModel):
    script: dict                       # {title, production, overview, parts} — ScriptRepo shape
    script_config: dict | None = None  # the knobs that produced it, kept restorable alongside


class SaveStoryboardRequest(BaseModel):
    storyboard: dict                   # StoryboardDocument dump (same as VideoPromptsStepRequest)
    # Media links carry forward keyed on (scene_id, shot.no) — an edit that renumbers/removes a shot
    # silently orphans its generated image/video. The save 409s with the loss list instead; the UI
    # confirms with the user and retries with force=True.
    force: bool = False


class RunCreateRequest(BaseModel):
    """New in v2 — promotes 'the job' to a first-class Run row (replaces v1's implicit
    Google-Sheet row id / active_config.json job pointer)."""
    topic: str
    title: str = ""


class RunUpdateRequest(BaseModel):
    """PATCH: every field optional, and `exclude_unset` at the route means an omitted one is left
    alone rather than blanked — so the Runs table can send just the cell that changed.

    No `id`: it is the primary key six tables point at, and menus are keyed on it.
    `brand_id`/`character_id` route to `set_active_pointers`, not to a plain column write."""
    topic: str | None = None
    title: str | None = None
    # Same vocabulary as script.category with no translation table, because this IS where every
    # step now reads its category from (_merge_script_config). "other" maps to no rule block at
    # all — neither the food nor the drink one. "" means not set yet, and leaves config.yaml's
    # default standing.
    type: Literal["food", "drink", "other", ""] | None = None
    avatar: str | None = None
    brand_id: str | None = None
    # Which host this run is made with. None/"" = not chosen → the default character is used.
    character_id: str | None = None


class CharacterVoiceRequest(BaseModel):
    """The narration voice for Omni's `Voice direction:` block — mirrors core.config.VoiceConfig."""
    language: str = "Thai"
    gender: Literal["female", "male"] = "female"
    vo_pace: float = 2.5
    tone: str = ""
    style: str = ""


class CharacterMetaRequest(BaseModel):
    """The text half of the global character config — stored on the same config_refs row as the
    reference photo, so the host is described once instead of per run.

    `None` means "leave this alone", `""` means "clear it back to config.yaml". The name and the
    voice are saved from two separate cards in the panel, so a voice-only save must not blank the
    name it never sent."""
    name: str | None = None
    description: str | None = None
    voice: CharacterVoiceRequest | None = None


class SynthesizeRequest(BaseModel):
    topic: str
    run_id: str | None = None
    script_config: ScriptConfig | None = None


class DirectorCreateRequest(BaseModel):
    youtube_url: str
    title: str | None = None


# ─── Per-step requests (each step is fed the previous step's JSON) ──────────────

class ResearchStepRequest(BaseModel):
    topic: str
    run_id: str | None = None
    script_config: ScriptConfig | None = None
    # "ใช้ source link เดิม" / manual URLs — reuse these exact reference videos ({"title","url"},
    # title optional) instead of searching again. Skips generate_queries/search/filter/topup
    # entirely and goes straight to analyze → synthesize. Also how the user supplies their OWN
    # YouTube links (mix of reused + freshly-typed URLs is fine — both are just {title, url}).
    reuse_references: list[dict] | None = None
    # Which of the above (by exact url) to treat as the MASTER reference — the synthesized tutorial
    # is built primarily from this video, with the others contributing supplementary tips only.
    # "" / None = no master (pick whichever single method looks best overall, as before).
    master_url: str | None = None


class ScriptStepRequest(BaseModel):
    topic: str
    synthesis: str
    run_id: str | None = None
    script_config: ScriptConfig | None = None
    # Real host-voice quotes pooled from the research step's reference videos (see BATCH_PROMPT /
    # routes._aggregate_voice_samples) — grounds the script's voice in an ACTUAL cooking video
    # instead of only the fixed generic gold-standard example.
    voice_samples: list[str] = []
    # How the real reference videos' hosts RUN THE SHOW (pacing/transitions/engagement technique —
    # distinct from voice_samples' WHAT they say). See BATCH_PROMPT / routes._aggregate_presentation_notes.
    presentation_notes: list[str] = []


class ScriptRegeneratePartRequest(BaseModel):
    topic: str
    synthesis: str
    script: dict                       # full ScriptDocument dump — validated when reconstructed
    part_number: int                   # 1-based part to regenerate
    target_duration: str = "06:00-06:30"
    target_scenes: int | None = None
    min_shots: int | None = None       # per-part auto-fit: explicit shot floor for this part's prompt
    run_id: str | None = None
    script_config: ScriptConfig | None = None


class StoryboardStepRequest(BaseModel):
    topic: str
    script: dict  # a ScriptDocument dump — validated when reconstructed
    target_shots: int | None = None  # per-part auto-fit: soft target total shots for this doc (steers breakdown)
    run_id: str | None = None
    script_config: ScriptConfig | None = None


class ImagePromptsStepRequest(BaseModel):
    topic: str
    storyboard: dict          # a StoryboardDocument dump — shots stay fixed, only prompt_img is regenerated
    script: dict | None = None  # ScriptDocument dump for the production spec; optional if the storyboard carries `production`
    run_id: str | None = None
    script_config: ScriptConfig | None = None
    # What the user said their own attached reference photos are — the same notes /steps/image sends
    # as `ref_notes`. Without them the per-shot "↻ Prompt" rewrote the prompt blind to refs the user
    # had just attached. Request-level rather than per-shot because the only caller sends a one-shot
    # mini storyboard; a whole-board regenerate would have to carry these on each shot instead.
    ref_notes: list[str] = []


class ImagesStepRequest(BaseModel):
    topic: str
    storyboard: dict  # a StoryboardDocument dump — validated when reconstructed
    run_id: str | None = None
    script_config: ScriptConfig | None = None   # carries category (food|drink) → drink prompt rules
    regenerate_all: bool = True                 # False = skip shots that already have generated_img


class VideoPromptsStepRequest(BaseModel):
    topic: str
    storyboard: dict  # a StoryboardDocument dump — validated when reconstructed
    use_image_seed: bool = True   # frame mode — drives how prompt_video is written + Step 6 rendering
    use_last_frame: bool = False
    run_id: str | None = None
    script_config: ScriptConfig | None = None   # carries voice_desc (narrator voice pin) + category
    # Omni narration voice config (appended verbatim to each prompt_video). None = "not sent" →
    # _merge_voice_config falls back to the Character config, which is where the UI sets these now.
    # They stay on the request as an override escape hatch for tests and POC scripts.
    language: str | None = None
    vo_pace: float | None = None   # words per second
    tone: str | None = None
    style: str | None = None
    gender: str | None = None      # "female" | "male" — normalized in gemini_client._voice_gender


class RefManifestItem(BaseModel):
    shot_id: int
    prev_shot_id: int | None = None   # the shot whose rendered clip's last frame is this FIRST_FRAME (continuous) or IMAGE_REF (match_cut)
    match_cut: bool = False


class RefManifestRequest(BaseModel):
    """Ask for each shot's Omni image-ref manifest (tags + thumbnail URLs) for the Step5 check panel."""
    shots: list[RefManifestItem] = []


class PdfExportRequest(BaseModel):
    html: str                    # a self-contained export HTML doc (from the frontend's exportHtml)
    filename: str = "export"     # download filename (without extension)


class ImageStepRequest(BaseModel):
    topic: str
    scene_id: str
    no: int
    prompt_img: str
    on_screen_text: str = ""
    image_results: list = []   # [{subject, query, candidates:[...]}] for subject refs
    shot_kind: str = "person"  # "person" | "insert" — drops character_ref for inserts
    # dynamic flow (image_gen.dynamic_image): full shot + previous shot for classify/prev-ref
    shot: dict | None = None
    prev_shot: dict | None = None
    all_ingredients: list[str] = []   # full recipe list (storyboard.ingredients) — for overview shots
    ingredient_images: dict = {}      # {canonical ingredient → generated_img url} — reuse earlier ingredient images
    prev_state_url: str = ""          # last cooking-step image — carry dish state (diced→boiling→mashed)
    used_ingredients: list[str] = []  # ingredients already worked in a prior process shot (→ no raw-intro ref)
    all_equipment: list[str] = []     # full tools list (storyboard.equipment) — for the all-equipment overview
    equipment_images: dict = {}      # {canonical tool → generated_img url} — reuse earlier equipment images
    extra_refs: list[str] = []        # user-uploaded ref image URLs to ADD into (re)generation
    # What each extra ref IS, index-aligned with extra_refs. Unlike the edit routes' ref_notes (read
    # only by the instruction author), these reach BOTH readers: the prompt author, so it can
    # describe the object, and the manifest label, so the image model can tell one attachment from
    # another instead of seeing several identically-worded "additional reference" lines.
    ref_notes: list[str] = []
    removed_refs: list[str] = []      # keys "{kind}|{label}" of auto refs the user removed
    # per-shot overrides — replace an auto ref with a user-picked image URL ("" = auto)
    ref_person: str = ""              # replaces the character ref for THIS shot
    ref_scene: str = ""               # replaces the kitchen/scene ref for THIS shot
    ref_state: str = ""               # replaces the auto dish-state frame for THIS shot
    # food|drink → selects the drink-specific prompt rule block. None = "not specified", and the
    # route falls back to config.yaml's script.category — the same source every bulk step reads via
    # _merge_script_config. It used to default to the literal "food" and no caller ever sent it, so
    # on a `category: drink` config the bulk steps used drink rules while a per-shot regenerate
    # silently used food ones inside the same run.
    category: str | None = None
    prompt_override: str | None = None  # user-edited FULL image prompt ("ครบ") → sent to the model verbatim
    qc_feedback: str = ""             # QC-FEEDBACK block from a failed review — appended to the prompt when the user opts in on a manual regen
    shot_id: int | None = None        # DB StoryboardShot.id — when set, the generated image is persisted via MediaAssetRepo/StoryboardRepo
    run_id: str | None = None         # when set, tags the MediaAsset + persists a usage_records row for this call


class ImageEditRequest(BaseModel):
    """Edit a shot's EXISTING generated_img in place with a short instruction (gpt-image-2 / Gemini
    image-edit) instead of re-rendering from prompt_img — for a small fix (e.g. "remove the spoon",
    "make the sauce darker") without redoing the whole shot. `shot_id` is required — the base image
    comes from the DB row's current_image_asset_id."""
    topic: str
    scene_id: str
    no: int
    shot_id: int
    instruction: str
    # Extra reference images for THIS edit only — shown to the model AFTER the base image, so they
    # are "image 2", "image 3", … in the role list the provider wrapper builds (image 1 is always
    # the current image being edited). Not persisted; the next edit starts from an empty list.
    extra_refs: list[str] = []
    # What each extra ref IS, index-aligned with extra_refs. Only the instruction author reads these
    # (the model gets bytes + the role line), so it can write "the bowl in image 2" instead of
    # referring to an attachment it knows nothing about.
    ref_notes: list[str] = []
    run_id: str | None = None


class ImageEditPromptRequest(BaseModel):
    """Polish a rough image-edit note into one clear prompt, with the attached refs described."""
    instruction: str
    original_desc: str = ""      # what the base image is — usually the shot's prompt_img
    ref_notes: list[str] = []    # index-aligned with the edit's extra_refs
    mode: str = ""               # "" = whole-image edit | "mask" | "outpaint" — shapes the rules given to the author


class ImagePromptReviseRequest(BaseModel):
    """Rewrite a shot's FULL image prompt to fix what was wrong with the picture it produced."""
    full_prompt: str
    feedback: str


class ImageRegionEditRequest(BaseModel):
    """Region edit of a shot's EXISTING generated_img — gpt-image-2 only (mask-based edits aren't
    supported through the Gemini image path). Two modes:
    - "outpaint": server pads the current image to (target_width, target_height) with a transparent
      border and asks gpt-image to fill it in — no mask needed from the client.
    - "mask": client draws a freehand mask (PNG, same pixel size as the current image, alpha=0 over
      the region to regenerate) and sends it as `mask_data_url`; the rest of the image is preserved.
    """
    topic: str
    scene_id: str
    no: int
    shot_id: int
    instruction: str
    mode: str                              # "outpaint" | "mask"
    target_width: int | None = None        # outpaint: new canvas width (>= current width)
    target_height: int | None = None       # outpaint: new canvas height (>= current height)
    mask_data_url: str | None = None       # mask: "data:image/png;base64,...."
    # Extra reference images for THIS edit — sent after the image being edited, so they are
    # "image 2", "image 3", … and the mask still binds to image 1 (see edit_image_region).
    extra_refs: list[str] = []
    ref_notes: list[str] = []              # index-aligned with extra_refs; read only by the prompt author
    run_id: str | None = None


class ImageCropRequest(BaseModel):
    """Reframe a shot's EXISTING generated_img: keep the selected rectangle and scale it back up to
    the image's original pixel size. No model is involved — it is a Pillow crop + resize — so unlike
    the region edit this works on every provider and costs nothing.

    The rectangle is in FRACTIONS of the image (0-1), not pixels: the client reads the size from the
    <img> it is displaying, the server from the bytes, and nothing guarantees those agree. The client
    derives it from a zoom level applied to the whole frame, so it always carries the original aspect
    ratio and the result cannot be squashed."""
    topic: str
    scene_id: str
    no: int
    shot_id: int
    x: float
    y: float
    w: float
    h: float
    run_id: str | None = None


class VideoStepRequest(BaseModel):
    topic: str
    scene_id: str
    no: int
    prompt_video: str
    shot_kind: str = "person"     # "person" | "insert" — the video QC uses it (no person allowed in an insert)
    image_url: str = ""           # the shot's generated_img (first frame)
    last_image_url: str = ""      # the NEXT shot's generated_img (last frame — seamless join)
    use_image_seed: bool = True   # use image_url as the first frame (image-to-video)
    use_last_frame: bool = True   # use last_image_url as the final frame
    # optional per-request overrides of config.yaml video_gen
    duration_seconds: int | None = None
    resolution: str | None = None
    aspect_ratio: str | None = None
    generate_audio: bool | None = None
    person_generation: str | None = None
    shot_id: int | None = None    # DB StoryboardShot.id — when set, the generated video is persisted via MediaAssetRepo/StoryboardRepo
    run_id: str | None = None     # when set, tags the MediaAsset + persists a usage_records row for this call
    # "" (default) = obey config.yaml video_gen.model as usual; "omni" = force this ONE shot onto
    # Gemini Omni Flash (Interactions API, multi-ref) regardless of the configured model — lets a
    # single storyboard mix Veo shots and Omni shots. QC-regen loop is Veo-only; Omni skips it.
    engine: str = ""
    # Omni-only: the PREVIOUS shot's generated_video URL — its clip's actual LAST FRAME (extracted
    # server-side) becomes THIS shot's FIRST_FRAME for real shot-to-shot continuity, the way the
    # Omni POC does it (see _omni_prev_frame_ref). Ignored for engine="" (Veo uses image_url/
    # last_image_url instead — a different continuity mechanism, image-seed + interpolation).
    prev_video_url: str = ""
    match_cut: bool = False


class VideoEditRequest(BaseModel):
    """Conversational video edit (Gemini Omni Flash Interactions API) — edits the shot's CURRENT
    generated_video in place with a short text instruction, instead of re-rendering from prompt_img/
    prompt_video. `shot_id` is required (the base clip + edit-conversation state both come from the
    DB row) — this feature only exists for shots already generated and persisted."""
    topic: str
    scene_id: str
    no: int
    shot_id: int
    instruction: str              # short edit instruction, e.g. "zoom in slower" / "remove the spoon at the end"
    extra_refs: list[str] = []    # optional NEW reference image URLs for THIS edit turn only
    run_id: str | None = None


class VideoEditPromptRequest(BaseModel):
    instruction: str
    # Descriptions of the images attached to THIS edit turn, in send order — only the author sees
    # them (the run call sends the bytes), so it can write "the hat <IMAGE_REF_0>" instead of
    # referring to attachments it knows nothing about. Only the `description` key is read.
    refs: list[dict] = []


class ImageUndoRequest(BaseModel):
    """Put back the image this shot's current one replaced. Swaps current ↔ previous, so pressing it
    again redoes. Covers every whole-image replacement — regenerate, quick edit, outpaint/mask edit,
    crop, upload — because they all go through set_shot_image."""
    shot_id: int


class VideoUndoRequest(BaseModel):
    """Put back the clip this shot's current one replaced. Swaps current ↔ previous, so pressing it
    again redoes. Covers whole-clip replacements only (render / edit / upload / crop / trim) —
    revoice and burnt subtitles rewrite the file in place with ffmpeg and leave no earlier asset to
    swap back to."""
    shot_id: int


class VideoCropRequest(BaseModel):
    """Reframe a shot's EXISTING clip: keep the selected rectangle on EVERY frame and scale it back
    to the clip's original pixel size. Pure ffmpeg — no model, works on every provider, costs
    nothing. One static box for the whole clip, like the image crop (no keyframes).

    The rectangle is in FRACTIONS of the frame (0-1), same contract as ImageCropRequest and for the
    same reason: the client derives it from a zoom level on the whole frame, so it always carries
    the original aspect ratio."""
    topic: str
    scene_id: str
    no: int
    shot_id: int
    x: float
    y: float
    w: float
    h: float
    run_id: str | None = None


class TrimSegment(BaseModel):
    start: float
    end: float


class VideoTrimRequest(BaseModel):
    """Cut a shot's EXISTING clip down to the given KEEP segments (seconds, sorted,
    non-overlapping), concatenated in order. The client works in "ranges to delete" and sends the
    complement, so one schema covers "delete this middle part" and "end the clip here". Pure ffmpeg
    re-encode; audio is trimmed with the video."""
    topic: str
    scene_id: str
    no: int
    shot_id: int
    segments: list[TrimSegment]
    run_id: str | None = None


class VideoGroupRequest(BaseModel):
    topic: str
    scene_id: str
    no: int                  # the group's first shot number (used for the output filename)
    prompts: list[str]       # ordered prompt_video for each shot in the continuous group
    image_url: str = ""      # the FIRST shot's generated_img (first frame of the take)
    images: list[str] = []   # per-shot generated_img (parallel to prompts) — for the individual fallback when the chain breaks
    nos: list[int] = []      # per-shot `no` (parallel to prompts) — so a fallback clip lands on the right shot
    shot_kinds: list[str] = []  # per-shot shot_kind (parallel) — for the fallback shot's QC/person rule
    # DB StoryboardShot.id per shot (parallel to prompts/nos) — when set, each shot's render
    # (the shared group clip, or its individual fallback) is persisted via MediaAssetRepo/StoryboardRepo.
    shot_ids: list[int] = []
    run_id: str | None = None  # when set, persists a usage_records row for this call
    # "" (default) = Veo extension-chain as usual; "omni" = Omni can't chain (no extension API), so
    # consecutive shots are bin-packed into batches that fit its real ~10s-per-clip cap and each
    # batch renders as ONE merged clip (see OMNI_SAFE_CLIP_SECONDS in routes.py).
    engine: str = ""
    # per-shot voice_over (parallel to prompts) — used ONLY to estimate how many shots fit in one
    # Omni clip before rendering (Omni has no duration knob to request an exact length).
    voice_overs: list[str] = []


class RevoiceRequest(BaseModel):
    topic: str
    scene_id: str
    no: int
    video_url: str               # the shot's generated_video (/api/media/... clip to re-voice)
    voice_over: str = ""         # single shot's spoken line → TTS
    voice_overs: list[str] = []  # GROUP clip: each shot's line in order (multi-segment) — wins over voice_over
    voice: str = ""              # Gemini prebuilt voice ("" = config tts.voice)
    style: str = ""              # delivery/tone descriptor ("" = config tts.style)
    mode: str = "replace"        # "replace" = swap the clip's spoken voice (demucs+STT); "add" = overlay
                                  # narration onto the clip KEEPING its music+ambience, centered (no demucs)


class KitchenFixturesRequest(BaseModel):
    fixtures: list[str]


class SubtitleRequest(BaseModel):
    """Burn ONE clip's caption, timed to the window where speech actually happens (per-clip subtitle)."""
    video_url: str
    text: str = ""
    scene_id: str = ""
    no: int = 0
    start: float | None = None   # exact speech window from /revoice spans; omit → detect via whisper VAD
    end: float | None = None


class MergeRequest(BaseModel):
    topic: str
    video_urls: list[str]  # ordered /api/media/... clip URLs to concatenate
    # transition INTO each clip (same length as video_urls; index 0 ignored — nothing precedes it).
    # "cut" | "match_cut" → hard concat; "dissolve" → crossfade; "fade" → fade-through-black;
    # "j_cut" | "l_cut" → hard cut with the audio bridged across it.
    transitions: list[str] = []
    # on_screen_text per clip (same length as video_urls) — burned in if video_gen.subtitles_in_video.
    captions: list[str] = []
    # per-clip post effects (same length as video_urls): {"speed": float, "ken_burns": bool}
    effects: list[dict] = []
    run_id: str | None = None  # when set, persists a usage_records row for this call
    burn_subtitles: bool = False  # UI "subtitle ทั้งเรื่อง" — burn captions into the merge (overrides config)
    # raw = ต่อคลิปตามลำดับดิบๆ ข้ามทุกชั้นเสริม (transition/fade/color/subtitle/effect) — ใช้
    # normalized-concat filter (scale+fps+audio format เท่ากันทุก input) แทน copy-concat เพราะ
    # Veo clips คละ resolution/codec และเสียงยาวกว่าภาพเล็กน้อยต่อคลิป (เลื่อนสะสมถ้า copy-concat)
    raw: bool = False
    # ต่อคลิป Intro นำหน้า + End credit ปิดท้าย (config_refs kind intro/end_credit) — ข้ามเงียบๆ
    # เมื่อยังไม่ได้ตั้ง config ไว้ จึงเปิดเป็น default ได้โดยไม่กระทบระบบเดิม
    include_bumpers: bool = True


def sse(data: dict) -> str:
    """Serialize a dict as one Server-Sent-Events frame."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
