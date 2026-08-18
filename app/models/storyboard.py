"""Storyboard model — each script scene broken into individual camera shots.

Produced by the storyboard node from a ScriptDocument. Content is split from the
script faithfully (no additions/removals); each shot also carries a `prompt_img`
used later to generate the shot's image.
"""

from pydantic import BaseModel, Field, field_validator

from app.models.script import ImageElement


class RefUsed(BaseModel):
    kind: str = Field(description='"kitchen" | "person" | "prev" | "keyword" | "ingredient" | "state"')
    label: str = ""
    url: str = Field(default="", description="preview URL (config/keyword/prev/state); empty → UI uses fixed endpoint")
    source: str = Field(default="", description='"config" | "search" | "generated" | "prev" | ""')


class ShotImagePlan(BaseModel):
    """Dynamic image-gen plan for one shot (classify → prompt+ref). Persisted so the
    UI can show the chosen type + the reference images that were actually used."""
    classified: bool = False
    has_human: bool = False
    has_ingredients: bool = False
    is_process: bool = False
    shows_dish: bool = False                   # the in-progress dish/mixture is visible (incl. recap shots)
    shows_kitchen: bool = False                # the kitchen room/background is visible (False = tight close-up/flat-lay/plain bg)
    shot_scale: str = ""                       # "closeup" | "medium" | "wide" — how much of the room is in frame (drives bg-lock router)
    same_framing_as_prev: bool = False         # same angle/scale/composition as prev shot (→ can EDIT the prev frame)
    scene_match: str = ""                      # id of the configured scene the classifier picked for this shot (background ref)
    reuse_prev: bool = False                  # reuse the previous shot's image as-is (no generation)
    image_generate_type: str = ""             # "reuse_prev" | "use_ref_img" | "new_generate"
    # deltas vs the previous shot (only meaningful for use_ref_img)
    ingredient_changed: bool = False
    ingredient_change: str = ""
    equipment_changed: bool = False
    equipment_change: str = ""
    process_changed: bool = False
    process_change: str = ""
    ref_keywords: list[str] = Field(default_factory=list)   # subjects chosen as keyword refs
    refs_used: list[RefUsed] = Field(default_factory=list)  # resolved refs fed to the model (for UI)
    # C1 (cinema): output-QC verdict on the FINAL kept render for this shot.
    qc_passed: bool = True                     # False = every attempt failed review; last render kept
    qc_attempts: int = 0                       # QC review rounds run for this shot (0 = QC off/skipped)
    qc_issues: list[str] = Field(default_factory=list)  # issues found on the FINAL kept render
    # exact assembled prompt of the last render — without this field, nodes.py's
    # `{k: v … if k in ShotImagePlan.model_fields}` filter silently drops it on the bulk path
    full_prompt: str = ""


class StoryboardShot(BaseModel):
    shot_id: int | None = Field(default=None, description="DB StoryboardShot.id — carried through so per-shot lookups (image plan / Omni ref manifest) work without a run_id")
    no: int = Field(description="Shot order within the scene, starting at 1")
    time: str = Field(description="Shot time range, e.g. '2:00 - 2:08'")
    motion_description: str = Field(description="Motion & Description — what happens / how it is shot")
    voice_over: str = Field(description="The portion of the scene's voice over spoken in this shot")
    on_screen_text: str = Field(description="The portion of the scene's on-screen text shown in this shot")
    key_message: str = Field(description="Key message for this shot (may be empty)")
    shot_kind: str = Field(
        default="person",
        description=(
            "'person' if the host/character is in the frame, or 'insert' for a close-up / "
            "flat-lay of food, ingredients or tools ONLY (no person). Drives which reference "
            "images are fed at generation."
        ),
    )
    join_with_prev: str = Field(
        default="cut",
        description=(
            "How THIS shot connects to the PREVIOUS shot (the cut/transition kind): "
            "'continuous' (same setup, no cut — chained as one Veo take), "
            "'match_cut' (the previous clip ends settling into this shot's opening frame), "
            "'dissolve' (a soft cross-dissolve), or 'cut' (a clean hard cut). "
            "An 'insert' / no-person shot is ALWAYS 'cut' or 'match_cut', never 'continuous'."
        ),
    )
    screen_direction: str = Field(
        default="",
        description=(
            "'left' | 'right' | 'neutral' — which way the subject faces / the action moves across "
            "the frame. Kept consistent across consecutive shots of the same setup (180° rule) so "
            "the host never flips sides between cuts. '' = unspecified (older storyboards)."
        ),
    )
    prompt_img: str = Field(description="Image-generation prompt to render this shot's frame")
    prompt_full: str = Field(
        default="",
        description=(
            "User-edited FULL image prompt ('ครบ' = prompt_img + text rules + ref wrapper). When "
            "non-empty it is sent to the image model VERBATIM (skips auto-assembly); empty = auto."
        ),
    )
    ingredient_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical recipe ingredients (verbatim from production.ingredients) this shot shows or "
            "uses — lets a cooking shot reuse the exact image generated when the ingredient was introduced."
        ),
    )
    equipment_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical tools/equipment this shot shows or uses (verbatim from production.equipment) — "
            "like ingredient_refs but for tools. Equipment never changes state, so a later shot reuses "
            "the exact image generated when the tool was introduced."
        ),
    )
    image_subjects: list[str] = Field(
        default_factory=list,
        description=(
            "Atomic subject OBJECTS in this shot (food, ingredients, tools, finished "
            "items) for reference-image search — one standalone object per entry, "
            "WRITTEN IN THE TOPIC'S LANGUAGE (Thai if the topic is Thai), e.g. "
            "['เส้นหมี่แห้ง', 'น้ำตาลมะพร้าว']. EXCLUDE people / characters / hands and the "
            "scene, setting, or background (kitchen, counter, room, environment, lighting). "
            "No camera jargon."
        ),
    )
    img: str = Field(default="", description="Primary reference image URL (populated by image search)")
    image_results: list[ImageElement] = Field(
        default_factory=list,
        description="Per-element reference search results (populated by image search)",
    )
    dish_state: str = Field(
        default="",
        description=(
            "The dish/drink's CURRENT physical state at THIS shot — a short English phrase "
            "(e.g. 'concentrated dark-green tea, no milk yet, in a measuring cup'). Computed once "
            "across the whole recipe timeline so the state only moves forward; the authoritative "
            "source for how the food looks in prompt_img, replacing the finished-dish paste."
        ),
    )
    generated_img: str = Field(default="", description="URL of the image generated from prompt_img")
    image_plan: ShotImagePlan = Field(default_factory=ShotImagePlan)

    @field_validator("image_plan", mode="before")
    @classmethod
    def _default_image_plan(cls, v):
        return ShotImagePlan() if v is None else v
    prompt_video: str = Field(default="", description="Veo prompt — motion/camera for this shot's video clip")
    target_seconds: float = Field(
        default=0.0,
        description="How long this clip should run, estimated by the prompt author from this shot's "
                    "own script. Omni has no duration parameter, so it reaches the model only as a "
                    "sentence in prompt_video. 0 = never estimated (generic 4-10s wording).")
    generated_video: str = Field(default="", description="URL of the video (.mp4) generated from prompt_video")
    speed: float = Field(default=1.0, description="Playback speed for this clip at assembly (1.0 = normal, 0.5 = slow-mo, 2.0 = fast)")
    ken_burns: bool = Field(default=False, description="Apply a slow post-zoom (Ken Burns push-in) to this clip at assembly")

    # C1a (shot grammar): screen_direction keys motion continuity — normalize casing/synonyms
    # from the LLM ("Left", "L-to-R") so the downstream 180° checks can't be defeated.
    @field_validator("screen_direction", mode="before")
    @classmethod
    def _norm_screen_direction(cls, v):
        s = str(v or "").strip().lower()
        if s in {"left", "right", "neutral"}:
            return s
        if "left" in s and "right" not in s:
            return "left"
        if "right" in s and "left" not in s:
            return "right"
        return "neutral" if s else ""

    # Normalize shot_kind so a capitalized "Insert" from the LLM can't slip past the many
    # `== "insert"` checks downstream (which would then feed a person ref into an object shot).
    @field_validator("shot_kind", mode="before")
    @classmethod
    def _norm_shot_kind(cls, v):
        return "insert" if str(v or "").strip().lower() == "insert" else "person"


class StoryboardScene(BaseModel):
    scene_id: str
    name: str
    transition_in: str = Field(
        default="cut",
        description="How this scene opens from the previous scene: 'cut' | 'dissolve' | 'fade' | 'match_cut'. "
                    "Derived from the first shot's join_with_prev.",
    )
    music: str = Field(
        default="",
        description="Background-music direction carried from the script scene (mood/tempo/genre + SFX notes) — "
                    "folded into every shot's prompt_video so Veo scores the scene consistently.",
    )
    shots: list[StoryboardShot] = Field(default_factory=list)


class StoryboardDocument(BaseModel):
    title: str
    ingredients: list[str] = Field(
        default_factory=list,
        description="Full recipe ingredient list (name+qty) carried from production — for overview shots",
    )
    equipment: list[str] = Field(
        default_factory=list,
        description="Full tools/equipment list carried from production — for the 'all equipment' overview shot",
    )
    production: dict = Field(
        default_factory=dict,
        description="Production spec (character/theme/lighting/dish/...) carried from the script so prompt_img "
                    "can be regenerated downstream without the script. Empty on storyboards made before this existed.",
    )
    scenes: list[StoryboardScene] = Field(default_factory=list)
