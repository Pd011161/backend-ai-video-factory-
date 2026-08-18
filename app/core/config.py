from pathlib import Path
from typing import Literal
from urllib.parse import quote

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to project root regardless of CWD
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class DatabaseConfig(BaseModel):
    """Connection details minus the password, which belongs in .env (DB_PASSWORD).

    Split into fields rather than one URL so the secret never sits in a committed file, and so the
    password is percent-encoded here instead of by hand — a raw `@` or `%` in a hand-written URL
    breaks the connection string and, for `%`, crashes alembic before any migration runs.
    """
    backend: Literal["sqlite", "postgres"] = "sqlite"

    # postgres
    host: str = "localhost"
    port: int = 5432
    name: str = "videofactory"
    user: str = "videofactory"

    # sqlite — relative paths resolve against the working directory, as SQLAlchemy expects
    path: str = "./app.db"

    def url(self, password: str = "") -> str:
        if self.backend == "sqlite":
            return f"sqlite:///{self.path}"
        cred = quote(self.user, safe="")
        if password:
            cred = f"{cred}:{quote(password, safe='')}"
        return f"postgresql+psycopg://{cred}@{self.host}:{self.port}/{self.name}"


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8090
    log_level: str = "INFO"
    # SEC-4 (opt-in). api_key="" → no auth (dev default, unchanged). When set, every /api call must carry
    # cookie vf_key == api_key (the frontend prompts once and stores it). cors_origins locks the browser
    # origins allowed to call the API; "*" stays open for local dev.
    api_key: str = ""
    cors_origins: list[str] = ["*"]


class GeminiConfig(BaseModel):
    credentials_file: str = "credentials/credentials.json"
    model: str = "gemini-2.5-flash"
    location: str = "us-central1"
    temperature: float = 0.3
    max_output_tokens: int = 8000
    batch_size: int = 3
    # Script writing is a creative-writing task (hook, voice, pacing), not a deterministic one
    # (classify/filter/etc want the low default temperature above for consistent, safe answers).
    # A separate higher temperature here lets the model vary phrasing/rhythm instead of always
    # producing the single most statistically "safe" (= generic-sounding) completion.
    script_temperature: float = 0.85
    # Script generation produces a large structured object — give it more room
    # and a bounded thinking budget so content isn't truncated.
    script_max_output_tokens: int = 32000
    script_thinking_budget: int = 1024


class VideoConfig(BaseModel):
    max_results: int = 10
    search_query_variants: int = 3
    max_duration_seconds: int = 600


class FilterConfig(BaseModel):
    max_output_tokens: int = 1024
    thinking_budget: int = 512


class SearchConfig(BaseModel):
    max_search_rounds: int = 3
    # Stop top-up early once this many consecutive filter rounds add NO new relevant video
    # (search still finds candidates, but none are on-topic). 0 = disabled.
    stop_after_barren_rounds: int = 2


class PipelineConfig(BaseModel):
    analyze_concurrency: int = 4
    recursion_limit: int = 25
    validate_video_prompts: bool = True      # QA-check prompt_video after generation and fix invalid ones
    normalize_prompt_img: bool = True        # normalize prompt_img character/setting consistency across shots per scene
    # After Gemini watches each video, drop clips it flags OFF_TOPIC (wrong subject despite a matching
    # title). False = keep every analyzed clip (previous behavior).
    off_topic_gate: bool = True


class ScriptConfig(BaseModel):
    enabled: bool = True
    # What the topic IS → picks the prompt rule block (prompts.CATEGORY_RULES). "other" = neither,
    # and gets NO block. Same vocabulary as Run.type, which is where this value now comes from.
    category: Literal["food", "drink", "other"] = "food"
    storyboard: bool = True       # break each scene into shots with image prompts
    include_tools_scene: bool = True  # True = separate วัตถุดิบ + อุปกรณ์ scenes; False = ingredients only
    director_id: str = ""         # optional saved director-style guide to attach
    parts: int = 1
    duration_per_part: str = "06:00-06:30"
    target_duration: str = "06:00-06:30"  # auto-fit acceptance window for the estimated runtime (per part)
    # "custom" = duration_per_part is a fixed target (current behavior — hard floor, "reach AT LEAST").
    # "source" = derive the target from the reference videos' own pacing (research step's `pacing` block)
    # instead, with a soft ±30% guard rail instead of a hard floor, so the script doesn't get padded with
    # filler just to hit a fixed window when the source content is naturally shorter.
    duration_mode: Literal["source", "custom"] = "source"
    source_avg_duration_seconds: float = 0     # from research.pacing.avg_duration_seconds (0 = unavailable)
    source_phase_pct: dict[str, float] = Field(default_factory=dict)  # from research.pacing.phase_pct
    words_per_part: int = 0          # target spoken voice_over words PER PART (0 = off)
    words_per_second: float = 0      # speaking pace — words/second (0 = off; word↔time + per-shot voice_over sizing)
    min_shots_per_part: int = 60     # floor on shots PER PART (0 = no floor); min_shots × clip ≈ min duration
    min_scenes_per_part: int = 17    # floor on scenes PER PART (0 = no floor); scenes drive how many shots
    character_name: str = ""
    character_desc: str = ""
    voice_desc: str = ""             # เสียงพากย์: gender/age/tone (e.g. "เสียงผู้หญิงไทยวัย 35 อบอุ่น ชัดถ้อยชัดคำ") — pins Veo's per-clip voice; falls back to brand vo_tone → character_desc
    theme: str = ""
    mood: str = ""
    material_palette: str = ""
    lighting: str = ""


class ImageSearchConfig(BaseModel):
    # Reference image per scene via Tavily. API key comes from env
    # TAVILY_API_KEY (secret, not committed) — no engine id, no OAuth.
    enabled: bool = False
    provider: str = "tavily"
    search_depth: str = "basic"   # basic|advanced
    candidates: int = 3           # reference image candidates kept per visual element
    timeout: int = 15
    concurrency: int = 4
    validation: bool = True           # LLM-validate subjects / queries / candidates
    max_validation_rounds: int = 5    # max regenerate rounds for subjects & queries

    # Vision confirmation: after the cheap text-description filter, download the
    # surviving candidate images and let Gemini actually LOOK at them.
    vision_validation: bool = True    # gate the multimodal confirm pass
    download_timeout: int = 10        # per-image HTTP download timeout (s)
    max_image_bytes: int = 5_000_000  # skip downloads larger than this
    resize_px: int = 512              # longest edge sent to Gemini (downscale to save tokens)
    vision_batch_images: int = 12     # max images packed into one vision call


class GeminiImageConfig(BaseModel):
    # Vertex / google-genai image model (gemini-2.5-flash-image).
    model: str = "gemini-2.5-flash-image"
    aspect_ratio: str = "16:9"    # native, no letterbox — 1:1 | 16:9 | 9:16 | 4:3 | 3:4


class OpenAIImageConfig(BaseModel):
    model: str = "gpt-image-2"
    size: str = "1536x864"        # 1024x1024 | 1536x864 (16:9) | 1536x1024 (3:2 landscape) | 1024x1536
    quality: str = "low"          # "low" | "medium" | "high" | "auto" (high≈5min/img, low≈fast)


class OutputQCConfig(BaseModel):
    # C1 (cinema): vision QC on GENERATED shot images — Gemini LOOKS at each render and
    # fails it on rule violations (person in an insert, missing/extra subjects, anatomy
    # artifacts, burned-in text). A failed shot is re-generated with the QC feedback
    # appended to its prompt, up to max_regens extra attempts; the last render is kept
    # (flagged) if every attempt fails. Fail-open: a QC call/parse error never blocks.
    enabled: bool = False
    max_regens: int = 2          # extra generation attempts after a FAIL (0 = judge only, never regen)


class ImageGenConfig(BaseModel):
    # Generate an image per storyboard shot from its prompt_img.
    enabled: bool = False
    provider: str = "gemini"      # which sub-config to use: "gemini" (Vertex) | "openai" (gpt-image-*)
    gemini: GeminiImageConfig = Field(default_factory=GeminiImageConfig)
    openai: OpenAIImageConfig = Field(default_factory=OpenAIImageConfig)
    concurrency: int = 3
    # Reference images applied to EVERY generated shot (path under project root, or URL).
    # These are the ONLY references used in generation — they keep the person and the
    # background identical across all shots.
    character_ref: str = ""       # the person — kept consistent every shot
    scene_ref: str = ""           # the place / background — kept consistent every shot
    # True  → render the shot's on_screen_text (Thai) into the generated image.
    # False → no text in the image (overlay the caption later in editing).
    on_screen_text_in_image: bool = False
    # True  → inject the Theme text into prompt_img generation (current behavior).
    # False → drop Theme from prompt_img so the scene comes purely from scene_ref
    #         (avoids the Theme text fighting the scene_ref image).
    use_theme_in_prompt: bool = True
    # Feed the shot's searched reference photos (the matched ingredient/tool images)
    # into generation, on top of character_ref + scene_ref, so real objects ground
    # the image. max_subject_refs caps how many per shot (quality ~3-4 total refs).
    subject_refs_in_image: bool = True
    max_subject_refs: int = 2
    # Dynamic per-shot flow (classify → prompt+ref → map → generate). False = legacy
    # render_shot_image path (one fresh image per shot, no prev-frame chaining).
    dynamic_image: bool = False
    use_prev_as_ref: bool = True   # allow reusing/adapting the previous shot's image in a scene
    max_total_refs: int = 6        # cap on refs to the image model (prev+person+kitchen+ingredient+equipment+state+keyword); 6 fits a cooking shot's anchors, higher risks gpt-image compositing stray bits
    # Fixtures that ALREADY exist in scene_ref (the kitchen). When a shot's ref keywords name one of
    # these, it is dropped (no stock/extra ref) and the prompt is told to use the existing one — so the
    # model doesn't invent a second/mismatched sink, faucet, stove, etc.
    kitchen_fixtures: list[str] = Field(default_factory=lambda: [
        "sink", "อ่างล้างจาน", "อ่าง", "faucet", "tap", "ก๊อกน้ำ", "ก๊อก", "stove", "hob", "เตา", "เตาแก๊ส",
        "oven", "เตาอบ", "counter", "countertop", "เคาน์เตอร์", "cabinet", "cupboard", "ตู้", "ตู้ครัว",
        "window", "หน้าต่าง", "range hood", "hood", "เครื่องดูดควัน", "microwave", "ไมโครเวฟ",
        "refrigerator", "fridge", "ตู้เย็น", "wall", "ผนัง", "floor", "พื้น",
    ])
    # Longest-edge resize (px) for reference images sent to the image model.
    # ref_resize_px → scene_ref + subject reference photos; character_ref gets its
    # own (larger) size so the person's face/outfit keeps more detail.
    ref_resize_px: int = 512
    character_ref_resize_px: int = 768
    # C1 (cinema): output QC loop on generated shot images (see OutputQCConfig).
    output_qc: OutputQCConfig = Field(default_factory=OutputQCConfig)


class PostColorConfig(BaseModel):
    # C2 (cinema): post color pipeline at assembly time. Two stages:
    #   1. COLOR MATCH (per clip, BEFORE assembly) — measure each clip's mean luma/saturation
    #      (ffmpeg signalstats) and pull outliers toward the batch median with a small `eq`
    #      correction, so the look doesn't jump across cuts (the most visible AI tell).
    #   2. GRADE (final.mp4, ONE re-encode) — apply the brand LUT (.cube via lut3d) +
    #      saturation/contrast trim + film grain (noise) for a unified filmic finish.
    # Everything is CPU-only ffmpeg — no extra API cost.
    enabled: bool = False
    match_colors: bool = True          # stage 1 on/off
    max_brightness_shift: float = 0.05 # clamp on the per-clip eq brightness correction (±, 0..1 scale)
    max_saturation_ratio: float = 1.12 # clamp on the per-clip saturation correction (ratio, both directions)
    min_correction: float = 0.02       # corrections smaller than this are skipped — invisible to the eye,
                                       # and every skipped correction saves a lossy per-clip re-encode
    lut: str = ""                      # .cube LUT path (project-relative ok); active brand's `lut` field overrides
    saturation: float = 1.0            # final-grade saturation trim (1.0 = off)
    contrast: float = 1.0              # final-grade contrast trim (1.0 = off)
    grain: float = 0.0                 # film grain strength (ffmpeg noise alls=N, ~3-6 subtle; 0 = off)


class EpisodeGateConfig(BaseModel):
    # C3 (cinema/TOR): measure the REAL final.mp4 with ffprobe after merge and emit a
    # pass/fail report (SSE `episode_gate` event + folded into `merged`). NON-BLOCKING:
    # a failing episode still saves — the gate exists so nobody discovers a 720p file or
    # a 7-minute part at delivery time. 0 = that check is off.
    enabled: bool = False
    min_seconds: float = 0             # e.g. 360  (TOR 6:00)
    max_seconds: float = 0             # e.g. 390  (TOR 6:30)
    required_width: int = 0            # e.g. 1920 (TOR 1080p 16:9)
    required_height: int = 0           # e.g. 1080
    required_fps: float = 0            # e.g. 24 (tolerance ±0.5)


class VideoGenConfig(BaseModel):
    # Generate a video clip per storyboard shot from its prompt_video (Veo).
    enabled: bool = True
    # "gemini_api" → Gemini Developer API (env GEMINI_API_KEY; where veo-3.1-lite lives)
    # "vertex"     → Vertex AI (service account project) — different model availability
    provider: str = "gemini_api"
    model: str = "veo-3.1-lite-generate-preview"
    # first+last-frame (last_frame) is only offered by the STANDARD model — fast/lite
    # 400 "use case not supported". Shots using last_frame switch to this model.
    interpolation_model: str = "veo-3.1-generate-preview"
    # Model used ONLY when a shot's per-call `engine: "omni"` override forces it onto Gemini Omni
    # Flash (see /steps/video) — independent of `model` above, so a storyboard can mix Veo shots
    # (model) with Omni shots (omni_model) without a global config switch.
    omni_model: str = "gemini-omni-flash-preview"
    resolution: str = "720p"             # 720p | 1080p
    duration_seconds: int = 6            # seconds per generated clip
    aspect_ratio: str = "16:9"           # 16:9 | 9:16
    generate_audio: bool = False         # True = Veo adds audio; False = silent (overlay VO later) — Vertex only
    person_generation: str = "allow_adult"  # "allow_adult" | "dont_allow" — needed so person frames aren't filtered
    use_image_seed: bool = True          # use the shot's generated_img as the first frame (image-to-video)
    output_gcs_uri: str = ""             # optional gs:// bucket — only if Veo/Vertex requires one
    poll_interval_seconds: int = 8       # how often to poll the long-running operation

    # ── Cinematic prompt quality (Phase 0) ──
    # Enforce the Veo 5-element structure + camera/lens/lighting/audio vocabulary when
    # writing prompt_video. False = the plain motion prompt (previous behavior).
    cinematic_prompt: bool = True
    # A fixed colour-grade / lighting token folded into EVERY shot's prompt_video so the
    # look stays consistent across the whole film. Empty → derived from the production
    # spec (lighting / material_palette / mood) at prompt time.
    color_grade: str = "warm 3200K key light, soft shadows, shallow depth of field, cinematic color"
    # Veo negative prompt — plain comma-separated terms (per Veo guidance: describe what you don't
    # want, never "no"/"don't"). Sent with EVERY video generation (main + chain), so keep it to
    # ALWAYS-unwanted artifacts — context-dependent physics (e.g. flames: wrong on tea, right on a
    # wok) is enforced in VIDEO_PROMPT_PROMPT's PHYSICS RULES instead. "" = off.
    negative_prompt: str = ("subtitles, captions, on-screen text, watermark, logo, morphing objects, "
                            "warping, deformed hands, extra fingers, duplicate person, floating objects, "
                            "teleporting, jump cut, scene change, cartoon, drawing, low quality")
    # Vision QC on GENERATED clips (mirror of image_gen.output_qc): Gemini WATCHES each per-shot render
    # and fails it on state drift / object morph / containment / foreign scene / burned-in text.
    # RETIRED (2026-08): the operator stopped using QC — code kept intact, re-enable via config.yaml.
    # Default off so a missing/reset yaml cannot silently bring the per-render QC cost back.
    output_qc: OutputQCConfig = Field(default_factory=OutputQCConfig)

    # ── Transitions / chaining (Phase 1–2) ──
    # False → every shot boundary is a plain hard cut (previous behavior); the join_with_prev
    # metadata is ignored at render/assembly.
    transitions_enabled: bool = True
    default_join: str = "cut"            # fallback when a shot has no join_with_prev tag
    min_chain_shots: int = 2             # only extension-chain a run of >= this many `continuous` shots
    rai_retries: int = 3                 # retries when Veo drops a clip on the safety/audio (RAI) filter
    retry_backoff_seconds: int = 15      # base wait before each retry (× attempt; ×2 for a code-13 backend error) — stop hammering a struggling Veo backend

    # ── Assembly / ffmpeg (Phase 3) ──
    dissolve_seconds: float = 0.4        # xfade length for a join=dissolve boundary
    scene_fade_seconds: float = 0.5      # fade/xfade length at a scene boundary (transition_in=fade/dissolve)
    bookend_fade_seconds: float = 0.5    # fade-in at the very start / fade-out at the very end (0 = off)
    audio_crossfade: bool = True         # acrossfade the audio together with a dissolve
    reencode_fps: int = 24               # fps used when re-encoding clips to match (0 = try stream-copy concat)

    # ── Extended editing (Phase 4) ──
    jl_cut_seconds: float = 0.5          # audio lead/trail overlap for a J-cut / L-cut boundary
    subtitles_in_video: bool = False     # burn each clip's on_screen_text into final.mp4 as soft subtitles
    subtitle_font: str = "assets/fonts/NotoSansThai-Regular.ttf"  # bundled Thai font (family auto-detected)
    subtitle_font_size: int = 56         # POC parity — large, readable over video
    # Narrow each clip's subtitle span to where speech ACTUALLY occurs (Silero VAD, ~2 MB model
    # bundled in faster-whisper — cheap enough for Render free tier) instead of the clip's full
    # on-screen duration — so a caption appears on speech-onset and clears when the line ends.
    subtitle_sync_to_speech: bool = True
    ffmpeg_bin: str = "ffmpeg"           # ffmpeg binary; resolver falls back to a libass build if this lacks it

    # ── Post color pipeline + episode gate (C2/C3 cinema) ──
    post_color: PostColorConfig = Field(default_factory=PostColorConfig)
    episode_gate: EpisodeGateConfig = Field(default_factory=EpisodeGateConfig)


class LangfuseConfig(BaseModel):
    # Secrets (public_key / secret_key) are NOT read from here — they come from
    # the env vars LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY. Only non-secret
    # settings live in config.yaml.
    enabled: bool = False
    host: str = "https://cloud.langfuse.com"
    environment: str = "development"
    sample_rate: float = 1.0
    debug: bool = False
    # Log images/videos as media in traces — requires S3 media storage on the Langfuse
    # instance. Leave false for self-hosted without it (else "media upload Connection refused").
    log_media: bool = False


class S3RefsConfig(BaseModel):
    # Public S3 base for ingredient/equipment reference photos. A menu subject's id is the object key:
    # {base_url}/{subject_id}.{ext}; each ext tried in order until one is a real image ("" = off).
    enabled: bool = False
    base_url: str = ""                                        # e.g. https://bucket.s3.<region>.amazonaws.com/refs
    extensions: list[str] = ["jpg", "jpeg", "png", "webp"]    # object-key extensions to try, in order


class TTSConfig(BaseModel):
    # Text-to-speech for voice-over (Gemini TTS via Vertex AI).
    # Uses the same credentials.json service account as the text-generation client — no GEMINI_API_KEY needed.
    provider: str = "gemini"
    model: str = "gemini-2.5-flash-preview-tts"
    voice: str = "Kore"          # Gemini prebuilt voice (Kore / Aoede / Leda / Puck / Charon / ...)
    style: str = ""              # optional delivery instruction prepended to the text (e.g. "อ่านอบอุ่น เป็นกันเอง")
    sample_rate: int = 24000     # Gemini TTS PCM rate — don't change unless the model output changes
    # Re-voice (app/services/revoice.py): detect WHEN Veo starts speaking (STT for timing only,
    # text discarded) and land the TTS there instead of always at t=0. False = always t=0 (safety
    # valve if detection proves flaky/slow).
    detect_timestamp: bool = True


class VoiceConfig(BaseModel):
    """The narrator's voice for Omni's `Voice direction:` block (OMNI_VOICE_BLOCK).

    NOT TTSConfig — that is the re-voice path (a different model, a different consumer). These
    five values were literal defaults in THREE places (Step 5's useState, VideoPromptsStepRequest,
    and the voice_cfg dict in generate_video_prompts) until they became config; this is now the
    single floor, overridden by config_refs and then by anything the request explicitly sends.

    Changing these only affects prompts written AFTER the change — the block is baked into
    prompt_video at Step 5 and render never rebuilds it."""
    language: str = "Thai"
    gender: Literal["female", "male"] = "female"
    vo_pace: float = 2.5             # words per second
    tone: str = "bright, cheerful, playful, energetic, and friendly"
    style: str = "natural conversational delivery with expressive intonation and subtle vocal emphasis"


class Settings(BaseSettings):
    """v2 settings: pydantic-settings for env/.env (database_url), plus the
    optional config.yaml-driven pipeline/service settings ported from v1.
    All Drive-specific config (ResearchDriveConfig/DriveCategoryConfig) has been
    dropped — v2 has no Drive dependency.
    """

    # Absolute, not ".env": a relative path resolves against the working directory, so running
    # `alembic upgrade head` from the repo root instead of backend/ silently loaded no .env at all —
    # DB_PASSWORD came back empty and the connection failed while config.yaml (which IS found from
    # anywhere, being anchored to ROOT_DIR) still said postgres.
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # Non-secret connection parts live in config.yaml; the password comes from the environment and
    # the two are joined below. `database_url` stays as a full-URL escape hatch because Docker
    # Compose and CI want to hand over one string — when DATABASE_URL is set it wins outright.
    database: DatabaseConfig = DatabaseConfig()
    db_password: str = ""          # env: DB_PASSWORD
    database_url: str = ""         # env: DATABASE_URL — overrides everything above when non-empty

    app: AppConfig = AppConfig()
    gemini: GeminiConfig = GeminiConfig()
    video: VideoConfig = VideoConfig()
    filter: FilterConfig = FilterConfig()
    search: SearchConfig = SearchConfig()
    pipeline: PipelineConfig = PipelineConfig()
    script: ScriptConfig = ScriptConfig()
    image_search: ImageSearchConfig = ImageSearchConfig()
    image_gen: ImageGenConfig = ImageGenConfig()
    video_gen: VideoGenConfig = VideoGenConfig()
    tts: TTSConfig = TTSConfig()
    voice: VoiceConfig = VoiceConfig()
    langfuse: LangfuseConfig = LangfuseConfig()
    s3_refs: S3RefsConfig = S3RefsConfig()

    @model_validator(mode="after")
    def _assemble_database_url(self) -> "Settings":
        """Everything downstream (db/base.py, migrations/env.py) reads `database_url`, so build it
        here once from the config.yaml parts + DB_PASSWORD unless a full DATABASE_URL was given."""
        if not self.database_url:
            self.database_url = self.database.url(self.db_password)
        return self

    @classmethod
    def load(cls) -> "Settings":
        config_path = ROOT_DIR / "config.yaml"
        yaml_data: dict = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        # cls(**yaml_data) still goes through BaseSettings.__init__, so env vars / .env are layered
        # in via the normal settings sources — EXCEPT that pydantic-settings ranks explicit init
        # kwargs ABOVE the environment, so anything named in config.yaml silently wins over its env
        # var. That is fine (and intended) for the `database:` parts, but a full `database_url`
        # there would beat DATABASE_URL and point at the wrong database with no error at all.
        for secret in ("database_url", "db_password"):
            if secret in yaml_data:
                raise ValueError(
                    f"config.yaml must not set `{secret}` — it would silently override the "
                    f"environment. Put it in .env instead (config.yaml holds the non-secret "
                    f"`database:` parts only)."
                )
        return cls(**yaml_data)


settings = Settings.load()
