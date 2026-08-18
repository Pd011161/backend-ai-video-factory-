"""LangGraph pipeline state + helpers.

The state is a plain TypedDict threaded through every node. Nodes return partial
dicts that LangGraph merges into the running state (last-write-wins per key).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.config import get_stream_writer

from app.core.config import ScriptConfig
from app.core.container import ServiceContainer
from langchain_core.runnables import RunnableConfig


class PipelineState(TypedDict, total=False):
    # inputs
    topic: str
    target: int
    script_config: ScriptConfig
    use_image_seed: bool      # video-prompt frame mode (chosen at Step 5)
    use_last_frame: bool
    # Omni video-prompt authoring (Step 5): image-ref manifest per shot + narration voice config.
    # MUST be declared here — LangGraph drops any state key not in this TypedDict.
    omni_manifests: dict      # {"<scene_id>:<no>": {"header": str, "listing": str}}
    # What the user said their own attached reference photos are — reaches the prompt author so a
    # per-shot "↻ Prompt" rewrite can mention refs it never sees. See ImagePromptsStepRequest.
    ref_notes: list[str]
    vo_language: str
    vo_pace: float
    vo_tone: str
    vo_style: str
    vo_gender: str

    # search / filter
    queries: list[str]
    pool: list[dict]          # unfiltered candidates pending relevance filter
    seen_ids: set[str]        # every video id we've already considered
    confirmed: list[dict]     # videos that passed the relevance filter
    round_num: int            # current top-up search round (1-based)
    barren_rounds: int        # consecutive filter rounds that added NO new relevant video

    # analysis / synthesis
    summaries: list[str]
    video_pacing: list[dict]  # per-video {title, duration_seconds, phases} — feeds research's `pacing` summary
    synthesis: str
    voice_samples: list[str]  # real host-voice quotes pooled from research — grounds Script's voice/tone
    presentation_notes: list[str]  # real "how they run the show" notes pooled from research — grounds Script's pacing/hosting-flow craft
    master_url: str  # research: which confirmed video (by url) to treat as the master reference ("" = none)
    master_index: int | None  # resolved master (explicit match OR auto-suggested) — index into `confirmed`/`video_pacing`
    ingredients: list[str]    # research: structured list extracted from the synthesis (seeds the run's Menu)
    equipment: list[str]
    # Script: the run already has a Menu → its lists are authoritative. generate_script sends them to
    # the LLM AND force-overwrites production with them, so a Menu deletion sticks across regenerates.
    menu_locked: bool
    menu_ingredients: list[str]
    menu_equipment: list[str]
    script_doc_obj: Any       # the ScriptDocument pydantic object, threaded between script nodes
    script_doc: dict | None   # final dumped form
    part_number: int          # staircase per-part regen: which part to regenerate (1-based)
    target_duration: str      # staircase per-part regen: acceptance window for that part, e.g. "06:00-06:30"
    target_scenes: int        # staircase per-part regen: target scene count for that part (0 = use config floor)
    target_shots: int         # storyboard breakdown: soft target total shots for the doc (0 = off)
    min_shots: int            # staircase per-part regen: explicit shot floor passed to the regen prompt (0 = off)
    storyboard_obj: Any       # the StoryboardDocument object, threaded storyboard → images
    storyboard: dict | None   # final dumped form
    regenerate_all: bool      # images step: False = skip shots that already have generated_img (default True)

    # terminal error (set by a node that hard-stops the flow)
    error: str | None


def services(config: RunnableConfig) -> ServiceContainer:
    """Pull the injected service container out of the runnable config."""
    return config["configurable"]["services"]


def emit(event: dict[str, Any]) -> None:
    """Forward an SSE event to the streaming consumer (no-op outside a stream)."""
    writer = get_stream_writer()
    if writer is not None:
        writer(event)
