"""Compile the video-factory LangGraph pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import PipelineState


def build_pipeline():
    g = StateGraph(PipelineState)

    g.add_node("generate_queries", nodes.generate_queries)
    g.add_node("search", nodes.search)
    g.add_node("filter", nodes.relevance_filter)
    g.add_node("topup", nodes.topup)
    g.add_node("analyze", nodes.analyze)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("script", nodes.generate_script)
    g.add_node("images", nodes.fetch_images)
    g.add_node("storyboard", nodes.storyboard)
    g.add_node("image_prompts", nodes.image_prompts)
    g.add_node("generate_images", nodes.generate_images)
    g.add_node("no_relevant", nodes.fail_no_relevant)

    g.add_edge(START, "generate_queries")
    g.add_edge("generate_queries", "search")

    g.add_conditional_edges(
        "search", nodes.after_search,
        {"filter": "filter", "no_videos": END},
    )
    g.add_conditional_edges(
        "filter", nodes.after_filter,
        {"analyze": "analyze", "topup": "topup", "no_relevant": "no_relevant"},
    )
    g.add_conditional_edges(
        "topup", nodes.after_topup,
        {"filter": "filter", "analyze": "analyze", "no_relevant": "no_relevant"},
    )
    g.add_conditional_edges(
        "analyze", nodes.after_analyze,
        {"synthesize": "synthesize", "end": END},
    )
    g.add_conditional_edges(
        "synthesize", nodes.after_synthesize,
        {"script": "script", "end": END},
    )
    g.add_edge("script", "storyboard")
    g.add_edge("storyboard", "image_prompts")
    g.add_edge("image_prompts", "images")
    g.add_edge("images", "generate_images")
    g.add_edge("generate_images", END)
    g.add_edge("no_relevant", END)

    return g.compile()


# ─── Per-step sub-graphs ────────────────────────────────────────────────────────
# The same node functions wired into smaller graphs so each pipeline step can run
# on its own — fed by the previous step's JSON and producing the next step's JSON.


def build_research():
    """Step 1: topic → synthesis (search → filter ↔ topup → analyze → synthesize)."""
    g = StateGraph(PipelineState)
    g.add_node("generate_queries", nodes.generate_queries)
    g.add_node("search", nodes.search)
    g.add_node("filter", nodes.relevance_filter)
    g.add_node("topup", nodes.topup)
    g.add_node("analyze", nodes.analyze)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("no_relevant", nodes.fail_no_relevant)

    g.add_edge(START, "generate_queries")
    g.add_edge("generate_queries", "search")
    g.add_conditional_edges("search", nodes.after_search, {"filter": "filter", "no_videos": END})
    g.add_conditional_edges("filter", nodes.after_filter, {"analyze": "analyze", "topup": "topup", "no_relevant": "no_relevant"})
    g.add_conditional_edges("topup", nodes.after_topup, {"filter": "filter", "analyze": "analyze", "no_relevant": "no_relevant"})
    g.add_conditional_edges("analyze", nodes.after_analyze, {"synthesize": "synthesize", "end": END})
    g.add_edge("synthesize", END)
    g.add_edge("no_relevant", END)
    return g.compile()


def build_research_reuse():
    """Step 1 alt entry — "ใช้ source link เดิม": skip generate_queries/search/filter/topup
    entirely and re-analyze a caller-supplied set of reference videos (the route pre-seeds
    `confirmed` from a previous research result's `references`) straight into analyze →
    synthesize. Reuses the exact same source videos instead of searching again."""
    g = StateGraph(PipelineState)
    g.add_node("analyze", nodes.analyze)
    g.add_node("synthesize", nodes.synthesize)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", nodes.after_analyze, {"synthesize": "synthesize", "end": END})
    g.add_edge("synthesize", END)
    return g.compile()


def build_script():
    """Step 2: synthesis → ScriptDocument."""
    g = StateGraph(PipelineState)
    g.add_node("script", nodes.generate_script)
    g.add_edge(START, "script")
    g.add_edge("script", END)
    return g.compile()


def build_regenerate_script_part():
    """Regenerate one part of an existing ScriptDocument (staircase per-part auto-fit)."""
    g = StateGraph(PipelineState)
    g.add_node("regenerate_script_part", nodes.regenerate_script_part)
    g.add_edge(START, "regenerate_script_part")
    g.add_edge("regenerate_script_part", END)
    return g.compile()


def build_storyboard():
    """Step 3: ScriptDocument → StoryboardDocument (shots + image prompts + ref images).

    `storyboard` writes prompt_img as a side-output of the same LLM call that writes the shot text
    (breakdown_scene), which is a weaker author than the dedicated one and never sets `dish_state`.
    Chaining `image_prompts` right after rewrites every prompt_img with that dedicated author, on a
    real dish_state timeline — the step the operator asked to have split out, without splitting the
    UI into two buttons. build_result still persists ONE version, the post-rewrite one.
    """
    g = StateGraph(PipelineState)
    g.add_node("storyboard", nodes.storyboard)
    g.add_node("image_prompts", nodes.image_prompts)
    g.add_node("images", nodes.fetch_images)
    g.add_edge(START, "storyboard")
    g.add_edge("storyboard", "image_prompts")
    g.add_edge("image_prompts", "images")
    g.add_edge("images", END)
    return g.compile()


def build_image_prompts():
    """Step 3b: regenerate prompt_img on an existing StoryboardDocument (decoupled from text breakdown)."""
    g = StateGraph(PipelineState)
    g.add_node("image_prompts", nodes.image_prompts)
    g.add_edge(START, "image_prompts")
    g.add_edge("image_prompts", END)
    return g.compile()


def build_images():
    """Step 4: StoryboardDocument → generated shot images."""
    g = StateGraph(PipelineState)
    g.add_node("generate_images", nodes.generate_images)
    g.add_edge(START, "generate_images")
    g.add_edge("generate_images", END)
    return g.compile()


def build_video_prompts():
    """Step 5: StoryboardDocument → per-shot video (motion) prompts."""
    g = StateGraph(PipelineState)
    g.add_node("generate_video_prompts", nodes.generate_video_prompts)
    g.add_edge(START, "generate_video_prompts")
    g.add_edge("generate_video_prompts", END)
    return g.compile()
