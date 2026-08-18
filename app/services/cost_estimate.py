"""Rough $/THB cost estimate for usage_records, computed from EDITABLE rates in the `cost_rates`
table (see app/repositories/cost_rate_repo.py) instead of hardcoded constants — provider prices
change, and hardcoding them means the estimate silently goes stale with no way to fix it short of a
code deploy. Editing a rate in the Usage Report UI takes effect on the next report load.

Still an ESTIMATE, not an invoice, no matter how current the rates are:
  - Veo/Omni video calls never carry token usage at all (record_usage(..., None, ...) — see
    gemini_client.py's veo.*/omni.* calls), so their cost is a single flat-per-clip guess, not a
    real per-second/per-token computation.
  - gpt-image-2 calls DO sometimes carry real token usage (openai_image.py's usage_dict), but our
    usage_records schema stores one blended "prompt"/"completion" pair, not OpenAI's actual
    text-input / image-input / image-output token breakdown — so even the token-based branch is an
    approximation of the real bill, not a reproduction of it.

DEFAULT_RATES below is a snapshot fetched 2026-07-23 from ai.google.dev/gemini-api/docs/pricing
(Gemini 2.5 Flash standard tier, Veo 3.1 Fast tier @720p, Gemini Omni Flash Preview) and OpenAI's
published gpt-image-2 token pricing (as summarized by third-party trackers — OpenAI's own pricing
page should be treated as the source of truth if these ever look wrong). These seed the `cost_rates`
table ONCE, the first time it's read empty; after that, DB values win even if this snapshot goes
out of date, and only an explicit edit in the UI changes them again.
"""
from __future__ import annotations

# key -> (label, unit, default value)
DEFAULT_RATES: dict[str, tuple[str, str, float]] = {
    "gemini_text_input_per_1m": ("Gemini 2.5 Flash — input", "USD / 1M tokens", 0.30),
    "gemini_text_output_per_1m": ("Gemini 2.5 Flash — output", "USD / 1M tokens", 2.50),
    "omni_input_per_1m": ("Gemini Omni Flash — input", "USD / 1M tokens", 1.50),
    "omni_output_text_per_1m": ("Gemini Omni Flash — output (text)", "USD / 1M tokens", 9.00),
    "omni_output_video_per_1m": ("Gemini Omni Flash — output (video)", "USD / 1M tokens", 17.50),
    "omni_video_flat_per_call": ("Omni video call (fallback — no token data returned)", "USD / call", 0.15),
    "veo_per_clip": ("Veo clip (Fast tier, 720p, ~6s — config default)", "USD / clip", 0.60),
    "openai_image_input_per_1m": ("gpt-image-2 — input tokens", "USD / 1M tokens", 8.00),
    "openai_image_output_per_1m": ("gpt-image-2 — output tokens", "USD / 1M tokens", 30.00),
    "openai_image_low_flat": ("gpt-image-2 image (fallback — low quality, no token data)", "USD / image", 0.005),
    "usd_to_thb": ("USD → THB", "THB per USD", 36.0),
}


def estimate_usd(agent: str, prompt_tokens: int, completion_tokens: int, calls: int, rates: dict[str, float]) -> float:
    """Best-effort USD estimate for one usage bucket (or a single record, with calls=1).
    `rates` is the live key→value map from the `cost_rates` table (see CostRateRepo.as_dict)."""
    has_tokens = prompt_tokens > 0 or completion_tokens > 0

    if agent.startswith("veo."):
        return rates.get("veo_per_clip", 0.0) * calls

    if agent.startswith("omni."):
        if has_tokens:
            out_rate = rates.get("omni_output_video_per_1m", 0.0) if "video" in agent else rates.get("omni_output_text_per_1m", 0.0)
            return (prompt_tokens / 1_000_000) * rates.get("omni_input_per_1m", 0.0) + (completion_tokens / 1_000_000) * out_rate
        return rates.get("omni_video_flat_per_call", 0.0) * calls

    if agent.startswith("openai."):
        if has_tokens:
            return (prompt_tokens / 1_000_000) * rates.get("openai_image_input_per_1m", 0.0) + (
                completion_tokens / 1_000_000
            ) * rates.get("openai_image_output_per_1m", 0.0)
        return rates.get("openai_image_low_flat", 0.0) * calls

    # Default: Gemini text/vision calls (agent prefixed "gemini.") — token-billed.
    return (prompt_tokens / 1_000_000) * rates.get("gemini_text_input_per_1m", 0.0) + (
        completion_tokens / 1_000_000
    ) * rates.get("gemini_text_output_per_1m", 0.0)


def usd_to_thb(usd: float, rates: dict[str, float]) -> float:
    return usd * rates.get("usd_to_thb", 36.0)
