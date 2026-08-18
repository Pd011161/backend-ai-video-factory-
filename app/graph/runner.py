"""Drive the compiled pipeline and stream SSE events to the caller.

Wraps the whole graph execution in a single root Langfuse span so every node and
LLM generation nests under one trace — start of flow to end.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from loguru import logger

from app.core import observability as obs
from app.core.config import ScriptConfig
from app.core.container import ServiceContainer
from app.core.usage import UsageTracker, set_tracker


async def run_pipeline(
    graph: Any,
    container: ServiceContainer,
    topic: str,
    script_config: ScriptConfig,
    *,
    run_id: str | None = None,
) -> AsyncIterator[dict]:
    settings = container.settings
    initial: dict[str, Any] = {
        "topic": topic,
        "target": settings.video.max_results,
        "script_config": script_config,
    }
    config = {
        "configurable": {"services": container},
        "recursion_limit": settings.pipeline.recursion_limit,
    }

    # Per-run token accounting, published so every (possibly threaded) Gemini call
    # can record into it. Isolated per request via the task's context.
    tracker = UsageTracker()
    set_tracker(tracker)
    _t0 = time.perf_counter()

    had_error = False
    with obs.step_trace("video-factory", run_id=run_id, step="synthesize", topic=topic) as root:
        try:
            async for event in graph.astream(initial, config=config, stream_mode="custom"):
                if isinstance(event, dict) and event.get("type") == "error":
                    had_error = True
                yield event

            # Always report token usage (even on error — partial work still costs).
            usage = tracker.summary()
            logger.info(
                f"Run token usage — total={usage['totals']['total']:,} "
                f"across {usage['totals']['calls']} calls, {len(usage['agents'])} agents"
            )
            yield {"type": "usage", "topic": topic, "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage}

            if not had_error:
                yield {"type": "done"}

            root.set_trace_io(
                input={"topic": topic},
                output={"status": "error" if had_error else "ok", "tokens": usage["totals"]["total"]},
            )
        except asyncio.CancelledError:
            # client disconnected (Stop button / closed tab) — astream is torn down with us
            logger.info("Pipeline run cancelled by client disconnect")
            root.update(level="WARNING", status_message="cancelled")
            raise
        except Exception as e:  # noqa: BLE001 — surface any failure as a stream error
            logger.exception("Pipeline error")
            root.update(level="ERROR", status_message=str(e))
            yield {"type": "error", "message": str(e)}


async def run_step(
    graph: Any,
    container: ServiceContainer,
    initial: dict[str, Any],
    build_result: Callable[[dict], Any],
    *,
    run_id: str | None = None,
    step: str | None = None,
) -> AsyncIterator[dict]:
    """Run a single-step sub-graph and stream its events, ending with the step's
    output JSON (``step_result``). Same node events as the full pipeline, but the
    caller hands in the ready initial state and a function that turns the final
    state into the JSON payload for the next step.
    """
    config = {
        "configurable": {"services": container},
        "recursion_limit": container.settings.pipeline.recursion_limit,
    }
    tracker = UsageTracker()
    set_tracker(tracker)
    _t0 = time.perf_counter()

    # Sanitized input for the trace — full scalars, big pydantic objects as a marker.
    trace_input = {
        k: (v if isinstance(v, (str, int, float, bool, type(None))) else f"<{type(v).__name__}>")
        for k, v in initial.items()
    }

    had_error = False
    final_state: dict = {}
    with obs.step_trace(f"step.{step}" if step else "video-factory-step", run_id=run_id, step=step, **trace_input) as root:
        try:
            async for mode, chunk in graph.astream(initial, config=config, stream_mode=["custom", "values"]):
                if mode == "custom":
                    if isinstance(chunk, dict) and chunk.get("type") == "error":
                        had_error = True
                    yield chunk
                else:  # "values" — latest full-state snapshot
                    final_state = chunk

            usage = tracker.summary()
            logger.info(
                f"Step token usage — total={usage['totals']['total']:,} "
                f"across {usage['totals']['calls']} calls, {len(usage['agents'])} agents"
            )
            yield {"type": "usage", "elapsed_ms": round((time.perf_counter() - _t0) * 1000), **usage}

            result = build_result(final_state) if not had_error else None
            # Full step I/O on the trace — the exact output JSON + token usage.
            root.set_trace_io(
                input=trace_input,
                output={"status": "error" if had_error else "ok",
                        "tokens": usage["totals"]["total"], "result": result},
            )

            if not had_error:
                yield {"type": "step_result", "data": result}
                yield {"type": "done"}
        except asyncio.CancelledError:
            # client disconnected (Stop button / closed tab) — astream is torn down with us
            logger.info("Step run cancelled by client disconnect")
            root.update(level="WARNING", status_message="cancelled")
            raise
        except Exception as e:  # noqa: BLE001 — surface any failure as a stream error
            logger.exception("Step error")
            root.update(level="ERROR", status_message=str(e))
            yield {"type": "error", "message": str(e)}
