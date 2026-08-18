"""Langfuse observability — initialized once at startup, no-op safe when disabled.

We hold our own reference to the configured Langfuse client rather than relying
on `langfuse.get_client()`, so tracing works deterministically regardless of how
many clients exist in the process. Every helper degrades to a no-op context
manager when Langfuse is disabled or unconfigured, so call sites stay clean and
never need to branch on whether tracing is on.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from loguru import logger

from app.core.config import LangfuseConfig

_client: Any | None = None
_log_media: bool = False  # gate media (image/video) logging — needs S3 on the Langfuse instance


def init_langfuse(cfg: LangfuseConfig) -> Any | None:
    """Initialize the global Langfuse client. Returns the client or None.

    Secrets are read from the environment only (LANGFUSE_PUBLIC_KEY /
    LANGFUSE_SECRET_KEY), never from config.yaml. Host may be overridden by
    LANGFUSE_HOST, otherwise it falls back to the non-secret value in config.
    """
    global _client, _log_media
    _log_media = bool(getattr(cfg, "log_media", False))

    if not cfg.enabled:
        logger.info("Langfuse disabled (langfuse.enabled=false)")
        _client = None
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST") or cfg.host

    if not (public_key and secret_key):
        logger.warning(
            "Langfuse enabled but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set "
            "in the environment — tracing disabled"
        )
        _client = None
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("langfuse not installed — tracing disabled")
        _client = None
        return None

    try:
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            environment=cfg.environment,
            sample_rate=cfg.sample_rate,
            debug=cfg.debug,
            tracing_enabled=True,
        )
    except Exception as e:
        logger.warning(f"Langfuse init failed — tracing disabled: {e}")
        _client = None
        return None

    try:
        ok = _client.auth_check()
    except Exception as e:
        logger.debug(f"Langfuse auth_check raised: {e}")
        ok = True  # don't block startup on a transient check

    if not ok:
        logger.warning(
            "Langfuse auth_check failed (bad/missing keys) — traces will not be delivered"
        )
    else:
        logger.info(f"Langfuse initialized (env={cfg.environment}, host={host})")
    return _client


def get_langfuse() -> Any | None:
    return _client


def shutdown_langfuse() -> None:
    """Flush any buffered events on shutdown."""
    if _client is not None:
        try:
            _client.flush()
            logger.debug("Langfuse flushed")
        except Exception as e:
            logger.warning(f"Langfuse flush failed: {e}")


class _NoopObservation:
    """Stand-in yielded when tracing is disabled — accepts any call."""

    def update(self, *args: Any, **kwargs: Any) -> "_NoopObservation":
        return self

    def set_trace_io(self, *args: Any, **kwargs: Any) -> "_NoopObservation":
        return self

    def __getattr__(self, _name: str) -> Any:  # tolerate any other span method
        return lambda *a, **k: self


@contextmanager
def span(name: str, *, input: Any = None, metadata: dict | None = None) -> Iterator[Any]:
    """Open a Langfuse span as the current observation (no-op when disabled)."""
    if _client is None:
        yield _NoopObservation()
        return
    with _client.start_as_current_observation(
        name=name, as_type="span", input=input, metadata=metadata
    ) as obs:
        yield obs


@contextmanager
def step_trace(name: str, *, run_id: str | None = None, step: str | None = None, **meta: Any) -> Iterator[Any]:
    """Root span for one step/endpoint. Wraps the root in `propagate_attributes` so session_id=run_id
    and tags=[step] flow to EVERY child span — a whole video run then reads as one session in the
    Langfuse UI, filterable by step. `meta` (scene_id/no/shot_id…) rides as trace metadata (values
    coerced to str ≤200 by the SDK). No-op when tracing is disabled."""
    if _client is None:
        yield _NoopObservation()
        return
    tags = [t for t in [step] if t]
    meta_str = {k: str(v) for k, v in {"run_id": run_id, "step": step, **meta}.items() if v is not None}
    try:
        from langfuse import propagate_attributes
        cm = propagate_attributes(session_id=run_id, tags=tags or None, metadata=meta_str or None)
    except Exception:  # noqa: BLE001 — fall back to a plain span if the API shape changes
        cm = None
    if cm is None:
        with span(name, input=meta_str or None, metadata=meta_str or None) as obs:
            yield obs
        return
    with cm:
        with span(name, input=meta_str or None, metadata=meta_str or None) as obs:
            yield obs


@contextmanager
def generation(
    name: str,
    *,
    model: str | None = None,
    input: Any = None,
    metadata: dict | None = None,
) -> Iterator[Any]:
    """Open a Langfuse generation observation (no-op when disabled)."""
    if _client is None:
        yield _NoopObservation()
        return
    with _client.start_as_current_observation(
        name=name, as_type="generation", model=model, input=input, metadata=metadata
    ) as obs:
        yield obs


def record_cost(agent: str, usage: dict | None) -> None:
    """Attach an estimated USD cost to the CURRENT generation so it shows in the Langfuse UI
    (independent of the DB cost report). Called from record_usage (inside the generation context),
    so it covers every LLM/media call in one place. Uses the DEFAULT_RATES snapshot — best-effort,
    never breaks a call. `usage` is the {input,output,total} dict passed to usage_details."""
    if _client is None:
        return
    try:
        from app.services.cost_estimate import DEFAULT_RATES, estimate_usd
        rates = {k: v[2] for k, v in DEFAULT_RATES.items()}
        usd = estimate_usd(agent, int((usage or {}).get("input", 0) or 0), int((usage or {}).get("output", 0) or 0), 1, rates)
        _client.update_current_generation(cost_details={"total": usd})
    except Exception as e:  # noqa: BLE001
        logger.debug(f"record_cost failed for {agent}: {e}")


async def traced_stream(gen: Any, name: str, *, run_id: str | None = None, step: str | None = None, **meta: Any):
    """Wrap an async SSE generator in a step_trace root span WITHOUT re-indenting the endpoint body.
    Every observation created while iterating `gen` nests under this root and inherits session_id=
    run_id + tags (via step_trace/propagate_attributes). Use for per-shot endpoints that call the
    services directly (no run_step). No-op wrapping when tracing is disabled."""
    with step_trace(name, run_id=run_id, step=step or name, **meta):
        async for ev in gen:
            yield ev


def attach_media(url: str | None, kind: str = "media") -> None:
    """Put OUR system's S3 URL (from _persist_media/_storage — the same link the frontend plays) on
    the CURRENT observation as a clickable field + trace output. Not a Langfuse media upload — just
    the link. Call right after persisting a rendered image/video inside a traced context."""
    if _client is None or not url:
        return
    try:
        _client.update_current_span(metadata={f"{kind}_url": url}, output={"url": url})
    except Exception as e:  # noqa: BLE001
        logger.debug(f"attach_media failed: {e}")


def media(content_bytes: bytes | None, content_type: str):
    """Wrap raw bytes as a LangfuseMedia so it renders (image/video) in the trace.
    Returns None when tracing/media is disabled or there's nothing to wrap — so callers
    can drop it into input/output dicts unconditionally. Gated by langfuse.log_media
    because media needs S3 storage on the Langfuse instance (else upload is refused)."""
    if _client is None or not _log_media or not content_bytes:
        return None
    try:
        from langfuse.media import LangfuseMedia
        return LangfuseMedia(content_bytes=content_bytes, content_type=content_type)
    except Exception as e:  # noqa: BLE001 — never break a generation over media logging
        logger.warning(f"LangfuseMedia wrap failed: {e}")
        return None


