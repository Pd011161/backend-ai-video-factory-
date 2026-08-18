"""Per-run token accounting, broken down by agent (LLM-call type).

A ``UsageTracker`` lives for the duration of one pipeline run. It is published on
a ``ContextVar`` so every Gemini call — including ones that run in worker threads
via ``asyncio.to_thread`` or as parallel ``asyncio.gather`` tasks — can record its
token usage without threading the tracker through every function signature.

``asyncio.to_thread`` and task creation both *copy* the current context, so the
tracker reference propagates into threads/child tasks while staying isolated per
request (a different request runs in a different task with its own context).
Because worker threads update the same tracker concurrently, mutations are guarded
by a lock.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field

_current: contextvars.ContextVar["UsageTracker | None"] = contextvars.ContextVar(
    "usage_tracker", default=None
)


@dataclass
class AgentUsage:
    input: int = 0
    output: int = 0
    total: int = 0
    calls: int = 0
    duration_ms: float = 0.0


@dataclass
class UsageTracker:
    agents: dict[str, AgentUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, agent: str, usage: dict | None, duration_ms: float = 0.0) -> None:
        """Add one call's usage to ``agent`` (call still counts if tokens unknown)."""
        with self._lock:
            a = self.agents.setdefault(agent, AgentUsage())
            a.calls += 1
            a.duration_ms += duration_ms
            if usage:
                a.input += int(usage.get("input", 0) or 0)
                a.output += int(usage.get("output", 0) or 0)
                a.total += int(usage.get("total", 0) or 0)

    def summary(self) -> dict:
        """Serializable {agents, totals} breakdown for the SSE ``usage`` event."""
        with self._lock:
            agents = {
                name: {
                    "input": u.input,
                    "output": u.output,
                    "total": u.total,
                    "calls": u.calls,
                    "duration_ms": round(u.duration_ms),
                }
                for name, u in self.agents.items()
            }
        totals = {
            "input": sum(a["input"] for a in agents.values()),
            "output": sum(a["output"] for a in agents.values()),
            "total": sum(a["total"] for a in agents.values()),
            "calls": sum(a["calls"] for a in agents.values()),
            "duration_ms": sum(a["duration_ms"] for a in agents.values()),
        }
        return {"agents": agents, "totals": totals}


def set_tracker(tracker: UsageTracker) -> None:
    """Publish ``tracker`` as the current run's tracker (call once per run)."""
    _current.set(tracker)


def record_usage(agent: str, usage: dict | None, duration_ms: float = 0.0) -> None:
    """Record usage against the current run's tracker, if one is active, and attach an estimated
    USD cost to the current Langfuse generation (best-effort). Called from inside each generation's
    context in the services, so cost lands on the right observation."""
    tracker = _current.get()
    if tracker is not None:
        tracker.record(agent, usage, duration_ms)
    try:
        from app.core import observability as obs
        obs.record_cost(agent, usage)
    except Exception:  # noqa: BLE001 — never break a call over cost logging
        pass
