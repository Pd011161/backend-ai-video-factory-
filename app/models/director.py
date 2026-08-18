"""Director prompt — a reusable directorial style guide extracted from a reference
YouTube video, stored as JSON and optionally attached to script/storyboard generation.
"""

from __future__ import annotations

from pydantic import BaseModel

# The 7 style-guide sections produced by DIRECTOR_EXTRACT_PROMPT (besides summary).
SECTION_KEYS = [
    "tone_mood", "pacing_editing", "shots_framing", "narrative_vo",
    "hook_cta", "graphics_text_music", "do_dont",
]


class DirectorMeta(BaseModel):
    """Lightweight record for the history list (no body)."""
    id: str
    title: str
    source_url: str
    source_title: str = ""
    created_at: str = ""
    summary: str = ""


class DirectorPrompt(DirectorMeta):
    """Full record — the style guide as structured sections (key → Thai text)."""
    sections: dict = {}
