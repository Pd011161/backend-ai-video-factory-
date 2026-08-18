"""Service container — built once at startup, injected into graph nodes.

Holds long-lived dependencies (the Gemini client, image-search credentials,
settings) so nodes stay pure functions that receive their collaborators via
LangGraph's `configurable` config rather than reaching for module-level globals.
This also keeps heavy init (Vertex AI auth) out of import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.core.config import Settings
from app.services.gemini_client import GeminiClient


@dataclass
class ServiceContainer:
    settings: Settings
    gemini: GeminiClient
    tavily_api_key: str = ""
    openai_api_key: str = ""

    @classmethod
    def build(cls, settings: Settings) -> "ServiceContainer":
        return cls(
            settings=settings,
            gemini=GeminiClient(settings),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )
