# syntax=docker/dockerfile:1
# FastAPI backend. Python 3.12 rather than the 3.13 used locally: faster-whisper pulls in
# ctranslate2, whose prebuilt wheels lag a release behind, and a source build here would be slow
# and fragile. pyproject only requires >=3.11, so 3.12 is within contract.
FROM python:3.12-slim

# ffmpeg — merge, revoice, subtitle burn-in and last-frame extraction all shell out to it. The
# Debian build links libass, which the `subtitles` filter needs.
# pango/cairo/gdk-pixbuf — WeasyPrint's runtime deps for the PDF export route.
# fonts-noto-core — so Thai in the exported PDF isn't tofu. (Subtitle burn-in uses the font that
# ships in assets/fonts, not a system one.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libpango-1.0-0 libpangocairo-1.0-0 libpangoft2-1.0-0 \
        libgdk-pixbuf-2.0-0 libffi8 shared-mime-info \
        fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Speech timing uses the Silero VAD bundled inside the faster-whisper package (~2 MB, no
    # download). HF_HOME kept for anything that does hit the HF hub (e.g. demucs deps on revoice).
    HF_HOME=/app/.cache/huggingface

# Dependencies in their own layer so they rebuild only when the lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Source + runtime assets. Secrets, outputs/ and the DB are deliberately absent (see
# .dockerignore) — compose mounts them.
COPY app/ app/
COPY migrations/ migrations/
COPY assets/ assets/
COPY alembic.ini config.yaml ./
RUN uv sync --frozen --no-dev

EXPOSE 8090
# $PORT with an 8090 fallback: Render injects PORT and routes traffic to it; compose/local keep 8090.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD sh -c "python -c \"import urllib.request as u; u.urlopen('http://127.0.0.1:${PORT:-8090}/api/health', timeout=4)\"" || exit 1

# main.py deliberately does NOT create tables ("run `alembic upgrade head` before starting the
# server") — so migrate first, otherwise a fresh volume starts with no schema and every request 500s.
CMD ["sh", "-c", "uv run --no-sync alembic upgrade head && exec uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8090}"]
