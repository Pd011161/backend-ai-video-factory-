"""Single entrypoint: `uv run main.py` does everything, then starts the server.

Steps, in order, before the app boots:
  1. ensure a `.env` exists (bootstrap from `.env.example` on first run) and load it
  2. bring the SQLite schema up to date (`alembic upgrade head`) — no manual migrate step
  3. warn (non-fatal) about missing API keys so you find out now, not mid-render
  4. start uvicorn on app.main:app

Run it from the `backend/` directory (where `alembic.ini` and the `app` package live);
`uv run main.py` already does that.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent


def _ensure_env() -> None:
    """First run has no `.env` — bootstrap it from `.env.example` so the app can start,
    then remind the user to fill in the secrets. Load it either way."""
    from dotenv import load_dotenv

    env, example = BACKEND_DIR / ".env", BACKEND_DIR / ".env.example"
    if not env.exists() and example.exists():
        shutil.copyfile(example, env)
        print(f"⚠️  created {env.name} from .env.example — fill in your API keys before real use")
    load_dotenv(env)


def _run_migrations() -> None:
    """`alembic upgrade head` in-process — creates/updates app.db automatically."""
    from alembic import command
    from alembic.config import Config

    print("→ applying database migrations (alembic upgrade head)…")
    try:
        command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")
        print("✓ database up to date")
    except Exception as e:  # noqa: BLE001 — surface clearly; a broken DB should stop startup
        print(f"✗ migration failed: {e}", file=sys.stderr)
        raise


def _warn_missing_keys() -> None:
    """Non-fatal heads-up for the keys the app actually needs to generate anything."""
    missing = [k for k in ("GEMINI_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"⚠️  missing env keys: {', '.join(missing)} — set them in .env before generating")


def main() -> None:
    os.chdir(BACKEND_DIR)   # so alembic.ini + `app` import resolve regardless of caller cwd
    _ensure_env()
    _run_migrations()
    _warn_missing_keys()

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8090"))
    reload = os.getenv("RELOAD", "true").lower() not in ("0", "false", "no")
    print(f"→ starting server on http://{host}:{port} (reload={reload})")
    # import string (not the app object) so --reload can re-exec the worker
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
