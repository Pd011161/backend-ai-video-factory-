import os
import sys

# WeasyPrint (PDF export) loads Homebrew's pango/glib/cairo dylibs, which live outside macOS's default
# dyld search path. Set this before the app starts so the process (and any reload worker, which inherits
# this env) can find them. Requires a fresh start — a hot-reload won't re-read it.
if sys.platform == "darwin":
    _libs = [d for d in ("/opt/homebrew/lib", "/usr/local/lib") if os.path.isdir(d)]
    if _libs:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            [*_libs, os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")]).rstrip(":")

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Load .env before anything reads env.
load_dotenv()

# On Render (or any host where the repo's gitignored credentials/ dir is absent) the service-account
# JSONs arrive as env vars instead of files. Materialize them at the paths the rest of the code
# expects (settings.gemini.credentials_file, the notebook's drive-uploader SA) before anything reads
# them. A file already on disk (local dev) wins — the env var is ignored then.
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _env, _rel in (
    ("GEMINI_CREDENTIALS_JSON", "credentials/credentials.json"),
    ("DRIVE_SA_JSON", "credentials/elevated-legacy-474815-d0-5da39ee82587.json"),
):
    _val = os.getenv(_env)
    _dst = _ROOT / _rel
    if _val and not _dst.exists():
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dst.write_text(_val)

from app.api.routes import router
from app.core.config import ROOT_DIR, settings
from app.core.container import ServiceContainer
from app.core.logging import setup_logging
from app.core.observability import init_langfuse, shutdown_langfuse
from app.graph.builder import (
    build_image_prompts,
    build_images,
    build_pipeline,
    build_regenerate_script_part,
    build_research,
    build_research_reuse,
    build_script,
    build_storyboard,
    build_video_prompts,
)

# Generated images/videos are written here and served under /api/media for the frontend
# (only used when S3 storage is not configured — app/services/storage.py is unchanged from v1).
_MEDIA_DIR = ROOT_DIR / "outputs" / "media"
_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

setup_logging()


def _db_summary() -> str:
    """The database we actually connected to, with the password blanked, plus where the URL came
    from. The source matters: a DATABASE_URL left over in .env silently outranks the `database:`
    block in config.yaml, so someone can flip `backend: postgres`, see no error, and keep writing to
    SQLite. Naming the winner here is what turns that into a one-glance answer."""
    url = settings.database_url
    source = "config.yaml" if url == settings.database.url(settings.db_password) \
        else "DATABASE_URL env — overriding config.yaml"
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        url = f"{scheme}://{user}:***@{host}"
    return f"{url}  (from {source})"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Observability must be initialized before any traced code runs.
    init_langfuse(settings.langfuse)
    logger.info(f"database: {_db_summary()}")

    # Build long-lived dependencies once and stash them on app.state.
    # NOTE: the schema itself is Alembic-managed only — this app never calls
    # Base.metadata.create_all(); run `alembic upgrade head` before starting the server.
    app.state.container = ServiceContainer.build(settings)
    # Load durable global character/scene refs (config_refs) into settings — so an uploaded ref
    # survives restart instead of reverting to the config.yaml default.
    try:
        from app.db.base import SessionLocal
        from app.api.routes import load_config_refs_into_settings
        with SessionLocal() as _db:
            load_config_refs_into_settings(_db, settings.image_gen, settings.script, settings.voice)
    except Exception as e:  # noqa: BLE001 — never block startup on this
        logger.warning(f"load config_refs failed: {e}")
    app.state.pipeline = build_pipeline()
    # Per-step sub-graphs for the step-by-step UI (each runs one stage independently).
    app.state.steps = {
        "research": build_research(),
        "research_reuse": build_research_reuse(),
        "script": build_script(),
        "script_regenerate_part": build_regenerate_script_part(),
        "storyboard": build_storyboard(),
        "image_prompts": build_image_prompts(),
        "images": build_images(),
        "video_prompts": build_video_prompts(),
    }
    logger.info("Pipeline compiled and service container ready")

    yield

    shutdown_langfuse()
    logger.info("Server shutdown")


app = FastAPI(title="video-factory-v2", docs_url="/docs", redoc_url=None, lifespan=lifespan)

# FRONTEND_ORIGIN may hold several origins, comma-separated (a tunnel URL, a LAN address…).
# The localhost dev origins are ALWAYS included: a stale value here once pointed at :5174 (vite's
# fallback port from some earlier day) and every browser request from :5173 failed with a generic
# "Request failed" — an env file should be able to ADD origins, not lock local dev out.
_dev_origins = {"http://localhost:5173", "http://127.0.0.1:5173"}
_env_origins = {o.strip().strip('"') for o in os.getenv("FRONTEND_ORIGIN", "").split(",") if o.strip()}
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(_dev_origins | _env_origins),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(router, prefix="/api")
app.mount("/api/media", StaticFiles(directory=str(_MEDIA_DIR)), name="media")
