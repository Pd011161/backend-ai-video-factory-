# AI Video Factory — Backend

FastAPI service that drives the AI video production pipeline: research → script → storyboard → image prompts/generation → video prompts/generation → merge. Built around LangGraph pipelines, SQLAlchemy persistence, and pluggable media storage (local disk or S3).

## Stack

- **FastAPI** + **uvicorn** — HTTP API
- **SQLAlchemy** + **Alembic** — persistence (SQLite `app.db` by default, PostgreSQL optional)
- **LangGraph** + **langchain-google-vertexai** — pipeline orchestration and LLM calls (Gemini via Vertex AI)
- **Vertex AI** — Gemini (text/image), Veo + Gemini Omni Flash (video), Gemini TTS (voice-over)
- **OpenAI** — secondary LLM/image provider for select steps
- **Tavily** — reference image search
- **yt-dlp** — source video lookup during research
- **faster-whisper** — revoice alignment + subtitle speech-window (VAD) detection
- **Langfuse** — LLM tracing/observability (optional)
- **WeasyPrint** — PDF export
- **boto3** — optional S3-backed media storage

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (dependency manager — repo ships a `uv.lock`)
- macOS: WeasyPrint needs Homebrew's `pango`/`glib`/`cairo` (`brew install pango`) for PDF export; `app/main.py` wires `DYLD_FALLBACK_LIBRARY_PATH` automatically on startup.

## Setup

```bash
cd backend
uv sync
uv run main.py
```

That's it — `main.py` is a single entrypoint that, before the server boots:

1. creates `.env` from `.env.example` if it's missing (then reminds you to fill it in)
2. runs `alembic upgrade head`, so `app.db` is created/updated with no manual migrate step
3. warns (non-fatal) about missing `GEMINI_API_KEY` / `OPENAI_API_KEY`
4. starts uvicorn — `HOST`, `PORT`, `RELOAD` env vars override the defaults (`0.0.0.0:8090`, reload on)

Then fill in `.env` — it holds **secrets only**; non-secret settings live in `config.yaml`:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini image/LLM generation |
| `OPENAI_API_KEY` | OpenAI-backed steps |
| `TAVILY_API_KEY` | Reference image search |
| `FRONTEND_ORIGIN` | Browser origin allowed by CORS |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | LLM tracing (optional) |
| `AWS_ACCESS_KEY` / `AWS_SECRET_KEY` / `AWS_REGION` / `AWS_BUCKET` / `AWS_URL` | S3 media storage (optional — falls back to local `outputs/media`) |

### Credentials

Service-account keys live in `credentials/` (gitignored):

| File | Service account | Used by |
|---|---|---|
| `credentials/credentials.json` | Vertex AI | Gemini text/image, Veo video, Gemini TTS — no `GEMINI_API_KEY` needed for these |
| `credentials/elevated-legacy-*.json` | Drive uploader (Drive + Sheets) | the research notebook in `autopipeline/` only |

The Vertex path comes from `gemini.credentials_file` in `config.yaml` (default
`credentials/credentials.json`), resolved relative to `backend/`. Point it somewhere else
by setting that key — `GOOGLE_APPLICATION_CREDENTIALS` is **not** read.

## Configuration

`config.yaml` holds every non-secret setting, grouped by top-level key:

| Key | Covers |
|---|---|
| `gemini` | credentials path, model, location, temperatures, token budgets |
| `video` / `search` / `filter` | source video lookup and filtering during research |
| `pipeline` / `script` | pipeline behaviour, script structure and length |
| `image_search` / `image_gen` | reference search, image generation + output QC, post-colour |
| `video_gen` | Veo model, `omni_model`, resolution, clip duration, aspect ratio, audio |
| `tts` | voice-over synthesis (Vertex AI service account — no API key) |
| `langfuse` | tracing host / enable flag (secrets stay in `.env`) |

Values map 1:1 onto the Pydantic models in `app/core/config.py`.

## Database

SQLite by default, PostgreSQL optional — the same migrations run against both.

`uv run main.py` migrates automatically. To run it yourself:

```bash
uv run alembic upgrade head
```

Migrations live in `migrations/versions`. Create a new one with:

```bash
uv run alembic revision --autogenerate -m "description"
```

Autogenerate diffs against whichever backend is configured, so generate on the one you
actually deploy: SQLite can't express most `ALTER`s and migrations touching it are wrapped
in `render_as_batch` (table rebuild), which PostgreSQL neither needs nor uses.

### Choosing the backend

The `database:` block in `config.yaml` holds everything except the password:

| Key | Applies to | Default |
|---|---|---|
| `backend` | — | `sqlite` (`sqlite` \| `postgres`) |
| `path` | sqlite | `./app.db`, relative to `backend/` |
| `host` / `port` / `name` / `user` | postgres | `localhost` / `5432` / `videofactory` / `videofactory` |

The password is deliberately absent: it lives in `.env` as `DB_PASSWORD` and is joined on
at startup by `DatabaseConfig.url()` in `app/core/config.py`. That split keeps the secret
out of a committed file, and percent-encodes it for you — `@ : / %` need no hand-escaping.
(A raw `%` in a hand-written URL crashes Alembic before any migration runs, because
ConfigParser reads it as interpolation syntax.)

Resolution order, highest first:

1. **`DATABASE_URL`** env var — a complete URL, overriding the block entirely. Compose and
   CI use it. Setting it in `config.yaml` is rejected at load time: pydantic-settings ranks
   init kwargs above env vars, so a value there would silently outrank the real env var.
2. **`config.yaml` `database:` + `DB_PASSWORD`**.

Everything downstream (`app/db/base.py`, `migrations/env.py`) reads the single assembled
`settings.database_url`, so Alembic always follows the app — there is no second switch.

Check the resolved value without starting the server:

```bash
uv run python -c "from app.core.config import settings; print(settings.database_url)"
```

Startup logs the same thing with the password blanked and the source named, so a stale
`DATABASE_URL` in `.env` shows up as `(from DATABASE_URL env — overriding config.yaml)`
instead of as missing data.

### Migrating SQLite → PostgreSQL

With `backend: "postgres"` and `DB_PASSWORD` set:

```bash
uv run alembic upgrade head
```

That builds the schema, empty — Alembic does not move data. To bring existing content
across:

```bash
uv run python scripts/sqlite_to_postgres.py --source sqlite:///./app.db --target "$(uv run python -c 'from app.core.config import settings; print(settings.database_url)')"
```

It copies tables in `Base.metadata.sorted_tables` order (so FKs resolve), runs `setval` on
the sequence behind every integer primary key, and compares row counts per table at the
end. The `setval` step is what stops the first post-migration insert from colliding with
an existing id. It assumes an empty target — re-running raises PK conflicts.

`app.db` is left in place, so `backend: "sqlite"` restores the old data at any time. The
two stop tracking each other from the switch onward.

`app.db` is gitignored — it holds generated content, not schema. Get a populated copy from
a teammate if you need shared data.

## Running

```bash
uv run main.py                                          # recommended
uv run uvicorn app.main:app --reload --port 8090        # raw server, skips bootstrap/migrations
```

The API is served at `http://localhost:8090`. Every route is mounted under `/api`:

- health check — `GET /api/health`
- generated media — `GET /api/media/...` (local storage mode)

## Project layout

```
app/
  api/           # FastAPI routers (routes.py) + request/response schemas
  core/          # config, DI container, logging, observability (Langfuse), usage/cost tracking
  db/            # SQLAlchemy engine/session setup + models.py (the actual tables)
  graph/         # LangGraph pipeline builders, nodes, state, runner
  models/        # Pydantic domain models (script, storyboard, director) — not the DB layer
  repositories/  # data-access layer over db/models.py
  services/      # Gemini/OpenAI clients, TTS, revoice, storage (local/S3), export, image & video search
main.py          # single entrypoint: env bootstrap → migrations → uvicorn
config.yaml      # non-secret settings
credentials/     # service-account keys (gitignored)
migrations/      # Alembic migrations
autopipeline/    # standalone research notebook (Drive + Sheets batch pipeline)
scripts/         # maintenance/one-off scripts
assets/fonts/    # NotoSansThai — subtitle rendering
docs/            # sample reference images
outputs/         # generated media + exports (local storage mode)
```

## Key API surface

All paths below are relative to the `/api` prefix.

**Pipeline steps** — most stream progress via SSE:

- `POST /steps/{research,script,storyboard,image_plan,image_prompts,images,video_prompts,video,video_group,merge}`
- `POST /steps/script/regenerate_part` — regenerate one script part in place
- `POST /steps/image`, `/steps/image/edit`, `/steps/image/edit_region`, `/steps/image/upload` — single-shot image generation, whole-image and masked-region edits, manual upload
- `POST /steps/video/edit`, `/steps/video/edit_prompt`, `/steps/video/ref_manifest`, `/steps/video/subtitle`, `/steps/video_upload` — clip re-render, prompt-only edit, reference manifest, subtitle burn-in, manual upload
- `GET /steps/video_config` — resolved video-generation settings
- `GET/POST/PUT /steps/kitchen_fixtures` — scene fixture extraction and overrides

**Runs and documents:**

- `GET/POST /runs`, `GET /runs/{id}` — run CRUD
- `GET /runs/{id}/{research,script,storyboard}` plus `/versions` and `/versions/{version}` — versioned document history
- `PUT /shots/{shot_id}/prompt_video` — edit a single shot's video prompt
- `POST /runs/{id}/regen`, `GET /runs/{id}/regen_report` — targeted regeneration and its report
- `GET /runs/{id}/usage`, `GET /cost_rates`, `PUT /cost_rates/{key}` — token/cost accounting

**Configuration entities:**

- `GET/POST/PUT/DELETE /brands` (+ `/brands/{id}/scenes`), `/menus` (+ `/menus/{id}/subjects`), `/directors`, `/prompts`
- `POST /menus/from_script_upload`, `POST /menus/{id}/pull_s3` — bulk menu import
- `GET/PUT /active_config` — currently active brand/menu/director selection
- `POST /refs/{kind}`, `GET /refs/{kind}/preview` — reference asset upload/preview

**Auxiliary:**

- `POST /synthesize`, `POST /voice_desc`, `POST /revoice` — voice-over generation and re-recording
- `POST /export/pdf` — document export

Full route list: `app/api/routes.py`.
