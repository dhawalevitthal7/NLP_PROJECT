"""
FastAPI application entrypoint.

Logging is configured here at module load so that ALL loggers across the
app emit structured output to stdout.  Every service uses
`logging.getLogger(__name__)` which inherits from the root logger configured
here.
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration  — must happen BEFORE any app import that uses loggers
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,   # override any prior root-logger config (e.g. from uvicorn)
)

# Reduce noise from httpx / httpcore used by the OpenAI SDK
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("🚀 Semantic Answer Evaluation API starting up…")

# ─────────────────────────────────────────────────────────────────────────────
# App + CORS
# ─────────────────────────────────────────────────────────────────────────────
from app.api.routes import router  # noqa: E402  (imported after logging setup)

app = FastAPI(
    title="Semantic Answer Evaluation API",
    version="0.2.0",
    description=(
        "MVP pipeline for OCR + schema extraction + multi-technique "
        "semantic grading of student exam papers."
    ),
)

# Allow the browser frontend (served on same origin or localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# ─────────────────────────────────────────────────────────────────────────────
# Static frontend
# ─────────────────────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def home():
    return FileResponse(str(frontend_dir / "index.html"))
