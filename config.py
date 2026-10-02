"""Central configuration read from the environment."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so secrets stay out of the code and out of git."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# "local" (in-process pool) or "celery" (Redis broker + workers).
QUEUE_BACKEND = os.environ.get("QUEUE_BACKEND", "local")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
WORKER_COUNT = int(os.environ.get("WORKER_COUNT", "2"))

# Batch generation: hard cap on topics per request, and how many videos may be
# produced at the same time (keeps CPU/RAM in check on small hosts).
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "20"))
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))

# Output geometry. Lower these on small hosts: 1080x1920 at 30fps is what
# OOM-kills a 512 MB Render free instance during the final compose.
VIDEO_WIDTH = int(os.environ.get("VIDEO_WIDTH", "1080"))
VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", "1920"))
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", "30"))

# FFmpeg pressure knobs. threads=1 keeps a single encode inside the memory
# budget of a free instance; raise it when the host has real CPU and RAM.
FFMPEG_THREADS = int(os.environ.get("FFMPEG_THREADS", "1"))  # 1 keeps a single encode in RAM
FFMPEG_PRESET = os.environ.get("FFMPEG_PRESET", "veryfast")

# Storage: "local" disk or "s3" (any S3-compatible endpoint: AWS, R2, MinIO...).
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "videos")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")  # e.g. https://<account>.r2.cloudflarestorage.com
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL", "")  # CDN/base URL for direct links

# Stock-footage cache: reuse already-downloaded, validated Pexels/Pixabay clips
# so repeated topics do not re-hit the providers or re-download the same file.
PEXELS_CACHE_ENABLED = os.environ.get("PEXELS_CACHE_ENABLED", "true").strip().lower() not in (
    "0", "false", "no", "off",
)
PEXELS_CACHE_TTL_DAYS = int(os.environ.get("PEXELS_CACHE_TTL_DAYS", "30"))
PEXELS_CACHE_MAX_GB = float(os.environ.get("PEXELS_CACHE_MAX_GB", "5"))
PEXELS_CACHE_DIR = Path(
    os.environ.get("PEXELS_CACHE_DIR") or (ROOT / "data" / "cache" / "pexels")
)

# LLM used for script + metadata. Without a key, a local generator is used.
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
LLM_PROVIDER = (
    "nvidia" if os.environ.get("NVIDIA_API_KEY")
    else "gemini" if os.environ.get("GEMINI_API_KEY")
    else "groq" if os.environ.get("GROQ_API_KEY")
    else "openrouter" if os.environ.get("OPENROUTER_API_KEY")
    else "custom" if os.environ.get("LLM_BASE_URL")
    else "local"
)
