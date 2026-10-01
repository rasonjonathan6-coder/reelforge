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

# Storage: "local" disk or "s3" (any S3-compatible endpoint: AWS, R2, MinIO...).
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "videos")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")  # e.g. https://<account>.r2.cloudflarestorage.com
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL", "")  # CDN/base URL for direct links

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
