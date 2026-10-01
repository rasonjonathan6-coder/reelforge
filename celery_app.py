"""Celery application for the Redis-backed queue.

Start a worker with:
    celery -A celery_app worker --loglevel=info --concurrency=2
"""

from __future__ import annotations

from celery import Celery

from config import REDIS_URL

celery_app = Celery("reelforge", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,
)

# Register tasks (import side effect).
import tasks  # noqa: E402,F401
