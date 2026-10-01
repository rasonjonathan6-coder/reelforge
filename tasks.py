"""Celery tasks for the Redis-backed queue."""

from __future__ import annotations

from celery_app import celery_app
from jobs import produce


@celery_app.task(name="reelforge.produce", bind=True)
def produce_task(self, job_id: str, req: dict) -> str:  # noqa: ANN001
    produce(job_id, req)
    return job_id
