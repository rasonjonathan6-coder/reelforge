"""Job model and the shared production routine.

Both the local worker pool and the Celery worker call `produce`, so behaviour
stays identical whichever queue backend is configured.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import REDIS_URL, ROOT
from generate import generate
from pipeline import script_writer, storage, thumbnail, tts

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CLIP_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    progress: int = 0
    step: str = "En attente"
    video_url: str | None = None
    thumbnail_url: str | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Persistence: Redis when configured, otherwise an in-process dict.
# ---------------------------------------------------------------------------
_local: dict[str, Job] = {}
_redis = None


def _store():
    global _redis
    if _redis is None:
        import redis

        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _use_redis() -> bool:
    from config import QUEUE_BACKEND

    return QUEUE_BACKEND == "celery"


def save(job: Job) -> None:
    if _use_redis():
        try:
            _store().set(f"job:{job.id}", json.dumps(asdict(job)), ex=86400)
            return
        except Exception:  # noqa: BLE001 - fall back to memory if Redis is down
            pass
    _local[job.id] = job


def load(job_id: str) -> Job | None:
    if _use_redis():
        try:
            raw = _store().get(f"job:{job_id}")
            if raw:
                return Job(**json.loads(raw))
        except Exception:  # noqa: BLE001
            pass
    return _local.get(job_id)


def update(job_id: str, **changes) -> None:
    job = load(job_id) or Job(id=job_id)
    for key, value in changes.items():
        setattr(job, key, value)
    save(job)


# ---------------------------------------------------------------------------
# Production pipeline
# ---------------------------------------------------------------------------
def produce(job_id: str, req: dict) -> None:
    update(job_id, status="running", progress=5, step="Génération du script")
    work = OUTPUT_DIR / f".work_{job_id}"
    video_path = OUTPUT_DIR / f"{job_id}.mp4"
    thumb_path = OUTPUT_DIR / f"{job_id}.jpg"
    try:
        script = req.get("text") or ""
        if req.get("auto_script") and req.get("topic"):
            script = script_writer.write_script(
                req["topic"], duration=req.get("duration", 45)
            )
        if len(script) < 20:
            raise ValueError("Script trop court : fournis un texte ou un sujet.")

        update(job_id, progress=15, step="Métadonnées et miniature")
        metadata = script_writer.write_metadata(script, req.get("topic") or script[:60])
        thumbnail.generate_thumbnail(
            metadata["title"], metadata["thumbnail_prompt"], thumb_path, work
        )
        thumb_url = storage.upload(thumb_path, f"{job_id}.jpg")

        update(job_id, progress=35, step="Voix off et montage")
        clips_dir = req.get("clips_dir")
        clips = None
        if clips_dir:
            folder = Path(clips_dir)
            if not folder.is_dir():
                raise ValueError(f"Dossier de clips introuvable : {clips_dir}")
            clips = sorted(p for p in folder.iterdir() if p.suffix.lower() in CLIP_SUFFIXES)
            if not clips:
                raise ValueError(f"Aucun clip vidéo dans {clips_dir}")

        generate(
            text=script,
            out_path=video_path,
            voice=req.get("voice", tts.DEFAULT_VOICE),
            rate=req.get("rate", tts.DEFAULT_RATE),
            query=req.get("query", "city night vertical"),
            use_stock=req.get("use_stock", True),
            work_dir=work,
            clips=clips,
            logo_text=req.get("logo") or None,
        )

        update(job_id, progress=90, step="Publication")
        video_url = storage.upload(video_path, f"{job_id}.mp4")

        update(
            job_id,
            status="done",
            progress=100,
            step="Terminé",
            video_url=video_url,
            thumbnail_url=thumb_url,
            meta={
                "script": script,
                "title": metadata["title"],
                "description": metadata["description"],
                "hashtags": metadata["hashtags"],
                "size_kb": round(video_path.stat().st_size / 1024),
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        import traceback

        traceback.print_exc()
        update(job_id, status="error", error=str(exc), step="Échec")
    finally:
        shutil.rmtree(work, ignore_errors=True)
