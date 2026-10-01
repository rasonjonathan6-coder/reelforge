"""Job model and the shared production routine.

Both the local worker pool and the Celery worker call `produce`, so behaviour
stays identical whichever queue backend is configured.

A job renders into its own directory (`outputs/<batch>/<job>/`) holding the
script, audio, subtitles, metadata and final mp4, so batch jobs never mix
their files.
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

# Stage labels reported to the UI, and their progress percentage.
STEP_LABELS = {
    "script": "Génération du script",
    "tts": "Génération de la voix",
    "visuals": "Préparation des visuels",
    "subtitles": "Sous-titres",
    "compose": "Montage final",
    "metadata": "Métadonnées et miniature",
    "completed": "Terminé",
}
STEP_PROGRESS = {
    "script": 10,
    "tts": 35,
    "subtitles": 50,
    "visuals": 60,
    "compose": 75,
    "metadata": 92,
    "completed": 100,
}


def step_progress(step: str, sub: int | None = None) -> int:
    """Progress for a stage, optionally refined by a 0-100 intra-stage value."""
    base = STEP_PROGRESS.get(step, 0)
    if sub is None:
        return base
    order = ["script", "tts", "subtitles", "visuals", "compose", "metadata", "completed"]
    nxt = STEP_PROGRESS[order[order.index(step) + 1]] if step in order and step != "completed" else base
    return max(base, min(99, base + round((nxt - base) * (sub / 100))))


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | completed | failed
    progress: int = 0
    step: str = "queued"
    video_url: str | None = None
    thumbnail_url: str | None = None
    error: str | None = None
    topic: str = ""
    duration: float | None = None
    dir: str | None = None
    batch_id: str | None = None
    meta: dict = field(default_factory=dict)

    # Backwards-compatible aliases for the original single-job vocabulary.
    @property
    def video_path(self) -> Path | None:
        if not self.dir:
            return None
        path = Path(self.dir) / "final.mp4"
        return path if path.exists() else None


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
def _job_dir(job_id: str, batch_id: str | None) -> Path:
    """`outputs/<batch>/<job>/` for batch jobs, `outputs/<job>/` otherwise."""
    return (OUTPUT_DIR / batch_id / job_id) if batch_id else (OUTPUT_DIR / job_id)


def produce(job_id: str, req: dict) -> None:
    work = _job_dir(job_id, req.get("batch_id"))
    work.mkdir(parents=True, exist_ok=True)
    video_path = work / "final.mp4"
    thumb_path = work / "thumbnail.jpg"

    update(
        job_id,
        status="running",
        step="script",
        progress=step_progress("script"),
        dir=str(work),
    )

    def on_step(stage: str, progress: int) -> None:
        update(job_id, step=stage, progress=progress)

    try:
        script = req.get("text") or ""
        if req.get("auto_script") and req.get("topic"):
            script = script_writer.write_script(
                req["topic"],
                duration=req.get("duration", 45),
                language=req.get("language", "français"),
                style=req.get("style", ""),
                tone=req.get("tone", ""),
            )
        if len(script) < 20:
            raise ValueError("Script trop court : fournis un texte ou un sujet.")
        (work / "script.json").write_text(
            json.dumps({"topic": req.get("topic", ""), "script": script},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        update(job_id, step="metadata", progress=step_progress("metadata"))
        metadata = script_writer.write_metadata(script, req.get("topic") or script[:60])
        (work / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        thumbnail.generate_thumbnail(
            metadata["title"], metadata["thumbnail_prompt"], thumb_path, work
        )
        thumb_url = storage.upload(thumb_path, _rel(thumb_path))

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
            work_dir=work / ".work",
            clips=clips,
            logo_text=req.get("logo") or None,
            on_step=on_step,
            info_out=work,
        )
        # Intermediate artefacts required by the batch layout.
        (work / "scenes.json").write_text(
            json.dumps({"use_stock": req.get("use_stock", True),
                        "query": req.get("query", ""),
                        "clips": [c.name for c in (clips or [])]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        duration = _probe_duration(video_path)
        video_url = storage.upload(video_path, _rel(video_path))

        update(
            job_id,
            status="completed",
            step="completed",
            progress=100,
            video_url=video_url,
            thumbnail_url=thumb_url,
            duration=duration,
            topic=req.get("topic", ""),
            meta={
                "script": script,
                "title": metadata["title"],
                "description": metadata["description"],
                "hashtags": metadata["hashtags"],
                "size_kb": round(video_path.stat().st_size / 1024),
                "duration": duration,
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        import traceback

        traceback.print_exc()
        update(job_id, status="failed", error=str(exc), step="failed")
    finally:
        shutil.rmtree(work / ".work", ignore_errors=True)


def _rel(path: Path) -> str:
    """Path relative to OUTPUT_DIR, used as the storage key / public URL."""
    try:
        return str(path.relative_to(OUTPUT_DIR)).replace("\\", "/")
    except ValueError:
        return path.name


def _probe_duration(video_path: Path) -> float | None:
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, timeout=60,
        )
        return round(float(out.stdout.strip()), 2)
    except Exception:  # noqa: BLE001 - duration is informational
        return None


def slugify(text: str, max_length: int = 48) -> str:
    """Filesystem-safe, short slug for download names."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (text[:max_length].rstrip("_")) or "video"
