"""Batch generation: many independent reels from a list of topics.

Each topic becomes its own job with its own directory under
`outputs/<batch_id>/<job_id>/`, so jobs never share files and one failing job
never cancels the others.

Concurrency is bounded by MAX_CONCURRENT_JOBS: the same shared `produce`
routine runs, just not more than N at once, which keeps CPU/RAM in check.
"""

from __future__ import annotations

import json
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import MAX_BATCH_SIZE, MAX_CONCURRENT_JOBS
from jobs import OUTPUT_DIR, Job, load, produce, save, slugify

_BATCHES_DIR = OUTPUT_DIR

# One shared, bounded pool for every batch: never more than N reels at once.
_executor = ThreadPoolExecutor(max_workers=max(1, MAX_CONCURRENT_JOBS))

_TERMINAL = {"completed", "failed"}


def _batch_dir(batch_id: str) -> Path:
    return _BATCHES_DIR / batch_id


def _record_path(batch_id: str) -> Path:
    return _batch_dir(batch_id) / "batch.json"


def save_batch(record: dict) -> None:
    folder = _batch_dir(record["batch_id"])
    folder.mkdir(parents=True, exist_ok=True)
    _record_path(record["batch_id"]).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_batch(batch_id: str) -> dict | None:
    path = _record_path(batch_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def clean_topics(topics: list[str]) -> list[str]:
    """Trim, drop blanks/comments, de-duplicate while keeping order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in topics:
        topic = (raw or "").strip()
        if not topic or topic.startswith("#") or topic in seen:
            continue
        seen.add(topic)
        cleaned.append(topic)
    return cleaned


def create_batch(topics: list[str], common: dict) -> dict:
    """Validate, register and start a batch. Returns the batch record."""
    topics = clean_topics(topics)
    if not topics:
        raise ValueError("Aucun sujet valide fourni.")
    if len(topics) > MAX_BATCH_SIZE:
        raise ValueError(f"Maximum de {MAX_BATCH_SIZE} vidéos par lot.")

    batch_id = uuid.uuid4().hex[:12]
    _batch_dir(batch_id).mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    for topic in topics:
        job_id = uuid.uuid4().hex[:12]
        job_dir = _batch_dir(batch_id) / job_id
        save(Job(id=job_id, batch_id=batch_id, topic=topic, dir=str(job_dir)))

        req = {
            **common,
            "topic": topic,
            "text": "",
            "auto_script": True,
            "batch_id": batch_id,
        }
        _executor.submit(produce, job_id, req)
        jobs.append({"job_id": job_id, "topic": topic})

    record = {
        "batch_id": batch_id,
        "created": time.time(),
        "topics": topics,
        "jobs": jobs,
        "max_concurrent": MAX_CONCURRENT_JOBS,
    }
    save_batch(record)
    return record


def batch_status(batch_id: str) -> dict | None:
    """Live status of every job in the batch, plus aggregate counts."""
    record = load_batch(batch_id)
    if record is None:
        return None

    jobs: list[dict] = []
    counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
    for entry in record["jobs"]:
        job = load(entry["job_id"])
        if job is None:
            counts["failed"] += 1
            jobs.append({"job_id": entry["job_id"], "topic": entry["topic"],
                         "status": "failed", "step": "failed", "progress": 0,
                         "error": "Job introuvable"})
            continue
        counts[job.status] = counts.get(job.status, 0) + 1
        jobs.append({
            "job_id": job.id,
            "topic": job.topic or entry["topic"],
            "status": job.status,
            "step": job.step,
            "progress": job.progress,
            "duration": job.duration,
            "video_url": job.video_url,
            "thumbnail_url": job.thumbnail_url,
            "error": job.error,
            "meta": job.meta,
        })

    total = len(jobs)
    finished = counts["completed"] + counts["failed"]
    if finished == total:
        overall = "completed" if counts["failed"] == 0 else "completed_with_errors"
    elif counts["running"] or counts["completed"]:
        overall = "running"
    else:
        overall = "queued"

    return {
        "batch_id": batch_id,
        "status": overall,
        "counts": counts,
        "total": total,
        "jobs": jobs,
        "zip_available": counts["completed"] > 0,
    }


def build_zip(batch_id: str) -> Path | None:
    """Zip only the completed videos, numbered by batch order."""
    record = load_batch(batch_id)
    if record is None:
        return None

    zip_path = _batch_dir(batch_id) / f"ReelForge_Batch_{batch_id}.zip"
    added = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, entry in enumerate(record["jobs"], start=1):
            job = load(entry["job_id"])
            if job is None or job.status != "completed" or not job.dir:
                continue
            video = Path(job.dir) / "final.mp4"
            if not video.exists():
                continue
            name = f"{index:02d}_{slugify(job.topic or entry['topic'])}.mp4"
            archive.write(video, name)
            added += 1

    if added == 0:
        zip_path.unlink(missing_ok=True)
        return None
    return zip_path


def run_batch_sync(topics: list[str], common: dict, out_dir: Path | None = None) -> dict:
    """CLI helper: start a batch and block until every job reaches a terminal state."""
    record = create_batch(topics, common)
    batch_id = record["batch_id"]
    print(f"Batch {batch_id} : {len(record['jobs'])} vidéos (max {MAX_CONCURRENT_JOBS} en parallèle)")
    for entry in record["jobs"]:
        print(f"  - {entry['topic']}")

    while True:
        status = batch_status(batch_id)
        if status is None:
            raise RuntimeError("Lot introuvable")
        running = [j for j in status["jobs"] if j["status"] in {"queued", "running"}]
        for job in status["jobs"]:
            if job["status"] == "running":
                print(f"  ⟳ {job['topic'][:40]} — {job['step']} ({job['progress']}%)")
        if not running:
            break
        time.sleep(3)

    status = batch_status(batch_id)
    print(f"\nBatch {batch_id} terminé : "
          f"{status['counts']['completed']} ok, {status['counts']['failed']} échec(s)")
    for job in status["jobs"]:
        if job["status"] == "completed":
            print(f"  ✓ {job['topic'][:40]} — {job['duration']}s")
        else:
            print(f"  ✗ {job['topic'][:40]} — {job['error']}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(record["jobs"], start=1):
            job = load(entry["job_id"])
            if job and job.status == "completed" and job.dir:
                video = Path(job.dir) / "final.mp4"
                if video.exists():
                    dest = out_dir / f"{index:02d}_{slugify(job.topic or entry['topic'])}.mp4"
                    dest.write_bytes(video.read_bytes())
                    print(f"  -> {dest}")
    return status
