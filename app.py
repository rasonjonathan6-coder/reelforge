"""Free faceless-reel generator API.

Queue backend is selectable: an in-process pool (default, zero infrastructure)
or Celery + Redis for horizontal scaling. Storage is local disk or S3-compatible.
"""

from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from batch import batch_status, build_zip, create_batch
from config import (
    LLM_PROVIDER,
    MAX_BATCH_SIZE,
    MAX_CONCURRENT_JOBS,
    QUEUE_BACKEND,
    ROOT,
    STORAGE_BACKEND,
    WORKER_COUNT,
)
from jobs import CLIP_SUFFIXES, OUTPUT_DIR, Job, load, produce, save
from pipeline import script_writer, tts, visuals

app = FastAPI(title="ReelForge", version="2.2.0")

executor = ThreadPoolExecutor(max_workers=WORKER_COUNT)

CLIPS_DIR = OUTPUT_DIR / "clips"


class GenerateRequest(BaseModel):
    text: str = ""
    topic: str = ""
    auto_script: bool = False
    duration: int = Field(45, ge=15, le=90)
    voice: str = tts.DEFAULT_VOICE
    rate: str = tts.DEFAULT_RATE
    query: str = "city night vertical"
    use_stock: bool = True
    logo: str = ""
    clips_dir: str = ""


class ScriptRequest(BaseModel):
    topic: str = Field(..., min_length=3)
    duration: int = Field(45, ge=15, le=90)


class BatchRequest(BaseModel):
    topics: list[str] = Field(..., min_length=1)
    language: str = "français"
    duration: int = Field(30, ge=15, le=90)
    style: str = "storytelling"
    tone: str = "dynamic"
    voice: str = tts.DEFAULT_VOICE
    rate: str = tts.DEFAULT_RATE
    query: str = "city night vertical"
    use_stock: bool = True
    logo: str = ""
    clips_dir: str = ""


def _dispatch(job_id: str, payload: dict) -> None:
    if QUEUE_BACKEND == "celery":
        try:
            from tasks import produce_task

            produce_task.delay(job_id, payload)
            return
        except Exception as exc:  # noqa: BLE001 - broker down: keep serving
            print(f"[queue] Celery indisponible ({exc}); repli sur le pool local")
    executor.submit(produce, job_id, payload)


@app.post("/api/generate")
def create_job(req: GenerateRequest) -> dict:
    if not req.auto_script and len(req.text.strip()) < 20:
        raise HTTPException(400, "Fournis un texte (20 caractères min) ou active auto_script avec un sujet.")
    if req.auto_script and len(req.topic.strip()) < 3:
        raise HTTPException(400, "Renseigne un sujet pour la génération automatique.")

    job_id = uuid.uuid4().hex[:12]
    save(Job(id=job_id))
    _dispatch(job_id, req.model_dump())
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/batch-generate")
def create_batch_jobs(req: BatchRequest) -> dict:
    common = req.model_dump(exclude={"topics"})
    try:
        record = create_batch(req.topics, common)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"batch_id": record["batch_id"], "jobs": record["jobs"]}


@app.get("/api/batch/{batch_id}")
def get_batch(batch_id: str) -> dict:
    status = batch_status(batch_id)
    if status is None:
        raise HTTPException(404, "Lot introuvable")
    return status


@app.get("/api/batch/{batch_id}/download")
def download_batch(batch_id: str):
    if batch_status(batch_id) is None:
        raise HTTPException(404, "Lot introuvable")
    zip_path = build_zip(batch_id)
    if zip_path is None:
        raise HTTPException(409, "Aucune vidéo terminée à télécharger pour ce lot.")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"ReelForge_Batch_{batch_id}.zip",
    )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = load(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    # `done`/`error` keep the original single-job contract working unchanged.
    return {
        "job_id": job.id,
        "status": job.status,
        "done": job.status == "completed",
        "error": job.error,
        "progress": job.progress,
        "step": job.step,
        "video_url": job.video_url,
        "thumbnail_url": job.thumbnail_url,
        "duration": job.duration,
        "topic": job.topic,
        "batch_id": job.batch_id,
        "meta": job.meta,
    }


@app.post("/api/script")
def generate_script(req: ScriptRequest) -> dict:
    try:
        script = script_writer.write_script(req.topic, duration=req.duration)
        return {"script": script}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Génération du script impossible : {exc}")


@app.post("/api/upload-clips")
async def upload_clips(files: list[UploadFile] = File(...)) -> dict:
    """Accept AI-generated (or any) clips and return a folder the job can use."""
    folder = CLIPS_DIR / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload in files:
        name = Path(upload.filename or "clip.mp4").name
        if Path(name).suffix.lower() not in CLIP_SUFFIXES:
            continue
        dest = folder / name
        with open(dest, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved.append(name)
    if not saved:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(
            400, f"Aucun clip valide. Formats acceptés : {', '.join(sorted(CLIP_SUFFIXES))}"
        )
    return {"clips_dir": str(folder), "clips": saved, "count": len(saved)}


@app.get("/api/voices")
def voices() -> dict:
    return {
        "voices": [
            {"id": "fr-FR-DeniseNeural", "label": "Denise (femme, FR)"},
            {"id": "fr-FR-HenriNeural", "label": "Henri (homme, FR)"},
            {"id": "fr-FR-VivienneMultilingualNeural", "label": "Vivienne (femme, FR)"},
            {"id": "fr-FR-RemyMultilingualNeural", "label": "Rémy (homme, FR)"},
            {"id": "en-US-AriaNeural", "label": "Aria (femme, EN)"},
            {"id": "en-US-GuyNeural", "label": "Guy (homme, EN)"},
        ]
    }


@app.get("/api/config")
def public_config() -> dict:
    stock = [name for name in ("pexels", "pixabay") if visuals.provider_ready(name)]
    return {
        "queue": QUEUE_BACKEND,
        "storage": STORAGE_BACKEND,
        "llm": LLM_PROVIDER,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "stock_providers": stock,
        "stock_available": bool(stock),
    }


app.mount("/videos", StaticFiles(directory=str(OUTPUT_DIR)), name="videos")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
