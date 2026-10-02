"""Free faceless-reel generator API.

Queue backend is selectable: an in-process pool (default, zero infrastructure)
or Celery + Redis for horizontal scaling. Storage is local disk or S3-compatible.
"""

from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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
    PEXELS_CACHE_TTL_DAYS,
    QUEUE_BACKEND,
    ROOT,
    STORAGE_BACKEND,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WORKER_COUNT,
)
from jobs import CLIP_SUFFIXES, OUTPUT_DIR, Job, load, produce, save
from pipeline import music, script_writer, stock_cache, tts, video_prompts, visuals


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Keep the clip cache tidy at boot: drop expired/invalid entries and cap the
    # total size. Never blocks startup on a cache problem.
    try:
        removed = stock_cache.cleanup()
        if any(removed.values()):
            print(f"[PexelsCache] nettoyage au démarrage : {removed}")
    except Exception as exc:  # noqa: BLE001 - maintenance is best-effort
        print(f"[PexelsCache] nettoyage ignoré ({exc})")
    yield


app = FastAPI(title="ReelForge", version="2.2.0", lifespan=lifespan)

executor = ThreadPoolExecutor(max_workers=WORKER_COUNT)

CLIPS_DIR = OUTPUT_DIR / "clips"


class GenerateRequest(BaseModel):
    text: str = ""
    topic: str = ""
    auto_script: bool = False
    dialogue: bool = False
    dialogue_cast: str = ""
    duration: int = Field(45, ge=15, le=90)
    language: str = "français"
    style: str = "storytelling"
    tone: str = "dynamic"
    voice: str = tts.DEFAULT_VOICE
    rate: str = tts.DEFAULT_RATE
    query: str = "city night vertical"
    use_stock: bool = True
    visual_source: str = ""
    music: bool = True
    music_mood: str = ""
    logo: str = ""
    clips_dir: str = ""
    animated_characters: bool = False
    character_style: str = "anime"
    animation_provider: str = ""
    quality: str = ""
    characters_dir: str = ""


class ScriptRequest(BaseModel):
    topic: str = Field(..., min_length=3)
    duration: int = Field(45, ge=15, le=90)
    dialogue: bool = False
    dialogue_cast: str = ""
    rate: str = tts.DEFAULT_RATE


class VideoPromptsRequest(BaseModel):
    """Prompts for the free GPU notebook: either a topic or a ready script."""

    topic: str = ""
    script: str = ""
    duration: int = Field(45, ge=5, le=90)
    dialogue: bool = False
    dialogue_cast: str = ""
    rate: str = tts.DEFAULT_RATE
    max_scenes: int = Field(8, ge=1, le=20)


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
    visual_source: str = ""
    dialogue: bool = False
    dialogue_cast: str = ""
    music: bool = True
    music_mood: str = ""
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
    if req.dialogue:
        if len(req.topic.strip()) < 3:
            raise HTTPException(400, "Renseigne un sujet pour le dialogue automatique.")
    elif not req.auto_script and len(req.text.strip()) < 20:
        raise HTTPException(400, "Fournis un texte (20 caractères min) ou active auto_script avec un sujet.")
    if req.auto_script and not req.dialogue and len(req.topic.strip()) < 3:
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
        if req.dialogue:
            turns, script = script_writer.write_dialogue_script(
                req.topic, duration=req.duration, rate=req.rate
            )
            return {"script": script, "dialogue": True,
                    "speakers": script_writer.characters_of(turns)}
        script = script_writer.write_script(req.topic, duration=req.duration, rate=req.rate)
        return {"script": script, "dialogue": False}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Génération du script impossible : {exc}")


@app.post("/api/video-prompts")
def make_video_prompts(req: VideoPromptsRequest) -> dict:
    """Ready-to-use English prompts for the free GPU video notebook.

    The notebook cannot translate a French script, so this endpoint turns the
    script into Wan/LTX prompts and returns the matching settings; the user
    pastes the list into Colab, then uploads the clips back here.
    """
    script = req.script.strip()
    topic = req.topic.strip()
    try:
        if not script:
            if len(topic) < 3:
                raise HTTPException(
                    400, "Renseigne un sujet (3 caractères min) ou colle un script.")
            if req.dialogue:
                turns, script = script_writer.write_dialogue_script(
                    topic, duration=req.duration, rate=req.rate)
                del turns
            else:
                script = script_writer.write_script(
                    topic, duration=req.duration, rate=req.rate)
        result = video_prompts.prompts_for(
            script, topic=topic, duration=req.duration, max_scenes=req.max_scenes)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Préparation des prompts impossible : {exc}")
    result["script"] = script
    result["topic"] = topic
    result["notebook"] = "colab/ReelForge_Video_IA_Colab.ipynb"
    return result


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


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CHARACTERS_DIR = OUTPUT_DIR / "characters"


@app.post("/api/upload-character")
async def upload_character(files: list[UploadFile] = File(...)) -> dict:
    """Store character reference images and return how each one maps to a name.

    The filename (minus extension) is the character name; the job then paints
    that character with the colours sampled from the picture.
    """
    folder = CHARACTERS_DIR / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []
    for upload in files:
        name = Path(upload.filename or "character.png").name
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        dest = folder / name
        with open(dest, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved.append({"name": Path(name).stem, "path": str(dest)})
    if not saved:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(
            400, f"Aucune image valide. Formats acceptés : {', '.join(sorted(IMAGE_SUFFIXES))}"
        )
    return {"characters_dir": str(folder), "characters": saved, "count": len(saved)}


@app.get("/api/voices")
def voices() -> dict:
    return {"voices": tts.VOICES}


@app.get("/api/music")
def music_moods() -> dict:
    return {
        "moods": music.available_moods(),
        "default": music.DEFAULT_MOOD,
        "default_label": music.MOODS[music.DEFAULT_MOOD]["label"],
    }


@app.get("/api/config")
def public_config() -> dict:
    stock = [name for name in ("pexels", "pixabay") if visuals.provider_ready(name)]
    cache = stock_cache.stats_snapshot()
    return {
        "queue": QUEUE_BACKEND,
        "storage": STORAGE_BACKEND,
        "llm": LLM_PROVIDER,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "video": {"width": VIDEO_WIDTH, "height": VIDEO_HEIGHT, "fps": VIDEO_FPS},
        "stock_providers": stock,
        "stock_available": bool(stock),
        "stock_cache": {
            "enabled": cache["enabled"],
            "entries": cache["entries"],
            "size_mb": round(cache["bytes"] / 1024 / 1024, 2),
            "ttl_days": PEXELS_CACHE_TTL_DAYS,
        },
    }


app.mount("/videos", StaticFiles(directory=str(OUTPUT_DIR)), name="videos")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
