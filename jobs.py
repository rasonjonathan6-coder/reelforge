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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import REDIS_URL, ROOT
from generate import cast_dialogue, generate
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
    "animation": 70,
    "compose": 75,
    "metadata": 92,
    "completed": 100,
}

# Duration fitting: accept a narration whose measured audio lands within this
# many seconds of the request, else retry (initial try + 2 retries, never more).
DURATION_TOLERANCE = 1.0
MAX_DURATION_ATTEMPTS = 3


def step_progress(step: str, sub: int | None = None) -> int:
    """Progress for a stage, optionally refined by a 0-100 intra-stage value."""
    base = STEP_PROGRESS.get(step, 0)
    if sub is None:
        return base
    order = ["script", "tts", "subtitles", "visuals", "animation", "compose", "metadata", "completed"]
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


def _fit_speech(text: str, req: dict, target: float, work: Path, attempts: int = MAX_DURATION_ATTEMPTS):
    """Synthesize `text`, measure it, and retry until close to `target`.

    Only the narration length is adjusted (never time-stretched), and at most
    `attempts` passes are made. Returns the best `Speech` found plus the text
    that produced it, so the archived script matches the audio.
    """
    voice = req.get("voice", tts.DEFAULT_VOICE)
    rate = req.get("rate", tts.DEFAULT_RATE)
    pace = tts.BASE_WORDS_PER_SECOND * tts.rate_factor(rate)
    probe_dir = work / ".fit"
    probe_dir.mkdir(parents=True, exist_ok=True)

    best_speech = None
    best_text = text
    for attempt in range(attempts):
        speech = tts.synthesize(text, probe_dir / f"voice_{attempt}.mp3", voice=voice, rate=rate)
        print(f"[duration] essai {attempt + 1}/{attempts} : {speech.duration:.2f}s "
              f"({len(speech.words)} mots) pour une cible de {target}s")
        if best_speech is None or abs(speech.duration - target) < abs(best_speech.duration - target):
            best_speech, best_text = speech, text
        if abs(speech.duration - target) <= DURATION_TOLERANCE:
            break
        if attempt < attempts - 1:
            # Ask for exactly the word count the observed pace implies.
            needed = max(20, round(len(speech.words) * (target / max(speech.duration, 0.1))))
            text = script_writer.write_script(
                req.get("topic", ""),
                duration=max(1, round(needed / pace)),
                language=req.get("language", "français"),
                style=req.get("style", ""),
                tone=req.get("tone", ""),
                rate=rate,
            )
    # Keep the chosen take at a stable path: the probe folder is removed next
    # and the renderer needs the audio to survive until compose.
    if best_speech is not None:
        best_speech = tts.retime(best_speech, target, DURATION_TOLERANCE)
        final_audio = work / "voice_fit.mp3"
        shutil.copy(best_speech.audio_path, final_audio)
        best_speech.audio_path = final_audio
    shutil.rmtree(probe_dir, ignore_errors=True)
    return best_speech, best_text


def _load_character_references(folder: str | None) -> None:
    """Register every reference image in `folder` as a character look.

    Images are named after the character (`Léa.png`, `Tom.jpg`); the pipeline
    then paints that character with the colours sampled from the picture. The
    registry is cleared first so a job never inherits a previous job's look.
    """
    from pipeline import avatars

    avatars.clear_references()
    if not folder:
        return
    base = Path(folder)
    if not base.is_dir():
        return
    for image in sorted(base.iterdir()):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        try:
            avatars.load_reference(image.stem, image)
        except Exception:  # noqa: BLE001 - a bad image must not kill the job
            continue


def _dialogue_characters(turns: list) -> list[str]:
    """Character names in first-appearance order (all of them)."""
    return script_writer.characters_of(turns)


def _cast_dialogue(turns: list, base_voice: str, cast: str | None = None) -> list[tts.DialogueLine]:
    """Give each character a voice according to the requested `cast`.

    Delegates to `generate.cast_dialogue` so the CLI and the API share one
    casting implementation. A one-character script keeps that single voice;
    with two or more, `cast` picks the distribution (`mixte` by default).
    """
    return cast_dialogue(turns, base_voice, cast)


def _fit_dialogue(turns: list, req: dict, target: float, work: Path,
                  attempts: int = MAX_DURATION_ATTEMPTS):
    """Two-voice fitting: retry the dialogue until the audio lands on target.

    Returns `(DialogueSpeech, plain_script)` where `plain_script` is the
    `Nom: ...` text that produced the audio, so the archived script matches.
    """
    rate = req.get("rate", tts.DEFAULT_RATE)
    pace = tts.BASE_WORDS_PER_SECOND * tts.rate_factor(rate)
    probe_dir = work / ".fit"
    probe_dir.mkdir(parents=True, exist_ok=True)

    best_speech = None
    best_text = ""
    for attempt in range(attempts):
        lines = _cast_dialogue(turns, req.get("voice", tts.DEFAULT_VOICE),
                               req.get("dialogue_cast"))
        if not lines:
            break
        speech = tts.synthesize_dialogue(lines, probe_dir / f"voice_{attempt}.mp3", rate=rate)
        plain = "\n".join(f"{line.speaker}: {line.text}" for line in lines)
        print(f"[dialogue] essai {attempt + 1}/{attempts} : {speech.duration:.2f}s "
              f"({len(speech.words)} mots, {len(speech.speakers)} voix) pour une cible de {target}s")
        if best_speech is None or abs(speech.duration - target) < abs(best_speech.duration - target):
            best_speech, best_text = speech, plain
        if abs(speech.duration - target) <= DURATION_TOLERANCE:
            break
        if attempt < attempts - 1:
            needed = max(20, round(len(speech.words) * (target / max(speech.duration, 0.1))))
            turns, _ = script_writer.write_dialogue_script(
                req.get("topic", ""),
                duration=max(1, round(needed / pace)),
                language=req.get("language", "français"),
                rate=rate,
            )
    if best_speech is not None:
        best_speech = tts.retime(best_speech, target, DURATION_TOLERANCE)
        final_audio = work / "voice_fit.mp3"
        shutil.copy(best_speech.audio_path, final_audio)
        best_speech.audio_path = final_audio
    shutil.rmtree(probe_dir, ignore_errors=True)
    return best_speech, best_text


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
        dialogue_mode = bool(req.get("dialogue"))
        turns: list = []
        if dialogue_mode:
            # A user-edited dialogue (already `Nom: réplique` lines) is
            # authoritative; only generate when there is no usable script.
            turns = script_writer.parse_dialogue(script) if script.strip() else []
            if not turns:
                turns, script = script_writer.write_dialogue_script(
                    req.get("topic", ""),
                    duration=req.get("duration", 45),
                    language=req.get("language", "français"),
                    rate=req.get("rate", tts.DEFAULT_RATE),
                )
        elif req.get("auto_script") and req.get("topic"):
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
            json.dumps({"topic": req.get("topic", ""), "script": script,
                        "dialogue": dialogue_mode},
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

        target = float(req["duration"]) if req.get("duration") else None
        speech = None
        if req.get("auto_script") and req.get("topic") and target:
            if dialogue_mode:
                turns, _ = script_writer.write_dialogue_script(
                    req.get("topic", ""),
                    duration=max(1, round(target)),
                    language=req.get("language", "français"),
                    rate=req.get("rate", tts.DEFAULT_RATE),
                )
                speech, script = _fit_dialogue(turns, req, target, work)
            else:
                # The topic panel asks for an initial draft sized to the requested
                # duration; regenerate it here (otherwise the fitting loop inherits
                # a script built for a different duration and cannot converge).
                script = script_writer.write_script(
                    req.get("topic", ""),
                    duration=max(1, round(target)),
                    language=req.get("language", "français"),
                    style=req.get("style", ""),
                    tone=req.get("tone", ""),
                    rate=req.get("rate", tts.DEFAULT_RATE),
                )
                # Fit the narration to the requested duration (max 3 TTS passes);
                # a user-supplied text is respected as-is.
                speech, script = _fit_speech(script, req, target, work)
            (work / "script.json").write_text(
                json.dumps({"topic": req.get("topic", ""), "script": script,
                            "dialogue": dialogue_mode},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif dialogue_mode and turns:
            speech = tts.synthesize_dialogue(
                _cast_dialogue(turns, req.get("voice", tts.DEFAULT_VOICE),
                               req.get("dialogue_cast")),
                work / "voice.mp3",
                rate=req.get("rate", tts.DEFAULT_RATE),
            )

        speakers = list(getattr(speech, "speakers", None) or []) if speech else []

        _load_character_references(req.get("characters_dir"))

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
            topic=req.get("topic", ""),
            on_step=on_step,
            info_out=work,
            speech=speech,
            target_duration=target,
            speakers=speakers,
            music_enabled=bool(req.get("music", True)),
            music_mood=req.get("music_mood") or None,
            animated_characters=bool(req.get("animated_characters")),
            character_style=req.get("character_style") or "anime",
            animation_provider=req.get("animation_provider") or None,
            quality=req.get("quality") or "",
        )

        # Music report written by `generate`; absent when the bed was skipped.
        music_meta = {}
        music_file = work / "music.json"
        if music_file.exists():
            try:
                music_meta = json.loads(music_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - diagnostics only
                music_meta = {}

        # Intermediate artefacts required by the batch layout.
        visuals_meta = {}
        visuals_file = work / "visuals.json"
        if visuals_file.exists():
            try:
                visuals_meta = json.loads(visuals_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - diagnostics only
                visuals_meta = {}
        (work / "scenes.json").write_text(
            json.dumps({"use_stock": req.get("use_stock", True),
                        "query": req.get("query", ""),
                        "clips": [c.name for c in (clips or [])],
                        "source": visuals_meta.get("source", "local_fallback"),
                        "sources": visuals_meta.get("sources", []),
                        "queries": visuals_meta.get("queries", []),
                        "scene_origins": visuals_meta.get("scene_origins", []),
                        "cache": visuals_meta.get("cache", {})},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        duration = _probe_duration(video_path)
        video_url = storage.upload(video_path, _rel(video_path))

        durations = {}
        duration_file = work / "duration.json"
        if duration_file.exists():
            try:
                durations = json.loads(duration_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - diagnostics only
                durations = {}
        durations.setdefault("target_duration", target)
        durations["final_video_duration"] = duration
        if target and duration and abs(duration - target) > DURATION_TOLERANCE:
            print(
                f"[duration] écart de {duration - target:+.2f}s par rapport à la cible "
                f"({target}s) ; durée réelle conservée"
            )

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
                "target_duration": durations.get("target_duration"),
                "audio_duration": durations.get("audio_duration"),
                "final_video_duration": durations.get("final_video_duration"),
                "visual_source": visuals_meta.get("source", "local_fallback"),
                "visual_sources": visuals_meta.get("sources", []),
                "visual_queries": visuals_meta.get("queries", []),
                "visual_scene_origins": visuals_meta.get("scene_origins", []),
                "visual_cache": visuals_meta.get("cache", {}),
                "dialogue": dialogue_mode,
                "speakers": speakers,
                "voice_map": dict(getattr(speech, "voice_map", None) or {}) if speech else {},
                "music_mood": music_meta.get("mood"),
                "music_label": music_meta.get("label"),
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
