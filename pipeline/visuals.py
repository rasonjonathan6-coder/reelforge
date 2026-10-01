"""Background visuals for the reel.

Sources, in priority order:
  1. Clips supplied by the caller (e.g. AI-generated scenes).
  2. Pexels stock footage when PEXELS_API_KEY is set.
  3. Pixabay stock footage when PIXABAY_API_KEY is set.
  4. Multi-scene animated gradients with a slow zoom, grain and vignette.

Each scene of the narration gets its own search query, so a stock-backed reel
shows varied footage instead of one repeated clip. Segments are joined with
cross-dissolves (xfade). The source actually used is always reported back, so
the app never claims stock footage it did not really use.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

WIDTH, HEIGHT = 1080, 1920
FPS = 30

TRANSITION = 0.6  # cross-dissolve duration, seconds
SCENE_SECONDS = 4.0  # target on-screen time per generated scene
MAX_STOCK_SCENES = 8  # cap on stock searches per reel (keeps downloads sane)
MIN_CLIP_SECONDS = 1.0  # ignore stock entries too short to be usable
MIN_CLIP_HEIGHT = 720  # ignore stock renditions below this height
ZOOM_PER_SECOND = 0.02  # Ken Burns speed; must be slow enough to stay subtle
ZOOM_MAX = 1.15  # never zoom past this, whatever the scene length

PEXELS_URL = "https://api.pexels.com"
PIXABAY_URL = "https://pixabay.com"

PALETTES = [
    ("0x0f0c29", "0x302b63", "0x24243e", "0x1b1b3a"),
    ("0x1a1a2e", "0x16213e", "0x0f3460", "0x533483"),
    ("0x000428", "0x004e92", "0x0b8793", "0x1b2a4a"),
    ("0x232526", "0x414345", "0x2c3e50", "0x1c1c1c"),
    ("0x3a1c71", "0xd76d77", "0x4a2c8f", "0x1e1147"),
    ("0x42275a", "0x734b6d", "0x2c1b47", "0x14061f"),
    ("0x0b486b", "0xf56217", "0x3b1f2b", "0x1a1a2e"),
]

STOPWORDS = {
    "avec", "pour", "dans", "cette", "cela", "plus", "mais", "tout", "sans",
    "sont", "être", "faire", "votre", "vous", "nous", "leur", "quand", "alors",
    "comme", "aussi", "peut", "dont", "très", "bien", "juste", "même", "tous",
    "elle", "ils", "elles", "quoi", "chez", "vers", "deja", "déjà", "les",
    "des", "une", "est", "que", "qui", "sur", "par", "pas", "son", "ses",
    "tes", "cest", "the", "and", "for", "with", "that", "this", "from",
}

# Rotated so consecutive scenes never ask the provider for the same shot.
VISUAL_MODIFIERS = [
    "cinematic", "aerial view", "close up", "slow motion",
    "dramatic lighting", "wide shot", "macro", "timelapse",
]

SOURCE_LABELS = {
    "pexels": "Vidéos Pexels utilisées",
    "pixabay": "Vidéos Pixabay utilisées",
    "ai_clips": "Clips IA utilisés",
    "local_fallback": "Aucune vidéo stock disponible — fallback local utilisé",
}


@dataclass
class Background:
    """Result of building the reel background, with the sources really used."""

    path: Path
    sources: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    clips: list[str] = field(default_factory=list)

    @property
    def visual_source(self) -> str:
        if not self.sources:
            return "local_fallback"
        return self.sources[0] if len(self.sources) == 1 else "+".join(self.sources)

    @property
    def message(self) -> str:
        return source_message(self.sources)


def source_message(sources: list[str]) -> str:
    """Human-readable summary of which visual sources were actually used."""
    if not sources or sources == ["local_fallback"]:
        return SOURCE_LABELS["local_fallback"]
    parts: list[str] = []
    stock = [s for s in sources if s in ("pexels", "pixabay")]
    if stock:
        parts.append("Vidéos " + " + ".join(s.capitalize() for s in stock) + " utilisées")
    if "ai_clips" in sources:
        parts.append(SOURCE_LABELS["ai_clips"])
    if "local_fallback" in sources:
        parts.append("complété par le fallback animé")
    return " · ".join(parts) if parts else SOURCE_LABELS["local_fallback"]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def _per_scene(duration: float, count: int) -> float:
    return (duration + TRANSITION * (count - 1)) / count


def _join_xfade(segments: list[Path], duration: float, out_path: Path) -> Path:
    """Cross-dissolve the segments into a single clip of `duration` seconds."""
    if len(segments) == 1:
        _run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(segments[0]),
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-pix_fmt", "yuv420p", str(out_path),
        ])
        return out_path

    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]

    per_scene = _per_scene(duration, len(segments))
    steps = []
    prev = "[0:v]"
    for index in range(1, len(segments)):
        offset = index * (per_scene - TRANSITION)
        label = f"[v{index}]"
        steps.append(
            f"{prev}[{index}:v]xfade=transition=fade:duration={TRANSITION:.3f}"
            f":offset={offset:.3f}{label}"
        )
        prev = label
    filter_complex = ";".join(steps)

    _run([
        "ffmpeg", "-y", "-loglevel", "error", *inputs,
        "-filter_complex", filter_complex,
        "-map", prev,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        str(out_path),
    ])
    return out_path


def _scene_filters() -> str:
    """Slow zoom + grain + vignette: reads as footage rather than a flat loop."""
    return (
        f"zoompan=z='min(1+{ZOOM_PER_SECOND}*in_time,{ZOOM_MAX})':d=1"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"noise=alls=7:allf=t+u,"
        f"vignette=PI/4.5,"
        f"format=yuv420p"
    )


def generate_scenes(duration: float, out_path: Path, seed: int | None = None) -> Path:
    """Several animated gradient scenes with distinct palettes, cross-dissolved."""
    rng = random.Random(seed)
    count = max(1, math.ceil(duration / SCENE_SECONDS))
    per_scene = _per_scene(duration, count)
    seg_dir = out_path.parent / "scenes"
    seg_dir.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    for index in range(count):
        c0, c1, c2, c3 = rng.choice(PALETTES)
        src = (
            f"gradients=s={WIDTH}x{HEIGHT}:rate={FPS}:"
            f"c0={c0}:c1={c1}:c2={c2}:c3={c3}:nb_colors=4"
            f":speed={rng.uniform(0.010, 0.028):.4f}:duration={per_scene + 1:.3f}"
        )
        seg = seg_dir / f"scene_{index}.mp4"
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", src,
            "-vf", f"gblur=sigma=38,{_scene_filters()}",
            "-t", f"{per_scene:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            str(seg),
        ])
        segments.append(seg)

    return _join_xfade(segments, duration, out_path)


def generate_gradient(duration: float, out_path: Path, seed: int | None = None) -> Path:
    """Single-scene gradient (kept for thumbnails and simple callers)."""
    return generate_scenes(duration, out_path, seed=seed)


# ---------------------------------------------------------------------------
# Search queries, one per scene
# ---------------------------------------------------------------------------
def _keywords(text: str, limit: int = 3) -> list[str]:
    words = re.findall(r"[A-Za-zÀ-ÿ]{4,}", (text or "").lower())
    picked: list[str] = []
    for word in words:
        if word in STOPWORDS or word in picked:
            continue
        picked.append(word)
        if len(picked) >= limit:
            break
    return picked


def _scene_texts(script: str, count: int) -> list[str]:
    """Split the narration into `count` ordered chunks, one per scene."""
    parts = [p.strip() for p in re.split(r"[.!?…]+", script or "") if p.strip()]
    if not parts:
        return [""] * count
    buckets: list[list[str]] = [[] for _ in range(count)]
    for index, part in enumerate(parts):
        buckets[min(count - 1, index * count // len(parts))].append(part)
    return [" ".join(b) for b in buckets]


def _scene_count(duration: float) -> int:
    return max(3, min(MAX_STOCK_SCENES, math.ceil(max(duration, 1.0) / SCENE_SECONDS)))


def scene_queries(topic: str, script: str, count: int) -> list[str]:
    """One distinct, scene-derived search query per scene."""
    topic_words = _keywords(topic, 3)
    scenes = _scene_texts(script, count)
    queries: list[str] = []
    seen: set[str] = set()
    for index in range(count):
        words = _keywords(scenes[index], 2) or topic_words or ["abstract background"]
        modifier = VISUAL_MODIFIERS[index % len(VISUAL_MODIFIERS)]
        query = " ".join(words + [modifier]).strip()
        if query in seen:
            query = f"{query} {index + 1}"
        seen.add(query)
        queries.append(query)
    return queries


# ---------------------------------------------------------------------------
# Stock providers
# ---------------------------------------------------------------------------
def provider_ready(name: str) -> bool:
    return bool(os.environ.get(f"{name.upper()}_API_KEY"))


def _pexels_candidates(query: str, limit: int = 12) -> list[dict]:
    import requests

    base = os.environ.get("PEXELS_API_BASE", PEXELS_URL).rstrip("/")
    resp = requests.get(
        f"{base}/videos/search",
        headers={"Authorization": os.environ["PEXELS_API_KEY"]},
        params={"query": query, "per_page": max(limit, 15)},
        timeout=30,
    )
    resp.raise_for_status()

    out: list[dict] = []
    for video in resp.json().get("videos", []):
        files = [
            f for f in video.get("video_files", [])
            if (f.get("height") or 0) >= MIN_CLIP_HEIGHT and f.get("link")
        ]
        if not files:
            continue
        # Prefer a portrait rendition, then the one closest to the reel height.
        files.sort(key=lambda f: (
            0 if (f.get("height") or 0) > (f.get("width") or 0) else 1,
            abs((f.get("height") or 0) - HEIGHT),
        ))
        best = files[0]
        out.append({
            "id": str(video.get("id")),
            "url": best["link"],
            "duration": float(video.get("duration") or 0),
            "width": int(best.get("width") or 0),
            "height": int(best.get("height") or 0),
        })
    return out


def _pixabay_candidates(query: str, limit: int = 12) -> list[dict]:
    import requests

    base = os.environ.get("PIXABAY_API_BASE", PIXABAY_URL).rstrip("/")
    resp = requests.get(
        f"{base}/api/videos/",
        params={
            "key": os.environ["PIXABAY_API_KEY"],
            "q": query,
            "video_type": "film",
            "safesearch": "true",
            "per_page": max(limit, 20),
        },
        timeout=30,
    )
    resp.raise_for_status()

    out: list[dict] = []
    for hit in resp.json().get("hits", []):
        variants = hit.get("videos", {})
        chosen = None
        for key in ("large", "medium", "small", "tiny"):
            candidate = variants.get(key)
            if candidate and (candidate.get("height") or 0) >= MIN_CLIP_HEIGHT and candidate.get("url"):
                chosen = candidate
                break
        if not chosen:
            continue
        out.append({
            "id": str(hit.get("id")),
            "url": chosen["url"],
            "duration": float(hit.get("duration") or 0),
            "width": int(chosen.get("width") or 0),
            "height": int(chosen.get("height") or 0),
        })
    return out


def _candidates(name: str, query: str, limit: int = 12) -> list[dict]:
    return _pexels_candidates(query, limit) if name == "pexels" else _pixabay_candidates(query, limit)


def _select(candidates: list[dict], needed: int, seen: set[str]) -> list[dict]:
    """Pick usable, not-yet-used candidates, portrait first."""
    pool = [
        c for c in candidates
        if c.get("url") and c.get("id") not in seen
        and (not c.get("duration") or c["duration"] >= MIN_CLIP_SECONDS)
    ]
    pool.sort(key=lambda c: (
        0 if c["height"] > c["width"] else 1,
        abs(c["height"] - HEIGHT),
    ))
    picked = pool[:needed]
    for candidate in picked:
        seen.add(candidate["id"])
    return picked


def _probe_video(path: Path) -> dict | None:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(proc.stdout or "{}")
    except Exception:  # noqa: BLE001 - an unreadable file is simply unusable
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)
    try:
        duration = float((data.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if width <= 0 or height <= 0:
        return None
    return {"width": width, "height": height, "duration": duration}


def _valid_video(path: Path) -> bool:
    info = _probe_video(path)
    return bool(info and info["duration"] >= MIN_CLIP_SECONDS)


def _download(candidates: list[dict], work_dir: Path, tag: str) -> list[Path]:
    import requests

    work_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for candidate in candidates:
        dest = work_dir / f"{tag}_{candidate['id']}.mp4"
        try:
            with requests.get(candidate["url"], stream=True, timeout=90) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        handle.write(chunk)
        except Exception as exc:  # noqa: BLE001 - skip this clip, try the next
            print(f"[visuals] téléchargement {tag} échoué ({exc})")
            dest.unlink(missing_ok=True)
            continue
        if _valid_video(dest):
            clips.append(dest)
        else:
            print(f"[visuals] fichier {tag} inexploitable, ignoré")
            dest.unlink(missing_ok=True)
    return clips


def _collect_stock(queries: list[str], work_dir: Path) -> tuple[list[Path], list[str]]:
    """One real clip per scene when possible: Pexels first, then Pixabay."""
    clips: list[Path] = []
    sources: list[str] = []
    seen: set[str] = set()

    for query in queries:
        for name in ("pexels", "pixabay"):
            if not provider_ready(name):
                continue
            try:
                candidates = _candidates(name, query)
            except Exception as exc:  # noqa: BLE001 - fall through to the next provider
                print(f"[visuals] {name} indisponible ({exc}); fournisseur suivant")
                continue
            files = _download(_select(candidates, 1, seen), work_dir, name)
            if files:
                clips.extend(files)
                if name not in sources:
                    sources.append(name)
                break

    # Scenes with no footage reuse an already-downloaded clip rather than
    # dropping to a gradient, so the reel stays real video end to end.
    base = list(clips)
    if base:
        while len(clips) < len(queries):
            clips.append(base[len(clips) % len(base)])
    return clips, sources


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _cover_filter() -> str:
    """Fill 1080x1920 by cropping, never by stretching (aspect ratio kept).

    `d=1` lets every source frame through, so the clip's own motion is kept;
    the zoom is driven by `in_time` so it still progresses across the scene.
    """
    return (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='min(1+{ZOOM_PER_SECOND}*in_time,{ZOOM_MAX})':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"format=yuv420p"
    )


def _montage(clips: list[Path], duration: float, work_dir: Path, out_path: Path) -> Path:
    """Trim/loop each clip to its scene slot and cross-dissolve them."""
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    per_scene = _per_scene(duration, len(clips))
    segments: list[Path] = []
    for index, clip in enumerate(clips):
        seg = seg_dir / f"seg_{index}.mp4"
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(clip),
            "-t", f"{per_scene:.3f}",
            "-vf", _cover_filter(), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", str(seg),
        ])
        segments.append(seg)
    return _join_xfade(segments, duration, out_path)


def build_background_info(
    duration: float,
    work_dir: Path,
    query: str = "city night vertical",
    use_stock: bool = True,
    clips: list[Path] | None = None,
    topic: str = "",
    script: str = "",
) -> Background:
    """Produce the moving background and report which source was really used.

    `clips` (caller-supplied, e.g. AI scenes) takes priority over every stock
    source. When stock is requested but nothing usable comes back, the local
    animated fallback is used and clearly reported as such.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "background.mp4"

    if clips:
        selected = list(clips)
        _montage(selected, duration, work_dir, out_path)
        return Background(out_path, ["ai_clips"], [], [c.name for c in selected])

    queries: list[str] = []
    if use_stock:
        queries = scene_queries(topic or query, script, _scene_count(duration))
        stock, sources = _collect_stock(queries, work_dir)
        if stock:
            _montage(stock, duration, work_dir, out_path)
            return Background(out_path, sources, queries, [c.name for c in stock])
        print("[visuals] aucune vidéo stock exploitable — fallback local utilisé")

    generate_scenes(duration, out_path)
    return Background(out_path, ["local_fallback"], queries, [])


def build_background(
    duration: float,
    work_dir: Path,
    query: str = "city night vertical",
    use_stock: bool = True,
    clips: list[Path] | None = None,
    topic: str = "",
    script: str = "",
) -> Path:
    """Backwards-compatible wrapper returning only the background path."""
    return build_background_info(
        duration, work_dir, query=query, use_stock=use_stock,
        clips=clips, topic=topic, script=script,
    ).path
