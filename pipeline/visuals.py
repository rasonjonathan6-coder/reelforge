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

from config import PEXELS_CACHE_TTL_DAYS
from pipeline import stock_cache

WIDTH, HEIGHT = 1080, 1920
FPS = 30

# The pipeline always prefers a portrait rendition, so that is the orientation
# a request is keyed on. The rendition actually picked may still be landscape
# when no portrait exists; its real width/height are recorded in the cache meta.
REQUEST_ORIENTATION = "portrait"

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
    # Bright, saturated palettes: a dark, low-contrast gradient reads as a
    # frozen image once compressed, so keep the fallback vivid and luminous.
    ("0x6a11cb", "0x2575fc", "0x00c6ff", "0x7b2ff7"),
    ("0xff512f", "0xf09819", "0xff2e63", "0x7b1fa2"),
    ("0x00b09b", "0x96c93d", "0x1fa2ff", "0x12d8fa"),
    ("0x2af598", "0x009efd", "0x22e1ff", "0x3b2667"),
    ("0xf857a6", "0xff5858", "0x8e2de2", "0x4a00e0"),
    ("0xfc466b", "0x3f5efb", "0x00d2ff", "0x3a47d5"),
    ("0x11998e", "0x38ef7d", "0xf7b733", "0xfc4a1a"),
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
    scene_origins: list[str] = field(default_factory=list)
    cache_stats: dict = field(default_factory=dict)
    scene_weights: list[float] = field(default_factory=list)

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
    weights = [1.0] * len(segments)
    return _join_xfade_timed(segments, weights, duration, out_path)


def _join_xfade_timed(segments: list[Path], weights: list[float], duration: float,
                      out_path: Path) -> Path:
    """Cross-dissolve segments whose on-screen time follows `weights`.

    A dialogue gives each turn its own scene, and turns are not equally long, so
    the slot of each segment is proportional to its weight (its word count)
    rather than the flat `duration / count` used for narration.
    """
    if len(segments) == 1:
        _run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(segments[0]),
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-pix_fmt", "yuv420p", str(out_path),
        ])
        return out_path

    total = sum(weights) or float(len(weights))
    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]

    steps = []
    prev = "[0:v]"
    offset = 0.0
    for index in range(1, len(segments)):
        offset += (duration + TRANSITION * (len(segments) - 1)) * weights[index - 1] / total - TRANSITION
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
        f"noise=alls=14:allf=t+u,"
        f"vignette=PI/5.5,"
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
            f":speed={rng.uniform(0.06, 0.12):.4f}:duration={per_scene + 1:.3f}"
        )
        seg = seg_dir / f"scene_{index}.mp4"
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", src,
            # Light blur only: a heavy blur flattens the colours into one another
            # and the moving gradient becomes invisible.
            "-vf", f"gblur=sigma=6,{_scene_filters()}",
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


# A dialogue script labels each line with its character (`Léo: ...`). Those
# names are not visual subjects, so they are stripped before keyword extraction.
_SPEAKER_LABEL = re.compile(r"^\s*[\wÀ-ÿ'’ .-]{1,24}\s*[:\u2013\u2014-]\s*", re.M)


def _narration(script: str) -> str:
    return _SPEAKER_LABEL.sub("", script or "")


def _scene_texts(script: str, count: int) -> list[str]:
    """Split the narration into `count` ordered chunks, one per scene."""
    parts = [p.strip() for p in re.split(r"[.!?…\n]+", _narration(script)) if p.strip()]
    if not parts:
        return [""] * count
    buckets: list[list[str]] = [[] for _ in range(count)]
    for index, part in enumerate(parts):
        buckets[min(count - 1, index * count // len(parts))].append(part)
    return [" ".join(b) for b in buckets]


def _group_scenes(texts: list[str], weights: list[float], cap: int) -> tuple[list[str], list[float]]:
    """Fold an over-long scene list into at most `cap` contiguous scenes.

    Keeps the script order (each scene is a run of consecutive turns) and sums
    the weights, so the on-screen time stays proportional. Used to bound the
    number of stock downloads when a dialogue has many short turns.
    """
    return _fold_groups(texts, cap), _fold_groups(weights, cap, mode="sum")


def _fold_groups(values: list, cap: int, mode: str = "join") -> list:
    """Group `values` into `cap` contiguous buckets, keeping the script order.

    `mode="join"` concatenates text buckets (the first non-empty wins for
    `mode="first"`, used for places); `mode="sum"` adds numeric buckets (the
    weights). All callers use the same bucket boundaries, so texts, weights,
    places and actions stay aligned.
    """
    if len(values) <= cap:
        return list(values)
    out: list = []
    for index in range(cap):
        lo = index * len(values) // cap
        hi = (index + 1) * len(values) // cap
        chunk = values[lo:hi]
        if mode == "sum":
            out.append(sum(chunk))
        elif mode == "first":
            out.append(next((x for x in chunk if x), ""))
        else:
            out.append(" ".join(x for x in chunk if x).strip())
    return out


def _scene_count(duration: float) -> int:
    return max(3, min(MAX_STOCK_SCENES, math.ceil(max(duration, 1.0) / SCENE_SECONDS)))


def scene_queries(topic: str, script: str, count: int) -> list[str]:
    """One distinct search query per scene, mixing the topic and the scene text.

    Topic words keep the reel on subject even when a scene mentions an aside;
    the scene's own words give consecutive scenes different footage.
    """
    return queries_for_texts(topic, _scene_texts(script, count))


def scene_texts_for(script: str, duration: float,
                    cap: int | None = None) -> list[str]:
    """One text per scene for a whole reel, sized to `duration`.

    Public wrapper around the internal scene split, so callers that need the
    scene texts themselves (not search queries) do not reach into a private
    helper. `cap` folds an over-long list into that many contiguous scenes.
    """
    texts = _scene_texts(script, _scene_count(duration))
    if cap and len(texts) > cap:
        texts = _fold_groups(texts, cap)
    return texts


def _place_words(place: str) -> list[str]:
    """Words of a detected place, kept even when short (e.g. "rue", "mer")."""
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", (place or "").lower())
    return [w for w in words if w not in STOPWORDS]


def queries_for_texts(topic: str, texts: list[str], places: list[str] | None = None) -> list[str]:
    """Turn one text per scene into one search query per scene.

    `places` gives each scene its own setting (detected from that turn), so the
    background follows the script: a kitchen turn searches for a kitchen, a
    street turn for a street.
    """
    topic_words = _keywords(topic, 3)
    queries: list[str] = []
    seen: set[str] = set()
    for index, text in enumerate(texts):
        place = places[index] if places and index < len(places) else ""
        scene_words = _keywords(text, 2)
        if place:
            words = (topic_words[:1] + _place_words(place) + scene_words)[:3]
        else:
            words = (topic_words[:2] + scene_words)[:3]
        if not words:
            words = topic_words or ["abstract background"]
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
             "-show_entries", "stream=width,height,codec_name:format=duration",
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
    return {
        "width": width,
        "height": height,
        "duration": duration,
        "video_codec": streams[0].get("codec_name"),
    }


def _valid_video(path: Path) -> bool:
    info = _probe_video(path)
    return bool(info and info["duration"] >= MIN_CLIP_SECONDS)


def _fetch_one(candidate: dict, dest: Path) -> bool:
    """Download one candidate to `dest`. Returns True on success.

    `dest` is normally a cache temp file, so a failure never leaves a partial
    file where a valid cache entry is expected.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(candidate["url"], stream=True, timeout=90) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
    except Exception as exc:  # noqa: BLE001 - skip this clip, try the next
        print(f"[visuals] téléchargement échoué ({exc})")
        dest.unlink(missing_ok=True)
        return False
    return True


def _download(candidates: list[dict], work_dir: Path, tag: str) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for candidate in candidates:
        dest = work_dir / f"{tag}_{candidate['id']}.mp4"
        if not _fetch_one(candidate, dest):
            continue
        if _valid_video(dest):
            clips.append(dest)
        else:
            print(f"[visuals] fichier {tag} inexploitable, ignoré")
            dest.unlink(missing_ok=True)
    return clips


def _validate_clip(path: Path) -> dict | None:
    """ffprobe a clip and return its metadata only if it is usable."""
    info = _probe_video(path)
    if not info or info["duration"] < MIN_CLIP_SECONDS:
        return None
    return info


def _scene_clip(name: str, query: str, work_dir: Path, seen: set[str], stats) -> tuple[Path | None, str]:
    """Get one usable clip for a scene, via the cache when enabled.

    Returns `(path, origin)` where origin is `pexels_cache` (served from the
    persistent cache) or `pexels_api` (freshly searched and downloaded). The
    provider name is used verbatim, so Pixabay reports `pixabay_*`.
    """
    key = stock_cache.cache_key(name, query, REQUEST_ORIENTATION)

    def fetch():
        candidates = _candidates(name, query)
        picked = _select(candidates, 1, set(seen))
        if not picked:
            return None, None
        candidate = picked[0]
        tmp = stock_cache.temp_path(key)
        if not _fetch_one(candidate, tmp):
            return None, None
        return tmp, {
            "source_url": candidate.get("url"),
            "provider_id": candidate.get("id"),
        }

    path, status, meta = stock_cache.get(
        name, query, REQUEST_ORIENTATION, fetch, _validate_clip,
        ttl_days=PEXELS_CACHE_TTL_DAYS, stats=stats,
    )

    if path is not None:
        if meta and meta.get("provider_id"):
            seen.add(str(meta["provider_id"]))
        return path, f"{name}_cache" if status == stock_cache.HIT else f"{name}_api"

    if status == stock_cache.DISABLED:
        # Cache off: keep the original, uncached behaviour.
        candidates = _candidates(name, query)
        files = _download(_select(candidates, 1, seen), work_dir, name)
        return (files[0], f"{name}_api") if files else (None, "")

    return None, ""


def _collect_stock(queries: list[str], work_dir: Path) -> tuple[list[Path], list[str], stock_cache.CacheStats, list[str]]:
    """One real clip per scene when possible: Pexels first, then Pixabay.

    Each scene first asks the persistent cache; only a cache miss triggers a
    provider search and download. The returned `scene_origins` mirrors
    `queries` so a caller can tell which scenes came from the cache.
    """
    clips: list[Path] = []
    sources: list[str] = []
    origins: list[str] = []
    found: list[Path | None] = []
    seen: set[str] = set()
    stats = stock_cache.CacheStats()

    for query in queries:
        clip: Path | None = None
        origin = ""
        for name in ("pexels", "pixabay"):
            if not provider_ready(name):
                continue
            try:
                clip, origin = _scene_clip(name, query, work_dir, seen, stats)
            except Exception as exc:  # noqa: BLE001 - fall through to the next provider
                print(f"[visuals] {name} indisponible ({exc}); fournisseur suivant")
                continue
            if clip is not None:
                if name not in sources:
                    sources.append(name)
                break
        found.append(clip)
        origins.append(origin)

    # Scenes with no footage reuse an already-downloaded clip rather than
    # dropping to a gradient, so the reel stays real video end to end.
    base = [clip for clip in found if clip is not None]
    if base:
        for index, clip in enumerate(found):
            if clip is not None:
                clips.append(clip)
            else:
                clips.append(base[index % len(base)])
                origins[index] = "reused"
    return clips, sources, stats, origins


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _cover_filter(pan: bool = False) -> str:
    """Fill 1080x1920 by cropping, never by stretching (aspect ratio kept).

    `d=1` lets every source frame through, so the clip's own motion is kept;
    the zoom is driven by `in_time` so it still progresses across the scene.
    `pan` also slides the crop sideways while zoomed in, so a walking scene
    reads as the camera travelling with the character.
    """
    if pan:
        x_expr = "iw/2-(iw/zoom/2)+(iw-iw/zoom)/2*sin(2*PI*in_time/8)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    return (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='min(1+{ZOOM_PER_SECOND}*in_time,{ZOOM_MAX})':d=1:"
        f"x='{x_expr}':y='{y_expr}':s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"format=yuv420p"
    )


def _montage(clips: list[Path], duration: float, work_dir: Path, out_path: Path,
             weights: list[float] | None = None,
             pans: list[bool] | None = None) -> Path:
    """Trim/loop each clip to its scene slot and cross-dissolve them.

    `weights` gives each clip its share of the timeline (one weight per clip);
    when omitted every clip gets an equal slot, as for narration. `pans` adds a
    travelling crop to the scenes whose character is walking.
    """
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    if not weights or len(weights) != len(clips):
        weights = [1.0] * len(clips)
    if not pans or len(pans) != len(clips):
        pans = [False] * len(clips)
    total = sum(weights) or float(len(clips))
    span = duration + TRANSITION * (len(clips) - 1)
    slots = [span * w / total for w in weights]
    segments: list[Path] = []
    for index, clip in enumerate(clips):
        seg = seg_dir / f"seg_{index}.mp4"
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(clip),
            "-t", f"{slots[index]:.3f}",
            "-vf", _cover_filter(pan=pans[index]), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", str(seg),
        ])
        segments.append(seg)
    return _join_xfade_timed(segments, weights, duration, out_path)


def build_background_info(
    duration: float,
    work_dir: Path,
    query: str = "city night vertical",
    use_stock: bool = True,
    clips: list[Path] | None = None,
    topic: str = "",
    script: str = "",
    scene_texts: list[str] | None = None,
    scene_weights: list[float] | None = None,
    scene_places: list[str] | None = None,
    scene_pans: list[bool] | None = None,
) -> Background:
    """Produce the moving background and report which source was really used.

    `clips` (caller-supplied, e.g. AI scenes) takes priority over every stock
    source. When stock is requested but nothing usable comes back, the local
    animated fallback is used and clearly reported as such.

    `scene_texts` gives one text per scene (a dialogue passes one per turn, so
    the footage follows the script line by line); `scene_weights` sizes each
    scene's on-screen time. `scene_places` gives each scene its own setting
    (detected from that turn) so the decor follows the script; `scene_pans`
    adds a travelling camera to the walking scenes. All default to narration.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "background.mp4"

    if clips:
        selected = list(clips)
        _montage(selected, duration, work_dir, out_path, weights=scene_weights,
                 pans=scene_pans)
        return Background(out_path, ["ai_clips"], [], [c.name for c in selected])

    queries: list[str] = []
    if use_stock:
        texts = scene_texts or _scene_texts(script, _scene_count(duration))
        weights = scene_weights if scene_weights and len(scene_weights) == len(texts) else None
        places = scene_places if scene_places and len(scene_places) == len(texts) else None
        pans = scene_pans if scene_pans and len(scene_pans) == len(texts) else None
        if scene_texts:
            texts = _fold_groups(texts, MAX_STOCK_SCENES)
            weights = _fold_groups(weights or [1.0] * len(scene_texts), MAX_STOCK_SCENES, mode="sum")
            places = _fold_groups(places, MAX_STOCK_SCENES, mode="first") if places else None
            pans = _fold_groups(pans, MAX_STOCK_SCENES, mode="first") if pans else None
        queries = queries_for_texts(topic or query, texts, places=places)
        stock, sources, stats, origins = _collect_stock(queries, work_dir)
        if stock:
            weights = weights if weights and len(weights) == len(stock) else None
            pans = pans if pans and len(pans) == len(stock) else None
            _montage(stock, duration, work_dir, out_path, weights=weights, pans=pans)
            return Background(
                out_path, sources, queries, [c.name for c in stock],
                scene_origins=origins, cache_stats=stats.as_dict(),
                scene_weights=list(weights or []),
            )
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
    scene_texts: list[str] | None = None,
    scene_weights: list[float] | None = None,
    scene_places: list[str] | None = None,
    scene_pans: list[bool] | None = None,
) -> Path:
    """Backwards-compatible wrapper returning only the background path."""
    return build_background_info(
        duration, work_dir, query=query, use_stock=use_stock,
        clips=clips, topic=topic, script=script,
        scene_texts=scene_texts, scene_weights=scene_weights,
        scene_places=scene_places, scene_pans=scene_pans,
    ).path
