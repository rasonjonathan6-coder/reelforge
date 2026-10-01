"""Voice-over generation with edge-tts, returning word-level timings.

edge-tts streams WordBoundary events with 100-nanosecond offsets, which is
enough to drive word-by-word captions without a separate aligner.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import edge_tts

DEFAULT_VOICE = "fr-FR-VivienneMultilingualNeural"
DEFAULT_RATE = "-5%"  # slower reads as far more natural than the old +8%
TICKS_PER_SECOND = 10_000_000

# Calibrated from real edge-tts output at rate "+0%": about 3.05 words/second
# for French (183 wpm). The actual pace varies with the voice, so callers can
# re-measure after synthesis instead of trusting this estimate.
BASE_WORDS_PER_SECOND = 3.05


def rate_factor(rate: str) -> float:
    """Multiplier applied to the speaking pace by an edge-tts rate string."""
    match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%", str(rate or ""))
    if not match:
        return 1.0
    return max(0.5, min(2.0, 1.0 + float(match.group(1)) / 100.0))


@dataclass
class WordTiming:
    text: str
    start: float
    end: float


@dataclass
class Speech:
    audio_path: Path
    duration: float
    words: list[WordTiming]


def estimate_words(duration: float, rate: str = DEFAULT_RATE) -> int:
    """Word count that should land close to `duration` seconds of speech."""
    pace = BASE_WORDS_PER_SECOND * rate_factor(rate)
    return max(20, round(float(duration) * pace))


def audio_duration(path: Path) -> float | None:
    """Real duration of an audio file, measured with ffprobe.

    edge-tts stops emitting word boundaries at the last word, so the last
    word's end time under-reports the file by the trailing silence (0.3-0.9s
    measured). Measuring the container is the only accurate source.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        value = float(out.stdout.strip())
        return value if value > 0 else None
    except Exception:  # noqa: BLE001 - fall back to the word-boundary estimate
        return None


async def _synthesize(text: str, voice: str, rate: str, out_path: Path) -> Speech:
    communicate = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    words: list[WordTiming] = []

    with open(out_path, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / TICKS_PER_SECOND
                duration = chunk["duration"] / TICKS_PER_SECOND
                words.append(
                    WordTiming(text=chunk["text"], start=start, end=start + duration)
                )

    if not words:
        raise RuntimeError("edge-tts returned no word boundaries")

    return Speech(
        audio_path=out_path,
        duration=audio_duration(out_path) or words[-1].end,
        words=words,
    )


def retime(speech: Speech, target: float, tolerance: float = 1.0) -> Speech:
    """Nudge a voice-over onto `target` with a small tempo change.

    Sizing the script gets close, but content variance (sentence boundaries,
    pauses) still leaves up to ~2s on the table. A tempo change of at most 8%
    is inaudible on speech and lands the video within a second, so it is used
    as a final safety net; beyond that the natural read is kept untouched.
    Word timings are rescaled so karaoke subtitles stay in sync.
    """
    if target <= 0 or abs(speech.duration - target) <= tolerance:
        return speech
    factor = speech.duration / target
    if not 0.85 <= factor <= 1.15:
        return speech

    out_path = speech.audio_path.with_name(f"{speech.audio_path.stem}_retimed.mp3")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(speech.audio_path),
         "-filter:a", f"atempo={factor:.6f}", str(out_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"[duration] atempo indisponible ({result.stderr.strip()[:120]})")
        return speech

    scale = 1.0 / factor
    words = [WordTiming(w.text, w.start * scale, w.end * scale) for w in speech.words]
    duration = audio_duration(out_path) or target
    print(f"[duration] tempo {factor:.3f} appliqué : {speech.duration:.2f}s -> {duration:.2f}s")
    return Speech(audio_path=out_path, duration=duration, words=words)


def synthesize(
    text: str,
    out_path: Path,
    voice: str = DEFAULT_VOICE,
    rate: str = "+8%",
) -> Speech:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_synthesize(text, voice, rate, out_path))
