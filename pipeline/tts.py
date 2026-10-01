"""Voice-over generation with edge-tts, returning word-level timings.

edge-tts streams WordBoundary events with 100-nanosecond offsets, which is
enough to drive word-by-word captions without a separate aligner.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import edge_tts

DEFAULT_VOICE = "fr-FR-DeniseNeural"
TICKS_PER_SECOND = 10_000_000


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
        duration=words[-1].end,
        words=words,
    )


def synthesize(
    text: str,
    out_path: Path,
    voice: str = DEFAULT_VOICE,
    rate: str = "+8%",
) -> Speech:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_synthesize(text, voice, rate, out_path))
