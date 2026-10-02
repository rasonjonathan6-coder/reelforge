"""Voice-over generation with edge-tts, returning word-level timings.

edge-tts streams WordBoundary events with 100-nanosecond offsets, which is
enough to drive word-by-word captions without a separate aligner.

Multi-speaker narration (`synthesize_dialogue`) renders one take per line with
its own voice and concatenates them, so a script written as a dialogue gets a
distinct voice per character without any paid TTS API.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import unicodedata
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

# Pause inserted between two dialogue lines, long enough to read as a beat.
DIALOGUE_GAP = 0.32

# Catalogue exposed to the UI. `gender` drives automatic casting: the voice the
# user picked stays on the first character, the others get a contrasting one.
VOICES = [
    {"id": "fr-FR-VivienneMultilingualNeural", "label": "Vivienne (femme, FR)", "gender": "female", "locale": "fr-FR"},
    {"id": "fr-FR-DeniseNeural", "label": "Denise (femme, FR)", "gender": "female", "locale": "fr-FR"},
    {"id": "fr-FR-EloiseNeural", "label": "Éloïse (femme, FR)", "gender": "female", "locale": "fr-FR"},
    {"id": "fr-FR-HenriNeural", "label": "Henri (homme, FR)", "gender": "male", "locale": "fr-FR"},
    {"id": "fr-FR-RemyMultilingualNeural", "label": "Rémy (homme, FR)", "gender": "male", "locale": "fr-FR"},
    {"id": "en-US-AriaNeural", "label": "Aria (femme, EN)", "gender": "female", "locale": "en-US"},
    {"id": "en-US-GuyNeural", "label": "Guy (homme, EN)", "gender": "male", "locale": "en-US"},
]

VOICE_GENDER = {voice["id"]: voice["gender"] for voice in VOICES}
VOICE_LOCALE = {voice["id"]: voice["locale"] for voice in VOICES}

# Female and male pools, ordered by preference, used to cast extra characters.
_GENDER_POOL = {
    "female": [v["id"] for v in VOICES if v["gender"] == "female"],
    "male": [v["id"] for v in VOICES if v["gender"] == "male"],
}


def gender_of(voice: str) -> str:
    """Gender of a known voice, guessed from its name for anything else."""
    if voice in VOICE_GENDER:
        return VOICE_GENDER[voice]
    lowered = (voice or "").lower()
    if any(token in lowered for token in ("denise", "vivienne", "eloise", "aria", "jenny", "ava", "emma", "michelle", "female")):
        return "female"
    if any(token in lowered for token in ("henri", "remy", "guy", "brian", "andrew", "male")):
        return "male"
    return "female"


def voices_for(locale: str | None, gender: str) -> list[str]:
    """Voice ids for a gender, preferring the same locale as the narration."""
    pool = _GENDER_POOL.get(gender, _GENDER_POOL["female"])
    if not locale:
        return pool
    same = [v for v in pool if VOICE_LOCALE.get(v, "").startswith(locale[:2])]
    return same or pool


def cast_voices(base_voice: str, count: int, locale: str | None = None) -> list[str]:
    """Assign a distinct voice to `count` characters.

    The user's chosen voice keeps character 1; the following characters
    alternate the opposite gender and rotate through that pool, so a two-hander
    is always a man and a woman even when the caller supplies no mapping.
    """
    base = base_voice or DEFAULT_VOICE
    locale = locale or VOICE_LOCALE.get(base, "fr-FR")
    base_gender = gender_of(base)
    other_gender = "male" if base_gender == "female" else "female"
    others = voices_for(locale, other_gender)

    cast = [base]
    for index in range(1, max(1, count)):
        cast.append(others[(index - 1) % len(others)])
    return cast


# Accepted values for the dialogue cast, normalised to `mixte`, `female` or
# `male`. "mixte" is the default (man + woman); "femme"/"homme" force every
# character to one gender, which is what a same-sex two-hander needs.
def cast_genders(cast: str | None) -> str:
    """Normalise a cast choice to `mixte`, `female` or `male`."""
    plain = "".join(
        char for char in unicodedata.normalize("NFKD", (cast or "").lower())
        if not unicodedata.combining(char) and char.isalpha()
    )
    # "mixte (homme + femme)" normalises to "mixtehommefemme": check it first.
    if "mixte" in plain or plain in {"", "auto", "mix"}:
        return "mixte"
    # "female" contains "male", so the female check must come first.
    if "femme" in plain or "female" in plain or plain == "f":
        return "female"
    if "homme" in plain or "male" in plain or plain in {"h", "m"}:
        return "male"
    return "mixte"


def resolve_cast(base_voice: str, count: int, cast: str | None = None) -> list[str]:
    """Voice per character for a `cast` choice, one voice per character.

    - `mixte` (default): character 1 keeps the user's voice, character 2 the
      opposite gender.
    - `female` / `male`: every character speaks with that gender (femme+femme,
      homme+homme), starting from the user's voice when it already matches.
    """
    base = base_voice or DEFAULT_VOICE
    count = max(1, int(count))
    if count == 1:
        return [base]

    choice = cast_genders(cast)
    if choice == "mixte":
        return cast_voices(base, count)

    locale = VOICE_LOCALE.get(base, "fr-FR")
    pool = voices_for(locale, choice)
    if gender_of(base) == choice:
        ordered = [base] + [voice for voice in pool if voice != base]
    else:
        ordered = list(pool)
    # Rotate so every character still gets a voice even if the pool is short.
    return [ordered[index % len(ordered)] for index in range(count)]


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
    speaker: str = ""


@dataclass
class Speech:
    audio_path: Path
    duration: float
    words: list[WordTiming]


@dataclass
class DialogueLine:
    """One spoken turn: a character name and the exact words they say."""

    speaker: str
    text: str
    voice: str = ""


@dataclass
class DialogueSpeech(Speech):
    """A `Speech` plus the character names, in speaking order."""

    speakers: list[str] = None  # type: ignore[assignment]
    voice_map: dict = None  # type: ignore[assignment]


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
    words = [WordTiming(w.text, w.start * scale, w.end * scale, w.speaker) for w in speech.words]
    duration = audio_duration(out_path) or target
    print(f"[duration] tempo {factor:.3f} appliqué : {speech.duration:.2f}s -> {duration:.2f}s")
    result = Speech(audio_path=out_path, duration=duration, words=words)
    if isinstance(speech, DialogueSpeech):
        return DialogueSpeech(
            audio_path=out_path, duration=duration, words=words,
            speakers=speech.speakers, voice_map=speech.voice_map,
        )
    return result


def synthesize(
    text: str,
    out_path: Path,
    voice: str = DEFAULT_VOICE,
    rate: str = "+8%",
) -> Speech:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_synthesize(text, voice, rate, out_path))


def _silence(seconds: float, out_path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "anullsrc=r=24000:cl=mono", "-t", f"{max(0.05, seconds):.3f}",
         "-c:a", "libmp3lame", "-q:a", "6", str(out_path)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return out_path


def _concat(parts: list[Path], out_path: Path) -> Path:
    """Concatenate mp3 parts without re-encoding the speech itself."""
    listing = out_path.with_suffix(".txt")
    listing.write_text(
        "".join(f"file '{part.resolve().as_posix()}'\n" for part in parts),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c:a", "libmp3lame", "-q:a", "4", str(out_path)],
            check=True, capture_output=True, text=True, timeout=180,
        )
    finally:
        listing.unlink(missing_ok=True)
    return out_path


def synthesize_dialogue(
    lines: list[DialogueLine],
    out_path: Path,
    rate: str = DEFAULT_RATE,
) -> DialogueSpeech:
    """Render each line with its own voice and stitch them into one track.

    The per-line takes are concatenated (with a short beat between them) rather
    than mixed, so word timings stay exact: each line's boundaries are simply
    shifted by the duration already consumed.
    """
    if not lines:
        raise RuntimeError("dialogue vide")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = out_path.parent / "dialogue"
    parts_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    words: list[WordTiming] = []
    speakers: list[str] = []
    voice_map: dict[str, str] = {}
    offset = 0.0

    for index, line in enumerate(lines):
        voice = line.voice or DEFAULT_VOICE
        if line.speaker and line.speaker not in voice_map:
            voice_map[line.speaker] = voice
            speakers.append(line.speaker)

        take = asyncio.run(_synthesize(line.text, voice, rate, parts_dir / f"line_{index}.mp3"))
        for word in take.words:
            words.append(
                WordTiming(
                    text=word.text,
                    start=word.start + offset,
                    end=word.end + offset,
                    speaker=line.speaker,
                )
            )
        parts.append(take.audio_path)
        offset += take.duration
        if index < len(lines) - 1:
            parts.append(_silence(DIALOGUE_GAP, parts_dir / f"gap_{index}.mp3"))
            offset += DIALOGUE_GAP

    _concat(parts, out_path)
    duration = audio_duration(out_path) or offset
    return DialogueSpeech(
        audio_path=out_path, duration=duration, words=words,
        speakers=speakers, voice_map=voice_map,
    )
