"""Background music: mood detection, procedural generation, and voice ducking.

No paid asset is involved. A reel's mood is inferred from the topic and script
keywords, then a short chord loop is synthesised with FFmpeg's `aevalsrc` and
shaped with a low-pass, echo and tremolo so it sits under a voice-over. The mix
step uses `sidechaincompress` so the music automatically dips while narration
plays, which is what makes a bed sound intentional rather than pasted on.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SAMPLE_RATE = 44100

# How loud the bed sits under the voice. The sidechain stage drops it further
# whenever speech is present, so this is the "quiet passage" level.
MUSIC_GAIN = 0.30
# Fade lengths keep the bed from starting or ending abruptly.
FADE_IN = 1.5
FADE_OUT = 2.5

# Mood -> musical parameters. Roots are bass frequencies (Hz); `chords` are
# semitone offsets from the root, `wave` the relative amplitude of each partial.
MOODS: dict[str, dict] = {
    "calme": {
        "label": "Calme / inspirant",
        "root": 98.0,          # G2
        "chords": ([0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]),
        "lowpass": 1100,
        "echo": "0.6:0.45:90:0.22",
        "tremolo": "f=0.10:d=0.30",
        "partials": (0.9, 0.5, 0.3),
    },
    "epique": {
        "label": "Épique / dramatique",
        "root": 73.4,          # D2
        "chords": ([0, 3, 7], [-2, 2, 5], [3, 7, 10], [0, 3, 7]),
        "lowpass": 1500,
        "echo": "0.7:0.5:70:0.25",
        "tremolo": "f=0.16:d=0.35",
        "partials": (1.0, 0.7, 0.5),
    },
    "tension": {
        "label": "Tension / mystère",
        "root": 87.3,          # F2
        "chords": ([0, 1, 7], [0, 1, 6], [-1, 3, 8], [0, 1, 7]),
        "lowpass": 900,
        "echo": "0.8:0.6:120:0.3",
        "tremolo": "f=0.22:d=0.45",
        "partials": (0.9, 0.75, 0.6),
    },
    "energique": {
        "label": "Énergique / motivant",
        "root": 130.8,         # C3
        "chords": ([0, 4, 7], [7, 11, 14], [5, 9, 12], [0, 4, 7]),
        "lowpass": 2100,
        "echo": "0.5:0.35:50:0.2",
        "tremolo": "f=0.30:d=0.5",
        "partials": (1.0, 0.6, 0.4),
    },
    "fun": {
        "label": "Léger / fun",
        "root": 146.8,         # D3
        "chords": ([0, 4, 7], [2, 5, 9], [4, 7, 11], [0, 4, 7]),
        "lowpass": 2600,
        "echo": "0.45:0.3:45:0.18",
        "tremolo": "f=0.38:d=0.55",
        "partials": (1.0, 0.5, 0.35),
    },
}

# Keyword hints, French first (the product's default language). The score is
# weighted so an explicit topic word beats an incidental script word.
MOOD_KEYWORDS: dict[str, list[str]] = {
    "calme": [
        "calme", "paix", "zen", "meditation", "méditation", "respir", "douceur",
        "inspir", "confiance", "nature", "sommeil", "bien-etre", "bien-être",
        "serein", "gratitude", "positive", "espoir",
    ],
    "epique": [
        "epique", "épique", "hero", "héros", "guerre", "combat", "victoire",
        "destin", "legende", "légende", "puissance", "dragon", "roi", "bataille",
        "courage", "exploit", "sauver",
    ],
    "tension": [
        "mystere", "mystère", "mystérieux", "mysterieux", "peur", "horreur",
        "secret", "disparu", "dispara", "enquete", "enquête", "crime", "sombre",
        "menace", "danger", "etrange", "étrange", "suspect", "police", "ombre",
        "mensonge", "invisible", "personne",
    ],
    "energique": [
        "astuce", "astuces", "conseil", "productiv", "sport", "muscle", "gagner",
        "argent", "business", "reussir", "réussir", "succes", "succès", "vite",
        "rapide", "action", "objectif", "motivation", "challenge", "tiktok",
        "viral", "croissance", "entrepreneur",
    ],
    "fun": [
        "drôle", "drole", "humour", "rire", "blague", "fun", "amusant", "meme",
        "mème", "jeu", "challenge", "absurde", "insolite", "deguisement",
    ],
}

DEFAULT_MOOD = "energique"


def available_moods() -> list[dict]:
    return [{"id": key, "label": value["label"]} for key, value in MOODS.items()]


def _tokens(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def detect_mood(topic: str = "", script: str = "") -> str:
    """Pick the mood whose keywords best match the topic and script.

    The topic is weighted triple: it is the explicit statement of what the reel
    is about, while the script may mention a word only in passing.
    """
    topic_text = _tokens(topic)
    script_text = _tokens(script)
    if not topic_text and not script_text:
        return DEFAULT_MOOD

    best_mood = DEFAULT_MOOD
    best_score = 0
    for mood, keywords in MOOD_KEYWORDS.items():
        score = 0
        for word in keywords:
            if word in topic_text:
                score += 3
            if word in script_text:
                score += 1
        if score > best_score:
            best_score, best_mood = score, mood
    return best_mood


def _chord_expression(mood: dict, duration: float) -> str:
    """An `aevalsrc` expression cycling a 4-chord loop for `duration` seconds."""
    root = mood["root"]
    chords = mood["chords"]
    amps = mood["partials"]
    loop_seconds = max(8.0, min(16.0, duration / max(1, round(duration / 12.0))))
    step = loop_seconds / len(chords)

    def chord(freqs: list[float]) -> str:
        return "+".join(
            f"{amp:.3f}*sin(2*PI*{freq:.3f}*t)" for freq, amp in zip(freqs, amps)
        )

    triads = [[root * 2 ** (semi / 12) for semi in chord_offsets] for chord_offsets in chords]
    expression = chord(triads[-1])
    for index in range(len(triads) - 2, -1, -1):
        expression = f"if(lt(mod(t,{loop_seconds:.3f}),{step * (index + 1):.3f}),{chord(triads[index])},{expression})"
    return expression


def generate(
    duration: float,
    out_path: Path,
    mood: str | None = None,
    topic: str = "",
    script: str = "",
) -> Path:
    """Synthesise a music bed of exactly `duration` seconds.

    `mood` overrides detection; otherwise the mood is inferred from the text.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen = mood if mood in MOODS else detect_mood(topic, script)
    spec = MOODS[chosen]
    duration = max(1.0, float(duration))

    expression = _chord_expression(spec, duration)
    fade_out_start = max(0.0, duration - FADE_OUT)
    filters = [
        f"lowpass=f={spec['lowpass']}",
        f"aecho={spec['echo']}",
        f"tremolo={spec['tremolo']}",
        f"afade=t=in:st=0:d={min(FADE_IN, duration / 3):.2f}",
        f"afade=t=out:st={fade_out_start:.2f}:d={min(FADE_OUT, duration / 3):.2f}",
    ]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi",
         "-i", f"aevalsrc='{expression}':s={SAMPLE_RATE}:d={duration:.3f}",
         "-af", ",".join(filters),
         "-c:a", "libmp3lame", "-q:a", "4", str(out_path)],
        check=True, capture_output=True, text=True, timeout=180,
    )
    return out_path


def mix_into_video(
    video: Path,
    music: Path,
    out_path: Path,
    duration: float,
) -> Path:
    """Return `video` with its voice-over re-mixed against the music bed.

    The video stream is copied untouched; only the audio is rebuilt, with the
    bed ducked under the narration via `sidechaincompress`. `normalize=0` keeps
    the voice at full level instead of `amix` halving it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fade_out_start = max(0.0, duration - FADE_OUT)
    filter_complex = (
        f"[0:a]aformat=sample_rates={SAMPLE_RATE}:channel_layouts=mono,"
        f"anull[voice];"
        f"[1:a]aformat=sample_rates={SAMPLE_RATE}:channel_layouts=mono,"
        f"volume={MUSIC_GAIN},"
        f"afade=t=in:st=0:d={min(FADE_IN, duration / 3):.2f},"
        f"afade=t=out:st={fade_out_start:.2f}:d={min(FADE_OUT, duration / 3):.2f}[mus];"
        f"[mus][voice]sidechaincompress="
        f"threshold=0.05:ratio=8:attack=15:release=350:makeup=1[duck];"
        f"[voice][duck]amix=inputs=2:duration=first:normalize=0,"
        f"alimiter=limit=0.95[aout]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video), "-i", str(music),
         "-filter_complex", filter_complex,
         "-map", "0:v", "-map", "[aout]",
         "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
         "-movflags", "+faststart",
         str(out_path)],
        check=True, capture_output=True, text=True, timeout=300,
    )
    return out_path


def build(
    video: Path,
    work_dir: Path,
    duration: float,
    mood: str | None = None,
    topic: str = "",
    script: str = "",
) -> dict:
    """Generate the bed for this reel and mix it under the voice-over.

    Returns a small report (mood used, mixed video path) so the job can archive
    it. Raises on failure; callers treat music as optional and skip on error.
    """
    chosen = mood if mood in MOODS else detect_mood(topic, script)
    music_path = work_dir / "music.mp3"
    generate(duration, music_path, mood=chosen, topic=topic, script=script)
    mixed = work_dir / "with_music.mp4"
    mix_into_video(video, music_path, mixed, duration)
    return {"mood": chosen, "label": MOODS[chosen]["label"], "path": str(mixed)}
