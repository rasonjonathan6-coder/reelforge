"""Build a karaoke-style ASS subtitle track from word timings.

Words are grouped into short on-screen chunks (max_chars / max_gap) and the
active word is highlighted via ASS inline colour tags. When the words carry a
`speaker`, each character gets their own base colour so a two-hander is readable
at a glance.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.tts import WordTiming

PLAY_RES = (1080, 1920)
FONT = "DejaVu Sans"
FONT_SIZE = 96
HIGHLIGHT = "&H0000E5FF"  # BGR: orange-yellow
BASE_COLOR = "&H00FFFFFF"  # white
OUTLINE_COLOR = "&H00000000"
# Captions sit above the bottom safe area so TikTok/Reels UI never covers them.
MARGIN_V = 420

# One base colour per speaker, assigned in order of first appearance. ASS uses
# BGR byte order, and each stays bright enough to read over busy footage.
SPEAKER_COLORS = [
    "&H00FFFFFF",  # white
    "&H00FFE0A0",  # light blue-cyan (BGR)
    "&H00A0FFB0",  # light green
    "&H00C0A0FF",  # light pink
]


def speaker_palette(speakers: list[str]) -> dict[str, str]:
    """Map each speaker name to a distinct caption colour."""
    return {
        name: SPEAKER_COLORS[index % len(SPEAKER_COLORS)]
        for index, name in enumerate(speakers)
    }


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def _chunk(words: list[WordTiming], max_chars: int = 22, max_gap: float = 0.7):
    chunk: list[WordTiming] = []
    for word in words:
        if chunk:
            too_long = len(" ".join(w.text for w in chunk + [word])) > max_chars
            gap = word.start - chunk[-1].end
            if too_long or gap > max_gap:
                yield chunk
                chunk = []
        chunk.append(word)
    if chunk:
        yield chunk


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def build_ass(words: list[WordTiming], out_path: Path,
              speakers: list[str] | None = None) -> Path:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_RES[0]}
PlayResY: {PLAY_RES[1]}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{FONT},{FONT_SIZE},{BASE_COLOR},&H000000FF,{OUTLINE_COLOR},&H64000000,-1,0,0,0,100,100,0,0,1,7,4,2,80,80,{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    palette = speaker_palette(speakers or [])
    lines: list[str] = []
    for chunk in _chunk(words):
        for active in chunk:
            parts = []
            for word in chunk:
                color = HIGHLIGHT if word is active else palette.get(word.speaker, BASE_COLOR)
                parts.append(f"{{\\c{color}}}{_escape(word.text)}{{\\c{BASE_COLOR}}}")
            text = " ".join(parts)
            start = active.start
            end = min(active.end + 0.12, chunk[-1].end + 0.25)
            # Short pop on the whole line + fade in/out keeps captions lively
            # without the jitter a per-word scale animation would cause.
            lead = "{\\fad(80,60)\\fscx104\\fscy104\\t(0,140,\\fscx100\\fscy100)}"
            lines.append(
                f"Dialogue: 0,{_ts(start)},{_ts(end)},Caption,,0,0,0,,{lead}{text}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return out_path
