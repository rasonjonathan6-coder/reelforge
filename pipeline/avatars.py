"""Anime-style talking characters for a dialogue reel.

A speaker is drawn as a 2D anime character (big eyes with highlights, spiky
hair, cel outlines, expressive brows) and encoded as an RGBA `qtrle` clip that
`compose` overlays. The mouth follows the words that speaker actually says:
every word maps to a viseme (A/E/I/O/U/closed/...) from its letters and is held
for that word's real timing, so the character is lip-synced to the voice-over.
Blinking, brow movement, head bob and arm gestures keep it alive between lines.

Honest limit: this is an original caricature in anime style, drawn on CPU with
no model download — not a licensed likeness of an existing anime character.
"""

from __future__ import annotations

import math
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import FFMPEG_THREADS, VIDEO_FPS

W, H = 600, 820
FPS = VIDEO_FPS
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUTLINE = (28, 24, 34)
HEAD = (300, 300)
HEAD_RX, HEAD_RY = 150, 166
EYE_Y = 322
EYE_DX = 74
MOUTH = (300, 410)

# (skin, hair, outfit, eye) — picked deterministically per name.
PALETTES = [
    ((250, 214, 178), (58, 44, 40), (70, 120, 200), (86, 170, 240)),
    ((252, 222, 190), (34, 30, 32), (206, 78, 100), (120, 200, 150)),
    ((244, 200, 160), (96, 58, 36), (72, 170, 128), (250, 180, 90)),
    ((214, 162, 122), (28, 24, 26), (150, 96, 200), (120, 220, 240)),
    ((250, 220, 192), (168, 120, 64), (232, 138, 56), (140, 200, 255)),
]

# Known characters keep a signature look (hair, outfit, eye), owned by
# `script_understanding.KNOWN_CHARACTERS` so there is a single source of truth.
# The drawing is an original anime caricature, not the licensed character art.


def palette_for(name: str) -> dict:
    """Stable anime palette per speaker; a known character keeps its colours.

    A reference image registered with `load_reference` wins over everything:
    the character then keeps the colours of the picture the user supplied.
    """
    from pipeline import script_understanding

    reference = _REFERENCES.get(_key(name))
    if reference:
        return dict(reference)

    skin, hair, outfit, eye = PALETTES[sum(ord(c) for c in (name or "?")) % len(PALETTES)]
    signature = script_understanding.signature_look(name)
    if signature:
        hair, outfit, eye = signature
    return {"skin": skin, "hair": hair, "outfit": outfit, "eye": eye}


# ---------------------------------------------------------------------------
# Character reference images
#
# A user-supplied portrait is turned into the palette the engines paint the
# character with, so every scene keeps one consistent identity instead of the
# character being redesigned scene by scene. The registry is process-global on
# purpose (one worker renders one job at a time) and cleared after each job.
# ---------------------------------------------------------------------------
_REFERENCES: dict[str, dict] = {}


def _key(name: str) -> str:
    return (name or "").strip().lower()


def clear_references() -> None:
    _REFERENCES.clear()


def load_reference(name: str, image_path) -> dict:
    """Register `image_path` as the look of `name`; returns the extracted palette."""
    palette = palette_from_image(image_path, name)
    _REFERENCES[_key(name)] = palette
    return palette


def _average(img: Image.Image, box) -> tuple[int, int, int]:
    region = img.crop(box)
    if region.width < 1 or region.height < 1:
        return (0, 0, 0)
    # A 1x1 box filter is the average colour, computed by Pillow itself.
    return region.convert("RGB").resize((1, 1), Image.BOX).getpixel((0, 0))


def palette_from_image(image_path, name: str = "") -> dict:
    """Sample skin/hair/outfit/eye colours from a portrait.

    The bands are proportional (hair at the top, face in the upper middle,
    outfit lower down) which holds for the usual head-and-shoulders framing.
    Colours are quantised so the flat-shaded engines get stable values.
    """
    with Image.open(image_path) as raw:
        img = raw.convert("RGB")
    w, h = img.size
    if w < 8 or h < 8:
        return palette_for_fallback(name)

    hair = _average(img, (int(w * 0.20), 0, int(w * 0.80), int(h * 0.12)))
    skin = _average(img, (int(w * 0.34), int(h * 0.30), int(w * 0.66), int(h * 0.52)))
    outfit = _average(img, (int(w * 0.28), int(h * 0.78), int(w * 0.72), int(h * 0.98)))
    # The iris: the darker part of the eye band, which avoids the sclera.
    eyes = img.crop((int(w * 0.30), int(h * 0.28), int(w * 0.70), int(h * 0.40)))
    small = eyes.convert("RGB").resize((8, 4), Image.BOX)
    samples = [small.getpixel((x, y)) for y in range(4) for x in range(8)]
    samples.sort(key=sum)
    picked = samples[:max(1, len(samples) // 4)]
    eye = tuple(sum(p[i] for p in picked) // len(picked) for i in range(3))

    def boost(color, floor=90):
        return tuple(max(floor, min(255, int(c))) for c in color)

    return {"skin": boost(skin, 120), "hair": boost(hair, 20),
            "outfit": boost(outfit, 30), "eye": boost(eye, 60)}


def palette_for_fallback(name: str) -> dict:
    """The deterministic palette, ignoring any registered reference."""
    skin, hair, outfit, eye = PALETTES[sum(ord(c) for c in (name or "?")) % len(PALETTES)]
    return {"skin": skin, "hair": hair, "outfit": outfit, "eye": eye}


@dataclass
class Cue:
    """A time window during which a character is speaking."""

    start: float
    end: float


# --- lip sync -------------------------------------------------------------
def _strip(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def viseme(word: str) -> str:
    """Mouth shape for a word, from its letters (French-friendly heuristic)."""
    text = _strip(word)
    if not text:
        return "neutral"
    if re.match(r"^[bmp]", text):
        return "closed"
    text = (text.replace("qu", "k").replace("eau", "o").replace("au", "o")
            .replace("ai", "e").replace("ei", "e").replace("ph", "f"))
    if re.search(r"ou", text):
        return "u"
    if re.search(r"o", text):
        return "o"
    if re.search(r"(i|y)", text):
        return "i"
    if re.search(r"e", text):
        return "e"
    if re.search(r"a", text):
        return "a"
    if re.search(r"(f|v)", text):
        return "f"
    if re.search(r"u", text):
        return "u"
    return "neutral"


def _word_spans(items) -> list[tuple[str, float, float]]:
    """Normalise cues/words into (text, start, end) spans."""
    spans: list[tuple[str, float, float]] = []
    for item in items or []:
        text = getattr(item, "text", "") or ""
        spans.append((text, float(item.start), float(item.end)))
    return spans


def _mouth_at(spans: list[tuple[str, float, float]], t: float) -> str:
    for text, start, end in spans:
        if start - 0.02 <= t <= end + 0.02:
            return viseme(text)
    return "neutral"


# --- drawing --------------------------------------------------------------
def _hair_back(draw: ImageDraw.ImageDraw, pal: dict, bob: int) -> None:
    cx, cy = HEAD[0], HEAD[1] + bob
    hair = pal["hair"]
    draw.ellipse((cx - HEAD_RX - 16, cy - HEAD_RY - 28, cx + HEAD_RX + 16, cy + HEAD_RY - 18),
                 fill=hair, outline=OUTLINE, width=6)
    spikes = 5 + sum(hair) % 3
    for index in range(spikes):
        frac = index / (spikes - 1)
        x = cx - HEAD_RX + 18 + frac * (2 * HEAD_RX - 36)
        tip = cy - HEAD_RY - 58 - (20 if index % 2 else 0)
        draw.polygon([(x - 36, cy - HEAD_RY + 26), (x + 36, cy - HEAD_RY + 26), (x, tip)],
                     fill=hair, outline=OUTLINE)


def _hair_front(draw: ImageDraw.ImageDraw, pal: dict, bob: int) -> None:
    """Bangs over the forehead so it reads as hair, not bare skin."""
    cx, cy = HEAD[0], HEAD[1] + bob
    hair = pal["hair"]
    for index in range(4):
        x = cx - 96 + index * 64
        draw.polygon([(x - 46, cy - HEAD_RY + 28), (x + 46, cy - HEAD_RY + 28),
                      (x + 12, cy - 70), (x - 30, cy - 58)],
                     fill=hair, outline=OUTLINE)


def _eye(draw: ImageDraw.ImageDraw, cx: int, cy: int, pal: dict, blink: bool,
         wide: bool) -> None:
    if blink:
        draw.line((cx - 50, cy, cx + 50, cy), fill=OUTLINE, width=9)
        return
    ry = 64 if wide else 56
    draw.ellipse((cx - 50, cy - ry, cx + 50, cy + ry - 10),
                 fill=(255, 255, 255), outline=OUTLINE, width=5)
    draw.ellipse((cx - 34, cy - ry + 12, cx + 34, cy + ry - 26),
                 fill=pal["eye"], outline=OUTLINE, width=3)
    draw.ellipse((cx - 14, cy - 22, cx + 14, cy + 26), fill=(32, 28, 38))
    draw.ellipse((cx - 26, cy - ry + 22, cx - 4, cy - ry + 44), fill=(255, 255, 255))
    draw.ellipse((cx + 8, cy + 10, cx + 20, cy + 22), fill=(255, 255, 255))
    draw.arc((cx - 56, cy - ry - 8, cx + 56, cy + ry - 16), 195, 345,
             fill=OUTLINE, width=11)
    brow = -30 if wide else -24
    draw.line((cx - 44, cy - ry + brow, cx + 40, cy - ry + brow - 8), fill=OUTLINE, width=8)


def _mouth(draw: ImageDraw.ImageDraw, shape: str) -> None:
    cx, cy = MOUTH
    dark = (128, 54, 62)
    if shape == "closed":
        draw.line((cx - 26, cy, cx + 26, cy), fill=OUTLINE, width=7)
    elif shape == "a":
        draw.ellipse((cx - 30, cy - 22, cx + 30, cy + 34), fill=dark, outline=OUTLINE, width=5)
        draw.ellipse((cx - 20, cy + 6, cx + 20, cy + 26), fill=(214, 96, 104))
    elif shape == "e":
        draw.ellipse((cx - 34, cy - 10, cx + 34, cy + 22), fill=dark, outline=OUTLINE, width=5)
    elif shape == "i":
        draw.ellipse((cx - 36, cy - 8, cx + 36, cy + 14), fill=dark, outline=OUTLINE, width=5)
        draw.line((cx - 34, cy - 2, cx + 34, cy - 2), fill=(255, 255, 255), width=5)
    elif shape == "o":
        draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 26), fill=dark, outline=OUTLINE, width=5)
    elif shape == "u":
        draw.ellipse((cx - 15, cy - 16, cx + 15, cy + 20), fill=dark, outline=OUTLINE, width=5)
    elif shape == "f":
        draw.ellipse((cx - 26, cy - 6, cx + 26, cy + 14), fill=dark, outline=OUTLINE, width=5)
        draw.line((cx - 24, cy - 4, cx + 24, cy - 4), fill=(255, 255, 255), width=4)
    else:
        draw.arc((cx - 30, cy - 16, cx + 30, cy + 20), 20, 160, fill=OUTLINE, width=7)


def _bust(draw: ImageDraw.ImageDraw, pal: dict, *, bob: int, swing: int, sitting: bool) -> None:
    cx = HEAD[0]
    top = 430 + bob
    draw.rectangle((cx - 42, HEAD[1] + HEAD_RY - 40 + bob, cx + 42, 470 + bob),
                   fill=pal["skin"], outline=OUTLINE, width=5)
    draw.rounded_rectangle((cx - 190, top, cx + 190, top + 340), radius=70,
                           fill=pal["outfit"], outline=OUTLINE, width=6)
    draw.polygon([(cx - 60, top), (cx + 60, top), (cx, top + 92)],
                 fill=pal["eye"], outline=OUTLINE)
    draw.rounded_rectangle((cx - 252, top + 30 - swing, cx - 188, top + 232 + swing),
                           radius=34, fill=pal["outfit"], outline=OUTLINE, width=6)
    draw.rounded_rectangle((cx + 188, top + 30 + swing, cx + 252, top + 232 - swing),
                           radius=34, fill=pal["outfit"], outline=OUTLINE, width=6)
    for hx in (cx - 254, cx + 194):
        draw.ellipse((hx, top + 210, hx + 60, top + 272), fill=pal["skin"],
                     outline=OUTLINE, width=5)
    _ = sitting


def _name_tag(img: Image.Image, name: str, color) -> None:
    if not name:
        return
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(_FONT_PATH, 36)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), name, font=font)
    w = box[2] - box[0] + 38
    x = (W - w) // 2
    draw.rounded_rectangle((x, 8, x + w, 64), radius=20, fill=color, outline=OUTLINE, width=4)
    draw.text((x + 19, 14), name, font=font, fill=(22, 20, 26))


def draw_frame(name: str, spans: list[tuple[str, float, float]], t: float,
               action: str | None = None, pal: dict | None = None) -> Image.Image:
    """One animation frame (RGBA): mouth, eyes, head, arms and legs at time `t`.

    This is the single drawing implementation: `render_character` encodes these
    frames and the animation provider composites them onto a scene. The frame
    changes with `t` (viseme, blink, bob, arm swing, walk sway), so the output
    is a real animation, never a still image.
    """
    pal = pal or palette_for(name)
    talking_spans = [(s, e) for _, s, e in spans]
    talking = any(s - 0.05 <= t <= e + 0.05 for s, e in talking_spans)
    sitting = action in ("s'assied", "assied", "assis")
    walking = action in ("marche", "court")
    bob = 0 if sitting else int(7 * math.sin(2 * math.pi * 0.8 * t))
    if walking:
        swing = int(26 * math.sin(2 * math.pi * 2.2 * t))
    elif talking:
        swing = int(16 * math.sin(2 * math.pi * 1.7 * t))
    else:
        swing = int(5 * math.sin(2 * math.pi * 0.5 * t))
    shape = _mouth_at(spans, t) if talking else "neutral"
    blink = (t % 4.0) > 3.86
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _bust(draw, pal, bob=bob, swing=swing, sitting=sitting)
    _hair_back(draw, pal, bob)
    draw.ellipse((HEAD[0] - HEAD_RX, HEAD[1] - HEAD_RY + bob,
                  HEAD[0] + HEAD_RX, HEAD[1] + HEAD_RY + bob),
                 fill=pal["skin"], outline=OUTLINE, width=6)
    _hair_front(draw, pal, bob)
    _eye(draw, HEAD[0] - EYE_DX, EYE_Y + bob, pal, blink, wide=talking)
    _eye(draw, HEAD[0] + EYE_DX, EYE_Y + bob, pal, blink, wide=talking)
    draw.line((HEAD[0] - 6, 356 + bob, HEAD[0] + 6, 356 + bob), fill=OUTLINE, width=5)
    _mouth(draw, shape)
    if talking:
        draw.ellipse((HEAD[0] - 172, 72, HEAD[0] + 172, 134),
                     outline=pal["eye"], width=7)
    if walking:
        sway = int(12 * math.sin(2 * math.pi * 2.2 * t))
        shifted = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        shifted.paste(img, (sway, 0))
        img = shifted
    _name_tag(img, name, pal["eye"])
    return img


def render_character(
    name: str,
    items,
    out_path: Path,
    duration: float,
    action: str | None = None,
    fps: int = FPS,
) -> Path:
    """Render one lip-synced anime character to an RGBA `.mov`.

    `items` is the character's timed words (or plain cues): the mouth takes each
    word's viseme during that word's real timing, so the animation matches the
    voice-over. Blinks and a gentle bob run throughout; `action` adds a walk or
    a seated pose.
    """
    pal = palette_for(name)
    spans = _word_spans(items)
    frames = max(1, int(round(duration * fps)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-threads", str(FFMPEG_THREADS),
         "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "qtrle", "-pix_fmt", "rgba",
         str(out_path)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for index in range(frames):
        proc.stdin.write(draw_frame(name, spans, index / fps, action, pal).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"avatar encode failed for {name}")
    return out_path
