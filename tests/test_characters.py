"""Tests for script understanding, animated characters and their overlay.

Everything runs offline with real Pillow/ffmpeg: no network, no model download.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from pipeline import avatars, compose, script_understanding


class _Word:
    """Minimal timed word, same shape as `tts.WordTiming`."""

    def __init__(self, text: str, start: float, end: float, speaker: str = "Léo"):
        self.text, self.start, self.end, self.speaker = text, start, end, speaker


def _probe(path: Path, entries: str) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", entries,
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip().splitlines()[0]


def _frame(path: Path, seconds: float, out: Path) -> Image.Image:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(seconds),
         "-i", str(path), "-frames:v", "1", str(out)],
        check=True, capture_output=True, timeout=60,
    )
    return Image.open(out).convert("RGB")


# --- script understanding -------------------------------------------------
def test_place_is_detected_from_the_script():
    assert script_understanding.detect_place("Il ouvre le frigo de la cuisine.") == "cuisine"
    assert script_understanding.detect_place("Elle dort dans son lit.") == "chambre"
    assert script_understanding.detect_place("Rien de special ici.") == ""


def test_action_is_detected_from_the_script():
    assert script_understanding.detect_action("Goku marche vers la porte.") == "marche"
    assert script_understanding.detect_action("Elle s'assied sur le canape.") == "s'assied"
    assert script_understanding.detect_action("Il mange une pomme.") == "mange"


def test_characters_and_signature_look():
    script = "Goku: J'ai faim.\nVegeta: Tais-toi.\nGoku: Non."
    assert script_understanding.detect_characters(script) == ["Goku", "Vegeta"]
    assert script_understanding.signature_look("Goku") is not None
    assert script_understanding.signature_look("Inconnu") is None


def test_known_character_keeps_its_signature_colours():
    goku = avatars.palette_for("Goku")
    assert goku["outfit"] == (235, 130, 40)  # orange gi
    assert avatars.palette_for("Goku") == goku  # stable across calls
    plain = avatars.palette_for("Zoé")
    assert plain["outfit"] != goku["outfit"] or plain["hair"] != goku["hair"]


# --- animated characters --------------------------------------------------
def test_viseme_follows_french_letters():
    assert avatars.viseme("bonjour") == "closed"   # starts with b
    assert avatars.viseme("oui") == "u"
    assert avatars.viseme("eau") == "o"
    assert avatars.viseme("travail") == "e"        # ai -> e
    assert avatars.viseme("maison") == "closed"    # m -> bilabial, lips shut
    assert avatars.viseme("vite") == "i"
    assert avatars.viseme("chat") == "a"
    assert avatars.viseme("") == "neutral"


def test_lip_sync_follows_the_spoken_words(tmp_path):
    """The mouth shape must come from the word being said at that instant."""
    words = [
        _Word("oui", 0.0, 0.4),    # u
        _Word("chat", 0.4, 0.8),   # a
        _Word("vite", 0.8, 1.2),   # i
    ]
    out = avatars.render_character("Léo", words, tmp_path / "lips.mov", 1.5)
    assert out.exists() and out.stat().st_size > 0
    spans = avatars._word_spans(words)
    assert avatars._mouth_at(spans, 0.2) == "u"
    assert avatars._mouth_at(spans, 0.6) == "a"
    assert avatars._mouth_at(spans, 1.0) == "i"
    assert avatars._mouth_at(spans, 1.4) == "neutral"  # silence between lines


def test_render_character_is_rgba_with_alpha(tmp_path):
    out = avatars.render_character("Goku", [avatars.Cue(0.0, 2.0)], tmp_path / "c.mov", 3.0)
    assert out.exists()
    assert _probe(out, "stream=codec_name") == "qtrle"
    assert _probe(out, "stream=pix_fmt") in ("rgba", "argb")  # alpha is kept
    assert abs(float(_probe(out, "stream=duration")) - 3.0) < 0.1


def test_character_moves_and_talks(tmp_path):
    # Two clips: one always talking, one always silent. Their frames must differ
    # (mouth/arms animate), proving the character is not a frozen sticker.
    talking = avatars.render_character("Léo", [avatars.Cue(0.0, 3.0)], tmp_path / "t.mov", 3.0)
    silent = avatars.render_character("Léo", [], tmp_path / "s.mov", 3.0)
    a = _frame(talking, 0.5, tmp_path / "a.png")
    b = _frame(talking, 1.0, tmp_path / "b.png")
    c = _frame(silent, 0.5, tmp_path / "c.png")
    assert a.tobytes() != b.tobytes(), "talking character is frozen"
    assert a.tobytes() != c.tobytes(), "talking vs idle look identical"


def test_walking_action_bends_the_legs(tmp_path):
    walk = avatars.render_character("Léo", [], tmp_path / "w.mov", 3.0, action="marche")
    still = avatars.render_character("Léo", [], tmp_path / "i.mov", 3.0)
    frames = [(_frame(walk, t, tmp_path / f"w{t}.png").tobytes()) for t in (0.1, 0.4, 0.7)]
    assert len(set(frames)) == len(frames), "walking legs never move"
    assert _frame(walk, 0.4, tmp_path / "w2.png").tobytes() != \
        _frame(still, 0.4, tmp_path / "i2.png").tobytes()


# --- staging and overlay --------------------------------------------------
def test_stage_positions_stay_inside_the_frame():
    for count in (1, 2, 3, 5):
        positions = compose.stage_positions(count)
        assert len(positions) == count
        for x, y in positions:
            assert 0 <= x < compose.WIDTH and 0 <= y < compose.HEIGHT


def test_compose_overlays_characters_on_the_background(tmp_path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=c=navy:size=1080x1920:rate=30:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "bg.mp4")],
        check=True, capture_output=True, timeout=120,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=200:duration=3", str(tmp_path / "a.m4a")],
        check=True, capture_output=True, timeout=120,
    )
    ass = tmp_path / "c.ass"
    ass.write_text(
        "[Script Info]\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,"
        "PrimaryColour,OutlineColour,BorderStyle,Outline,Shadow,Alignment,"
        "MarginL,MarginR,MarginV\n"
        "Style: Caption,DejaVu Sans,90,&H00FFFFFF,&H000000FF,1,3,1,2,80,80,420\n"
        "[Events]\nFormat: Layer,Start,End,Style,Text\n"
        "Dialogue: 0,0:00:00.00,0:00:03.00,Caption,Salut\n",
        encoding="utf-8",
    )
    char = avatars.render_character("Goku", [avatars.Cue(0.0, 3.0)], tmp_path / "g.mov", 3.0)
    plain = compose.compose(tmp_path / "bg.mp4", tmp_path / "a.m4a", ass,
                            tmp_path / "plain.mp4", 3.0)
    withchar = compose.compose(tmp_path / "bg.mp4", tmp_path / "a.m4a", ass,
                               tmp_path / "with.mp4", 3.0, characters=[char])
    assert _probe(withchar, "stream=width") == "1080"
    assert _probe(withchar, "stream=height") == "1920"
    assert abs(float(_probe(withchar, "stream=duration")) - 3.0) < 0.15
    # A pixel where the character stands differs once it is overlaid.
    x, y = compose.stage_positions(1)[0]
    px = (x + 130, y + 130)
    assert _frame(withchar, 0.5, tmp_path / "f1.png").getpixel(px) != \
        _frame(plain, 0.5, tmp_path / "f2.png").getpixel(px)
