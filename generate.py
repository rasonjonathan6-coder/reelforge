"""Orchestrator: text -> voice-over -> visuals -> captions -> vertical reel.

Usage:
    python generate.py --text "..." --out output/reel.mp4
    python generate.py --script examples/script.txt --out output/reel.mp4

Runs entirely on CPU with no API key required.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from pipeline import compose, subtitles, tts, visuals

ROOT = Path(__file__).resolve().parent
DEFAULT_FFMPEG = ROOT / ".." / "bin" / "ffmpeg"


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    candidates = [Path("/workspace/bin/ffmpeg"), DEFAULT_FFMPEG.resolve()]
    for candidate in candidates:
        if candidate.exists():
            import os

            os.environ["PATH"] = f"{candidate.parent}:{os.environ['PATH']}"
            return
    raise RuntimeError("ffmpeg not found. Install it or run with /workspace/bin on PATH.")


def generate(
    text: str,
    out_path: Path,
    voice: str = tts.DEFAULT_VOICE,
    rate: str = "+8%",
    query: str = "city night vertical",
    use_stock: bool = True,
    work_dir: Path | None = None,
) -> Path:
    _ensure_ffmpeg()
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="reel_"))
    tmp.mkdir(parents=True, exist_ok=True)

    print("[1/4] Synthesizing voice-over...")
    speech = tts.synthesize(text, tmp / "voice.mp3", voice=voice, rate=rate)
    duration = max(speech.duration, 1.0)
    print(f"      -> {duration:.1f}s of speech, {len(speech.words)} words")

    print("[2/4] Building karaoke subtitles...")
    ass = subtitles.build_ass(speech.words, tmp / "captions.ass")

    print("[3/4] Preparing visuals...")
    background = visuals.build_background(duration, tmp, query=query, use_stock=use_stock)
    print(f"      -> {background.name}")

    print("[4/4] Composing final video...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compose.compose(background, speech.audio_path, ass, out_path, duration)
    print(f"Done: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a faceless vertical reel")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Narration text")
    group.add_argument("--script", type=Path, help="Path to a text file with narration")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "reel.mp4")
    parser.add_argument("--voice", default=tts.DEFAULT_VOICE)
    parser.add_argument("--rate", default="+8%")
    parser.add_argument("--query", default="city night vertical", help="Stock search terms")
    parser.add_argument("--no-stock", action="store_true", help="Skip Pexels, use generated visuals")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    text = args.text if args.text else args.script.read_text(encoding="utf-8").strip()
    work = None if args.keep_work else None
    generate(
        text=text,
        out_path=args.out,
        voice=args.voice,
        rate=args.rate,
        query=args.query,
        use_stock=not args.no_stock,
        work_dir=work,
    )


if __name__ == "__main__":
    main()
