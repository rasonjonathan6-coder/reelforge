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

from pipeline import compose, overlay, subtitles, tts, visuals

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
    rate: str = tts.DEFAULT_RATE,
    query: str = "city night vertical",
    use_stock: bool = True,
    work_dir: Path | None = None,
    clips: list[Path] | None = None,
    logo_text: str | None = None,
) -> Path:
    _ensure_ffmpeg()
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="reel_"))
    tmp.mkdir(parents=True, exist_ok=True)

    print("[1/5] Synthesizing voice-over...")
    speech = tts.synthesize(text, tmp / "voice.mp3", voice=voice, rate=rate)
    duration = max(speech.duration, 1.0)
    print(f"      -> {duration:.1f}s of speech, {len(speech.words)} words")

    print("[2/5] Building karaoke subtitles...")
    ass = subtitles.build_ass(speech.words, tmp / "captions.ass")

    print("[3/5] Preparing visuals...")
    background = visuals.build_background(
        duration, tmp, query=query, use_stock=use_stock, clips=clips
    )
    print(f"      -> {background.name}")

    print("[4/5] Composing final video...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compose.compose(background, speech.audio_path, ass, out_path, duration)

    if logo_text or overlay.available():
        print("[5/5] Adding progress bar and branding...")
        try:
            overlaid = tmp / "overlay.mp4"
            overlay.add_overlay(out_path, overlaid, duration, logo_text=logo_text)
            shutil.move(str(overlaid), str(out_path))
        except Exception as exc:  # noqa: BLE001 - overlay is cosmetic, never fatal
            print(f"      -> overlay skipped ({exc})")

    print(f"Done: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a faceless vertical reel")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Narration text")
    group.add_argument("--script", type=Path, help="Path to a text file with narration")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "reel.mp4")
    parser.add_argument("--voice", default=tts.DEFAULT_VOICE)
    parser.add_argument("--rate", default=tts.DEFAULT_RATE)
    parser.add_argument("--query", default="city night vertical", help="Stock search terms")
    parser.add_argument("--no-stock", action="store_true", help="Skip Pexels, use generated visuals")
    parser.add_argument(
        "--clips",
        type=Path,
        help="Folder of your own clips (e.g. AI-generated scenes) to use as visuals",
    )
    parser.add_argument("--logo", help="Brand text shown at the top of the video")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    text = args.text if args.text else args.script.read_text(encoding="utf-8").strip()
    clips = None
    if args.clips:
        clips = sorted(
            p for p in args.clips.iterdir()
            if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
        )
        if not clips:
            raise SystemExit(f"Aucun clip vidéo trouvé dans {args.clips}")

    work = None if args.keep_work else None
    generate(
        text=text,
        out_path=args.out,
        voice=args.voice,
        rate=args.rate,
        query=args.query,
        use_stock=not args.no_stock,
        work_dir=work,
        clips=clips,
        logo_text=args.logo,
    )


if __name__ == "__main__":
    main()
