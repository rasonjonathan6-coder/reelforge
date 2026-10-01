"""Orchestrator: text -> voice-over -> visuals -> captions -> vertical reel.

Usage:
    python generate.py --text "..." --out output/reel.mp4
    python generate.py --script examples/script.txt --out output/reel.mp4

Runs entirely on CPU with no API key required.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline import compose, overlay, subtitles, tts, visuals

ROOT = Path(__file__).resolve().parent
DEFAULT_FFMPEG = ROOT / ".." / "bin" / "ffmpeg"

# Real pipeline stages, in order. Both the single and batch modes report these.
STEPS = ("script", "tts", "visuals", "subtitles", "compose", "metadata")


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
    topic: str = "",
    on_step=None,
    info_out: Path | None = None,
    speech=None,
    target_duration: float | None = None,
) -> Path:
    """Render one reel.

    `on_step(stage, progress)` is called at each real stage so callers (the job
    runner, the batch runner) can report genuine progress instead of guessing.
    `info_out`, when given, receives the intermediate artifacts (voice-over,
    subtitles, background) so a caller can archive them per job.
    `speech` lets a caller pass an already-synthesized voice-over (used by the
    duration-fitting loop so the TTS is not run twice); when omitted it is
    synthesized here. `target_duration` is only recorded for diagnostics.
    `topic` seeds the stock-footage searches, together with the narration.
    """
    _ensure_ffmpeg()
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="reel_"))
    tmp.mkdir(parents=True, exist_ok=True)

    def report(stage: str, progress: int) -> None:
        if on_step:
            on_step(stage, progress)

    report("tts", 35)
    print("[1/5] Synthesizing voice-over...")
    if speech is None:
        speech = tts.synthesize(text, tmp / "voice.mp3", voice=voice, rate=rate)
    duration = max(speech.duration, 1.0)
    print(f"      -> {duration:.1f}s of speech, {len(speech.words)} words")

    report("subtitles", 50)
    print("[2/5] Building karaoke subtitles...")
    ass = subtitles.build_ass(speech.words, tmp / "captions.ass")

    report("visuals", 60)
    print("[3/5] Preparing visuals...")
    visuals_info = visuals.build_background_info(
        duration, tmp, query=query, use_stock=use_stock, clips=clips,
        topic=topic, script=text,
    )
    background = visuals_info.path
    print(f"      -> {background.name} ({visuals_info.message})")

    report("compose", 75)
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

    if info_out:
        info_out.mkdir(parents=True, exist_ok=True)
        shutil.copy(speech.audio_path, info_out / "audio.mp3")
        shutil.copy(ass, info_out / "subtitles.ass")
        (info_out / "visuals.json").write_text(
            json.dumps(
                {
                    "source": visuals_info.visual_source,
                    "sources": visuals_info.sources,
                    "queries": visuals_info.queries,
                    "clips": visuals_info.clips,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (info_out / "duration.json").write_text(
            json.dumps(
                {
                    "target_duration": target_duration,
                    "audio_duration": round(speech.duration, 2),
                    "final_video_duration": _probe_duration(out_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Done: {out_path}")
    return out_path


def _probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return round(float(out.stdout.strip()), 2)
    except Exception:  # noqa: BLE001 - duration is informational
        return None


def _batch_main(args) -> None:
    from batch import run_batch_sync

    topics = [
        line.strip()
        for line in args.batch.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not topics:
        raise SystemExit(f"Aucun sujet dans {args.batch}")

    common = {
        "duration": args.duration,
        "language": args.language,
        "style": args.style,
        "tone": args.tone,
        "voice": args.voice,
        "rate": args.rate,
        "query": args.query,
        "use_stock": not args.no_stock,
        "logo": args.logo or "",
    }
    run_batch_sync(topics, common, out_dir=args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate faceless vertical reels")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Narration text")
    group.add_argument("--script", type=Path, help="Path to a text file with narration")
    group.add_argument("--batch", type=Path, help="Path to a file with one topic per line")
    parser.add_argument("--topic", default="", help="Topic, used to steer stock searches")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "reel.mp4")
    parser.add_argument("--voice", default=tts.DEFAULT_VOICE)
    parser.add_argument("--rate", default=tts.DEFAULT_RATE)
    parser.add_argument("--query", default="city night vertical", help="Stock search terms")
    parser.add_argument("--no-stock", action="store_true", help="Skip stock, use generated visuals")
    parser.add_argument(
        "--clips",
        type=Path,
        help="Folder of your own clips (e.g. AI-generated scenes) to use as visuals",
    )
    parser.add_argument("--logo", help="Brand text shown at the top of the video")
    parser.add_argument("--keep-work", action="store_true")
    # Batch-only options.
    parser.add_argument("--language", default="français", help="Script language (batch)")
    parser.add_argument("--duration", type=int, default=30, help="Target seconds (batch)")
    parser.add_argument("--style", default="storytelling", help="Script style (batch)")
    parser.add_argument("--tone", default="dynamic", help="Script tone (batch)")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output" / "batches", help="Batch output folder"
    )
    # argparse reads a leading-dash value like "-5%" as an option; join it back.
    argv = sys.argv[1:]
    for flag in ("--rate",):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv) and re.match(r"^-\d", argv[i + 1]):
                argv[i] = f"{flag}={argv[i + 1]}"
                del argv[i + 1]
    args = parser.parse_args(argv)

    if args.batch:
        _batch_main(args)
        return

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
        topic=args.topic or "",
    )


if __name__ == "__main__":
    main()
