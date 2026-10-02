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

from pipeline import compose, music, overlay, script_understanding, script_writer, subtitles, tts, visuals
from pipeline import animation

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


def _turn_scenes(speech) -> tuple[list[str] | None, list[float] | None]:
    """One visual scene per dialogue turn, sized by that turn's word count.

    The scene follows the script line by line: each turn's own words seed its
    scene's search and its word count sets the scene's on-screen time, so the
    footage changes when a character starts speaking. A new scene starts on each
    speaker change, which covers any number of characters. Returns
    `(None, None)` for a single-voice narration, keeping uniform sentence scenes.
    """
    words = [w for w in (getattr(speech, "words", None) or []) if getattr(w, "speaker", "")]
    if not words:
        return None, None
    texts: list[str] = []
    weights: list[float] = []
    current = ""
    for word in words:
        if word.speaker != current:
            current = word.speaker
            texts.append("")
            weights.append(0.0)
        weights[-1] += 1.0
        texts[-1] = (texts[-1] + " " + word.text).strip()
    if len(texts) < 2:
        return None, None
    return texts, weights


def _turn_places(texts: list[str]) -> list[str]:
    """Setting detected per turn, so the decor follows the script line by line."""
    return [script_understanding.detect_place(t) for t in texts]


def _turn_pans(texts: list[str]) -> list[bool]:
    """True for turns whose character walks, so the camera travels with them."""
    return [script_understanding.detect_action(t) in ("marche", "court") for t in texts]


def _render_characters(speech, tmp: Path, duration: float, understanding: dict) -> list[Path]:
    """Render one animated character per speaker, or [] for a narration.

    Each character's speaking cues come from its own timed words, so it mouths
    the lines the script gives it and idles (or performs the script's action)
    the rest of the time. Rendered as RGBA clips and overlaid by `compose`.
    """
    from pipeline import avatars

    words = [w for w in (getattr(speech, "words", None) or []) if getattr(w, "speaker", "")]
    if not words:
        return []
    by_speaker: dict[str, list] = {}
    spoken: dict[str, str] = {}
    for word in words:
        by_speaker.setdefault(word.speaker, []).append(word)
        spoken[word.speaker] = (spoken.get(word.speaker, "") + " " + word.text).strip()
    fallback = understanding.get("action") or None
    out_dir = tmp / "characters"
    paths: list[Path] = []
    actions: list[str] = []
    for index, (name, said) in enumerate(by_speaker.items()):
        # Each character performs the action from its own lines, not a global one.
        action = script_understanding.detect_action(spoken.get(name, "")) or fallback
        actions.append(action or "-")
        # The words themselves drive the visemes, so the mouth matches the voice.
        paths.append(avatars.render_character(
            name, said, out_dir / f"char_{index}.mov", duration, action=action,
        ))
    print(f"      -> {len(paths)} personnage(s) animé(s) [actions: {', '.join(actions)}]")
    return paths


def _dialogue_lines(text: str, speech) -> list[tuple[str, str]]:
    """Recover the ordered `(speaker, line)` pairs the scenes must animate."""
    turns = script_writer.parse_dialogue(text) if text and ":" in text else []
    if turns:
        return [(getattr(t, "speaker", ""), getattr(t, "text", "")) for t in turns]
    # Fall back to grouping the timed words by speaker, in spoken order.
    lines: list[tuple[str, str]] = []
    for word in getattr(speech, "words", None) or []:
        speaker = getattr(word, "speaker", "")
        if not speaker:
            continue
        if lines and lines[-1][0] == speaker:
            lines[-1] = (speaker, f"{lines[-1][1]} {word.text}".strip())
        else:
            lines.append((speaker, word.text))
    return lines


def _generate_animated(speech, text, out_path, tmp, info_out, duration, *,
                       character_style="anime", animation_provider=None,
                       music_enabled=True, music_mood=None, logo_text=None,
                       topic="", speakers=None, on_step=None) -> Path:
    """Real animated-character reel: every scene is drawn frame by frame.

    Captions, the progress bar/logo, the voice-over and the music bed are all
    applied in a single FFmpeg pass inside the engine, so a 1080x1920 reel is
    not re-encoded four times (which is what made long renders look stuck).
    """
    lines = _dialogue_lines(text, speech)
    if not lines:
        lines = [("Narrateur", text.strip()[:160] or "Reel")]
    voices = {s: (v or "") for s, v in zip(speakers or [], speakers or [])}
    ass = tmp / "captions.ass"
    if not ass.exists():
        ass = subtitles.build_ass(speech.words, ass, speakers=speakers)

    # Generate the bed up front so it can be mixed in the same pass as the video.
    music_bed = None
    music_meta = {}
    if music_enabled:
        try:
            chosen = music_mood if music_mood in music.MOODS else music.detect_mood(topic, text)
            music_bed = music.generate(duration, tmp / "music.mp3",
                                       mood=chosen, topic=topic, script=text)
            music_meta = {"mood": chosen, "label": music.MOODS[chosen]["label"]}
        except Exception as exc:  # noqa: BLE001 - music is optional
            print(f"      -> musique ignorée ({exc})")
            music_bed = None

    result = animation.generate_animated_reel(
        speech, lines, out_path, tmp / "animation",
        target_duration=duration, voice_map=voices,
        environment=script_understanding.detect_place(text),
        visual_style=character_style, provider_name=animation_provider,
        captions=ass, audio_path=speech.audio_path, music_bed=music_bed,
        logo_text=logo_text, on_step=on_step,
    )
    print(f"      -> {len(result.scenes)} scènes animées, provider « {result.provider} »")
    if music_meta:
        print(f"      -> musique « {music_meta['label']} » (voix duckée)")

    if info_out:
        info_out.mkdir(parents=True, exist_ok=True)
        animation.write_animation_info(result, info_out)
        shutil.copy(speech.audio_path, info_out / "audio.mp3")
        shutil.copy(ass, info_out / "subtitles.ass")
        if music_meta:
            (info_out / "music.json").write_text(
                json.dumps(music_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


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
    speakers: list[str] | None = None,
    music_enabled: bool = True,
    music_mood: str | None = None,
    animated_characters: bool = False,
    character_style: str = "anime",
    animation_provider: str | None = None,
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
    `speakers`, when the voice-over is a two-character dialogue, colours each
    character's captions differently. `music_enabled` adds a generated,
    ducked music bed whose mood is derived from `topic`/`text` unless
    `music_mood` names one explicitly.
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
    ass = subtitles.build_ass(speech.words, tmp / "captions.ass", speakers=speakers)

    report("visuals", 60)
    print("[3/5] Preparing visuals...")
    if animated_characters:
        print("      -> mode personnages animés (vraie animation, pas de Ken Burns)")
        return _generate_animated(
            speech, text, out_path, tmp, info_out, duration,
            character_style=character_style, animation_provider=animation_provider,
            music_enabled=music_enabled, music_mood=music_mood,
            logo_text=logo_text, topic=topic, speakers=speakers, on_step=report,
        )
    scene_texts, scene_weights = _turn_scenes(speech)
    # The script drives the setting and the camera: a kitchen turn searches for a
    # kitchen, a walking turn pans, so the decor follows the script line by line.
    understanding = script_understanding.describe(text)
    place = understanding["place"]
    scene_places = _turn_places(scene_texts) if scene_texts else None
    scene_pans = _turn_pans(scene_texts) if scene_texts else None
    # Fall back to the script-level place when no turn names one.
    if scene_places and place and not any(scene_places):
        scene_places = [place] * len(scene_places)
    visuals_info = visuals.build_background_info(
        duration, tmp, query=query, use_stock=use_stock, clips=clips,
        topic=topic, script=text,
        scene_texts=scene_texts, scene_weights=scene_weights,
        scene_places=scene_places, scene_pans=scene_pans,
    )
    background = visuals_info.path
    print(f"      -> {background.name} ({visuals_info.message})"
          + (f" [lieu: {place}]" if place else ""))

    characters = _render_characters(speech, tmp, duration, understanding)

    report("compose", 75)
    print("[4/5] Composing final video...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compose.compose(background, speech.audio_path, ass, out_path, duration,
                    characters=characters)

    if logo_text or overlay.available():
        print("[5/5] Adding progress bar and branding...")
        try:
            overlaid = tmp / "overlay.mp4"
            overlay.add_overlay(out_path, overlaid, duration, logo_text=logo_text)
            shutil.move(str(overlaid), str(out_path))
        except Exception as exc:  # noqa: BLE001 - overlay is cosmetic, never fatal
            print(f"      -> overlay skipped ({exc})")

    # Music last: the video is already cut to length, so the bed matches exactly.
    # A failure here is cosmetic and never fails the render.
    music_meta: dict = {}
    if music_enabled:
        try:
            music_meta = music.build(
                out_path, tmp, duration,
                mood=music_mood, topic=topic, script=text,
            )
            shutil.move(music_meta["path"], str(out_path))
            print(f"      -> musique « {music_meta['label']} » (voix duckée)")
        except Exception as exc:  # noqa: BLE001 - music is optional
            print(f"      -> musique ignorée ({exc})")
            music_meta = {}

    if info_out:
        info_out.mkdir(parents=True, exist_ok=True)
        shutil.copy(speech.audio_path, info_out / "audio.mp3")
        shutil.copy(ass, info_out / "subtitles.ass")
        if music_meta:
            (info_out / "music.json").write_text(
                json.dumps({k: v for k, v in music_meta.items() if k != "path"},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (info_out / "visuals.json").write_text(
            json.dumps(
                {
                    "source": visuals_info.visual_source,
                    "sources": visuals_info.sources,
                    "queries": visuals_info.queries,
                    "clips": visuals_info.clips,
                    "scene_origins": visuals_info.scene_origins,
                    "cache": visuals_info.cache_stats,
                    "scene_weights": visuals_info.scene_weights,
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


def _dialogue_characters(turns: list) -> list[str]:
    """Character names in first-appearance order (all of them)."""
    return script_writer.characters_of(turns)


def cast_dialogue(turns: list, base_voice: str, cast: str | None = None) -> list:
    """Give each character a voice according to the requested `cast`.

    A one-character script keeps that single voice. With two or more, `cast`
    picks the distribution: `mixte` (man + woman, the default), `femme`
    (femme+femme) or `homme` (homme+homme). This is the single casting entry
    point shared by the CLI and `jobs.produce`, so behaviour never diverges.
    """
    characters = _dialogue_characters(turns)
    if not characters:
        return []
    voices = tts.resolve_cast(base_voice, len(characters), cast)
    voice_by_name = dict(zip(characters, voices))
    return [
        tts.DialogueLine(turn.speaker, turn.text, voice_by_name[turn.speaker])
        for turn in turns
    ]


def _cast_from_args(args) -> str | None:
    """Normalise the CLI `--cast` choice to what the pipeline expects."""
    return tts.cast_genders(getattr(args, "cast", None))


def _dialogue_speech(text: str, args, work: Path):
    """Build the dialogue voice-over for the CLI.

    A user-supplied `Nom: réplique` script is authoritative; otherwise the
    topic is turned into a dialogue sized to `--duration`. Returns
    `(speech, script)` so the caller keeps the exact text that was voiced.
    """
    turns = script_writer.parse_dialogue(text) if text.strip() else []
    script = text
    if not turns:
        turns, script = script_writer.write_dialogue_script(
            args.topic or "",
            duration=args.duration,
            language=args.language,
            rate=args.rate,
        )
    lines = cast_dialogue(turns, args.voice, _cast_from_args(args))
    speech = tts.synthesize_dialogue(
        lines, work / "voice.mp3", rate=args.rate
    )
    return speech, script


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
        "dialogue": bool(args.dialogue),
        "dialogue_cast": _cast_from_args(args),
        "music": args.music,
        "music_mood": args.music_mood or "",
    }
    run_batch_sync(topics, common, out_dir=args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate faceless vertical reels")
    group = parser.add_mutually_exclusive_group(required=False)
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
    # Animated-character mode (same behaviour as the API's `animated_characters`).
    parser.add_argument(
        "--animated-characters", action="store_true",
        help="Render real animated cartoon characters (frame-by-frame lip-sync, "
             "walk/head/arms) instead of the stock Ken-Burns slideshow.",
    )
    parser.add_argument(
        "--character-style", default="anime",
        choices=["anime", "cartoon_2d", "cartoon_3d", "stylized"],
        help="Animated-character art style (default: anime)",
    )
    parser.add_argument(
        "--animation-provider", default="",
        choices=["", "local", "remote"],
        help="Animation backend; 'local' is the free CPU engine (default)",
    )
    # Dialogue / monologue (same behaviour as the API's `dialogue` field).
    parser.add_argument(
        "--dialogue", action="store_true",
        help="Treat the text as a dialogue (`Nom: réplique` per line); a topic "
             "auto-generates one. The voice count follows the characters.",
    )
    parser.add_argument(
        "--cast", choices=["mixte", "femme", "homme"], default="mixte",
        help="Dialogue voice distribution: mixte = homme + femme (default), "
             "femme = femme + femme, homme = homme + homme",
    )
    # Music (on by default, matching the API); --no-music turns it off.
    parser.add_argument(
        "--music", dest="music", action="store_true", default=True,
        help="Add a generated, ducked background music bed (default: on)",
    )
    parser.add_argument(
        "--no-music", dest="music", action="store_false",
        help="Disable background music",
    )
    parser.add_argument(
        "--music-mood", default="",
        choices=["", *music.MOODS.keys()],
        help="Force a music mood instead of deriving it from the script",
    )
    # Batch-only options.
    parser.add_argument("--language", default="français", help="Script language (batch)")
    parser.add_argument("--duration", type=int, default=30, help="Target seconds (batch)")
    parser.add_argument("--style", default="storytelling", help="Script style (batch)")
    parser.add_argument("--tone", default="dynamic", help="Script tone (batch)")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output" / "batches", help="Batch output folder"
    )
    return parser


def _parse_args(parser: argparse.ArgumentParser, argv: list[str]):
    # argparse reads a leading-dash value like "-5%" as an option; join it back.
    for flag in ("--rate",):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv) and re.match(r"^-\d", argv[i + 1]):
                argv[i] = f"{flag}={argv[i + 1]}"
                del argv[i + 1]
    return parser.parse_args(argv)


def main() -> None:
    parser = build_parser()
    args = _parse_args(parser, sys.argv[1:])

    if args.batch:
        _batch_main(args)
        return

    # A single reel needs some input: a text/script, or a topic (dialogue can
    # auto-generate from a topic). Replaces argparse's required=True so that
    # `--dialogue --topic "..."` works without a text file.
    if not args.text and not args.script and not args.topic:
        parser.error("fournis --text, --script, --batch ou --topic")

    text = args.text if args.text else (args.script.read_text(encoding="utf-8").strip() if args.script else "")
    clips = None
    if args.clips:
        clips = sorted(
            p for p in args.clips.iterdir()
            if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
        )
        if not clips:
            raise SystemExit(f"Aucun clip vidéo trouvé dans {args.clips}")

    work = None if args.keep_work else None
    speakers = None
    speech = None
    if args.dialogue:
        # Dialogue needs the voice-over before `generate` so the per-character
        # captions and the cast are known up front.
        dialogue_work = args.out.parent / f".{args.out.stem}_work"
        dialogue_work.mkdir(parents=True, exist_ok=True)
        speech, text = _dialogue_speech(text, args, dialogue_work)
        speakers = list(getattr(speech, "speakers", None) or [])
        print(f"[dialogue] {len(speakers)} personnage(s), cast {_cast_from_args(args)} "
              f"-> {speech.duration:.2f}s")

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
        speech=speech,
        speakers=speakers,
        music_enabled=args.music,
        music_mood=args.music_mood or None,
        animated_characters=args.animated_characters,
        character_style=args.character_style,
        animation_provider=args.animation_provider or None,
    )


if __name__ == "__main__":
    main()
