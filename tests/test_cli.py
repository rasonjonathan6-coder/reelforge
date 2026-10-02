"""Tests for the CLI parity with the API/Web dialogue, cast and music options.

Parsing and propagation are checked offline (no TTS, no ffmpeg): the casting
logic itself is exercised in `test_dialogue_music.py`, and here we only assert
that the CLI hands the right values to the shared pipeline.
"""

from __future__ import annotations

import generate
from pipeline import script_writer, tts


def _parse(argv: list[str]):
    parser = generate.build_parser()
    return generate._parse_args(parser, argv)


# --- parsing of the new options -------------------------------------------

def test_dialogue_defaults_off():
    args = _parse(["--text", "Bonjour tout le monde, ceci est un texte assez long."])
    assert args.dialogue is False


def test_dialogue_flag_parsed():
    args = _parse(["--dialogue", "--topic", "le café et le sommeil"])
    assert args.dialogue is True


def test_cast_defaults_to_mixte():
    args = _parse(["--text", "Bonjour tout le monde, ceci est un texte assez long."])
    assert args.cast == "mixte"
    assert generate._cast_from_args(args) == "mixte"


def test_cast_accepts_femme_and_homme():
    for label, expected in (("femme", "female"), ("homme", "male"), ("mixte", "mixte")):
        args = _parse(["--text", "Un texte assez long pour passer la validation.", "--cast", label])
        assert generate._cast_from_args(args) == expected


def test_music_defaults_on():
    args = _parse(["--text", "Un texte assez long pour passer la validation."])
    assert args.music is True


def test_no_music_disables_bed():
    args = _parse(["--text", "Un texte assez long pour passer la validation.", "--no-music"])
    assert args.music is False


def test_music_flag_re_enables_bed():
    args = _parse(["--text", "Un texte assez long pour passer la validation.",
                   "--no-music", "--music"])
    assert args.music is True


def test_music_mood_parsed():
    args = _parse(["--text", "Un texte assez long pour passer la validation.",
                   "--music-mood", "epique"])
    assert args.music_mood == "epique"


def test_dialogue_can_run_from_topic_without_text():
    # No --text/--script: allowed as long as a topic is given.
    args = _parse(["--dialogue", "--topic", "deux amis parlent du café"])
    assert args.text is None
    assert args.topic == "deux amis parlent du café"


# --- propagation into the shared casting ----------------------------------

def _turns():
    script = "Léo: Le café ne réveille pas.\nMaya: Ah bon ?\nSam: C'est l'habitude."
    return script_writer.parse_dialogue(script)


def test_monologue_keeps_one_voice():
    turns = script_writer.parse_dialogue("Léo: Une seule personne parle vraiment ici.")
    lines = generate.cast_dialogue(turns, tts.DEFAULT_VOICE, "mixte")
    assert len(lines) == 1
    assert lines[0].voice == tts.DEFAULT_VOICE


def test_two_characters_get_two_distinct_voices():
    turns = script_writer.parse_dialogue("Léo: Bonjour tout le monde.\nMaya: Salut à toi.")
    lines = generate.cast_dialogue(turns, tts.DEFAULT_VOICE, "mixte")
    voices = [line.voice for line in lines]
    assert len(set(voices)) == 2


def test_three_characters_get_three_voices():
    lines = generate.cast_dialogue(_turns(), tts.DEFAULT_VOICE, "mixte")
    assert len({line.voice for line in lines}) == 3


def test_cast_does_not_change_character_count():
    turns = _turns()
    for cast in ("mixte", "femme", "homme"):
        lines = generate.cast_dialogue(turns, tts.DEFAULT_VOICE, cast)
        assert len({line.speaker for line in lines}) == 3


def test_cast_femme_is_all_female():
    lines = generate.cast_dialogue(_turns(), tts.DEFAULT_VOICE, "femme")
    assert {tts.gender_of(line.voice) for line in lines} == {"female"}


def test_cast_homme_is_all_male():
    lines = generate.cast_dialogue(_turns(), tts.DEFAULT_VOICE, "homme")
    assert {tts.gender_of(line.voice) for line in lines} == {"male"}


def test_cast_mixte_uses_both_genders():
    lines = generate.cast_dialogue(_turns(), tts.DEFAULT_VOICE, "mixte")
    assert {tts.gender_of(line.voice) for line in lines} == {"female", "male"}


def test_batch_common_carries_dialogue_cast_and_music():
    # The batch path must forward the same options as the single path.
    args = _parse(["--batch", "topics.txt", "--dialogue", "--cast", "femme",
                   "--no-music", "--music-mood", "calme"])
    common = {
        "dialogue": bool(args.dialogue),
        "dialogue_cast": generate._cast_from_args(args),
        "music": args.music,
        "music_mood": args.music_mood or "",
    }
    assert common == {"dialogue": True, "dialogue_cast": "female",
                      "music": False, "music_mood": "calme"}
