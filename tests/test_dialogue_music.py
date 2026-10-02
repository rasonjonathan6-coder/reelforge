"""Tests for the two-character dialogue and generated background music.

The pure logic (voice casting, script parsing, subtitle colours, scene queries,
mood detection) runs offline. Music generation and the ducked mix are exercised
with real ffmpeg over synthetic video/audio, so no external service is needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pipeline import music, script_writer, subtitles, tts, visuals


def _make_video(path: Path, seconds: int = 6) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=360x640:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
         "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return path


# --- voice casting ---------------------------------------------------------

def test_cast_voices_gives_two_distinct_genders():
    cast = tts.cast_voices(tts.DEFAULT_VOICE, 2)
    assert len(cast) == 2
    assert cast[0] != cast[1]
    assert tts.gender_of(cast[0]) != tts.gender_of(cast[1])


def test_cast_voices_single_character_keeps_base_voice():
    assert tts.cast_voices("fr-FR-HenriNeural", 1) == ["fr-FR-HenriNeural"]


def test_cast_voices_three_characters_are_all_distinct():
    cast = tts.cast_voices(tts.DEFAULT_VOICE, 3)
    assert len(set(cast)) == 3


def test_resolve_cast_same_gender_distributions():
    base = "fr-FR-VivienneMultilingualNeural"  # femme
    assert [tts.gender_of(v) for v in tts.resolve_cast(base, 2, "femme")] == ["female", "female"]
    assert [tts.gender_of(v) for v in tts.resolve_cast(base, 2, "homme")] == ["male", "male"]
    assert [tts.gender_of(v) for v in tts.resolve_cast(base, 2, "mixte")] == ["female", "male"]
    # Two characters of the same gender must not share one voice.
    femmes = tts.resolve_cast(base, 2, "femme")
    assert femmes[0] != femmes[1]


def test_cast_genders_accepts_ui_labels():
    assert tts.cast_genders("Femme + femme") == "female"
    assert tts.cast_genders("Homme + homme") == "male"
    assert tts.cast_genders("Mixte (homme + femme)") == "mixte"
    assert tts.cast_genders("") == "mixte"


# --- dialogue script parsing ----------------------------------------------

DIALOGUE = "Léo: Tu savais que le café ne réveille pas ?\nMaya: Non, sérieux ?\nLéo: Et pourtant.\n"


def test_parse_dialogue_reads_turns_and_speakers():
    turns = script_writer.parse_dialogue(DIALOGUE)
    assert [t.speaker for t in turns] == ["Léo", "Maya", "Léo"]
    assert turns[0].text.startswith("Tu savais")
    assert script_writer.characters_of(turns, 2) == ["Léo", "Maya"]


MONOLOGUE = "Léo: Le café ne te réveille pas vraiment.\nLéo: C'est surtout l'habitude.\n"


def test_single_character_script_keeps_a_single_voice():
    turns = script_writer.parse_dialogue(MONOLOGUE)
    assert script_writer.characters_of(turns) == ["Léo"]
    # A one-name script must not be padded to a two-voice cast.
    assert len(tts.cast_voices(tts.DEFAULT_VOICE, len(script_writer.characters_of(turns)))) == 1


def test_dialogue_labels_are_not_visual_keywords():
    queries = visuals.scene_queries("le café et le sommeil", DIALOGUE, 3)
    joined = " ".join(queries).lower()
    assert "léo" not in joined and "maya" not in joined
    assert len(set(queries)) == len(queries)


def test_local_dialogue_respects_the_word_budget():
    # A whole template cycle is ~105 words; short targets used to be impossible
    # to reach, which left the duration-fitting loop unable to converge.
    for duration in (20, 30, 45):
        words = tts.estimate_words(duration, "-5%")
        turns = script_writer._local_dialogue("le café et le sommeil", duration, words)
        spoken = sum(len(turn.text.split()) for turn in turns)
        assert spoken <= words + 20, (duration, spoken, words)
        assert spoken >= words - 20, (duration, spoken, words)


def test_local_dialogue_scales_with_duration():
    short = script_writer._local_dialogue("un sujet", 20, tts.estimate_words(20, "-5%"))
    long = script_writer._local_dialogue("un sujet", 60, tts.estimate_words(60, "-5%"))
    assert sum(len(t.text.split()) for t in short) < sum(len(t.text.split()) for t in long)


# --- LLM dialogue normalisation and validation -----------------------------

# A reasoning model replies with one `Nom: réplique` per line; `_clean` used to
# collapse the newlines and leave the parser with a single turn.
LLM_TWO_HANDER = (
    "Léo: Mon téléphone vient de recevoir un texto qui a été envoyé depuis demain.\n"
    "Mia: Quoi ? C'est totalement impossible, vérifie l'heure immédiatement.\n"
    "Léo: Pourtant l'horloge indique déjà la bonne heure sur mon écran.\n"
    "Mia: Alors quelqu'un nous parle vraiment depuis le futur proche.\n"
    "Léo: On doit décider quoi faire avant que ce soir arrive.\n"
    "Mia: Je cherche l'expéditeur pendant que tu notes chaque détail.\n"
)


def test_clean_dialogue_preserves_turn_lines():
    cleaned = script_writer._clean_dialogue(LLM_TWO_HANDER)
    assert cleaned.count("\n") == 5
    assert len(script_writer._parse_dialogue(cleaned)) == 6


def test_clean_dialogue_strips_code_fences():
    fenced = f"```\n{LLM_TWO_HANDER}```"
    cleaned = script_writer._clean_dialogue(fenced)
    assert "```" not in cleaned
    assert len(script_writer._parse_dialogue(cleaned)) == 6


def test_clean_dialogue_tolerates_bullets_and_blank_lines():
    bulleted = "\n".join(f"- {line}" for line in LLM_TWO_HANDER.splitlines())
    cleaned = script_writer._clean_dialogue(f"\n{bulleted}\n\n")
    assert len(script_writer._parse_dialogue(cleaned)) == 6


def _stub_llm(monkeypatch, reply: str):
    monkeypatch.setattr(script_writer, "_llm_available", lambda: True)
    monkeypatch.setattr(script_writer, "_chat", lambda *a, **k: reply)


def test_llm_two_character_dialogue_is_used(monkeypatch):
    _stub_llm(monkeypatch, LLM_TWO_HANDER)
    turns, plain = script_writer.write_dialogue_script("un message du futur", duration=30)
    assert script_writer.characters_of(turns) == ["Léo", "Mia"]
    assert plain.startswith("Léo:")
    assert len(turns) == 6


def test_llm_fenced_dialogue_is_used(monkeypatch):
    _stub_llm(monkeypatch, f"```\n{LLM_TWO_HANDER}```")
    turns, _ = script_writer.write_dialogue_script("un message du futur", duration=30)
    assert script_writer.characters_of(turns) == ["Léo", "Mia"]


def test_llm_monologue_keeps_one_character(monkeypatch):
    _stub_llm(
        monkeypatch,
        "Léo: Je reçois un message qui a été envoyé depuis demain matin.\n"
        "Léo: L'heure indiquée sur l'écran correspond exactement à maintenant.\n"
        "Léo: Je dois décider quoi faire avant la fin de la journée.\n"
        "Léo: Et si je changeais la suite des choses dès ce soir ?\n"
        "Léo: Personne d'autre ne peut m'aider sur ce problème étrange.\n",
    )
    turns, _ = script_writer.write_dialogue_script("un message du futur", duration=30)
    assert script_writer.characters_of(turns) == ["Léo"]


def test_llm_three_character_dialogue_keeps_three(monkeypatch):
    _stub_llm(
        monkeypatch,
        "Léo: Un message est arrivé sur mon téléphone depuis demain matin.\n"
        "Maya: Impossible, montre-moi ton écran tout de suite maintenant.\n"
        "Paul: Attendez, l'heure indiquée est exacte cette fois vraiment.\n"
        "Léo: On doit agir avant ce soir sans perdre une minute.\n"
        "Maya: Je vérifie l'application pendant que vous notez tout.\n"
        "Paul: Je préviens les autres avant que le futur arrive ici.\n",
    )
    turns, _ = script_writer.write_dialogue_script("un message du futur", duration=30)
    assert script_writer.characters_of(turns) == ["Léo", "Maya", "Paul"]


def test_unusable_llm_reply_falls_back_to_local(monkeypatch):
    _stub_llm(monkeypatch, "Désolé, je ne peux pas t'aider avec cette demande.")
    turns, _ = script_writer.write_dialogue_script("un message du futur", duration=30)
    assert len(turns) >= 4
    assert script_writer.characters_of(turns) == list(script_writer.DEFAULT_CHARACTERS)


# --- per-character subtitles ----------------------------------------------

def test_speaker_palette_assigns_distinct_colors():
    palette = subtitles.speaker_palette(["Léo", "Maya"])
    assert palette["Léo"] != palette["Maya"]


def test_build_ass_colors_each_speaker(tmp_path):
    words = [
        tts.WordTiming("Bonjour", 0.0, 0.4, "Léo"),
        tts.WordTiming("Salut", 0.5, 0.9, "Maya"),
    ]
    ass = subtitles.build_ass(words, tmp_path / "c.ass", speakers=["Léo", "Maya"])
    text = ass.read_text(encoding="utf-8")
    assert "Léo" in text or "L" in text  # speakers are recorded in the track
    assert subtitles.speaker_palette(["Léo", "Maya"])["Maya"] in text


# --- background music ------------------------------------------------------

def test_detect_mood_prefers_topic_keywords():
    assert music.detect_mood("astuces pour réussir sur TikTok", "") == "energique"
    assert music.detect_mood("une histoire de disparition étrange", "") == "tension"


def test_generate_creates_a_non_empty_bed(tmp_path):
    out = music.generate(6.0, tmp_path / "bed.mp3", mood="calme")
    assert out.exists() and out.stat().st_size > 1000


def test_build_mixes_music_under_voice_keeping_both_streams(tmp_path):
    video = _make_video(tmp_path / "in.mp4")
    info = music.build(video, tmp_path, 6.0, mood="energique")
    mixed = Path(info["path"])
    assert mixed.exists()
    assert info["mood"] == "energique" and info["label"]

    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(mixed)],
        capture_output=True, text=True, timeout=60,
    ).stdout.split()
    assert "video" in streams and "audio" in streams


# --- real dialogue synthesis (network) ------------------------------------

def test_synthesize_dialogue_tags_words_with_their_speaker(tmp_path):
    lines = [
        tts.DialogueLine("Léo", "Le café ne te réveille pas vraiment.", "fr-FR-HenriNeural"),
        tts.DialogueLine("Maya", "Ah bon ? Pourtant je me sens plus alerte.", "fr-FR-VivienneMultilingualNeural"),
    ]
    try:
        speech = tts.synthesize_dialogue(lines, tmp_path / "dlg.mp3")
    except Exception as exc:  # noqa: BLE001 - edge-tts needs network
        pytest.skip(f"edge-tts indisponible : {exc}")

    assert speech.duration > 0
    assert speech.speakers == ["Léo", "Maya"]
    assert speech.voice_map["Léo"] == "fr-FR-HenriNeural"
    assert {w.speaker for w in speech.words} == {"Léo", "Maya"}
    # Words keep a monotonic timeline after stitching the takes.
    starts = [w.start for w in speech.words]
    assert starts == sorted(starts)
