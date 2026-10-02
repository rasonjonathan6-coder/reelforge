"""Prompts for the free GPU video notebook (Wan / LTX).

The notebook generates the actual photoreal clips; these tests only pin the
bridge that feeds it: French script in, usable English caption out.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from pipeline import video_prompts
from app import app


def test_translate_known_terms():
    assert "street" in video_prompts.translate("une rue")
    assert "at night" in video_prompts.translate("la nuit")
    assert "rain" in video_prompts.translate("sous la pluie")


def test_translate_drops_the_plural_leftover():
    # "videos" -> "video" used to leave a stray "s" word behind.
    words = video_prompts.translate("des videos gratuites").split()
    assert "s" not in words
    assert "free" in words


def test_translate_keeps_unknown_words():
    assert "blabla" in video_prompts.translate("blabla sans mots connus")


def test_prompt_for_detects_place_and_action():
    prompt = video_prompts.prompt_for("Il marche dans une rue de nuit")
    assert "street" in prompt
    assert "walking" in prompt
    assert video_prompts.STYLE_SUFFIX.split(",")[0] in prompt


def test_prompt_for_never_repeats_a_chunk():
    prompt = video_prompts.prompt_for("Il cuisine dans la cuisine le matin")
    chunks = [c.strip().lower() for c in prompt.split(",")]
    assert len(chunks) == len(set(chunks))


def test_prompt_for_caps_the_scene_length():
    long_scene = " ".join(f"mot{n}" for n in range(60))
    prompt = video_prompts.prompt_for(long_scene)
    scene_words = len(prompt.split(",")[0].split())
    assert scene_words <= video_prompts.MAX_SCENE_WORDS


def test_prompt_for_falls_back_when_nothing_matches():
    prompt = video_prompts.prompt_for("")
    assert "abstract cinematic background" in prompt


def test_prompts_for_returns_one_prompt_per_scene():
    script = "Première idée sur le café. Deuxième idée sur le sommeil. Troisième idée."
    out = video_prompts.prompts_for(script, topic="café", duration=20)
    assert out["count"] == len(out["prompts"]) >= 3
    assert out["settings"]["width"] == 832 and out["settings"]["height"] == 480


def test_prompts_for_caps_scene_count():
    script = " ".join(f"Phrase numéro {n}." for n in range(30))
    out = video_prompts.prompts_for(script, duration=90, max_scenes=3)
    assert out["count"] == 3


def test_prompts_for_carries_a_negative_prompt():
    out = video_prompts.prompts_for("Une phrase simple.", duration=15)
    assert "watermark" in out["negative_prompt"]
    assert "text" in out["negative_prompt"]


def test_api_video_prompts_from_a_script():
    client = TestClient(app)
    resp = client.post("/api/video-prompts", json={
        "script": "Léa: Salut !\nMarc: On parle de création vidéo gratuite.",
        "topic": "création vidéo",
        "duration": 20,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert data["notebook"].endswith(".ipynb")
    assert data["settings"]["model"]


def test_api_video_prompts_needs_a_topic_or_a_script():
    client = TestClient(app)
    assert client.post("/api/video-prompts", json={"topic": "ab"}).status_code == 400
