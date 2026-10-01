"""Tests for batch generation (and the untouched single-job contract).

These run fully offline: they patch the LLM writer and never touch the network.
Rendering is monkeypatched so the suite stays fast; the real end-to-end render
is exercised separately with ffprobe.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import jobs
from pipeline import script_writer


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate every run: no writes into the repo's output/.
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "output")
    import batch

    monkeypatch.setattr(batch, "_BATCHES_DIR", tmp_path / "output")

    # Deterministic, offline script + metadata.
    monkeypatch.setattr(
        script_writer, "write_script",
        lambda topic, duration=45, language="français", style="", tone="": f"Script pour {topic}. " * 6,
    )
    monkeypatch.setattr(
        script_writer, "write_metadata",
        lambda script, topic: {
            "title": topic, "description": "desc", "hashtags": ["a", "b"],
            "thumbnail_prompt": "x",
        },
    )

    # Fake renderer: writes a real file so filesystem checks are meaningful.
    def fake_generate(text, out_path, on_step=None, info_out=None, **kwargs):
        if on_step:
            for stage in ("tts", "subtitles", "visuals", "compose"):
                on_step(stage, 50)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
        if info_out:
            Path(info_out).mkdir(parents=True, exist_ok=True)
            (Path(info_out) / "audio.mp3").write_bytes(b"audio")
            (Path(info_out) / "subtitles.ass").write_text("[Script Info]", encoding="utf-8")
        return Path(out_path)

    monkeypatch.setattr(jobs, "generate", fake_generate)
    monkeypatch.setattr(jobs, "storage", type("S", (), {
        "upload": staticmethod(lambda path, key: f"/videos/{key}")
    }))
    monkeypatch.setattr(
        jobs.thumbnail, "generate_thumbnail",
        lambda title, prompt, out_path, work_dir: Path(out_path).write_bytes(b"jpg"),
    )

    import app as app_module

    return TestClient(app_module.app)


def _wait_batch(client, batch_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/batch/{batch_id}").json()
        if status["status"] in {"completed", "completed_with_errors"}:
            return status
        time.sleep(0.2)
    raise AssertionError("batch did not finish in time")


def test_batch_creates_independent_jobs(client):
    resp = client.post("/api/batch-generate", json={
        "topics": ["Espace", "Inventions", "Animaux"],
        "duration": 30, "voice": "fr-FR-VivienneMultilingualNeural", "rate": "-5%",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"batch_id", "jobs"}
    assert len(body["jobs"]) == 3
    assert all(set(j) == {"job_id", "topic"} for j in body["jobs"])

    status = _wait_batch(client, body["batch_id"])
    assert status["counts"]["completed"] == 3
    assert status["counts"]["failed"] == 0

    # Each job lives in its own directory with the full artifact set.
    dirs = {j["job_id"]: Path(jobs._job_dir(j["job_id"], body["batch_id"])) for j in body["jobs"]}
    assert len(set(dirs.values())) == 3
    for job_id, folder in dirs.items():
        assert (folder / "final.mp4").exists()
        assert (folder / "script.json").exists()
        assert (folder / "metadata.json").exists()
        assert (folder / "audio.mp3").exists()
        assert (folder / "subtitles.ass").exists()
        assert (folder / "scenes.json").exists()

    # Per-video metadata carries title/description/hashtags.
    meta = json.loads((dirs[body["jobs"][0]["job_id"]] / "metadata.json").read_text())
    assert {"title", "description", "hashtags"} <= set(meta)


def test_batch_isolated_failure(client):
    """A failing job must not cancel the others."""
    original = jobs.generate
    calls = {"n": 0}

    def flaky(text, out_path, on_step=None, info_out=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("Pexels → fallback local échoué (test)")
        return original(text, out_path, on_step=on_step, info_out=info_out, **kwargs)

    jobs.generate = flaky
    try:
        body = client.post("/api/batch-generate", json={
            "topics": ["A", "B", "C"], "duration": 30,
        }).json()
        status = _wait_batch(client, body["batch_id"])
    finally:
        jobs.generate = original

    assert status["counts"]["completed"] == 2
    assert status["counts"]["failed"] == 1
    assert status["status"] == "completed_with_errors"
    failed = [j for j in status["jobs"] if j["status"] == "failed"]
    assert failed and failed[0]["error"]
    completed = [j for j in status["jobs"] if j["status"] == "completed"]
    assert all(j["video_url"] for j in completed)


def test_zip_contains_only_completed_videos(client):
    original = jobs.generate
    calls = {"n": 0}

    def flaky(text, out_path, on_step=None, info_out=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original(text, out_path, on_step=on_step, info_out=info_out, **kwargs)

    jobs.generate = flaky
    try:
        body = client.post("/api/batch-generate", json={"topics": ["A", "B", "C"]}).json()
        status = _wait_batch(client, body["batch_id"])
    finally:
        jobs.generate = original

    resp = client.get(f"/api/batch/{body['batch_id']}/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        assert len(names) == status["counts"]["completed"] == 2
        assert all(n.endswith(".mp4") for n in names)
        assert names[0].startswith("01_") or names[0].startswith("03_")
        assert not any("key" in n.lower() or ".env" in n for n in names)


def test_batch_size_limit(client):
    too_many = [f"sujet {i}" for i in range(25)]
    resp = client.post("/api/batch-generate", json={"topics": too_many})
    assert resp.status_code == 400
    assert "20" in resp.json()["detail"]


def test_batch_rejects_empty(client):
    resp = client.post("/api/batch-generate", json={"topics": ["   ", "# commentaire"]})
    assert resp.status_code == 400


def test_single_job_contract_unchanged(client):
    """The original single-video flow must keep working."""
    resp = client.post("/api/generate", json={
        "text": "Un texte suffisamment long pour passer la validation minimale.",
        "topic": "Test",
    })
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + 20
    job = None
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.2)

    assert job["status"] == "completed"
    assert job["done"] is True
    assert job["video_url"] == f"/videos/{job_id}/final.mp4"
    assert job["batch_id"] is None
    assert job["meta"]["title"]


def test_config_exposes_batch_limits(client):
    cfg = client.get("/api/config").json()
    assert "max_batch_size" in cfg
    assert "max_concurrent_jobs" in cfg


def test_slugify_is_safe():
    assert jobs.slugify("5 faits étonnants sur l'espace") == "5_faits_etonnants_sur_l_espace"
    assert "/" not in jobs.slugify("a/b\\c:d")
    assert len(jobs.slugify("x" * 200)) <= 48
