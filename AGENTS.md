# AGENTS.md

Repository-specific knowledge for ReelForge (CPU-only, free faceless-reel pipeline).

## What it is
FastAPI app + FFmpeg pipeline that renders 1080x1920 reels from a topic or a
script: script (local heuristic or free LLM), edge-tts voice-over, karaoke
subtitles, visuals, final montage, thumbnail and metadata. No GPU, no paid API.

## Commands
```bash
python -m pytest -q                       # full suite (~45 s, mostly ffmpeg)
python -m compileall -q pipeline generate.py jobs.py app.py batch.py
python -m uvicorn app:app --host 0.0.0.0 --port 12000   # local server
python generate.py --text "..." --out output/reel.mp4
```
`ffmpeg`/`ffprobe` must be on PATH. Tests that render real video are slow by
design; they assert real durations with ffprobe rather than mocking ffmpeg.

## Layout
- `app.py` — FastAPI routes (`/api/generate`, `/api/batch-generate`, `/api/jobs/{id}`, `/api/batch/{id}`, `/api/config`, `/api/upload-clips`).
- `jobs.py` — job model, persistence, `produce()` (shared by single and batch).
- `batch.py` — batch registry, bounded dispatch, ZIP.
- `generate.py` — orchestrator + CLI.
- `pipeline/` — `script_writer`, `tts`, `subtitles`, `visuals`, `stock_cache`, `compose`, `overlay`, `thumbnail`, `storage`.
- `web/index.html` — single-file UI (no build step).

## Conventions
- Each job renders into its own dir `output/<batch>/<job>/`; never mix files.
- Progress is reported through real pipeline stages via `on_step(stage, progress)`.
- Visual source is tracked and surfaced honestly in `meta.visual_source`
  (`pexels` / `pixabay` / `ai_clips` / `local_fallback`) and in `scenes.json` +
  `visuals.json`. Do not claim stock footage that was not actually used.
- Duration accuracy: `pipeline/tts.py` retimes the voice-over to the target;
  keep final duration within ~1 s of the requested 30/45/60 s.

## Visuals priority order
1. caller-supplied `clips` (AI scenes), 2. Pexels (`PEXELS_API_KEY`),
3. Pixabay (`PIXABAY_API_KEY`), 4. multi-scene animated gradients.
Stock search is per scene (`visuals.scene_queries`), one query per narration
chunk, portrait renditions preferred, joined with `xfade` cross-dissolves.
`PEXELS_API_BASE` / `PIXABAY_API_BASE` override the provider host (used to test
against a local fake server without real keys).

## Stock clip cache (`pipeline/stock_cache.py`)
Downloaded clips are kept in a persistent local cache (`data/cache/pexels/`,
git-ignored) so repeated topics do not re-hit the provider or re-download the
same file. One entry = a SHA-256 key over `(provider, query, orientation)` +
an mp4 + a JSON sidecar (provider url/id, width/height/duration, timestamps).
- Entries are ffprobe-validated before publishing; writes are atomic
  (tmp file + `os.replace`), so an interrupted download is never a valid entry.
- A per-key lock means concurrent jobs download a key exactly once.
- `PEXELS_CACHE_ENABLED` / `PEXELS_CACHE_TTL_DAYS` / `PEXELS_CACHE_MAX_GB` /
  `PEXELS_CACHE_DIR` tune it. `cleanup()` (TTL + LRU size cap) runs at API
  startup; it never deletes an entry a running job holds.
- Per-scene provenance is reported in `meta.visual_scene_origins`
  (`pexels_cache` / `pexels_api` / `reused`) and `meta.visual_cache` counters.
- Tests isolate the cache via `tests/conftest.py`; never let a test touch the
  real `data/cache/`.

## Gotchas
- Do NOT pass a per-scene frame count to `zoompan`'s `d`; it freezes the frame
  (each input frame gets its own sequence). Use `d=1` for continuous motion.
- Stock clips are looped with `-stream_loop -1` then cut with `-t`, so short
  clips fill a longer scene without freezing.
- Docker daemon needs `sudo -n`; build with `sudo -n docker build -t reelforge .`.
- `.env` holds secrets and is git-ignored; never commit it or echo its values.
