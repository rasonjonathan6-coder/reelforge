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
python generate.py --script dial.txt --dialogue --animated-characters --out output/anim.mp4
```
`ffmpeg`/`ffprobe` must be on PATH. Tests that render real video are slow by
design; they assert real durations with ffprobe rather than mocking ffmpeg.

## Layout
- `app.py` — FastAPI routes (`/api/generate`, `/api/batch-generate`, `/api/jobs/{id}`, `/api/batch/{id}`, `/api/config`, `/api/upload-clips`).
- `jobs.py` — job model, persistence, `produce()` (shared by single and batch).
- `batch.py` — batch registry, bounded dispatch, ZIP.
- `generate.py` — orchestrator + CLI.
- `pipeline/` — `script_writer`, `tts`, `subtitles`, `music`, `visuals`, `stock_cache`, `compose`, `overlay`, `thumbnail`, `storage`.
- `pipeline/animation/` — the "Animated Character" engine (see below); additive, does not replace `visuals`.
- `web/index.html` — single-file UI (no build step).

## Conventions
- Each job renders into its own dir `output/<batch>/<job>/`; never mix files.
- Progress is reported through real pipeline stages via `on_step(stage, progress)`.
- Visual source is tracked and surfaced honestly in `meta.visual_source`
  (`pexels` / `pixabay` / `ai_clips` / `ai_images` / `local_fallback`) and in `scenes.json` +
  `visuals.json`. Do not claim stock footage that was not actually used.
- Duration accuracy: `pipeline/tts.py` retimes the voice-over to the target;
  keep final duration within ~1 s of the requested 30/45/60 s.

## Animated Character engine (`pipeline/animation/`)
A second, opt-in video type: real animated cartoon characters instead of the
stock Ken-Burns slideshow. Selected by `animated_characters` (API/`jobs`), the
UI "Type de vidéo" select, or `--animated-characters` (CLI). It is purely
additive: when the flag is off, the code path is byte-for-byte the old one.

Pipeline: `CharacterProfile` (one stable look per character, built once) ->
`Scene.breakdown` (one scene per dialogue turn, sized to the target duration;
each scene carries action, emotion, camera, environment, line, voice, prompt,
`lip_sync_required`) -> `AnimationProvider.generate_scene` (a real per-frame
animation) -> FFmpeg assembly (concat + burned captions + muxed voice + music).

- `providers.LocalAnimationProvider` is the free, CPU-only default. It draws
  every frame itself (animated backdrop + `avatars.draw_frame`), so the output
  is never a still image. `ANIMATION_PROVIDER=remote` + `ANIMATION_REMOTE_URL`
  selects `RemoteAnimationProvider` (HTTP JSON in, MP4 out) for external
  services; `ANIMATION_REMOTE_TOKEN` is read from the environment only.
- Failures raise `AnimationFailed` with `{"code": "ANIMATION_GENERATION_FAILED",
  scene_id, provider, error, retryable}`. There is no silent slideshow
  fallback; set `ALLOW_STATIC_FALLBACK=true` to opt into one explicitly.
- Lip-sync reuses `pipeline/avatars.py` (`viseme`, `draw_frame`), driven by the
  real TTS word timings, so the mouth matches the voice.
- Reports are archived per job: `animation_scenes.json`, `characters.json`.
- `providers.Local3DAnimationProvider` (style `cartoon_3d`, or
  `--animation-provider local3d`) drives `pipeline/animation/render3d.py`, a
  software 3D engine with no GPU and no external service. One scene is a real
  perspective render: triangle rasteriser + z-buffer, Lambert shading with a
  key/fill/rim rig, specular, a floor/wall set, a cast shadow and a background
  gradient. `RenderSettings.preset(name)` picks `draft`/`standard`/
  `cinematic`/`photoreal`/`anime`; `photoreal` and `cinematic` supersample
  (1.25x / 1.15x) before downsampling. `FilmLook` then adds bloom, shallow
  depth of field (autofocus follows the head's screen position), a filmic
  highlight roll-off (blended ACES curve — lit skin must not clip to flat
  white, or the face loses its eyes/brows/mouth at close-up), warm/cool grade,
  chromatic aberration, unsharp mask, vignette and grain. The set geometry is
  built once per scene (`static_tris`) and reused for every frame.
- Character references: a portrait named after the speaker locks that
  character's palette across all scenes. API/`jobs`: `characters_dir`; CLI:
  `--character-references <folder>`; the registry (`pipeline/avatars`) is
  cleared before each job. `PEXELS_API_KEY` / NVIDIA keys live only in the
  git-ignored `.env` and are never read from source.

## Visuals priority order
1. caller-supplied `clips` (AI scenes), 2. `visual_source="ai_images"`
(free Pollinations images), 3. Pexels (`PEXELS_API_KEY`),
4. Pixabay (`PIXABAY_API_KEY`), 5. multi-scene animated gradients.
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

## Free AI images (`pipeline/ai_images.py`) — the in-site default
No GPU, no key, no account: Pollinations (`https://image.pollinations.ai`)
returns a real generated image per scene, which `image_to_clip()` turns into
a short clip with a slow zoom + grain so the reel still moves. This is the
free path that needs nothing outside the app.
- Only the `sdxl` model is free now; `flux`/`turbo` return HTTP 402.
- The free endpoint throttles (402 when the anonymous quota is busy), so
  `fetch_image()` retries (`AI_IMAGE_ATTEMPTS`) and pauses between scenes.
- `AI_IMAGE_BASE_URL` / `AI_IMAGE_MODEL` / `AI_IMAGE_ATTEMPTS` tune it.
- Honest limit: these are animated stills (camera motion only), not true
  subject motion. Never present them as real footage.
- If every image fails, the job falls back to `local_fallback`; it never
  silently swaps in stock footage the user did not choose.
- Tests serve a local PNG via a fake HTTP server; they never hit the network.

## Free AI video clips (`pipeline/video_prompts.py` + Colab)
Real photoreal clips need a GPU, which the CPU-only server does not have. The
free path is a Colab notebook, and this module is the bridge:
`POST /api/video-prompts` turns a topic or a ready script into English
Wan/LTX prompts plus the model settings and negative prompt, and the notebook
(`colab/ReelForge_Video_IA_Colab.ipynb`) pastes that JSON into cell 3.
Generated clips come back through `POST /api/upload-clips`, then flow into
`visuals.Background` as the `ai_clips` source (highest priority).
- `translate()` is keyword translation, not MT: whole-word matching only
  (`algorithme` must not be hit by the `algo` entry), multi-word entries first.
- `MAX_SCENE_WORDS` caps each caption; longer captions add artefacts.
- The model/settings live in this module so the API and the notebook cannot
  drift. Never put a paid API key in the notebook path — it is the free one.

## Dialogue and music
- A character script is `Nom: réplique` lines (`script_writer.write_dialogue_script`,
  parsed by `parse_dialogue`). `tts.synthesize_dialogue` renders each turn with its
  own voice and concatenates the takes (never mixes), so word timings stay exact.
  The cast follows the script: one name keeps a single voice, two or more names
  take the `dialogue_cast` distribution — `mixte` (default: man + woman, the
  user's voice for character #1 and the opposite gender for #2), `femme`
  (femme+femme) or `homme` (homme+homme) via `tts.resolve_cast` /
  `tts.cast_genders` (the latter normalises accents and the UI labels).
  `WordTiming.speaker` drives per-character subtitle colours
  (`subtitles.speaker_palette`). Never pad a one-character script to two voices.
- Music (`pipeline/music.py`) is synthesised by FFmpeg (`aevalsrc` chord loop +
  lowpass/echo/tremolo), never a paid asset. Mood is inferred from topic/script
  unless `music_mood` is set. It is mixed *after* composition so the bed matches
  the final video length; `sidechaincompress` ducks it under the voice.
  Music is cosmetic: a failure must never fail the job.
- `tremolo` needs named options (`f=…:d=…`): a positional second value is a parse
  error ("No option name near ..."), so the value must carry the `d=` prefix.
- An LLM dialogue reply must be normalised with `_clean_dialogue`, never `_clean`:
  `_clean` collapses all whitespace and would merge every `Nom: réplique` line
  into one, leaving `_parse_dialogue` with a single turn and forcing the local
  fallback. `_clean_dialogue` normalises line by line and drops code fences.
- Reasoning models (NVIDIA nemotron) emit `reasoning_content` before `content`,
  so a small `max_tokens` can be consumed entirely by reasoning and return an
  empty `content` with `finish_reason="length"`. The NVIDIA budget starts at 4096
  (`NVIDIA_MAX_TOKENS`) and grows on empty replies.
- Dialogue duration fitting only runs when `auto_script` is set (`produce()`),
  which the topic panel does by default; a hand-edited script stays authoritative
  and is never retimed.
- A dialogue drives the visuals per turn: `generate._turn_scenes` groups the
  timed words by speaker change (one scene per turn, sized by that turn's word
  count) and `visuals.build_background_info` takes `scene_texts`/`scene_weights`
  to search and time each scene. More turns than `MAX_STOCK_SCENES` are folded
  into contiguous groups by `_group_scenes` so downloads stay bounded. A
  narration (no speaker tags) keeps the uniform sentence-based scenes.
- The script also drives the decor and the camera: `pipeline/script_understanding`
  detects a place/action per turn (keyword table, no model). `generate._turn_places`
  feeds `scene_places` so each scene searches its own setting (kitchen turn ->
  kitchen), and `_turn_pans` feeds `scene_pans` so walking turns get a travelling
  crop (`_cover_filter(pan=True)`). Texts, weights, places and pans are folded with
  the same bucket boundaries, so they never drift out of alignment.
- Characters are animated stickers, not footage: `pipeline/avatars` draws one
  cartoon per speaker with Pillow and encodes it as an RGBA `qtrle` clip (mouth
  opens on that speaker's timed words, blink/bob when idle, walk/sit from the
  action). `generate._render_characters` builds the per-speaker speaking cues and
  `compose.compose(characters=...)` overlays them above the caption band. A known
  name (Goku, Naruto, ...) keeps its signature colours and tag via
  `script_understanding.signature_look` — an original caricature, not a licensed
  likeness. This is the free, CPU-only alternative to a talking-avatar model.

## Gotchas
- Do NOT pass a per-scene frame count to `zoompan`'s `d`; it freezes the frame
  (each input frame gets its own sequence). Use `d=1` for continuous motion.
- The animated fallback must stay *visibly* animated: dark low-contrast palettes
  plus `gblur=sigma=38` collapse into a near-black still image once encoded.
  Keep the palettes bright and the blur light (`sigma=6`), or "Visuels générés"
  looks like a frozen picture.
- `auto_script` fitting only converges when `duration` reaches the backend. The
  UI must send it; a topic-script drafted for another duration cannot be retimed
  by rewriting alone (LLM output length varies), so `produce()` re-drafts at the
  target before `_fit_speech`.
- Dialogue fitting converges only if the local generator can produce a script
  *shorter* than the target. `_local_dialogue` must fill up to the word budget,
  not stop at a whole template cycle (~105 words), otherwise short reels (~30 s)
  can never reach their target and `_fit_speech` loops on the same length.
- Stock clips are looped with `-stream_loop -1` then cut with `-t`, so short
  clips fill a longer scene without freezing.
- Docker daemon needs `sudo -n`; build with `sudo -n docker build -t reelforge .`.
- `.env` holds secrets and is git-ignored; never commit it or echo its values.
- In an FFmpeg `filter_complex`, a stream label can be consumed only once.
  The voice has to feed both `sidechaincompress` and `amix`, so it is emitted
  through `asplit=2[voice][voice2]`. Without the split, `amix` fails to bind and
  FFmpeg exits with "Stream specifier '...' matches no streams" (exit 234).
- A long render can *look* stuck because the fit loop re-drafts and re-synthesises
  the dialogue up to `MAX_DURATION_ATTEMPTS` (3) times before the animation even
  starts. The UI shows elapsed time next to the step for this reason.
