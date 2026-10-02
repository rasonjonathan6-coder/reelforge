# ReelForge — Générateur de vidéos faceless gratuit et illimité

Transforme un sujet (ou un texte) en vraie vidéo verticale 9:16 (1080×1920, 30 fps,
H.264/AAC) de 30 à 60 s : **script IA, voix off, sous-titres animés, visuels,
vignette et métadonnées** (titre, description, hashtags).

**Aucune clé API requise. Aucune limite. Aucun GPU.**

## Ce que ça fait / ce que ça ne fait pas

| Fonction | Statut |
|---|---|
| Vraie vidéo 30-60 s, format Reels/TikTok/Shorts | ✅ |
| Script généré depuis un sujet | ✅ générateur local intégré (sans clé) ou LLM gratuit optionnel |
| Voix off IA (edge-tts, 400+ voix, 100+ langues) | ✅ gratuit, illimité |
| Dialogue ou monologue (voix par personnage selon le script) | ✅ cases « Dialogue » dans l'interface |
| Distribution des voix : mixte, femme+femme, homme+homme | ✅ sélecteur « Voix des personnages » |
| Sous-titres animés synchronisés mot par mot | ✅ fade + mot actif en couleur |
| Couleur de sous-titre par personnage | ✅ un personnage = une couleur |
| Musique de fond générée (humeur déduite du script, duckée sous la voix) | ✅ synthétisée par FFmpeg, sans asset payant |
| Visuels qui bougent (scènes animées, transitions, grain) | ✅ |
| **Images IA photoréalistes** (Pollinations, **sans clé ni compte**) | ✅ un visuel par scène, zoom lent — généré dans l'app |
| Vidéos stock gratuites (Pexels **ou** Pixabay, clés gratuites) | ✅ une recherche par scène, la source utilisée est affichée |
| Cache local des clips stock (réutilise les plans déjà téléchargés) | ✅ moins d'appels API et de téléchargements sur les sujets répétés |
| Import de clips IA générés (Colab / Wan / LTX) | ✅ ils remplacent les visuels auto |
| Habillage : barre de progression + signature | ✅ |
| Vignette générée par IA + titre incrusté | ✅ |
| Titre, description, hashtags auto | ✅ |
| Personnage IA généré qui parle (lip-sync) | ❌ nécessite un GPU ou une API payante |
| Générer une scène vidéo IA *sur ce serveur* | ❌ CPU-only ; à faire sur Colab (voir plus bas) |

### La voix est-elle « vraie » ?

C'est une **voix neuronale** (Microsoft Neural via edge-tts), pas une voix robot
des années 2010 : respirations, intonations et liaisons correctes. Sur un reel de
30 s, elle passe pour humaine. Il n'existe **aucun** TTS français à la fois
naturel, illimité et gratuit sans GPU : les voix vraiment indiscernables
(ElevenLabs, Cartesia) sont payantes ou nécessitent un GPU.

La voix par défaut est `fr-FR-VivienneMultilingualNeural` à `-5 %`, plus
naturelle que l'ancienne Denise à `+8 %` (le débit rapide trahit la synthèse).

### De la vraie vidéo IA, gratuitement

La génération vidéo IA exige un GPU : c'est une contrainte matérielle, pas une
limite de ReelForge. Le notebook `colab/ReelForge_Video_IA_Colab.ipynb` génère de
vrais plans avec **Wan 2.1** (Apache 2.0, gratuit, commercialisable) sur le **T4
gratuit de Colab**. Tu télécharges les clips, tu les déposes dans le champ
« Clips IA générés » de l'interface, et ReelForge les monte avec ta voix et tes
sous-titres. Colab fournit le GPU, ReelForge fait le reste.

## Démarrage

```bash
pip install -r requirements.txt
export PATH=/workspace/bin:$PATH      # ffmpeg (voir plus bas)
cp .env.example .env                  # puis renseigne NVIDIA_API_KEY
python app.py                         # http://localhost:8000
```

Le fichier `.env` est lu automatiquement au démarrage et n'est **jamais
committé** (il est dans `.gitignore`). C'est là que va la clé NVIDIA.

FFmpeg : si absent et sans droits root :

```bash
pip install imageio-ffmpeg
python -c "import imageio_ffmpeg,shutil,os;os.makedirs('/workspace/bin',exist_ok=True);shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(),'/workspace/bin/ffmpeg');os.chmod('/workspace/bin/ffmpeg',0o755)"
```

> Remarque : les builds FFmpeg minimalistes n'ont pas `drawtext`. C'est pourquoi la
> vignette est dessinée avec Pillow, pas avec FFmpeg.

## Trois modes d'exécution (config par variables d'environnement)

### 1. Local (par défaut) — zéro infrastructure

```bash
python app.py
```

File d'attente en mémoire, stockage disque. Parfait pour un VPS unique.

### 2. File distribuée — Redis + Celery (montée en charge)

```bash
redis-server --daemonize yes
QUEUE_BACKEND=celery celery -A celery_app worker --loglevel=info --concurrency=4
QUEUE_BACKEND=celery REDIS_URL=redis://localhost:6379/0 python app.py
```

Le serveur web met la tâche en file et répond en millisecondes ; les workers
fabriquent les vidéos. On ajoute des workers pour absorber la charge, sans
changer une ligne de code. Si Redis tombe, l'API bascule automatiquement sur le
pool local.

### 3. Stockage S3 (AWS, Cloudflare R2, MinIO…)

```bash
STORAGE_BACKEND=s3 \
S3_BUCKET=mon-bucket \
S3_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com \
S3_PUBLIC_BASE_URL=https://cdn.example.com \
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
python app.py
```

Sans `S3_PUBLIC_BASE_URL`, l'API renvoie une URL présignée (7 jours).

## Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `QUEUE_BACKEND` | `local` | `local` ou `celery` |
| `REDIS_URL` | `redis://localhost:6379/0` | broker Celery |
| `WORKER_COUNT` | `2` | workers du pool local |
| `MAX_CONCURRENT_JOBS` | `2` | rendus simultanés maximum |
| `VIDEO_WIDTH` / `VIDEO_HEIGHT` | `1080` / `1920` | géométrie de sortie ; baissez sur une petite machine |
| `VIDEO_FPS` | `30` | images par seconde |
| `FFMPEG_PRESET` | `veryfast` | compromis vitesse/qualité x264 |
| `FFMPEG_THREADS` | `1` | threads x264 par encodage (1 = empreinte RAM minimale) |
| `STORAGE_BACKEND` | `local` | `local` ou `s3` |
| `S3_BUCKET`, `S3_PREFIX`, `S3_ENDPOINT_URL`, `S3_REGION` | — | stockage objet |
| `S3_PUBLIC_BASE_URL` | — | base CDN pour liens directs |
| `AI_IMAGE_BASE_URL` | `https://image.pollinations.ai` | service d'images IA gratuit (aucune clé) |
| `AI_IMAGE_MODEL` | `sdxl` | modèle d'image ; `flux`/`turbo` exigent désormais un compte payant |
| `AI_IMAGE_ATTEMPTS` | `6` | tentatives par image (le service gratuit limite le débit) |
| `PEXELS_API_KEY` | — | active les vidéos stock Pexels (clé gratuite) |
| `PIXABAY_API_KEY` | — | active les vidéos stock Pixabay (clé gratuite) |
| `PEXELS_CACHE_ENABLED` | `true` | cache local des clips stock |
| `PEXELS_CACHE_TTL_DAYS` | `30` | durée de vie d'une entrée (`0` = jamais) |
| `PEXELS_CACHE_MAX_GB` | `5` | plafond de taille, purge LRU au démarrage |
| `PEXELS_CACHE_DIR` | `data/cache/pexels` | dossier du cache |
| `NVIDIA_API_KEY` | — | script IA via **NVIDIA NIM** (build.nvidia.com, clé gratuite) |
| `NVIDIA_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | modèle NIM |
| `GEMINI_API_KEY` | — | script IA via Google Gemini (offre gratuite) |
| `GROQ_API_KEY` | — | script IA via Groq (offre gratuite) |
| `OPENROUTER_API_KEY` | — | script IA via OpenRouter |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME` | — | endpoint LLM compatible OpenAI |

### Script : trois options

Sans aucune clé, le générateur local intégré écrit le script (accroche, corps,
question finale) : zéro dépendance réseau, aucune limite.

Pour des scripts plus riches, une clé gratuite suffit — le code choisit tout seul
le fournisseur, dans cet ordre : **NVIDIA → Gemini → Groq → OpenRouter → endpoint
personnalisé → local**.

```bash
NVIDIA_API_KEY=nvapi-xxx python app.py   # build.nvidia.com → clé gratuite (crédits offerts)
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b NVIDIA_API_KEY=nvapi-xxx python app.py  # modèle plus rapide
# ou
GEMINI_API_KEY=xxx python app.py         # aistudio.google.com → clé gratuite
```

`GET /api/config` renvoie `llm` pour indiquer le fournisseur actif
(`nvidia`, `gemini`, `groq`, `openrouter`, `custom` ou `local`).

## API

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/api/script` | Sujet → script |
| POST | `/api/generate` | Crée un job, renvoie `job_id` |
| POST | `/api/batch-generate` | Crée un lot de jobs (un par sujet) |
| GET | `/api/batch/{id}` | Statut de chaque job du lot + compteurs |
| GET | `/api/batch/{id}/download` | ZIP des vidéos terminées du lot |
| GET | `/api/jobs/{id}` | Statut, progression, vidéo, vignette, métadonnées |
| GET | `/api/voices` | Liste des voix (avec genre et locale) |
| GET | `/api/music` | Humeurs musicales disponibles |
| POST | `/api/upload-clips` | Upload de clips IA → dossier réutilisable |
| GET | `/api/config` | Backends actifs + limites batch + état du cache |
| GET | `/videos/{...}.mp4` | Téléchargement |

```bash
# Script depuis un sujet
curl -X POST localhost:8000/api/script -H 'Content-Type: application/json' \
  -d '{"topic":"3 astuces pour dormir mieux","duration":30}'

# Vidéo
curl -X POST localhost:8000/api/generate -H 'Content-Type: application/json' \
  -d '{"text":"Ton script...","voice":"fr-FR-VivienneMultilingualNeural","rate":"-5%","use_stock":false}'

# Dialogue : le nombre de voix suit le script (1 personnage = 1 voix)
curl -X POST localhost:8000/api/generate -H 'Content-Type: application/json' \
  -d '{"topic":"le café réveille-t-il vraiment ?","dialogue":true,"duration":30,"music":true}'

# Distribution des voix : "mixte" (défaut, homme + femme), "femme" (femme+femme)
# ou "homme" (homme+homme). Le champ est aussi accepté par /api/script.
curl -X POST localhost:8000/api/generate -H 'Content-Type: application/json' \
  -d '{"topic":"deux amies parlent du café","dialogue":true,"dialogue_cast":"femme","duration":30}'

# Musique de fond : humeur auto (déduite du script) ou forcée
curl -X POST localhost:8000/api/generate -H 'Content-Type: application/json' \
  -d '{"text":"Ton script...","music":true,"music_mood":"epique"}'

# Vidéo montée depuis tes propres clips IA
curl -X POST localhost:8000/api/upload-clips -F "files=@clip1.mp4" -F "files=@clip2.mp4"
# -> {"clips_dir":".../output/clips/xxxx","count":2}
curl -X POST localhost:8000/api/generate -H 'Content-Type: application/json' \
  -d '{"text":"Ton script...","clips_dir":".../output/clips/xxxx","logo":"@ma.chaine"}'

# Lot : un sujet par ligne = une vidéo indépendante
curl -X POST localhost:8000/api/batch-generate -H 'Content-Type: application/json' \
  -d '{"topics":["5 faits étonnants sur l espace","Les animaux les plus rapides"],
       "language":"fr-FR","duration":30,"style":"storytelling","tone":"dynamic",
       "voice":"fr-FR-VivienneMultilingualNeural","rate":"-5%"}'
# -> {"batch_id":"...","jobs":[{"job_id":"...","topic":"..."}]}
curl -s localhost:8000/api/batch/<batch_id>            # statut détaillé
curl -sO localhost:8000/api/batch/<batch_id>/download  # ZIP des vidéos terminées
```

## Génération en lot

Chaque sujet devient un job indépendant, rangé dans son propre dossier :

```
output/
  BATCH_ID/
    JOB_ID_1/  script.json  audio.mp3  scenes.json  subtitles.ass  metadata.json  final.mp4
    JOB_ID_2/  ...
```

- **Statuts réels** : `queued` → `running` → `completed` / `failed`.
- **Étapes réelles** : `script`, `tts`, `subtitles`, `visuals`, `compose`, `metadata`,
  `completed` (aucune progression simulée — chaque étape est émise par le pipeline).
- **Échec isolé** : un job en échec ne bloque jamais les autres ; le lot reste
  téléchargeable pour les vidéos réussies.
- **Concurrence bornée** : `MAX_CONCURRENT_JOBS` (défaut 2) limite les rendus
  simultanés pour ne pas saturer le CPU/RAM.
- **Limite de taille** : `MAX_BATCH_SIZE` (défaut 20) — au-delà, l'API renvoie 400.
- **ZIP** : `Télécharger tout` n'inclut que les MP4 terminés, nommés
  `01_sujet.mp4`, `02_...` ; aucune clé API n'y figure.

## Durée des vidéos

La narration est ajustée à la durée demandée plutôt que coupée ou ralentie :

1. le script est calibré sur le débit réel de la voix choisie (`pipeline/tts.py`,
   ~3,05 mots/s à `+0%`, corrigé par le `rate`) ;
2. après synthèse, la durée **réelle** de `audio.mp3` est mesurée avec `ffprobe`
   (edge-tts s'arrête au dernier mot et sous-estime le fichier de 0,3 à 0,9 s) ;
3. si l'écart dépasse 1 s, le script est régénéré avec le nombre de mots que le
   débit observé implique — au maximum 3 passes, jamais de boucle infinie ;
4. la voix n'est jamais étirée ni transposée ; si l'écart persiste, la vidéo est
   conservée telle quelle et la durée réelle est signalée.

Chaque job expose le diagnostic dans `meta` et dans `duration.json` :

```json
{ "target_duration": 30, "audio_duration": 29.7, "final_video_duration": 29.8 }
```

## Cache local des clips stock

Les plans Pexels/Pixabay déjà téléchargés sont conservés dans `data/cache/pexels/`
(git-ignoré) : sur un sujet déjà traité, la vidéo est montée **sans aucun appel à
l'API ni nouveau téléchargement**. Chaque entrée est validée par `ffprobe` avant
d'être publiée, et l'écriture est atomique (fichier temporaire + `os.replace`),
donc un téléchargement interrompu n'est jamais pris pour un clip valide.

- **Clé** : SHA-256 de `(fournisseur, requête, orientation)` — la requête n'est
  jamais utilisée comme nom de fichier, et aucune clé API n'est stockée.
- **TTL** : `PEXELS_CACHE_TTL_DAYS` (défaut 30 j) ; **plafond** :
  `PEXELS_CACHE_MAX_GB` (défaut 5 Go), purge LRU au démarrage du serveur.
- **Concurrence** : un même plan n'est téléchargé qu'une fois, même si plusieurs
  jobs tournent en parallèle.
- **Provenance** : `meta.visual_scene_origins` indique, scène par scène,
  `pexels_cache` / `pexels_api` / `reused`, et `meta.visual_cache` les compteurs
  (`hits`, `misses`, `stored`…). `GET /api/config` expose l'état du cache.

## Déploiement

L'image embarque FFmpeg et les polices, donc rien à installer côté serveur.

```bash
docker build -t reelforge .
docker run -p 8000:8000 -e NVIDIA_API_KEY=nvapi-xxx reelforge
```

Testé : le conteneur génère une vraie vidéo 1080×1920 30 fps (H.264/AAC) de bout
en bout. `.env` est exclu de l'image via `.dockerignore` — la clé passe
uniquement par la variable d'environnement du service.

Sur **Render** : `render.yaml` décrit le service (runtime Docker, plan free,
`NVIDIA_API_KEY` à saisir dans le dashboard). Sur **Fly.io / Railway / VPS** :
même image, la variable `PORT` est respectée.

⚠️ **Le plan gratuit Render (512 Mo) ne tient pas le 1080×1920.** Deux rendus
simultanés (ou un seul montage final en 1080×1920) déclenchent l'OOM killer :
le conteneur redémarre, le job disparaît (stocké en mémoire) et l'interface
affiche une erreur de parsing JSON. `render.yaml` configure donc `WORKER_COUNT=1`,
`MAX_CONCURRENT_JOBS=1`, `VIDEO_WIDTH/HEIGHT=720x1280`, `FFMPEG_THREADS=1` et
`FFMPEG_PRESET=veryfast`. Remettez 1080×1920 sur une instance payante.

⚠️ Sur les hébergeurs à disque éphémère (Render free, Fly sans volume), les
vidéos de `output/` **et le cache de clips** disparaissent au redémarrage. Pour
conserver les vidéos, active le backend S3 (`STORAGE_BACKEND=s3`) ; le cache, lui,
se reconstruit tout seul (il ne fait qu'éviter des téléchargements répétés).

## Architecture

```
web/index.html        → interface (mode simple + mode lot, voix, aperçu, métadonnées)
app.py                → API FastAPI (dispatch local ou Celery)
batch.py              → lots : registre, concurrence bornée, statuts, ZIP
celery_app.py         → application Celery
tasks.py              → tâche Celery
jobs.py               → modèle de job + routine de production partagée
config.py             → configuration par variables d'environnement
generate.py           → orchestrateur + CLI
pipeline/script_writer.py → script + métadonnées (générateur local, LLM optionnel)
pipeline/thumbnail.py → vignette (fond IA + titre Pillow)
pipeline/tts.py       → edge-tts + timings mot par mot (et dialogue multi-voix)
pipeline/subtitles.py → sous-titres ASS karaoké (fade + mot actif, couleur par personnage)
pipeline/music.py     → musique de fond générée (humeur détectée) + ducking sous la voix
pipeline/ai_images.py → images IA gratuites (Pollinations, sans clé) → clips à zoom lent
pipeline/visuals.py   → stock Pexels/Pixabay (une recherche par scène), images IA ou scènes animées
pipeline/stock_cache.py → cache local des clips stock (clé SHA-256, TTL, purge LRU)
pipeline/overlay.py   → barre de progression + signature (drawtext)
pipeline/compose.py   → montage final 9:16
pipeline/storage.py   → stockage local ou S3
```

## Résilience

- LLM indisponible ou limité → métadonnées générées localement (heuristique), la
  vidéo se fait quand même.
- API image indisponible → fond de vignette en dégradé.
- Pexels/Pixabay indisponible ou sans clé → scènes animées générées par FFmpeg,
  et la source réellement utilisée est indiquée dans l'UI et dans `meta.visual_source`.
- Images IA (Pollinations) indisponibles → repli direct sur les scènes animées :
  la source choisie n'est jamais remplacée en silence par une autre.
- FFmpeg sans `drawtext` → habillage ignoré, la vidéo se termine quand même.
- Redis indisponible → repli automatique sur le pool local.

## Ligne de commande

```bash
python generate.py --script examples/script.txt --out output/reel.mp4 --no-stock
python generate.py --text "Ton script ici..." --out output/reel.mp4
python generate.py --text "..." --out output/reel.mp4 --clips ~/clips_ia --logo "@ma.chaine"

# Lot : un sujet par ligne dans topics.txt
python generate.py --batch topics.txt \
  --language fr-FR --duration 30 --style storytelling --tone dynamic \
  --voice fr-FR-VivienneMultilingualNeural --rate "-5%" \
  --output output/batches
```
