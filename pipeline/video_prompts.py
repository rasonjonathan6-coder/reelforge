"""Turn a French reel script into English prompts for free video models.

The free GPU video models (Wan, LTX-Video, CogVideoX) are trained on English
captions, so a French script has to be translated before it can drive them.
This module does that with a small keyword table — no model download, no API
call — and reuses the same scene split and place/action detection the rest of
the pipeline already uses, so the clips a user generates match the reel they
are going to edit.

Honest limit: this is keyword translation, not machine translation. A sentence
outside the table keeps its original words; the cinematic modifiers are what
keep the result usable.
"""

from __future__ import annotations

import re

from pipeline import script_understanding, visuals

# Negative prompt shared by every clip: the usual diffusion artefacts plus the
# things that would ruin a faceless reel (burned-in text, watermarks).
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed hands, extra limbs, extra fingers, "
    "watermark, text, subtitles, caption, logo, oversaturated, jpeg artifacts, "
    "still image, slideshow"
)

# What every clip should look like so consecutive clips cut together.
STYLE_SUFFIX = (
    "cinematic, vertical 9:16 framing, shallow depth of field, "
    "natural lighting, film grain, smooth camera motion"
)
_STYLE_WORDS = set(STYLE_SUFFIX.replace(",", " ").split())

# Video models are trained on short captions; past roughly this many words the
# extra text stops helping and starts injecting artefacts.
MAX_SCENE_WORDS = 12

# French -> English terms, applied longest-first so multi-word entries win.
# Keys are accent-stripped and lowercased to match `script_understanding._norm`.
_TERMS: dict[str, str] = {
    # Places
    "salle de bain": "bathroom", "salle a manger": "dining room",
    "salle": "living room", "cuisine": "kitchen", "chambre": "bedroom",
    "bureau": "office", "restaurant": "restaurant", "cafe": "coffee shop",
    "rue": "street", "trottoir": "sidewalk", "avenue": "avenue",
    "boulevard": "boulevard", "voiture": "car", "route": "road",
    "parc": "park", "jardin": "garden", "foret": "forest", "arbre": "trees",
    "plage": "beach", "sable": "sand", "mer": "sea", "ocean": "ocean",
    "vague": "waves", "ecole": "school", "classe": "classroom",
    "hopital": "hospital", "magasin": "shop", "boutique": "boutique",
    "supermarche": "supermarket", "maison": "house", "appartement": "apartment",
    "ville": "city", "village": "village", "montagne": "mountains",
    "ciel": "sky", "nuage": "clouds", "pont": "bridge", "gare": "train station",
    "aeroport": "airport", "usine": "factory", "ferme": "farm",
    # Time and weather
    "nuit": "at night", "matin": "in the morning", "soir": "in the evening",
    "coucher de soleil": "at sunset", "aube": "at sunrise",
    "pluie": "rain", "orage": "storm", "neige": "snow", "brouillard": "fog",
    "soleil": "sunlight", "hiver": "winter", "ete": "summer",
    "automne": "autumn", "printemps": "spring",
    # Actions
    "marche": "walking", "marcher": "walking", "avance": "walking forward",
    "court": "running", "danse": "dancing", "dort": "sleeping",
    "mange": "eating", "boit": "drinking", "lit": "reading",
    "telephone": "talking on the phone", "rit": "laughing", "pleure": "crying",
    "crie": "shouting", "parle": "talking", "assied": "sitting down",
    "regarde": "looking at something", "travail": "working",
    "ecrit": "writing", "cuisine": "kitchen",
    # Light and texture
    "neon": "neon lights", "reflet": "reflections", "lumiere": "light",
    "ombre": "shadows", "couleur": "vivid colors", "flou": "soft focus",
    "gros plan": "close up", "vue aerienne": "aerial view",
    "ralenti": "slow motion", "reve": "dreamlike",
    # Recurring reel vocabulary
    "creation": "content creation", "gratuit": "free", "gratuite": "free",
    "gratuites": "free", "video": "video", "astuce": "tips",
    "conseil": "advice", "chaine": "channel", "abonne": "subscribers",
    "audience": "audience", "visage": "face", "camera": "camera",
    "argent": "money", "temps": "time",
    "journee": "day", "semaine": "week", "projet": "project",
    "bienvenue": "welcoming shot", "salut": "greeting", "aujourd'hui": "today",
    "apprendre": "learning", "montrer": "showing", "creer": "creating",
    "reussir": "succeeding", "commencer": "starting", "gagner": "winning",
    # Social-media vocabulary that shows up in almost every reel script.
    "percer": "breaking through", "astuces": "tips", "tendance": "trending",
    "premiere": "first", "premier": "first", "deuxieme": "second",
    "troisieme": "third", "algorithm": "algorithm", "algo": "algorithm",
    "retention": "retention", "recompense": "rewards", "commentaire": "comments",
    "abonnes": "subscribers", "followers": "followers", "vues": "views",
    "publie": "posting", "publier": "posting", "poste": "posting",
    "remixe": "remixing", "audio": "audio", "voix": "voice",
    "instrument": "instrument", "demarquer": "standing out",
    "regulierement": "regularly", "regularite": "consistency",
    "quantite": "quantity", "calendrier": "schedule", "hebdomadaire": "weekly",
    "audience": "audience", "cible": "target", "scroll": "scrolling",
    "analyser": "analyzing", "analyse": "analyzing", "ajuster": "adjusting",
    "appliquer": "applying", "teste": "testing", "iterer": "iterating",
    "interagir": "engaging", "booster": "boosting", "portee": "reach",
    "organique": "organic", "detour": "worth watching",
    "conseils": "advice", "leviers": "levers", "videaste": "creator",
    "contenu": "content", "insights": "analytics", "twist": "personal touch",
    "choc": "striking", "intrigante": "intriguing", "flot": "feed",
    "secondes": "seconds", "seconde": "second", "heure": "hour",
    "heures": "hours", "jours": "days", "gens": "people",
    "attention": "attention", "valeur": "value",
    "nouveau": "new", "immediatement": "instantly", "premieres": "first",
    "algorithme": "algorithm", "hebdomadaire": "weekly",
    "hebdomadaires": "weekly", "ajoute": "adding", "ajouter": "adding",
    "personnelle": "personal", "personnel": "personal",
    "veux": "want", "voici": "here are", "change": "changes",
    "changent": "changes", "utilise": "using", "utiliser": "using",
    "sons": "sounds", "capte": "grab", "capter": "grab", "bat": "beats",
    "deux": "two", "trois": "three", "qualite": "quality",
    "visuel": "visual", "ajuste": "adjusting", "ajuster": "adjusting",
    "regulier": "regular", "immediat": "immediate",
}

# Words that carry no visual information once translated.
_FILLER = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "que",
    "qui", "pour", "avec", "dans", "sur", "sans", "pas", "plus", "tu", "je",
    "il", "elle", "on", "nous", "vous", "est", "sont", "as", "ai", "a",
    "the", "and", "with", "for", "your", "you", "s", "d", "l",
    "ton", "ta", "tes", "votre", "vos", "mon", "ma", "mes", "son", "sa",
    "ses", "ce", "cet", "cette", "ces", "ne", "se", "me", "te", "y", "en",
    "aussi", "mais", "donc", "car", "tout", "tous", "toute", "toutes",
    "bien", "moins", "tres", "meme", "comme", "faire", "fait", "va", "vas",
    "vont", "chaque", "reellement", "dans", "alors", "puis", "enfin",
    "aux", "cette", "quand", "avant", "apres",
}


def translate(text: str) -> str:
    """Translate the known French terms in `text` to English, keep the rest.

    Matching is done on whole words, not substrings: "algorithme" must not be
    rewritten by the shorter "algo" entry. Multi-word entries are resolved
    first so "vue aerienne" wins over the standalone "vue".
    """
    norm = script_understanding._norm(text)
    for french in sorted((k for k in _TERMS if " " in k), key=len, reverse=True):
        norm = re.sub(rf"\b{re.escape(french)}\b", f" {_TERMS[french]} ", norm)
    out: list[str] = []
    for word in re.split(r"[^a-z0-9]+", norm):
        if not word:
            continue
        if word in _TERMS:
            out.append(_TERMS[word])
        elif word not in _FILLER and len(word) > 1:
            out.append(word)
    return " ".join(out)


def prompt_for(text: str, topic: str = "", index: int = 0) -> str:
    """One English video prompt for one scene of the script."""
    place = script_understanding.detect_place(text)
    action = script_understanding.detect_action(text)
    lead: list[str] = []
    used: set[str] = set()
    for value in (translate(place) if place else "",
                  translate(action) if action else "",
                  translate(topic)):
        if value:
            lead.append(value)
            used.update(value.lower().split())
    # A translated scene repeats the place and topic words the lead already
    # named; dropping them leaves the model budget for the actual subject.
    scene = " ".join(w for w in translate(text).split() if w.lower() not in used)
    parts = list(lead)
    if scene:
        parts.append(" ".join(scene.split()[:MAX_SCENE_WORDS]))
    if not parts:
        parts.append("abstract cinematic background")
    shot = translate(visuals.VISUAL_MODIFIERS[index % len(visuals.VISUAL_MODIFIERS)])
    if shot:
        parts.append(shot)
    parts.append(STYLE_SUFFIX)
    # Drop an exact repeated chunk ("cinematic" often comes both from the
    # shot modifier and from the style suffix).
    seen: set[str] = set()
    chunks: list[str] = []
    for part in parts:
        for chunk in part.split(","):
            chunk = chunk.strip()
            if chunk and chunk.lower() not in seen:
                seen.add(chunk.lower())
                chunks.append(chunk)
    return ", ".join(chunks)


def prompts_for(script: str, topic: str = "", duration: float = 45.0,
                max_scenes: int | None = None) -> dict:
    """Video-model prompts for a whole reel script.

    Returns the prompt list plus the negative prompt and the model settings a
    free GPU notebook should use, so the API and the notebook cannot drift.
    """
    texts = visuals.scene_texts_for(script, duration, cap=max_scenes)
    prompts = [prompt_for(t, topic=topic, index=i) for i, t in enumerate(texts)]
    return {
        "prompts": prompts,
        "negative_prompt": NEGATIVE_PROMPT,
        "count": len(prompts),
        "settings": {
            "height": 480, "width": 832, "num_frames": 81, "fps": 16,
            "guidance": 5.0, "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "license": "Apache-2.0",
        },
    }
