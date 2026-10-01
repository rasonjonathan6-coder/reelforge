"""Script and metadata generation.

Order of preference:
  1. A configured LLM (any OpenAI-compatible endpoint, or Gemini) if a key is set.
  2. A local, template-based writer that needs no network and no key.

The local writer is deliberately the default so the app never breaks when a
free third-party API changes its quota policy.
"""

from __future__ import annotations

import json
import os
import random
import re

import requests

SYSTEM = (
    "Tu es un scénariste de vidéos courtes verticales (TikTok, Reels, Shorts). "
    "Tu écris en français, ton direct, phrases courtes, une idée par phrase. "
    "Tu commences par une accroche forte et tu termines par une question."
)


# ---------------------------------------------------------------------------
# Remote LLM (optional): NVIDIA NIM, Gemini, or any OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _extract(payload: dict) -> str | None:
    try:
        content = payload["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    text = str(content).strip() if content else ""
    return text or None


def _llm_available() -> bool:
    return bool(
        os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )


def _chat_nvidia(prompt: str, system: str) -> str:
    """Call NVIDIA NIM.

    Reasoning models (e.g. nemotron) spend part of the budget on
    `reasoning_content`; if the budget runs out, `content` comes back empty.
    We therefore retry with a growing budget before giving up.
    """
    url = f"{os.environ.get('NVIDIA_BASE_URL', NVIDIA_BASE_URL).rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"}
    body = {
        "model": os.environ.get("NVIDIA_MODEL", NVIDIA_DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.95,
    }

    budget = int(os.environ.get("NVIDIA_MAX_TOKENS", "2048"))
    last_reason = ""
    for _ in range(3):
        resp = requests.post(url, headers=headers, json={**body, "max_tokens": budget}, timeout=180)
        resp.raise_for_status()
        payload = resp.json()
        text = _extract(payload)
        if text:
            return text
        try:
            last_reason = payload["choices"][0].get("finish_reason", "")
        except (KeyError, IndexError):
            last_reason = ""
        budget *= 2

    raise RuntimeError(f"réponse vide du modèle NVIDIA (finish_reason={last_reason})")


def _chat_gemini(prompt: str, system: str) -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": os.environ["GEMINI_API_KEY"]},
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _chat_openai_compatible(prompt: str, system: str) -> str:
    base = os.environ.get("LLM_BASE_URL")
    key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not base:
        if os.environ.get("GROQ_API_KEY"):
            base = "https://api.groq.com/openai/v1"
        elif os.environ.get("OPENROUTER_API_KEY"):
            base = "https://openrouter.ai/api/v1"
        else:
            raise RuntimeError("Aucun LLM configuré")

    resp = requests.post(
        f"{base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}"} if key else {},
        json={
            "model": os.environ.get("LLM_MODEL_NAME", "llama-3.1-8b-instant"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
        },
        timeout=90,
    )
    resp.raise_for_status()
    text = _extract(resp.json())
    if not text:
        raise RuntimeError("réponse vide du modèle")
    return text


def _chat(prompt: str, system: str = SYSTEM) -> str:
    if os.environ.get("NVIDIA_API_KEY"):
        return _chat_nvidia(prompt, system)
    if os.environ.get("GEMINI_API_KEY"):
        return _chat_gemini(prompt, system)
    return _chat_openai_compatible(prompt, system)


# ---------------------------------------------------------------------------
# Local writer (always available)
# ---------------------------------------------------------------------------
HOOKS = [
    "Personne ne te le dira, mais {topic} peut tout changer.",
    "Tu crois tout savoir sur {topic} ? Détrompe-toi.",
    "Voici ce que la plupart des gens ignorent sur {topic}.",
    "Arrête tout : ce que tu vas entendre sur {topic} va te surprendre.",
    "Deux minutes pour comprendre {topic}, une habitude pour la vie.",
]

BODY = [
    "Le problème, c'est que la plupart des gens s'y prennent mal.",
    "La bonne nouvelle, c'est que la méthode est simple.",
    "Commence petit : une action concrète, aujourd'hui.",
    "Répète chaque jour, même cinq minutes suffisent.",
    "Entoure-toi de personnes qui avancent dans la même direction.",
    "Note tes progrès, c'est ce qui te gardera motivé.",
    "Ne cherche pas la perfection, cherche la régularité.",
    "Ce qui compte, c'est ce que tu fais quand personne ne regarde.",
]

CLOSERS = [
    "Alors, quelle est ta première action aujourd'hui ?",
    "Et toi, tu commences quand ? Dis-le-moi en commentaire.",
    "Quelle astuce vas-tu tester dès ce soir ?",
    "Prêt à changer les choses ? Écris oui en commentaire.",
]


def _local_script(topic: str, duration: int) -> str:
    topic = topic.strip().rstrip(".!?")
    target_words = max(40, int(duration * 2.6))
    rng = random.Random(topic)

    sentences = [rng.choice(HOOKS).format(topic=topic)]
    body = BODY[:]
    rng.shuffle(body)
    index = 0
    while len(" ".join(sentences).split()) < target_words - 12 and index < len(body):
        sentences.append(body[index])
        index += 1
    sentences.append(rng.choice(CLOSERS))
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = text.replace("*", "").replace("_", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip('"')


def write_script(
    topic: str,
    duration: int = 45,
    language: str = "français",
    style: str = "",
    tone: str = "",
) -> str:
    """Generate a narration script sized for the target duration."""
    words = int(duration * 2.6)  # ~150 words/min spoken pace
    if _llm_available():
        extra = ""
        if style:
            extra += f"Style : {style}\n"
        if tone:
            extra += f"Ton : {tone}\n"
        prompt = (
            f"Sujet : {topic}\nLangue : {language}\n{extra}"
            f"Écris la narration d'une vidéo de {duration} secondes, environ {words} mots. "
            "Pas de titres, pas de didascalies : uniquement le texte à lire à voix haute, "
            "en un seul paragraphe."
        )
        try:
            return _clean(_chat(prompt))
        except Exception as exc:  # noqa: BLE001 - fall back to the local writer
            print(f"[script] LLM indisponible ({exc}); génération locale")
    return _clean(_local_script(topic, duration))


STOPWORDS = {
    "avec", "pour", "dans", "cette", "cela", "plus", "mais", "tout", "sans",
    "sont", "être", "faire", "votre", "vous", "nous", "leur", "quand", "alors",
    "comme", "aussi", "peut", "dont", "très", "bien", "juste", "même", "tous",
    "elle", "ils", "elles", "quoi", "chez", "vers", "deja", "déjà", "va", "les",
    "des", "une", "est", "que", "qui", "sur", "par", "pas", "son", "ses", "tes",
    "cest",
}


def _local_metadata(script: str, topic: str) -> dict:
    first = re.split(r"[.!?]", script.strip())[0].strip()
    title = (first or topic or "Vidéo")[:90]

    words = re.findall(r"[A-Za-zÀ-ÿ]{5,}", script.lower())
    counts: dict[str, int] = {}
    for word in words:
        if word not in STOPWORDS:
            counts[word] = counts.get(word, 0) + 1
    keywords = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:5]
    hashtags = keywords + ["shorts", "reels", "pourtoi", "viral"]
    seen: set[str] = set()
    hashtags = [h for h in hashtags if not (h in seen or seen.add(h))][:8]

    return {
        "title": title,
        "description": script[:220],
        "hashtags": hashtags,
        "thumbnail_prompt": f"cinematic vertical poster about {topic or title}, dramatic lighting",
    }


def write_metadata(script: str, topic: str) -> dict:
    """Title, description, hashtags and a thumbnail prompt.

    Never raises: falls back to local heuristics so production is never blocked.
    """
    data: dict = {}
    if _llm_available():
        prompt = (
            f"Sujet : {topic}\nScript : {script}\n\n"
            "Renvoie STRICTEMENT un JSON valide avec les clés : "
            '"title" (max 90 caractères), "description" (2 phrases), '
            '"hashtags" (tableau de 8 hashtags sans le #), '
            '"thumbnail_prompt" (prompt d\'image en anglais). '
            "Aucun texte hors du JSON."
        )
        try:
            raw = _chat(prompt, system="Tu réponds uniquement par du JSON valide.")
            match = re.search(r"\{.*\}", raw, flags=re.S)
            data = json.loads(match.group(0)) if match else {}
        except Exception as exc:  # noqa: BLE001
            print(f"[metadata] LLM indisponible ({exc}); métadonnées locales")

    fallback = _local_metadata(script, topic)
    for key, value in fallback.items():
        if not data.get(key):
            data[key] = value
    return data
