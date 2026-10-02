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
from dataclasses import dataclass

import requests

from pipeline import tts

SYSTEM = (
    "Tu es un scénariste de vidéos courtes verticales (TikTok, Reels, Shorts). "
    "Tu écris en français, ton direct, phrases courtes, une idée par phrase. "
    "Tu commences par une accroche forte et tu termines par une question."
)

DIALOGUE_SYSTEM = (
    "Tu es un scénariste de vidéos courtes verticales (TikTok, Reels, Shorts). "
    "Tu écris des dialogues en français : répliques courtes et naturelles, une "
    "idée par réplique. Utilise un seul personnage si le sujet s'y prête (un "
    "monologue), ou deux pour un échange. "
    "Le dialogue commence par une accroche forte et finit par une question."
)

# Two default characters for the local (offline) dialogue writer. The first is
# voiced with the user's chosen voice, the second with a contrasting gender.
DEFAULT_CHARACTERS = ("Léo", "Mia")


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

    # Reasoning models (nemotron) emit `reasoning_content` *before* `content`, so
    # a budget sized for the visible answer alone can be fully consumed by the
    # reasoning and return `finish_reason="length"` with an empty `content`.
    budget = int(os.environ.get("NVIDIA_MAX_TOKENS", "4096"))
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


def _local_script(topic: str, duration: int, words: int | None = None) -> str:
    topic = topic.strip().rstrip(".!?")
    target_words = words or tts.estimate_words(duration)
    rng = random.Random(topic)

    sentences = [rng.choice(HOOKS).format(topic=topic)]
    body = BODY[:]
    rng.shuffle(body)
    # Cycle through the body pool so long targets (e.g. 60s) stay reachable;
    # a fixed 8-sentence cap used to leave long videos ~40% short.
    index = 0
    while len(" ".join(sentences).split()) < target_words - 12 and index < 400:
        sentences.append(body[index % len(body)])
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


_FENCE = re.compile(r"^\s*`{3,}\w*\s*$")


def _clean_dialogue(text: str) -> str:
    """Normalise a dialogue reply one line at a time.

    `_clean` collapses all whitespace, which merges every `Nom: réplique` line
    into a single line and leaves `_parse_dialogue` with one turn. Normalising
    per line keeps the structure, drops markdown code fences and tolerates
    bullets, quotes and stray blank lines.
    """
    lines: list[str] = []
    for raw in (text or "").splitlines():
        if _FENCE.match(raw):
            continue
        line = _clean(raw)
        if line:
            lines.append(line)
    return "\n".join(lines)


def write_script(
    topic: str,
    duration: int = 45,
    language: str = "français",
    style: str = "",
    tone: str = "",
    rate: str = "",
) -> str:
    """Generate a narration script sized for the target duration.

    `rate` (the edge-tts rate string, e.g. "-5%") keeps the word budget in sync
    with the chosen voice pace: a slower read needs fewer words for the same
    number of seconds.
    """
    words = tts.estimate_words(duration, rate or tts.DEFAULT_RATE)
    if _llm_available():
        extra = ""
        if style:
            extra += f"Style : {style}\n"
        if tone:
            extra += f"Ton : {tone}\n"
        prompt = (
            f"Sujet : {topic}\nLangue : {language}\n{extra}"
            f"Écris la narration d'une vidéo de {duration} secondes, environ {words} mots "
            f"({words * 2} signes environ). "
            "Pas de titres, pas de didascalies : uniquement le texte à lire à voix haute, "
            "en un seul paragraphe."
        )
        try:
            llm_text = _clean(_chat(prompt))
            # A truncated/empty completion would poison the duration fitting;
            # fall back to the local writer when the reply is far too short.
            if len(llm_text.split()) >= max(15, round(words * 0.5)):
                return llm_text
            print("[script] réponse LLM trop courte; génération locale")
        except Exception as exc:  # noqa: BLE001 - fall back to the local writer
            print(f"[script] LLM indisponible ({exc}); génération locale")
    return _clean(_local_script(topic, duration, words))


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


# ---------------------------------------------------------------------------
# Dialogue scripts (two characters, one voice each)
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    speaker: str
    text: str


# Local dialogue templates. `{a}` is the first character, `{b}` the second.
DIALOGUE_TEMPLATES = [
    [
        ("a", "Tu savais que {topic} pouvait changer ta vie ?"),
        ("b", "Vraiment ? Tout le monde en parle, mais personne n'explique comment."),
        ("a", "C'est justement le problème : on te donne la théorie, jamais la méthode."),
        ("b", "Alors donne-la moi. Par où je commence ?"),
        ("a", "Par une seule action, aujourd'hui, cinq minutes. Pas plus."),
        ("b", "Cinq minutes ? Ça me paraît trop simple pour marcher."),
        ("a", "C'est parce que c'est simple que ça marche. La régularité fait le reste."),
        ("b", "Et si j'oublie un jour ?"),
        ("a", "Tu reprends le lendemain, sans culpabiliser. C'est tout."),
        ("b", "D'accord. Je commence ce soir. Et toi, tu commences quand ?"),
    ],
    [
        ("a", "Arrête tout. Ce que je vais te dire sur {topic} va te surprendre."),
        ("b", "Vas-y, je t'écoute."),
        ("a", "La plupart des gens abandonnent au bout de trois jours."),
        ("b", "Pourquoi ? Ça n'a pas l'air si difficile."),
        ("a", "Parce qu'ils visent la perfection au lieu de la régularité."),
        ("b", "Donc je dois faire petit, mais tous les jours."),
        ("a", "Exactement. Petit et tous les jours bat grand et rare."),
        ("b", "Je note. Une action concrète, chaque jour."),
        ("a", "Et tu me diras dans une semaine ce que ça a changé."),
        ("b", "Marché conclu. Prêt à essayer avec moi ?"),
    ],
]


def _local_dialogue(topic: str, duration: int, words: int | None = None,
                    characters: tuple[str, str] | None = None) -> list[Turn]:
    topic = (topic or "ce sujet").strip().rstrip(".!?")
    target_words = words or tts.estimate_words(duration)
    first, second = characters or DEFAULT_CHARACTERS
    rng = random.Random(topic)
    template = rng.choice(DIALOGUE_TEMPLATES)

    # Fill up to the word budget instead of a fixed number of exchanges: a whole
    # template cycle is ~105 words, which used to make short dialogues (~30 s)
    # unreachable and left the duration-fitting loop unable to converge.
    turns: list[Turn] = []
    index = 0
    spoken = 0
    while spoken < target_words and index < 400:
        speaker_key, sentence = template[index % len(template)]
        text = sentence.format(topic=topic)
        turns.append(Turn(first if speaker_key == "a" else second, text))
        spoken += len(text.split())
        index += 1
    return turns


def _parse_dialogue(raw: str) -> list[Turn]:
    """Parse `Nom: réplique` lines, tolerating bullets, quotes and blank lines."""
    turns: list[Turn] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        match = re.match(r"^([\wÀ-ÿ'’ .\-]{1,24})\s*[:\u2013\u2014-]\s*(.+)$", line)
        if not match:
            continue
        speaker, text = match.group(1).strip(), match.group(2).strip()
        text = text.strip('"').strip()
        if speaker and text and len(text.split()) >= 2:
            turns.append(Turn(speaker, text))
    return turns


def _order_characters(turns: list[Turn], limit: int | None = None) -> list[str]:
    """First-appearance order, optionally capped at `limit` distinct characters."""
    order: list[str] = []
    for turn in turns:
        if turn.speaker not in order:
            order.append(turn.speaker)
    return order[:limit] if limit else order


def parse_dialogue(text: str) -> list[Turn]:
    """Public wrapper: parse `Nom: réplique` lines from user-edited text."""
    return _parse_dialogue(text)


def characters_of(turns: list[Turn], limit: int | None = None) -> list[str]:
    """Public wrapper: distinct character names in first-appearance order."""
    return _order_characters(turns, limit)


def write_dialogue_script(
    topic: str,
    duration: int = 45,
    language: str = "français",
    rate: str = "",
    characters: tuple[str, str] | None = None,
) -> tuple[list[Turn], str]:
    """Two-character dialogue sized for `duration`.

    Returns `(turns, plain_script)` where `plain_script` keeps the `Nom: ...`
    labels so a user can edit the dialogue by hand and have it re-parsed.
    """
    words = tts.estimate_words(duration, rate or tts.DEFAULT_RATE)

    turns: list[Turn] = []
    if _llm_available():
        names = characters or DEFAULT_CHARACTERS
        prompt = (
            f"Sujet : {topic}\nLangue : {language}\n"
            f"Personnages : {names[0]} et {names[1]}.\n"
            f"Écris un dialogue de {duration} secondes, environ {words} mots au total "
            f"({words * 2} signes environ), entre ces deux personnages.\n"
            "Format STRICT, une réplique par ligne :\n"
            f"{names[0]}: première réplique\n"
            f"{names[1]}: réponse\n"
            "Pas de didascalies, pas de titres, pas de narration : uniquement les répliques."
        )
        try:
            parsed = _parse_dialogue(_clean_dialogue(_chat(prompt, system=DIALOGUE_SYSTEM)))
            spoken = sum(len(turn.text.split()) for turn in parsed)
            if len(parsed) >= 4 and spoken >= max(15, round(words * 0.5)):
                turns = parsed
            else:
                print("[dialogue] réponse LLM inutilisable; génération locale")
        except Exception as exc:  # noqa: BLE001 - fall back to the local writer
            print(f"[dialogue] LLM indisponible ({exc}); génération locale")

    if not turns:
        turns = _local_dialogue(topic, duration, words, characters)

    # The cast follows the script: a monologue keeps its single voice, a
    # two-hander gets a man and a woman, and any extra name is folded into the
    # last distinct speaker so no line is dropped.
    order = _order_characters(turns)
    if order:
        fallback = order[-1]
        turns = [Turn(turn.speaker if turn.speaker in order else fallback, turn.text) for turn in turns]

    plain = "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)
    return turns, plain
