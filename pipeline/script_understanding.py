"""Read a dialogue script to steer the reel: place, action, character look.

The reel should show *what the script says*: if the scene happens in a kitchen,
the background must be a kitchen; if a character walks, the character must walk.
This module turns the script text into a place, an action and (when a known
character is named) a signature look, using a small French keyword table — no
model download, no API call.

Honest limits: this is keyword detection, not image generation. A named anime
character gets its signature *colours and tag*, not a licensed likeness.
"""

from __future__ import annotations

import re
import unicodedata

# Order matters: the first place whose keywords appear wins, so the most
# specific settings are tested before generic ones.
PLACES: list[tuple[str, tuple[str, ...]]] = [
    ("cuisine", ("cuisine", "casserole", "four", "frigo", "cafetiere", "vaisselle", "poele")),
    ("chambre", ("chambre", "lit", "dort", "reveil", "oreiller", "couverture", "dodo")),
    ("salle de bain", ("salle de bain", "douche", "baignoire", "mirroir", "brosse a dents", "savon")),
    ("bureau", ("bureau", "travail", "ordinateur", "reunion", "patron", "collègue", "collegue")),
    ("restaurant", ("restaurant", "cafe", "bistrot", "table", "menu", "serveur", "assiette")),
    ("rue", ("rue", "dehors", "exterieur", "marche", "trottoir", "avenue", "boulevard")),
    ("voiture", ("voiture", "auto", "conduit", "volant", "route", "taxi", "bus")),
    ("parc", ("parc", "jardin", "herbe", "arbre", "banc", "foret")),
    ("plage", ("plage", "sable", "mer", "ocean", "vague")),
    ("ecole", ("ecole", "classe", "prof", "eleve", "cours")),
    ("hopital", ("hopital", "docteur", "medecin", "infirmiere", "clinique")),
    ("magasin", ("magasin", "boutique", "supermarche", "caisse", "rayon")),
    ("salle", ("salle", "canape", "salon", "television", "tele", "maison", "appartement")),
]

# Ordered actions; the first match wins.
ACTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("s'assied", ("assied", "asseoir", "assis", "s'assoit", "se pose")),
    ("marche", ("marche", "marcher", "avance", "se dirige", "va vers", "part", "rentre")),
    ("court", ("court", "cours", "s'enfuit", "se sauve", "galope")),
    ("dort", ("dort", "dormir", "s'endort", "reve")),
    ("mange", ("mange", "manger", "avale", "goute", "goute", "boit", "petit dejeuner", "dine")),
    ("crie", ("crie", "hurle", "s'ecrie", "gueule")),
    ("pleure", ("pleure", "larmes", "sanglote")),
    ("rit", ("rit", "rire", "rigole", "eclate de rire")),
    ("telephone", ("telephone", "appelle", "sonne", "texto", "sms", "message", "portable")),
    ("lit", ("lit", "regarde", "observe", "feuillete", "journal")),
    ("danse", ("danse", "bouge", "s'amuse", "fete")),
]

# Signature look for well-known characters: (hair, outfit, accent). The named
# character keeps its identity tag; the drawing stays an original caricature.
KNOWN_CHARACTERS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "goku": ((20, 20, 24), (235, 130, 40), (70, 170, 255)),
    "vegeta": ((25, 20, 60), (60, 90, 210), (240, 200, 70)),
    "naruto": ((245, 210, 60), (235, 120, 45), (70, 150, 230)),
    "luffy": ((20, 20, 24), (200, 40, 45), (245, 200, 70)),
    "pikachu": ((245, 205, 60), (245, 205, 60), (215, 60, 60)),
    "sailor moon": ((250, 235, 120), (240, 240, 250), (240, 120, 180)),
    "elsa": ((240, 240, 250), (120, 200, 235), (200, 230, 250)),
    "batman": ((20, 20, 25), (40, 45, 60), (240, 210, 70)),
    "spiderman": ((200, 40, 45), (40, 70, 200), (240, 240, 240)),
    "mario": ((60, 40, 30), (200, 40, 45), (70, 110, 220)),
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def detect_place(script: str) -> str:
    """Return the reel's setting, or "" when the script names none."""
    text = _norm(script)
    for place, keywords in PLACES:
        if any(k in text for k in keywords):
            return place
    return ""


def detect_action(script: str) -> str:
    """Return the dominant action, or "" when the script names none."""
    text = _norm(script)
    for action, keywords in ACTIONS:
        if any(k in text for k in keywords):
            return action
    return ""


def detect_characters(script: str) -> list[str]:
    """Return the speaker names, in first-appearance order."""
    names: list[str] = []
    for line in (script or "").splitlines():
        match = re.match(r"\s*([\wÀ-ÿ'’ .-]{1,24}?)\s*[:\u2013\u2014-]\s+", line)
        if match:
            name = match.group(1).strip()
            if name and name not in names:
                names.append(name)
    return names


def signature_look(name: str):
    """Known-character look for `name`, else None (avatar uses its own palette)."""
    return KNOWN_CHARACTERS.get(_norm(name).strip())


def describe(script: str) -> dict:
    """Everything the reel needs from the script, in one call."""
    return {
        "place": detect_place(script),
        "action": detect_action(script),
        "characters": detect_characters(script),
    }
