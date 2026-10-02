"""SceneBreakdown: turn a script/dialogue into short animated scenes.

Each scene carries everything the animation provider and the editor need:
who acts, what they do, the emotion, the camera move, the environment, the
line spoken, the voice, an animation prompt and whether lip-sync applies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pipeline import script_understanding

CAMERAS = ("wide", "medium", "close_up", "over_the_shoulder", "tracking", "static",
           "zoom_in", "zoom_out")

_EMOTIONS = {
    "joie": ("rire", "heureux", "content", "sourire", "super", "génial"),
    "colere": ("colère", "furieux", "énervé", "tais-toi", "idiot", "imbécile"),
    "surprise": ("surprise", "incroyable", "quoi", "sérieux", "oh"),
    "tristesse": ("triste", "pleure", "désolé", "dommage"),
}


@dataclass
class Scene:
    scene_id: int
    duration: float
    character_action: str = "parle"
    emotion: str = "neutre"
    camera: str = "medium"
    environment: str = ""
    dialogue: str = ""
    voice: str = ""
    speaker: str = ""
    animation_prompt: str = ""
    lip_sync_required: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def detect_emotion(text: str) -> str:
    low = (text or "").lower()
    for emotion, words in _EMOTIONS.items():
        if any(w in low for w in words):
            return emotion
    return "neutre"


def camera_for(action: str, text: str) -> str:
    """Camera move for a scene, driven by the action then the line."""
    if action in ("marche", "court"):
        return "tracking"
    if action in ("mange", "regarde", "prend"):
        return "close_up"
    if "!" in (text or ""):
        return "zoom_in"
    return "medium"


def breakdown(lines: list[tuple[str, str]], total_duration: float,
              voices: dict[str, str] | None = None,
              environment: str = "") -> list[Scene]:
    """Split `(speaker, line)` pairs into scenes sized to reach `total_duration`.

    Durations are planned from the line length, then scaled so the scenes add
    up to the requested length instead of being cut afterwards.
    """
    voices = voices or {}
    if not lines:
        return []
    raw = [max(1.5, len(line.split()) / 2.5) for _, line in lines]
    scale = total_duration / sum(raw) if total_duration and sum(raw) else 1.0
    scenes: list[Scene] = []
    for index, (speaker, line) in enumerate(lines):
        action = script_understanding.detect_action(line) or "parle"
        place = script_understanding.detect_place(line) or environment
        scenes.append(Scene(
            scene_id=index + 1,
            duration=round(raw[index] * scale, 2),
            character_action=action,
            emotion=detect_emotion(line),
            camera=camera_for(action, line),
            environment=place,
            dialogue=line.strip(),
            voice=voices.get(speaker, ""),
            speaker=speaker,
            animation_prompt=f"{speaker} {action}, {detect_emotion(line)}, {place or 'studio'}",
            lip_sync_required=bool(line.strip()),
        ))
    return scenes
