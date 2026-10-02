"""CharacterProfile: the single, stable description of a character.

A profile is built once per video and reused by every scene, so the character
stays visually consistent instead of being redesigned scene by scene. The
palette (skin/hair/outfit/eye) comes from `pipeline.avatars`, which keeps a
signature look for known names and a deterministic one otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pipeline import avatars, script_understanding

VISUAL_STYLES = ("anime", "cartoon_2d", "cartoon_3d", "stylized")


@dataclass
class CharacterProfile:
    """Stable identity of one character across the whole video."""

    id: str
    name: str
    gender: str = "neutral"
    age_range: str = "adult"
    visual_style: str = "anime"
    body_type: str = "average"
    skin_style: str = "flat"
    hair: tuple[int, int, int] = (40, 34, 36)
    clothes: tuple[int, int, int] = (70, 120, 200)
    accessories: list[str] = field(default_factory=list)
    personality: str = "neutral"
    voice_id: str = ""
    seed: int = 0
    visual_prompt: str = ""
    eye: tuple[int, int, int] = (86, 170, 240)

    def palette(self) -> dict:
        """The exact colours every scene must reuse for this character."""
        return {
            "skin": self.skin_style if isinstance(self.skin_style, tuple) else (250, 214, 178),
            "hair": self.hair,
            "outfit": self.clothes,
            "eye": self.eye,
        }

    def as_dict(self) -> dict:
        return asdict(self)


def profile_for(name: str, voice_id: str = "", visual_style: str = "anime") -> CharacterProfile:
    """Build the one profile a video will reuse for `name`.

    The palette is derived once here (deterministic per name), so calling this
    for the same character always yields the same look.
    """
    pal = avatars.palette_for(name)
    seed = abs(hash(name)) % 10_000
    return CharacterProfile(
        id=f"char_{seed:04d}",
        name=name,
        gender=_guess_gender(name),
        visual_style=visual_style if visual_style in VISUAL_STYLES else "anime",
        skin_style=pal["skin"],
        hair=pal["hair"],
        clothes=pal["outfit"],
        eye=pal["eye"],
        voice_id=voice_id,
        seed=seed,
        accessories=["signature"] if script_understanding.signature_look(name) else [],
        visual_prompt=f"{name}, {visual_style} cartoon character, consistent look",
    )


def _guess_gender(name: str) -> str:
    return "neutral"
