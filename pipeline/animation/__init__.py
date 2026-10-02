"""Real character-animation engine (the "Animated Character" video type).

The stock pipeline (`pipeline.visuals`) makes a Ken-Burns slideshow: a photo
with zoom/pan. This module is a different, additive pipeline that produces a
genuinely animated cartoon: every frame is drawn anew (mouth viseme, blink,
head bob, arm swing, walk sway), so a static image is never the source of
motion.

Flow:
    character profile -> scene breakdown -> animation provider -> lip-sync
    -> scene clips -> ffmpeg assembly -> 1080x1920 MP4

Providers are pluggable via `ANIMATION_PROVIDER`. The local provider is the
free, CPU-only default and needs no API key or model download.
"""

from .character import CharacterProfile, profile_for
from .engine import AnimationFailed, AnimationResult, generate_animated_reel, write_animation_info
from .providers import AnimationProvider, AnimationProviderFactory, ProviderOutput
from .scenes import Scene, breakdown

__all__ = [
    "AnimationFailed",
    "AnimationProvider",
    "AnimationProviderFactory",
    "AnimationResult",
    "CharacterProfile",
    "ProviderOutput",
    "Scene",
    "breakdown",
    "generate_animated_reel",
    "profile_for",
    "write_animation_info",
]
