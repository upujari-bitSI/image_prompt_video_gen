# Lazy imports — torch/diffusers are only required when the GPU pipeline is
# actually used, not when other modules (e.g. mock tests) import subpackages.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import VideoDiffusionPipeline, GenerationRequest, GenerationResult
    from .character_consistency import CharacterConsistencyEncoder
    from .controlnet_guide import ControlNetGuide


def __getattr__(name: str):  # PEP 562 module-level __getattr__
    if name in ("VideoDiffusionPipeline", "GenerationRequest", "GenerationResult"):
        from .pipeline import VideoDiffusionPipeline, GenerationRequest, GenerationResult
        return locals()[name]
    if name == "CharacterConsistencyEncoder":
        from .character_consistency import CharacterConsistencyEncoder
        return CharacterConsistencyEncoder
    if name == "ControlNetGuide":
        from .controlnet_guide import ControlNetGuide
        return ControlNetGuide
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VideoDiffusionPipeline",
    "GenerationRequest",
    "GenerationResult",
    "CharacterConsistencyEncoder",
    "ControlNetGuide",
]
