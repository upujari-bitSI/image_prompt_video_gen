"""
Character Consistency Agent.
Wires reference images to scene subjects and primes the generation
request with the right IP-Adapter embeddings / prompt tokens.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from .base_agent import BaseAgent, AgentContext
from app.backend.diffusion_engine.character_consistency import AssetLibrary


class CharacterConsistencyAgent(BaseAgent):
    name = "CharacterConsistency"

    def __init__(self, asset_library: AssetLibrary) -> None:
        self._library = asset_library

    def process(self, ctx: AgentContext) -> Optional[str]:
        if ctx.scene_description is None:
            raise RuntimeError("ScenePlannerAgent must run first")

        characters = [
            s for s in ctx.scene_description.subjects
            if s.role == "character"
        ]

        matched = 0
        reference_images = []

        for subject in characters:
            # Try to match subject name to registered asset
            label = subject.asset_label or subject.name
            images = self._library.get_images(label)

            if images:
                reference_images.extend(images[:2])  # max 2 angles per char
                subject.description = (
                    f"{subject.description}, {label} consistent appearance"
                )
                matched += 1
                logger.debug(f"Character '{label}' matched to {len(images)} reference(s)")
            else:
                # Fallback: inject textual identity anchor
                token = self._library._encoder.to_prompt_tokens(  # type: ignore[attr-defined]
                    subject.name, ctx.style
                )
                ctx.scene_description.enhanced_prompt += f", {token}"
                logger.debug(f"Character '{subject.name}' using text fallback")

        # Store reference images on ctx for GenerationRequest
        ctx.scene_description.reference_images = reference_images  # type: ignore[attr-defined]

        return f"Character consistency: {matched}/{len(characters)} characters image-matched"
