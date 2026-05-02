"""Scene Planner Agent — parses and enriches the scene description."""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent, AgentContext
from app.backend.scene_parser import SceneParser
from config import settings


class ScenePlannerAgent(BaseAgent):
    name = "ScenePlanner"

    def __init__(self) -> None:
        self._parser = SceneParser()

    def process(self, ctx: AgentContext) -> Optional[str]:
        scene = self._parser.parse(ctx.raw_prompt, ctx.motion_text)

        # Segment into shots if duration > 10 s
        if ctx.duration_s > 10:
            shots = self._parser.segment_into_shots(scene, int(ctx.duration_s))
            scene.shots = shots

        # Inject style suffix from preset
        style_suffix = settings.style_presets.get(ctx.style, "")
        if style_suffix:
            scene.enhanced_prompt = f"{scene.enhanced_prompt}, {style_suffix}"

        ctx.scene_description = scene
        return (
            f"Scene parsed: {len(scene.subjects)} subjects, "
            f"{len(scene.shots)} shots, camera={scene.camera.movement}"
        )
