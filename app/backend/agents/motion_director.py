"""
Motion Director Agent.
Converts scene description + motion text into a MotionPlan and Keyframes.
"""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent, AgentContext
from app.backend.motion_engine import MotionPlanner
from app.backend.motion_engine.keyframe_generator import KeyframeGenerator


class MotionDirectorAgent(BaseAgent):
    name = "MotionDirector"

    def __init__(self) -> None:
        self._planner = MotionPlanner()
        self._kf_gen = KeyframeGenerator()

    def process(self, ctx: AgentContext) -> Optional[str]:
        if ctx.scene_description is None:
            raise RuntimeError("ScenePlannerAgent must run first")

        plan = self._planner.plan(
            scene=ctx.scene_description,
            motion_text=ctx.motion_text,
            duration_s=ctx.duration_s,
            fps=ctx.fps,
        )
        ctx.motion_plan = plan

        # Generate keyframes at every 4th frame (sparse sampling is enough)
        kf_indices = list(range(0, plan.total_frames, max(1, plan.fps // 6)))
        ctx.keyframes = self._kf_gen.generate(plan, kf_indices)

        return (
            f"Motion plan: {plan.total_frames} frames, "
            f"{len(ctx.keyframes)} keyframes, "
            f"intensity={plan.intensity:.1f}"
        )
