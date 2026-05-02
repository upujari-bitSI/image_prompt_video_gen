"""
Base Agent interface.
All pipeline agents inherit from BaseAgent and implement process().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


@dataclass
class AgentContext:
    """Shared mutable context passed between all agents in a pipeline run."""
    raw_prompt: str
    motion_text: str = ""
    duration_s: float = 10.0
    fps: int = 24
    width: int = 1280
    height: int = 720
    seed: int = 42
    style: str = "cinematic"
    use_upscale: bool = False
    use_interpolation: bool = False
    save_frames: bool = False

    # Populated as agents run
    scene_description: Any = None       # SceneDescription
    motion_plan: Any = None             # MotionPlan
    keyframes: list = field(default_factory=list)
    generation_request: Any = None      # GenerationRequest
    generation_result: Any = None       # GenerationResult
    final_video_path: Any = None        # Path

    # Quality / diagnostics
    quality_score: float = 0.0
    quality_notes: list[str] = field(default_factory=list)
    agent_logs: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    success: bool
    agent_name: str
    elapsed_s: float
    message: str = ""
    data: Any = None


class BaseAgent(ABC):
    name: str = "BaseAgent"

    def run(self, ctx: AgentContext) -> AgentResult:
        start = time.perf_counter()
        logger.info(f"[{self.name}] starting…")
        try:
            result = self.process(ctx)
            elapsed = time.perf_counter() - start
            ctx.agent_logs.append(f"[{self.name}] OK ({elapsed:.1f}s)")
            logger.success(f"[{self.name}] done in {elapsed:.1f}s")
            return AgentResult(
                success=True,
                agent_name=self.name,
                elapsed_s=elapsed,
                message=result or "OK",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            ctx.agent_logs.append(f"[{self.name}] FAILED: {exc}")
            logger.error(f"[{self.name}] failed: {exc}")
            return AgentResult(
                success=False,
                agent_name=self.name,
                elapsed_s=elapsed,
                message=str(exc),
            )

    @abstractmethod
    def process(self, ctx: AgentContext) -> Optional[str]:
        """Mutate ctx in-place; return a status message or None."""
