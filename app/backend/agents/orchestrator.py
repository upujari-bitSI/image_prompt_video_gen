"""
Pipeline Orchestrator.
Chains all agents into a single render job with progress tracking,
error recovery, and structured logging.

Job lifecycle:
  QUEUED → RUNNING → COMPLETED | FAILED
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from .base_agent import AgentContext, AgentResult
from .scene_planner import ScenePlannerAgent
from .character_consistency_agent import CharacterConsistencyAgent
from .motion_director import MotionDirectorAgent
from .rendering_agent import RenderingAgent
from .quality_checker import QualityCheckerAgent
from app.backend.diffusion_engine.character_consistency import AssetLibrary
from app.backend.diffusion_engine.pipeline import VideoDiffusionPipeline


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RenderJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.QUEUED
    context: Optional[AgentContext] = None
    results: list[AgentResult] = field(default_factory=list)
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0

    # Progress (0–100)
    progress: int = 0
    stage: str = "Queued"

    @property
    def elapsed_s(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def output_path(self) -> Optional[Path]:
        return self.context.final_video_path if self.context else None


class PipelineOrchestrator:
    """
    Single shared orchestrator — maintains a lazy singleton pipeline
    to avoid repeated model loads between jobs.
    """

    _STAGE_WEIGHTS = {
        "ScenePlanner": 5,
        "CharacterConsistency": 10,
        "MotionDirector": 10,
        "Rendering": 70,
        "QualityChecker": 5,
    }

    def __init__(self) -> None:
        self._asset_library = AssetLibrary()
        self._diffusion_pipeline = VideoDiffusionPipeline()
        self._lock = threading.Lock()
        self._jobs: dict[str, RenderJob] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def asset_library(self) -> AssetLibrary:
        return self._asset_library

    def submit(
        self,
        ctx: AgentContext,
        on_progress: Optional[Callable[[RenderJob], None]] = None,
    ) -> RenderJob:
        job = RenderJob(context=ctx)
        self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job, on_progress),
            daemon=True,
        )
        thread.start()
        logger.info(f"Job {job.job_id} queued")
        return job

    def run_sync(
        self,
        ctx: AgentContext,
        on_progress: Optional[Callable[[RenderJob], None]] = None,
    ) -> RenderJob:
        """Blocking version for use from CLI or unit tests."""
        job = RenderJob(context=ctx)
        self._jobs[job.job_id] = job
        self._run_job(job, on_progress)
        return job

    def get_job(self, job_id: str) -> Optional[RenderJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[RenderJob]:
        return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _run_job(
        self,
        job: RenderJob,
        on_progress: Optional[Callable[[RenderJob], None]],
    ) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        def diffusion_progress(chunk: int, total: int, msg: str) -> None:
            if total > 0:
                render_pct = int(chunk / total * 100)
            else:
                render_pct = 50
            job.stage = msg
            # Map into 15%–85% of overall progress
            job.progress = 15 + int(render_pct * 0.70)
            if on_progress:
                on_progress(job)

        agents = [
            ScenePlannerAgent(),
            CharacterConsistencyAgent(self._asset_library),
            MotionDirectorAgent(),
            RenderingAgent(self._diffusion_pipeline, diffusion_progress),
            QualityCheckerAgent(),
        ]

        cumulative_weight = 0
        total_weight = sum(self._STAGE_WEIGHTS.values())

        with self._lock:
            for agent in agents:
                job.stage = agent.name
                result = agent.run(job.context)  # type: ignore[arg-type]
                job.results.append(result)

                cumulative_weight += self._STAGE_WEIGHTS.get(agent.name, 5)
                job.progress = min(99, int(cumulative_weight / total_weight * 100))

                if on_progress:
                    on_progress(job)

                if not result.success:
                    job.status = JobStatus.FAILED
                    job.error = result.message
                    job.finished_at = time.time()
                    logger.error(f"Job {job.job_id} failed at {agent.name}: {result.message}")
                    return

        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.stage = "Done"
        job.finished_at = time.time()

        if on_progress:
            on_progress(job)

        logger.success(
            f"Job {job.job_id} completed in {job.elapsed_s:.1f}s → "
            f"{job.output_path}"
        )
