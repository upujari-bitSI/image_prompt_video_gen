from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import PipelineOrchestrator, RenderJob, JobStatus


def __getattr__(name: str):
    if name in ("PipelineOrchestrator", "RenderJob", "JobStatus"):
        from .orchestrator import PipelineOrchestrator, RenderJob, JobStatus
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PipelineOrchestrator", "RenderJob", "JobStatus"]
