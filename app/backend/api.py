"""
FastAPI REST Backend.
Exposes the pipeline as a JSON API for headless / programmatic use.

Endpoints:
  POST /api/jobs                — submit a render job
  GET  /api/jobs/{id}           — poll job status
  GET  /api/jobs                — list all jobs
  GET  /api/jobs/{id}/download  — download the output video
  POST /api/assets              — upload + register an asset
  GET  /api/assets              — list all registered assets
  DELETE /api/assets/{label}   — remove an asset
  GET  /api/health              — healthcheck + GPU info
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from config import settings
from app.backend.agents.base_agent import AgentContext
from app.backend.agents.orchestrator import PipelineOrchestrator, JobStatus

# Shared orchestrator instance (injected at startup from main.py)
_orchestrator: Optional[PipelineOrchestrator] = None


def get_orchestrator() -> PipelineOrchestrator:
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialised")
    return _orchestrator


def create_app(orchestrator: PipelineOrchestrator) -> FastAPI:
    global _orchestrator
    _orchestrator = orchestrator

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        docs_url="/api/docs",
    )

    # ── Request schemas ───────────────────────────────────────────────

    class JobRequest(BaseModel):
        prompt: str
        motion_text: str = ""
        style: str = "cinematic"
        duration_s: float = Field(10.0, ge=3, le=120)
        fps: int = Field(24, ge=12, le=60)
        width: int = Field(1280, ge=128, le=3840)
        height: int = Field(720, ge=128, le=2160)
        seed: int = 42
        use_upscale: bool = False
        use_interpolation: bool = False
        save_frames: bool = False

    class JobStatusResponse(BaseModel):
        job_id: str
        status: str
        progress: int
        stage: str
        elapsed_s: float
        error: str
        quality_score: float
        agent_logs: list[str]
        output_path: Optional[str]

    # ── Helpers ───────────────────────────────────────────────────────

    def _job_to_response(job) -> JobStatusResponse:
        ctx = job.context
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status.value,
            progress=job.progress,
            stage=job.stage,
            elapsed_s=job.elapsed_s,
            error=job.error,
            quality_score=ctx.quality_score if ctx else 0,
            agent_logs=ctx.agent_logs if ctx else [],
            output_path=str(job.output_path) if job.output_path else None,
        )

    # ── Routes ────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health():
        info: dict = {"status": "ok", "version": settings.version}
        try:
            info["cuda"] = torch.cuda.is_available()
            if info["cuda"]:
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )
        except Exception:
            pass
        return info

    @app.post("/api/jobs", status_code=202)
    async def submit_job(req: JobRequest):
        orch = get_orchestrator()
        ctx = AgentContext(
            raw_prompt=req.prompt,
            motion_text=req.motion_text,
            duration_s=req.duration_s,
            fps=req.fps,
            width=req.width,
            height=req.height,
            seed=req.seed,
            style=req.style,
            use_upscale=req.use_upscale,
            use_interpolation=req.use_interpolation,
            save_frames=req.save_frames,
        )
        job = orch.submit(ctx)
        return {"job_id": job.job_id, "status": job.status.value}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        job = get_orchestrator().get_job(job_id)
        if job is None:
            raise HTTPException(404, f"Job {job_id} not found")
        return _job_to_response(job)

    @app.get("/api/jobs")
    async def list_jobs():
        return [_job_to_response(j) for j in get_orchestrator().list_jobs()]

    @app.get("/api/jobs/{job_id}/download")
    async def download_job(job_id: str):
        job = get_orchestrator().get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(400, f"Job not completed (status={job.status})")
        if not job.output_path or not job.output_path.exists():
            raise HTTPException(500, "Output file missing")
        return FileResponse(
            str(job.output_path),
            media_type="video/mp4",
            filename=f"video_{job_id}.mp4",
        )

    @app.post("/api/assets")
    async def upload_asset(
        label: str = Form(...),
        role: str = Form("character"),
        tags: str = Form(""),
        files: list[UploadFile] = File(...),
    ):
        orch = get_orchestrator()
        images: list[Image.Image] = []
        for upload in files:
            data = await upload.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            images.append(img)

        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        orch.asset_library.register(label, images, role=role, tags=tag_list)
        return {"label": label, "role": role, "n_images": len(images)}

    @app.get("/api/assets")
    async def list_assets():
        lib = get_orchestrator().asset_library
        result = []
        for lbl in lib.list_assets():
            asset = lib._assets[lbl]  # type: ignore[attr-defined]
            result.append({
                "label": lbl,
                "role": asset["role"],
                "n_images": len(asset["images"]),
                "tags": asset["tags"],
            })
        return result

    @app.delete("/api/assets/{label}")
    async def delete_asset(label: str):
        lib = get_orchestrator().asset_library
        if label not in lib._assets:  # type: ignore[attr-defined]
            raise HTTPException(404, f"Asset '{label}' not found")
        del lib._assets[label]  # type: ignore[attr-defined]
        return {"deleted": label}

    return app
