"""
Rendering Agent.
Orchestrates the diffusion pipeline, upscaler, interpolator, and video assembler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from .base_agent import BaseAgent, AgentContext
from app.backend.diffusion_engine.pipeline import (
    VideoDiffusionPipeline,
    GenerationRequest,
)
from app.backend.video_builder.assembler import VideoAssembler, AssemblyConfig
from app.backend.video_builder.upscaler import VideoUpscaler
from app.backend.video_builder.interpolator import FrameInterpolator
from config import settings


class RenderingAgent(BaseAgent):
    name = "Rendering"

    def __init__(
        self,
        pipeline: Optional[VideoDiffusionPipeline] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self._pipeline = pipeline or VideoDiffusionPipeline()
        self._assembler = VideoAssembler()
        self._upscaler: Optional[VideoUpscaler] = None
        self._interpolator: Optional[FrameInterpolator] = None
        self._progress_callback = progress_callback

    def process(self, ctx: AgentContext) -> Optional[str]:
        if ctx.scene_description is None or ctx.motion_plan is None:
            raise RuntimeError("Scene and motion must be planned before rendering")

        scene = ctx.scene_description
        ref_images = getattr(scene, "reference_images", [])

        style_suffix = settings.style_presets.get(ctx.style, "")

        request = GenerationRequest(
            prompt=scene.enhanced_prompt or scene.raw_prompt,
            negative_prompt=scene.negative_prompt,
            reference_images=ref_images,
            keyframes=ctx.keyframes,
            width=ctx.width,
            height=ctx.height,
            fps=ctx.fps,
            duration_s=ctx.duration_s,
            num_inference_steps=settings.diffusion.num_inference_steps,
            guidance_scale=settings.diffusion.guidance_scale,
            seed=ctx.seed,
            style_suffix=style_suffix,
            ip_adapter_scale=settings.diffusion.ip_adapter_scale,
        )
        ctx.generation_request = request

        # Generate frames
        result = self._pipeline.generate(request, self._progress_callback)
        ctx.generation_result = result
        frames = result.frames

        if self._progress_callback:
            self._progress_callback(-1, -1, "Post-processing frames…")

        # Optional interpolation (increase FPS)
        if ctx.use_interpolation:
            interp = FrameInterpolator(multiplier=2)
            frames = interp.interpolate(frames, self._progress_callback)
            result.fps = ctx.fps * 2

        # Optional upscaling
        if ctx.use_upscale:
            upscaler = VideoUpscaler(scale=2)
            frames = upscaler.upscale_frames(frames, self._progress_callback)
            w, h = frames[0].size if frames else (ctx.width * 2, ctx.height * 2)
        else:
            w, h = ctx.width, ctx.height

        # Assemble video
        config = AssemblyConfig(
            fps=result.fps,
            width=w,
            height=h,
            crf=settings.video.crf,
            preset=settings.video.preset,
        )
        video_path = self._assembler.assemble(
            frames,
            output_path=result.output_dir / "output.mp4",
            config=config,
            save_frames=ctx.save_frames,
        )
        ctx.final_video_path = video_path

        return (
            f"Rendered {len(frames)} frames → {video_path.name} "
            f"({w}×{h} @ {result.fps}fps)"
        )
