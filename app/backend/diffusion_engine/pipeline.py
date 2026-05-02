"""
Frame Generation Engine.
Wraps AnimateDiff (text-to-video) and Stable Video Diffusion (image-to-video)
into a unified, chunked generation pipeline optimised for RTX 5090.

Key design decisions:
- Chunked generation (16-frame windows with overlap) to handle arbitrary durations.
- Optional IP-Adapter for character-image conditioning.
- Optional ControlNet (pose/depth) guidance per keyframe.
- Mixed precision (bf16) + xformers attention.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Literal, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from config import settings, OUTPUTS_DIR
from app.backend.motion_engine.keyframe_generator import Keyframe


# ---------------------------------------------------------------------------
# Request / Response types
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = (
        "blurry, watermark, text, deformed, ugly, extra limbs, duplicate, "
        "bad anatomy, flickering, low quality, nsfw"
    )

    # Image conditioning
    reference_images: list[Image.Image] = field(default_factory=list)
    reference_video_path: Optional[Path] = None

    # Motion / keyframe data
    keyframes: list[Keyframe] = field(default_factory=list)

    # Generation parameters
    width: int = 1280
    height: int = 720
    fps: int = 24
    duration_s: float = 10.0
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: int = 42

    # Style
    style_suffix: str = ""
    ip_adapter_scale: float = 0.6
    use_controlnet: bool = False

    # Pipeline selection
    mode: Literal["animatediff", "svd", "auto"] = "auto"


@dataclass
class GenerationResult:
    frames: list[Image.Image]
    fps: int
    duration_s: float
    seed: int
    output_dir: Path
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VideoDiffusionPipeline:
    """
    Lazy-loaded video diffusion pipeline.

    Models are loaded on first call to generate() to avoid startup delay.
    Supports AnimateDiff for general text-to-video and SVD for image-conditioned
    generation.
    """

    def __init__(self) -> None:
        self._pipe: object | None = None
        self._ip_adapter_loaded: bool = False
        self._dtype = self._resolve_dtype()
        self._device = settings.gpu.device
        self._mode: str | None = None
        self._configure_hf_token()

    @staticmethod
    def _configure_hf_token() -> None:
        """Pass HF_TOKEN to huggingface_hub if set, for authenticated downloads."""
        import os
        token = os.environ.get("HF_TOKEN", "")
        if token:
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
                logger.info("HuggingFace Hub: authenticated")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> GenerationResult:
        """
        Generate a video from a GenerationRequest.
        Returns a GenerationResult with PIL frames.
        """
        mode = self._select_mode(request)
        self._ensure_pipeline(mode, request)

        prompt = request.prompt
        if request.style_suffix:
            prompt = f"{prompt}, {request.style_suffix}"

        total_frames = int(request.duration_s * request.fps)
        chunk_size = settings.diffusion.num_frames_per_chunk
        overlap = settings.diffusion.overlap_frames

        logger.info(
            f"Generating {total_frames} frames ({request.duration_s}s @ {request.fps}fps) "
            f"mode={mode}, seed={request.seed}"
        )

        all_frames: list[Image.Image] = []
        generator = torch.Generator(device=self._device).manual_seed(request.seed)

        if mode == "animatediff":
            all_frames = self._generate_animatediff_chunked(
                request, prompt, total_frames, chunk_size, overlap,
                generator, progress_callback
            )
        elif mode == "svd":
            all_frames = self._generate_svd_chunked(
                request, total_frames, chunk_size, overlap,
                generator, progress_callback
            )

        output_dir = OUTPUTS_DIR / f"run_{int(time.time())}_{request.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.success(f"Generated {len(all_frames)} frames → {output_dir}")

        return GenerationResult(
            frames=all_frames,
            fps=request.fps,
            duration_s=request.duration_s,
            seed=request.seed,
            output_dir=output_dir,
            metadata={"mode": mode, "prompt": prompt, "steps": request.num_inference_steps},
        )

    def unload(self) -> None:
        """Release GPU memory."""
        self._pipe = None
        self._ip_adapter_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipeline unloaded, VRAM released")

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def _ensure_pipeline(self, mode: str, request: GenerationRequest) -> None:
        if self._pipe is not None and self._mode == mode:
            return

        self.unload()
        logger.info(f"Loading {mode} pipeline (dtype={self._dtype})…")

        if mode == "animatediff":
            self._pipe = self._load_animatediff(request)
        elif mode == "svd":
            self._pipe = self._load_svd()

        self._mode = mode

    def _load_animatediff(self, request: GenerationRequest) -> object:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

        adapter = MotionAdapter.from_pretrained(
            settings.diffusion.motion_adapter_id,
            torch_dtype=self._dtype,
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            settings.diffusion.base_model_id,
            motion_adapter=adapter,
            torch_dtype=self._dtype,
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
        )

        self._apply_gpu_optimisations(pipe)

        if request.reference_images and request.ip_adapter_scale > 0:
            self._load_ip_adapter(pipe)

        pipe.to(self._device)
        return pipe

    def _load_svd(self) -> object:
        from diffusers import StableVideoDiffusionPipeline

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            settings.diffusion.svd_model_id,
            torch_dtype=self._dtype,
            variant="fp16",
        )
        self._apply_gpu_optimisations(pipe)
        pipe.to(self._device)
        return pipe

    def _apply_gpu_optimisations(self, pipe: object) -> None:
        if settings.gpu.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()  # type: ignore[attr-defined]
                logger.debug("xformers enabled")
            except Exception as exc:
                logger.warning(f"xformers unavailable: {exc}")

        if settings.gpu.enable_vae_tiling:
            try:
                pipe.enable_vae_tiling()  # type: ignore[attr-defined]
            except AttributeError:
                pass

        if settings.gpu.enable_vae_slicing:
            try:
                pipe.enable_vae_slicing()  # type: ignore[attr-defined]
            except AttributeError:
                pass

        if settings.gpu.compile_model:
            try:
                pipe.unet = torch.compile(  # type: ignore[attr-defined]
                    pipe.unet, mode="reduce-overhead", fullgraph=True
                )
                logger.debug("torch.compile enabled on UNet")
            except Exception as exc:
                logger.warning(f"torch.compile failed: {exc}")

    def _load_ip_adapter(self, pipe: object) -> None:
        try:
            pipe.load_ip_adapter(  # type: ignore[attr-defined]
                settings.diffusion.ip_adapter_model_id,
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            pipe.set_ip_adapter_scale(settings.diffusion.ip_adapter_scale)  # type: ignore[attr-defined]
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter loaded for character conditioning")
        except Exception as exc:
            logger.warning(f"IP-Adapter load failed: {exc}")

    # ------------------------------------------------------------------
    # Generation — AnimateDiff chunked
    # ------------------------------------------------------------------

    def _generate_animatediff_chunked(
        self,
        request: GenerationRequest,
        prompt: str,
        total_frames: int,
        chunk_size: int,
        overlap: int,
        generator: torch.Generator,
        progress_callback: Optional[Callable],
    ) -> list[Image.Image]:
        from diffusers import AnimateDiffPipeline

        pipe: AnimateDiffPipeline = self._pipe  # type: ignore[assignment]
        all_frames: list[Image.Image] = []

        stride = chunk_size - overlap
        starts = list(range(0, total_frames, stride))

        for chunk_idx, start in enumerate(starts):
            end = min(start + chunk_size, total_frames)
            n_frames = end - start

            if progress_callback:
                progress_callback(
                    chunk_idx, len(starts),
                    f"Rendering chunk {chunk_idx+1}/{len(starts)} (frames {start}–{end})"
                )

            # Build per-chunk prompt from keyframe modifiers
            chunk_prompt = self._get_chunk_prompt(prompt, request.keyframes, start, end)

            kwargs: dict = dict(
                prompt=chunk_prompt,
                negative_prompt=request.negative_prompt,
                num_frames=n_frames,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )

            if self._ip_adapter_loaded and request.reference_images:
                kwargs["ip_adapter_image"] = request.reference_images

            output = pipe(**kwargs)
            chunk_frames: list[Image.Image] = output.frames[0]  # type: ignore[index]

            if all_frames and overlap > 0:
                # Blend overlapping frames with linear cross-fade
                chunk_frames = self._blend_overlap(
                    all_frames[-overlap:], chunk_frames[:overlap]
                ) + chunk_frames[overlap:]

            all_frames.extend(chunk_frames)

        return all_frames[:total_frames]

    # ------------------------------------------------------------------
    # Generation — SVD chunked
    # ------------------------------------------------------------------

    def _generate_svd_chunked(
        self,
        request: GenerationRequest,
        total_frames: int,
        chunk_size: int,
        overlap: int,
        generator: torch.Generator,
        progress_callback: Optional[Callable],
    ) -> list[Image.Image]:
        from diffusers import StableVideoDiffusionPipeline

        pipe: StableVideoDiffusionPipeline = self._pipe  # type: ignore[assignment]

        if not request.reference_images:
            raise ValueError("SVD mode requires at least one reference image")

        seed_image = request.reference_images[0].resize(
            (request.width, request.height), Image.LANCZOS
        )

        all_frames: list[Image.Image] = []
        stride = chunk_size - overlap
        starts = list(range(0, total_frames, stride))

        current_seed = seed_image

        for chunk_idx, start in enumerate(starts):
            end = min(start + chunk_size, total_frames)

            if progress_callback:
                progress_callback(
                    chunk_idx, len(starts),
                    f"SVD chunk {chunk_idx+1}/{len(starts)}"
                )

            output = pipe(
                image=current_seed,
                num_frames=end - start,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                generator=generator,
            )
            chunk_frames: list[Image.Image] = output.frames[0]  # type: ignore[index]

            if all_frames and overlap > 0:
                chunk_frames = self._blend_overlap(
                    all_frames[-overlap:], chunk_frames[:overlap]
                ) + chunk_frames[overlap:]

            all_frames.extend(chunk_frames)

            # Use last generated frame as seed for next chunk
            if chunk_frames:
                current_seed = chunk_frames[-1]

        return all_frames[:total_frames]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_mode(self, request: GenerationRequest) -> str:
        if request.mode != "auto":
            return request.mode
        if request.reference_images and not request.prompt:
            return "svd"
        return "animatediff"

    def _get_chunk_prompt(
        self,
        base_prompt: str,
        keyframes: list[Keyframe],
        start: int,
        end: int,
    ) -> str:
        mid = (start + end) // 2
        for kf in keyframes:
            if kf.frame_index == mid and kf.prompt_modifier:
                return f"{base_prompt}, {kf.prompt_modifier}"
        return base_prompt

    @staticmethod
    def _blend_overlap(
        tail: list[Image.Image],
        head: list[Image.Image],
    ) -> list[Image.Image]:
        """Linear alpha blend between the tail of the previous chunk and head of new."""
        n = min(len(tail), len(head))
        blended: list[Image.Image] = []
        for i in range(n):
            alpha = (i + 1) / (n + 1)
            t_arr = np.array(tail[i], dtype=np.float32)
            h_arr = np.array(head[i], dtype=np.float32)
            merged = (1 - alpha) * t_arr + alpha * h_arr
            blended.append(Image.fromarray(merged.astype(np.uint8)))
        return blended

    @staticmethod
    def _resolve_dtype() -> torch.dtype:
        if settings.gpu.dtype == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if settings.gpu.dtype in ("bf16", "fp16"):
            return torch.float16
        return torch.float32
