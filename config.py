"""
Global configuration — all tuneable constants live here.
RTX 5090 defaults: bf16, xformers, 24 GB VRAM headroom.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent
APP_DIR = ROOT_DIR / "app"
ASSETS_DIR = APP_DIR / "assets"
OUTPUTS_DIR = APP_DIR / "outputs"
MODELS_CACHE_DIR = ROOT_DIR / ".model_cache"

for _d in (ASSETS_DIR / "characters", ASSETS_DIR / "objects", ASSETS_DIR / "backgrounds",
           OUTPUTS_DIR, MODELS_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class GPUConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GPU_", extra="ignore")

    device: str = "cuda"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    enable_xformers: bool = True
    enable_attention_slicing: bool = False   # xformers supersedes this on RTX 5090
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = True
    compile_model: bool = False              # torch.compile — set True for throughput
    max_batch_size: int = 4
    vram_limit_gb: float = 20.0             # leave 4 GB headroom on 24 GB card


class DiffusionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DIFFUSION_", extra="ignore")

    # AnimateDiff base checkpoint — Realistic Vision is public, stable, and
    # widely cached on HF CDN; reliable for first-run downloads.
    base_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-3"

    # Stable Video Diffusion (image-to-video path)
    svd_model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"

    # IP-Adapter for character consistency
    ip_adapter_model_id: str = "h94/IP-Adapter"
    ip_adapter_scale: float = 0.6

    # ControlNet (pose / depth)
    controlnet_pose_id: str = "lllyasviel/control_v11p_sd15_openpose"
    controlnet_depth_id: str = "lllyasviel/control_v11f1p_sd15_depth"

    # Generation defaults
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    num_frames_per_chunk: int = 16     # AnimateDiff window size
    overlap_frames: int = 4            # for temporal blending between chunks

    # Upscaling
    realesrgan_model: str = "RealESRGAN_x4plus"


class VideoConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIDEO_", extra="ignore")

    default_duration_s: int = 10
    default_fps: int = 24
    default_resolution: tuple[int, int] = (1280, 720)
    max_duration_s: int = 120
    output_format: str = "mp4"
    crf: int = 18          # ffmpeg quality (lower = better)
    preset: str = "slow"


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: Literal["anthropic", "openai", "ollama", "local"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.3
    max_tokens: int = 1024
    # validation_alias lets pydantic-settings read ANTHROPIC_API_KEY directly,
    # bypassing the LLM_ prefix for this one field.
    api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",   # APP_HOST → host, APP_PORT → port, APP_DEBUG → debug
        extra="ignore",      # silently drop ANTHROPIC_API_KEY, GPU_*, etc. at this level
    )

    app_name: str = "AI Video Gen Studio"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False

    gpu: GPUConfig = Field(default_factory=GPUConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Style presets map to positive-prompt suffixes
    style_presets: dict[str, str] = {
        "cinematic": "cinematic lighting, anamorphic lens, film grain, 8K, photorealistic",
        "anime": "anime style, vibrant colors, cel shading, sharp lines, studio ghibli quality",
        "pixar": "Pixar 3D animation, soft lighting, subsurface scattering, high detail",
        "realistic": "hyperrealistic, RAW photo, DSLR, detailed skin texture, natural lighting",
        "toon": "toon shading, bold outlines, flat colors, cartoon style",
        "stylized": "painterly, concept art, dynamic composition, vibrant palette",
    }


settings = AppConfig()
