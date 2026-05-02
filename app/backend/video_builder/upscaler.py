"""
Video Upscaler.
Applies Real-ESRGAN upscaling to individual frames (or the final video).
Falls back to bicubic interpolation if Real-ESRGAN is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from config import settings


class VideoUpscaler:
    """
    Upscales frames using Real-ESRGAN (4× or 2×).

    GPU-accelerated; automatically degrades gracefully to bicubic on CPU.
    """

    def __init__(self, scale: int = 2) -> None:
        self._scale = scale
        self._model: object | None = None
        self._device = settings.gpu.device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upscale_frames(
        self,
        frames: list[Image.Image],
        progress_callback: Optional[callable] = None,
    ) -> list[Image.Image]:
        self._ensure_model()
        out: list[Image.Image] = []

        for i, frame in enumerate(frames):
            if progress_callback:
                progress_callback(i, len(frames), f"Upscaling frame {i+1}/{len(frames)}")
            out.append(self._upscale_single(frame))

        return out

    def upscale_single(self, frame: Image.Image) -> Image.Image:
        self._ensure_model()
        return self._upscale_single(frame)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            self._model = self._load_realesrgan()
        except Exception as exc:
            logger.warning(f"Real-ESRGAN unavailable ({exc}); using bicubic upscale")
            self._model = "bicubic"

    def _load_realesrgan(self) -> object:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore[import]
        from realesrgan import RealESRGANer  # type: ignore[import]

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path=None,   # auto-download
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=(settings.gpu.dtype != "fp32"),
            device=self._device,
        )
        logger.info(f"Real-ESRGAN loaded (×4) on {self._device}")
        return upsampler

    def _upscale_single(self, frame: Image.Image) -> Image.Image:
        if self._model == "bicubic":
            w, h = frame.size
            return frame.resize(
                (w * self._scale, h * self._scale), Image.LANCZOS
            )

        try:
            import cv2
            img_np = np.array(frame.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            out_bgr, _ = self._model.enhance(img_bgr, outscale=self._scale)  # type: ignore[union-attr]
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(out_rgb)
        except Exception as exc:
            logger.warning(f"Real-ESRGAN enhance failed: {exc}; using bicubic")
            w, h = frame.size
            return frame.resize(
                (w * self._scale, h * self._scale), Image.LANCZOS
            )
