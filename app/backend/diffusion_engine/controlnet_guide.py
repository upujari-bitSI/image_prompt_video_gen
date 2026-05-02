"""
ControlNet Guidance Module.
Provides per-frame conditioning images (pose, depth, canny edges) that
steer the diffusion process for spatial / motion consistency.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from config import settings
from app.backend.motion_engine.keyframe_generator import Keyframe


ControlType = Literal["pose", "depth", "canny", "none"]


class ControlNetGuide:
    """
    Lazy-loaded ControlNet conditioning pipeline.

    Generates conditioning images from:
    - keyframe skeleton data (pose)
    - monocular depth estimation (depth)
    - canny edge detection (canny)
    """

    def __init__(self) -> None:
        self._depth_estimator: object | None = None
        self._pose_detector: object | None = None
        self._device = settings.gpu.device
        self._dtype = torch.float16 if settings.gpu.dtype != "fp32" else torch.float32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_conditioning_image(
        self,
        control_type: ControlType,
        source_image: Optional[Image.Image],
        keyframe: Optional[Keyframe],
        width: int,
        height: int,
    ) -> Optional[Image.Image]:
        if control_type == "none":
            return None

        if control_type == "pose":
            return self._get_pose_image(keyframe, width, height)

        if control_type == "depth" and source_image is not None:
            return self._get_depth_image(source_image, width, height)

        if control_type == "canny" and source_image is not None:
            return self._get_canny_image(source_image)

        return None

    def get_controlnet_pipeline_kwargs(
        self,
        conditioning_images: list[Image.Image],
        controlnet_scale: float = 0.7,
    ) -> dict:
        """Return extra kwargs to inject into the diffusion pipeline call."""
        if not conditioning_images:
            return {}
        return {
            "image": conditioning_images[0] if len(conditioning_images) == 1 else conditioning_images,
            "controlnet_conditioning_scale": controlnet_scale,
        }

    # ------------------------------------------------------------------
    # Pose conditioning
    # ------------------------------------------------------------------

    def _get_pose_image(
        self, keyframe: Optional[Keyframe], width: int, height: int
    ) -> Optional[Image.Image]:
        if keyframe is None or not keyframe.character_positions:
            return None

        # Use ControlNet-aux OpenPose renderer if available
        try:
            from controlnet_aux import OpenposeDetector  # type: ignore[import]
            if self._pose_detector is None:
                self._pose_detector = OpenposeDetector.from_pretrained(
                    "lllyasviel/ControlNet"
                )
        except ImportError:
            pass

        # Fallback: draw minimal skeleton from keyframe positions
        from app.backend.motion_engine.keyframe_generator import KeyframeGenerator
        kg = KeyframeGenerator()
        return kg.generate_pose_image(keyframe, width, height)

    # ------------------------------------------------------------------
    # Depth conditioning
    # ------------------------------------------------------------------

    def _get_depth_image(
        self, image: Image.Image, width: int, height: int
    ) -> Optional[Image.Image]:
        try:
            self._ensure_depth_estimator()
            if self._depth_estimator is None:
                return self._canny_depth_fallback(image)

            from transformers import pipeline as hf_pipeline  # type: ignore[import]
            result = self._depth_estimator(image)  # type: ignore[operator]
            depth_arr = np.array(result["depth"])
            depth_norm = ((depth_arr - depth_arr.min()) /
                          (depth_arr.max() - depth_arr.min() + 1e-8) * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_norm).convert("RGB")
            return depth_img.resize((width, height), Image.LANCZOS)
        except Exception as exc:
            logger.warning(f"Depth estimation failed: {exc}")
            return None

    def _ensure_depth_estimator(self) -> None:
        if self._depth_estimator is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            self._depth_estimator = hf_pipeline(
                "depth-estimation",
                model="Intel/dpt-hybrid-midas",
                device=0 if self._device == "cuda" else -1,
                torch_dtype=self._dtype,
            )
            logger.info("Depth estimator loaded")
        except Exception as exc:
            logger.warning(f"Depth estimator unavailable: {exc}")

    def _canny_depth_fallback(self, image: Image.Image) -> Image.Image:
        """Canny edges as a stand-in for depth when DPT is unavailable."""
        return self._get_canny_image(image)

    # ------------------------------------------------------------------
    # Canny conditioning
    # ------------------------------------------------------------------

    @staticmethod
    def _get_canny_image(image: Image.Image) -> Image.Image:
        try:
            import cv2
            img_np = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, threshold1=100, threshold2=200)
            return Image.fromarray(edges).convert("RGB")
        except ImportError:
            logger.warning("OpenCV not available for canny edge detection")
            return image
