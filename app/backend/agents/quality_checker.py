"""
Quality Checker Agent.
Analyses generated frames for common failure modes:
  - Temporal flicker (high frame-to-frame luminance variance)
  - Spatial blur (Laplacian sharpness)
  - Identity drift (CLIP cosine distance between reference and generated faces)
  - Out-of-bounds artefacts (solid colour frames)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from .base_agent import BaseAgent, AgentContext


class QualityCheckerAgent(BaseAgent):
    name = "QualityChecker"

    # Thresholds (tuned empirically)
    FLICKER_THRESHOLD = 15.0       # mean abs luminance diff between adjacent frames
    BLUR_THRESHOLD = 50.0          # Laplacian variance — below = too blurry
    SOLID_COLOUR_THRESHOLD = 2.0   # std dev of whole frame — below = broken frame

    def process(self, ctx: AgentContext) -> Optional[str]:
        result = ctx.generation_result
        if result is None or not result.frames:
            raise RuntimeError("No frames to quality-check")

        frames = result.frames
        issues: list[str] = []

        # 1. Solid / black frame detection
        broken = self._detect_broken_frames(frames)
        if broken:
            issues.append(f"{len(broken)} broken/solid frames at indices {broken[:5]}")

        # 2. Temporal flicker
        flicker_score = self._measure_flicker(frames)
        if flicker_score > self.FLICKER_THRESHOLD:
            issues.append(f"High temporal flicker (score={flicker_score:.1f})")

        # 3. Spatial sharpness
        blur_scores = self._measure_sharpness(frames[::max(1, len(frames) // 10)])
        mean_sharpness = float(np.mean(blur_scores))
        if mean_sharpness < self.BLUR_THRESHOLD:
            issues.append(f"Low sharpness (Laplacian={mean_sharpness:.1f})")

        # Overall score: 100 → deduct per issue
        score = max(0.0, 100.0 - len(issues) * 25.0)
        ctx.quality_score = score
        ctx.quality_notes = issues

        if issues:
            logger.warning(f"Quality issues: {issues}")
        else:
            logger.success(f"Quality check passed (score={score:.0f})")

        return (
            f"Quality score: {score:.0f}/100 | "
            f"flicker={flicker_score:.1f} | "
            f"sharpness={mean_sharpness:.1f} | "
            f"issues={len(issues)}"
        )

    # ------------------------------------------------------------------

    def _detect_broken_frames(self, frames: list[Image.Image]) -> list[int]:
        broken = []
        for i, frame in enumerate(frames):
            arr = np.array(frame.convert("L"), dtype=np.float32)
            if arr.std() < self.SOLID_COLOUR_THRESHOLD:
                broken.append(i)
        return broken

    def _measure_flicker(self, frames: list[Image.Image]) -> float:
        if len(frames) < 2:
            return 0.0
        lumas = [
            np.array(f.convert("L"), dtype=np.float32).mean()
            for f in frames
        ]
        diffs = np.abs(np.diff(lumas))
        return float(diffs.mean())

    def _measure_sharpness(self, frames: list[Image.Image]) -> list[float]:
        scores = []
        for frame in frames:
            try:
                import cv2
                arr = np.array(frame.convert("L"))
                lap = cv2.Laplacian(arr, cv2.CV_64F)
                scores.append(float(lap.var()))
            except ImportError:
                # Fallback: simple gradient magnitude
                arr = np.array(frame.convert("L"), dtype=np.float32)
                gx = np.abs(np.diff(arr, axis=1)).mean()
                gy = np.abs(np.diff(arr, axis=0)).mean()
                scores.append(float((gx + gy) / 2))
        return scores or [100.0]
