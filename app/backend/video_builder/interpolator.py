"""
Frame Interpolator.
Increases effective frame rate using RIFE (Real-Time Intermediate Flow Estimation)
or falls back to PIL cross-dissolve interpolation.

RIFE: arXiv 2011.06294 — https://github.com/megvii-research/ECCV2022-RIFE
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from config import settings


class FrameInterpolator:
    """
    Doubles (or quadruples) frame count via optical-flow interpolation.

    When RIFE is installed in the environment it is preferred.
    Otherwise falls back to linear (cross-dissolve) interpolation which is
    fast but produces ghosting on fast motion.
    """

    def __init__(self, multiplier: int = 2) -> None:
        assert multiplier in (2, 4), "multiplier must be 2 or 4"
        self._multiplier = multiplier
        self._rife_available = self._check_rife()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate(
        self,
        frames: list[Image.Image],
        progress_callback: Optional[callable] = None,
    ) -> list[Image.Image]:
        logger.info(
            f"Interpolating {len(frames)} frames ×{self._multiplier} "
            f"using {'RIFE' if self._rife_available else 'linear'}"
        )

        if self._rife_available:
            return self._rife_interpolate(frames, progress_callback)
        return self._linear_interpolate(frames, progress_callback)

    # ------------------------------------------------------------------
    # RIFE path
    # ------------------------------------------------------------------

    def _rife_interpolate(
        self,
        frames: list[Image.Image],
        progress_callback: Optional[callable],
    ) -> list[Image.Image]:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = Path(tmpdir) / "in"
            out_dir = Path(tmpdir) / "out"
            in_dir.mkdir()
            out_dir.mkdir()

            for i, frame in enumerate(frames):
                frame.save(in_dir / f"{i:06d}.png")

            cmd = [
                "python", "-m", "rife",
                "--input", str(in_dir),
                "--output", str(out_dir),
                "--exp", str(int(np.log2(self._multiplier))),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                logger.warning(f"RIFE failed: {result.stderr}; falling back to linear")
                return self._linear_interpolate(frames, progress_callback)

            out_files = sorted(out_dir.glob("*.png"))
            return [Image.open(f).copy() for f in out_files]

    # ------------------------------------------------------------------
    # Linear cross-dissolve fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_interpolate(
        frames: list[Image.Image],
        progress_callback: Optional[callable],
        multiplier: int = 2,
    ) -> list[Image.Image]:
        out: list[Image.Image] = []
        n = len(frames)

        for i in range(n - 1):
            if progress_callback:
                progress_callback(i, n, f"Interpolating frame {i+1}/{n}")

            a = np.array(frames[i], dtype=np.float32)
            b = np.array(frames[i + 1], dtype=np.float32)
            out.append(frames[i])

            for j in range(1, multiplier):
                alpha = j / multiplier
                blended = ((1 - alpha) * a + alpha * b).astype(np.uint8)
                out.append(Image.fromarray(blended))

        if frames:
            out.append(frames[-1])

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_rife() -> bool:
        try:
            result = subprocess.run(
                ["python", "-m", "rife", "--help"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
