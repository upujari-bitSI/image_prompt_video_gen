"""
Video Assembly Module.
Stitches PIL Image frames → MP4 video using ffmpeg.

Features:
- Frame sequence → MP4 (H.264 / H.265)
- Motion blur pass (temporal averaging)
- Optional audio track attachment
- Project JSON sidecar for re-editing
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from config import settings, OUTPUTS_DIR


@dataclass
class AssemblyConfig:
    fps: int = 24
    width: int = 1280
    height: int = 720
    crf: int = 18
    preset: str = "slow"
    codec: str = "libx264"    # libx264 / libx265
    pixel_format: str = "yuv420p"
    audio_path: Optional[Path] = None
    apply_motion_blur: bool = False
    motion_blur_strength: int = 2    # number of frames to average


class VideoAssembler:
    """
    Converts a list of PIL frames into a final MP4 video file.
    Uses ffmpeg via subprocess for maximum compatibility.
    """

    def assemble(
        self,
        frames: list[Image.Image],
        output_path: Optional[Path] = None,
        config: Optional[AssemblyConfig] = None,
        save_frames: bool = False,
    ) -> Path:
        if not frames:
            raise ValueError("No frames to assemble")

        config = config or AssemblyConfig(
            fps=settings.video.default_fps,
            width=settings.video.default_resolution[0],
            height=settings.video.default_resolution[1],
            crf=settings.video.crf,
            preset=settings.video.preset,
        )

        if output_path is None:
            output_path = OUTPUTS_DIR / f"video_{int(time.time())}.mp4"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Optional pre-processing
        if config.apply_motion_blur:
            frames = self._apply_motion_blur(frames, config.motion_blur_strength)

        # Resize all frames to target resolution
        frames = self._normalise_frames(frames, config.width, config.height)

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()

            logger.info(f"Writing {len(frames)} frames to temp dir…")
            for i, frame in enumerate(frames):
                frame.save(frame_dir / f"frame_{i:06d}.png")

            if save_frames:
                saved_dir = output_path.parent / "frames"
                saved_dir.mkdir(exist_ok=True)
                for i, frame in enumerate(frames):
                    frame.save(saved_dir / f"frame_{i:06d}.png")
                logger.info(f"Frame sequence saved to {saved_dir}")

            self._run_ffmpeg(frame_dir, output_path, config)

        # Save project sidecar
        self._write_project_file(output_path, config, len(frames))

        logger.success(f"Video assembled: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # ffmpeg invocation
    # ------------------------------------------------------------------

    def _run_ffmpeg(
        self,
        frame_dir: Path,
        output_path: Path,
        config: AssemblyConfig,
    ) -> None:
        input_pattern = str(frame_dir / "frame_%06d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(config.fps),
            "-i", input_pattern,
        ]

        if config.audio_path and config.audio_path.exists():
            cmd += ["-i", str(config.audio_path), "-shortest"]

        cmd += [
            "-c:v", config.codec,
            "-crf", str(config.crf),
            "-preset", config.preset,
            "-pix_fmt", config.pixel_format,
            "-vf", f"scale={config.width}:{config.height}",
            str(output_path),
        ]

        logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (code {result.returncode}):\n{result.stderr}"
            )

    # ------------------------------------------------------------------
    # Frame pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_frames(
        frames: list[Image.Image], width: int, height: int
    ) -> list[Image.Image]:
        out = []
        for f in frames:
            if f.size != (width, height):
                f = f.resize((width, height), Image.LANCZOS)
            if f.mode != "RGB":
                f = f.convert("RGB")
            out.append(f)
        return out

    @staticmethod
    def _apply_motion_blur(
        frames: list[Image.Image], strength: int
    ) -> list[Image.Image]:
        """Temporal averaging over ±strength frames to simulate motion blur."""
        n = len(frames)
        arrs = [np.array(f, dtype=np.float32) for f in frames]
        blurred = []
        for i in range(n):
            lo = max(0, i - strength)
            hi = min(n, i + strength + 1)
            avg = np.mean(arrs[lo:hi], axis=0).astype(np.uint8)
            blurred.append(Image.fromarray(avg))
        return blurred

    # ------------------------------------------------------------------
    # Project sidecar
    # ------------------------------------------------------------------

    @staticmethod
    def _write_project_file(
        video_path: Path,
        config: AssemblyConfig,
        n_frames: int,
    ) -> None:
        project = {
            "video": str(video_path),
            "n_frames": n_frames,
            "duration_s": n_frames / config.fps,
            "config": asdict(config),
        }
        sidecar = video_path.with_suffix(".project.json")
        sidecar.write_text(json.dumps(project, indent=2, default=str))
