"""
Keyframe Generator.
Samples the MotionPlan at specific frame indices and produces
conditioning data (embeddings, masks, control images) for the diffusion engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

from .motion_planner import MotionPlan


@dataclass
class Keyframe:
    frame_index: int
    timestamp: float               # normalised [0, 1]
    camera_position: np.ndarray    # (3,)
    camera_rotation: np.ndarray    # (3,) pitch/yaw/roll degrees
    fov: float
    character_positions: dict[str, tuple[float, float]]  # name → (x, y) normalised
    character_actions: dict[str, str]
    prompt_modifier: str = ""      # injected into frame-level prompt


class KeyframeGenerator:
    """
    Samples a MotionPlan and produces per-frame Keyframe descriptors.
    Also generates simple skeleton / pose overlay images for ControlNet conditioning.
    """

    def generate(
        self,
        plan: MotionPlan,
        frame_indices: Optional[list[int]] = None,
    ) -> list[Keyframe]:
        if frame_indices is None:
            frame_indices = list(range(plan.total_frames))

        keyframes: list[Keyframe] = []

        for idx in frame_indices:
            if idx >= plan.total_frames:
                continue

            ts = plan.camera.timestamps[idx]
            cam_pos = plan.camera.positions[idx]
            cam_rot = plan.camera.rotations[idx]
            fov = float(plan.camera.fov_values[idx])

            char_positions: dict[str, tuple[float, float]] = {}
            char_actions: dict[str, str] = {}
            for char_path in plan.characters:
                if idx < len(char_path.positions):
                    x, y = char_path.positions[idx]
                    char_positions[char_path.name] = (float(x), float(y))
                    char_actions[char_path.name] = char_path.actions[idx]

            prompt_mod = self._build_prompt_modifier(cam_rot, fov, plan.intensity)

            keyframes.append(Keyframe(
                frame_index=idx,
                timestamp=float(ts),
                camera_position=cam_pos,
                camera_rotation=cam_rot,
                fov=fov,
                character_positions=char_positions,
                character_actions=char_actions,
                prompt_modifier=prompt_mod,
            ))

        return keyframes

    def generate_pose_image(
        self,
        keyframe: Keyframe,
        width: int,
        height: int,
    ) -> Image.Image:
        """
        Draw a minimal stick-figure pose overlay for each character.
        In production, replace with an actual pose estimator / skeleton renderer.
        """
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        for name, (nx, ny) in keyframe.character_positions.items():
            cx = int(nx * width)
            cy = int(ny * height)
            head_r = max(8, height // 20)

            # Head
            draw.ellipse(
                [cx - head_r, cy - head_r * 4, cx + head_r, cy - head_r * 2],
                outline=(255, 255, 0), width=2
            )
            # Torso
            torso_top = (cx, cy - head_r * 2)
            torso_bot = (cx, cy + head_r * 4)
            draw.line([torso_top, torso_bot], fill=(0, 255, 0), width=2)
            # Arms
            draw.line([(cx - head_r * 3, cy), (cx + head_r * 3, cy)],
                      fill=(0, 255, 255), width=2)
            # Legs
            draw.line([torso_bot, (cx - head_r * 2, cy + head_r * 8)],
                      fill=(255, 0, 0), width=2)
            draw.line([torso_bot, (cx + head_r * 2, cy + head_r * 8)],
                      fill=(255, 0, 0), width=2)

        return img

    def _build_prompt_modifier(
        self,
        cam_rot: np.ndarray,
        fov: float,
        intensity: float,
    ) -> str:
        parts: list[str] = []

        if abs(cam_rot[1]) > 5:   # yaw
            direction = "left" if cam_rot[1] < 0 else "right"
            parts.append(f"camera panning {direction}")

        if fov < 45:
            parts.append("telephoto lens, compressed perspective")
        elif fov > 70:
            parts.append("wide angle lens")

        if intensity > 1.5:
            parts.append("dynamic motion, action shot")

        return ", ".join(parts)
