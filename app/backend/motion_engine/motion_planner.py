"""
Motion Planning Module.
Converts motion text descriptions + camera directives into
time-indexed keyframe trajectories consumed by the diffusion engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger

from app.backend.scene_parser.parser import CameraDirective, SceneDescription


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CameraPath:
    """Interpolated camera transform over [0, 1] normalised time."""
    positions: np.ndarray      # (N, 3) — x, y, z translations
    rotations: np.ndarray      # (N, 3) — pitch, yaw, roll in degrees
    fov_values: np.ndarray     # (N,)   — field of view in degrees
    timestamps: np.ndarray     # (N,)   — normalised [0, 1]


@dataclass
class CharacterMotionPath:
    name: str
    positions: np.ndarray      # (N, 2) — normalised screen x, y
    scales: np.ndarray         # (N,)   — size factor
    actions: list[str]         # per-frame action label
    timestamps: np.ndarray


@dataclass
class MotionPlan:
    camera: CameraPath
    characters: list[CharacterMotionPath] = field(default_factory=list)
    intensity: float = 1.0     # 0 = subtle, 1 = normal, 2 = dynamic
    fps: int = 24
    total_frames: int = 240

    def camera_conditioning_prompt(self) -> str:
        """Human-readable camera motion for use in diffusion prompts."""
        movement_labels = {
            "pan_left": "camera slowly panning left",
            "pan_right": "camera slowly panning right",
            "zoom_in": "gradual zoom in, approaching subject",
            "zoom_out": "gradual zoom out, pulling away",
            "orbit": "orbiting camera movement",
            "dolly_in": "dolly push-in, depth of field shift",
            "dolly_out": "dolly pull-out",
            "tilt_up": "camera tilting upward",
            "tilt_down": "camera tilting downward",
            "handheld": "handheld shaky camera",
            "static": "static locked-off camera",
        }
        return movement_labels.get("static", "cinematic camera movement")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class MotionPlanner:
    """
    Converts textual motion descriptions into MotionPlan objects.

    The planner operates in two modes:
    1. Structured mode — when a SceneDescription is available.
    2. Text-only mode — parses free-form motion text.
    """

    # Keywords → animation primitives
    _CHARACTER_MOTION_MAP: dict[str, str] = {
        "walk": "walk", "walking": "walk", "run": "run", "running": "run",
        "sprint": "run", "idle": "idle", "stand": "idle", "sit": "idle",
        "talk": "talk", "speaking": "talk", "jump": "jump", "dance": "dance",
        "fight": "action", "attack": "action",
    }

    _INTENSITY_MAP: dict[str, float] = {
        "subtle": 0.3, "gentle": 0.4, "slow": 0.5, "moderate": 1.0,
        "normal": 1.0, "dynamic": 1.6, "fast": 1.8, "intense": 2.0,
        "extreme": 2.5,
    }

    def plan(
        self,
        scene: SceneDescription,
        motion_text: str = "",
        duration_s: float = 10.0,
        fps: int = 24,
        intensity: float = 1.0,
    ) -> MotionPlan:
        total_frames = int(duration_s * fps)
        timestamps = np.linspace(0.0, 1.0, total_frames)

        # Resolve intensity from text
        resolved_intensity = self._parse_intensity(motion_text, intensity)

        camera_path = self._build_camera_path(
            scene.camera, timestamps, resolved_intensity
        )

        character_paths = self._build_character_paths(
            scene, motion_text, timestamps
        )

        logger.info(
            f"Motion plan: {total_frames} frames @ {fps}fps, "
            f"camera={scene.camera.movement}, intensity={resolved_intensity:.1f}"
        )

        return MotionPlan(
            camera=camera_path,
            characters=character_paths,
            intensity=resolved_intensity,
            fps=fps,
            total_frames=total_frames,
        )

    # ------------------------------------------------------------------
    # Camera path generation
    # ------------------------------------------------------------------

    def _build_camera_path(
        self,
        directive: CameraDirective,
        timestamps: np.ndarray,
        intensity: float,
    ) -> CameraPath:
        n = len(timestamps)
        positions = np.zeros((n, 3))
        rotations = np.zeros((n, 3))
        fov_values = np.full(n, 55.0)   # default 55° FoV

        speed = directive.speed * intensity
        t = timestamps  # convenience alias

        mov = directive.movement

        if mov == "pan_left":
            rotations[:, 1] = -np.linspace(0, 30 * speed, n)   # yaw
        elif mov == "pan_right":
            rotations[:, 1] = np.linspace(0, 30 * speed, n)
        elif mov == "zoom_in":
            fov_values = 55.0 - np.linspace(0, 20 * speed, n)
        elif mov == "zoom_out":
            fov_values = 55.0 + np.linspace(0, 20 * speed, n)
        elif mov == "dolly_in":
            positions[:, 2] = np.linspace(0, -2 * speed, n)   # z forward
        elif mov == "dolly_out":
            positions[:, 2] = np.linspace(0, 2 * speed, n)
        elif mov == "orbit":
            angle = np.linspace(0, 2 * math.pi * speed * 0.3, n)
            radius = 3.0
            positions[:, 0] = radius * np.sin(angle)
            positions[:, 2] = radius * (np.cos(angle) - 1.0)
            rotations[:, 1] = np.degrees(angle)
        elif mov == "tilt_up":
            rotations[:, 0] = -np.linspace(0, 20 * speed, n)  # pitch
        elif mov == "tilt_down":
            rotations[:, 0] = np.linspace(0, 20 * speed, n)
        elif mov == "handheld":
            np.random.seed(42)
            noise = np.random.randn(n, 3) * 0.5 * speed
            # Low-pass filter to avoid seizure-inducing jitter
            from scipy.ndimage import uniform_filter1d
            positions = uniform_filter1d(noise, size=max(3, n // 20), axis=0)
            rot_noise = np.random.randn(n, 3) * 1.5 * speed
            rotations = uniform_filter1d(rot_noise, size=max(3, n // 20), axis=0)

        return CameraPath(
            positions=positions,
            rotations=rotations,
            fov_values=fov_values,
            timestamps=timestamps,
        )

    # ------------------------------------------------------------------
    # Character motion paths
    # ------------------------------------------------------------------

    def _build_character_paths(
        self,
        scene: SceneDescription,
        motion_text: str,
        timestamps: np.ndarray,
    ) -> list[CharacterMotionPath]:
        paths: list[CharacterMotionPath] = []
        n = len(timestamps)

        for subject in scene.subjects:
            if subject.role != "character":
                continue

            action = self._resolve_action(subject.action, motion_text)
            positions = self._character_trajectory(action, n)
            scales = np.ones(n)

            paths.append(CharacterMotionPath(
                name=subject.name,
                positions=positions,
                scales=scales,
                actions=[action] * n,
                timestamps=timestamps,
            ))

        return paths

    def _character_trajectory(
        self, action: str, n: int
    ) -> np.ndarray:
        """Return normalised (x, y) screen positions for a character action."""
        positions = np.zeros((n, 2))
        t = np.linspace(0, 1, n)

        if action == "walk":
            # Character walks from left-center to right-center
            positions[:, 0] = np.linspace(0.2, 0.8, n)
            positions[:, 1] = np.full(n, 0.5)
        elif action == "run":
            positions[:, 0] = np.linspace(0.1, 0.9, n)
            positions[:, 1] = 0.5 + np.sin(t * math.pi * 4) * 0.02  # slight bob
        elif action in ("talk", "idle"):
            positions[:, 0] = np.full(n, 0.5)
            positions[:, 1] = np.full(n, 0.5)
        elif action == "jump":
            positions[:, 0] = np.linspace(0.3, 0.7, n)
            positions[:, 1] = 0.5 - np.sin(t * math.pi) * 0.2
        elif action == "dance":
            positions[:, 0] = 0.5 + np.sin(t * math.pi * 6) * 0.15
            positions[:, 1] = 0.5 + np.cos(t * math.pi * 4) * 0.05
        else:
            positions[:, 0] = np.full(n, 0.5)
            positions[:, 1] = np.full(n, 0.5)

        return positions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_action(self, subject_action: str, motion_text: str) -> str:
        combined = f"{subject_action} {motion_text}".lower()
        for kw, action in self._CHARACTER_MOTION_MAP.items():
            if kw in combined:
                return action
        return "idle"

    def _parse_intensity(self, motion_text: str, default: float) -> float:
        text_lower = motion_text.lower()
        for kw, val in self._INTENSITY_MAP.items():
            if kw in text_lower:
                return val
        return default
