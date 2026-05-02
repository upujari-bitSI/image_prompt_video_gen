"""
Scene Understanding Module.
Parses free-form scene prompts into structured representations that
downstream pipeline stages (motion planner, diffusion engine) can consume.

Supports: Anthropic Claude (preferred), OpenAI, and a fast local regex fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from loguru import logger

from config import settings


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Subject:
    name: str
    role: Literal["character", "object", "environment"]
    action: str = ""
    description: str = ""
    asset_label: str = ""   # matches uploaded asset label from UI


@dataclass
class CameraDirective:
    movement: Literal["static", "pan_left", "pan_right", "zoom_in",
                      "zoom_out", "orbit", "dolly_in", "dolly_out",
                      "tilt_up", "tilt_down", "handheld"] = "static"
    start_angle: float = 0.0
    end_angle: float = 0.0
    speed: float = 1.0         # 0.0 (imperceptible) – 3.0 (fast)


@dataclass
class LightingDescription:
    time_of_day: str = "day"
    weather: str = "clear"
    mood: str = "neutral"
    key_light: str = "sunlight"


@dataclass
class SceneDescription:
    """Fully structured output of the scene parser."""
    raw_prompt: str
    enhanced_prompt: str = ""

    subjects: list[Subject] = field(default_factory=list)
    environment: str = ""
    camera: CameraDirective = field(default_factory=CameraDirective)
    lighting: LightingDescription = field(default_factory=LightingDescription)
    style_hint: str = "cinematic"

    # Shot segmentation for long scenes
    shots: list[str] = field(default_factory=list)

    # Negative prompt generated automatically
    negative_prompt: str = (
        "blurry, low quality, watermark, text, deformed, ugly, extra limbs, "
        "duplicate, bad anatomy, disfigured, mutation, flickering"
    )

    def to_diffusion_prompt(self, style_suffix: str = "") -> str:
        base = self.enhanced_prompt or self.raw_prompt
        if style_suffix:
            return f"{base}, {style_suffix}"
        return base


# ---------------------------------------------------------------------------
# LLM-backed parser
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a cinematography AI assistant.
Convert the user's scene description into a structured JSON object with these keys:
{
  "enhanced_prompt": "<rewritten prompt, cinematic, detailed>",
  "subjects": [
    {"name": "<name>", "role": "character|object|environment", "action": "<verb phrase>", "description": "<visual detail>"}
  ],
  "environment": "<setting description>",
  "camera": {
    "movement": "static|pan_left|pan_right|zoom_in|zoom_out|orbit|dolly_in|dolly_out|tilt_up|tilt_down|handheld",
    "speed": <0.0-3.0>
  },
  "lighting": {
    "time_of_day": "<dawn|morning|noon|afternoon|sunset|dusk|night|midnight>",
    "weather": "<clear|cloudy|rainy|foggy|stormy|snowy>",
    "mood": "<warm|cool|dramatic|soft|harsh|mysterious>",
    "key_light": "<sunlight|moonlight|neon|lamp|fire|overcast>"
  },
  "style_hint": "<cinematic|anime|pixar|realistic|toon|stylized>",
  "shots": ["<shot 1 description>", "<shot 2 description>", ...]
}
Return ONLY valid JSON. No markdown fences."""


class SceneParser:
    def __init__(self) -> None:
        self._client: object | None = None
        self._provider = settings.llm.provider
        self._init_client()

    def _init_client(self) -> None:
        try:
            if self._provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=settings.llm.api_key or None)
            elif self._provider == "openai":
                import openai
                self._client = openai.OpenAI()
            elif self._provider == "ollama":
                import ollama
                self._client = ollama
            logger.info(f"Scene parser using {self._provider} backend")
        except Exception as exc:
            logger.warning(f"LLM client init failed ({exc}); falling back to regex parser")
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, prompt: str, motion_hint: str = "") -> SceneDescription:
        """Parse a raw scene prompt into a SceneDescription."""
        full_prompt = prompt
        if motion_hint:
            full_prompt += f" Motion: {motion_hint}"

        parsed_data: dict = {}

        if self._client is not None:
            parsed_data = self._llm_parse(full_prompt)

        if not parsed_data:
            parsed_data = self._regex_parse(full_prompt)

        return self._build_scene(prompt, parsed_data)

    def segment_into_shots(self, scene: SceneDescription, target_duration_s: int) -> list[str]:
        """Break a long scene into individual shots of ~3–5 seconds each."""
        if scene.shots:
            return scene.shots

        # Heuristic: one shot per ~4 seconds
        n_shots = max(1, target_duration_s // 4)
        base = scene.enhanced_prompt or scene.raw_prompt

        if self._client is not None:
            shots = self._llm_segment(base, n_shots)
            if shots:
                return shots

        # Fallback: repeat prompt with slight variation
        return [f"{base}, shot {i+1} of {n_shots}" for i in range(n_shots)]

    # ------------------------------------------------------------------
    # LLM backends
    # ------------------------------------------------------------------

    def _llm_parse(self, prompt: str) -> dict:
        try:
            if self._provider == "anthropic":
                return self._parse_anthropic(prompt)
            elif self._provider == "openai":
                return self._parse_openai(prompt)
            elif self._provider == "ollama":
                return self._parse_ollama(prompt)
        except Exception as exc:
            logger.warning(f"LLM parse failed: {exc}")
        return {}

    def _parse_anthropic(self, prompt: str) -> dict:
        import anthropic
        assert isinstance(self._client, anthropic.Anthropic)
        resp = self._client.messages.create(
            model=settings.llm.model,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(resp.content[0].text)

    def _parse_openai(self, prompt: str) -> dict:
        import openai
        assert isinstance(self._client, openai.OpenAI)
        resp = self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=settings.llm.temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(resp.choices[0].message.content)

    def _parse_ollama(self, prompt: str) -> dict:
        resp = self._client.chat(  # type: ignore[union-attr]
            model=settings.llm.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return json.loads(resp["message"]["content"])

    def _llm_segment(self, prompt: str, n_shots: int) -> list[str]:
        """Ask the LLM to split the scene into n_shots shot descriptions."""
        segment_prompt = (
            f"Split this scene into exactly {n_shots} individual shot descriptions "
            f"for a video sequence. Return a JSON array of strings only.\n\nScene: {prompt}"
        )
        try:
            if self._provider == "anthropic":
                import anthropic
                assert isinstance(self._client, anthropic.Anthropic)
                resp = self._client.messages.create(
                    model=settings.llm.model,
                    max_tokens=512,
                    temperature=0.4,
                    messages=[{"role": "user", "content": segment_prompt}],
                )
                return json.loads(resp.content[0].text)
        except Exception as exc:
            logger.warning(f"Shot segmentation failed: {exc}")
        return []

    # ------------------------------------------------------------------
    # Regex / heuristic fallback
    # ------------------------------------------------------------------

    _CAMERA_KEYWORDS: dict[str, str] = {
        "pan left": "pan_left", "pan right": "pan_right",
        "zoom in": "zoom_in", "zoom out": "zoom_out",
        "push in": "dolly_in", "pull out": "dolly_out",
        "orbit": "orbit", "tilt up": "tilt_up", "tilt down": "tilt_down",
        "handheld": "handheld",
    }

    _TIME_KEYWORDS: dict[str, str] = {
        "dawn": "dawn", "sunrise": "dawn", "morning": "morning",
        "afternoon": "afternoon", "sunset": "sunset", "dusk": "dusk",
        "night": "night", "midnight": "midnight", "noon": "noon",
    }

    def _regex_parse(self, prompt: str) -> dict:
        p_lower = prompt.lower()

        # Camera
        camera_movement = "static"
        for kw, val in self._CAMERA_KEYWORDS.items():
            if kw in p_lower:
                camera_movement = val
                break

        # Time of day
        time_of_day = "day"
        for kw, val in self._TIME_KEYWORDS.items():
            if kw in p_lower:
                time_of_day = val
                break

        # Style hint
        style_hint = "cinematic"
        for style in ("anime", "pixar", "realistic", "toon", "stylized"):
            if style in p_lower:
                style_hint = style
                break

        # Subjects: extract nouns preceded by "a/an/the" via simple regex
        subjects = []
        noun_matches = re.findall(
            r"\b(?:a|an|the)\s+([a-z]+(?:\s[a-z]+)?)\b", p_lower
        )
        for nm in noun_matches[:4]:
            subjects.append({
                "name": nm,
                "role": "character",
                "action": "",
                "description": nm,
            })

        return {
            "enhanced_prompt": prompt,
            "subjects": subjects,
            "environment": "",
            "camera": {"movement": camera_movement, "speed": 1.0},
            "lighting": {
                "time_of_day": time_of_day,
                "weather": "clear",
                "mood": "neutral",
                "key_light": "sunlight",
            },
            "style_hint": style_hint,
            "shots": [],
        }

    # ------------------------------------------------------------------
    # Build SceneDescription from raw dict
    # ------------------------------------------------------------------

    def _build_scene(self, raw_prompt: str, data: dict) -> SceneDescription:
        subjects = [
            Subject(
                name=s.get("name", ""),
                role=s.get("role", "character"),
                action=s.get("action", ""),
                description=s.get("description", ""),
            )
            for s in data.get("subjects", [])
        ]

        cam_data = data.get("camera", {})
        camera = CameraDirective(
            movement=cam_data.get("movement", "static"),
            speed=float(cam_data.get("speed", 1.0)),
        )

        light_data = data.get("lighting", {})
        lighting = LightingDescription(
            time_of_day=light_data.get("time_of_day", "day"),
            weather=light_data.get("weather", "clear"),
            mood=light_data.get("mood", "neutral"),
            key_light=light_data.get("key_light", "sunlight"),
        )

        return SceneDescription(
            raw_prompt=raw_prompt,
            enhanced_prompt=data.get("enhanced_prompt", raw_prompt),
            subjects=subjects,
            environment=data.get("environment", ""),
            camera=camera,
            lighting=lighting,
            style_hint=data.get("style_hint", "cinematic"),
            shots=data.get("shots", []),
        )
