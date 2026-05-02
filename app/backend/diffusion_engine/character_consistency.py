"""
Character Consistency Engine.
Maintains visual identity of characters across all generated frames using:
1. IP-Adapter feature injection (primary path)
2. Multi-angle image fusion (front/side/back → single embedding)
3. Prompt-based identity anchoring (fallback)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image

from config import settings


class CharacterConsistencyEncoder:
    """
    Encodes one or more reference images of a character into a compact
    representation for IP-Adapter conditioning.

    Multi-angle fusion: averages CLIP image embeddings from front, side,
    back views to produce a single, view-invariant representation.
    """

    def __init__(self) -> None:
        self._clip_model: object | None = None
        self._clip_processor: object | None = None
        self._device = settings.gpu.device
        self._dtype = torch.float16 if settings.gpu.dtype != "fp32" else torch.float32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, images: list[Image.Image]) -> Optional[torch.Tensor]:
        """
        Encode a list of character images (can be multi-angle) into a
        fused embedding tensor suitable for IP-Adapter.

        Returns (1, 1, 768) or (1, N, 768) embedding, or None on failure.
        """
        if not images:
            return None

        self._ensure_clip()

        if self._clip_model is None:
            logger.warning("CLIP unavailable; character consistency disabled")
            return None

        embeddings = []
        for img in images:
            emb = self._encode_single(img)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None

        # Average-pool multi-angle embeddings
        stacked = torch.stack(embeddings, dim=0)
        fused = stacked.mean(dim=0, keepdim=True)
        return fused   # (1, seq_len, dim)

    def blend_embeddings(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor,
        alpha: float = 0.7,
    ) -> torch.Tensor:
        """Blend two character embeddings (e.g., costume change between scenes)."""
        return alpha * primary + (1 - alpha) * secondary

    def to_prompt_tokens(self, character_name: str, style: str = "cinematic") -> str:
        """
        Lightweight fallback: generate textual identity anchors.
        Used when IP-Adapter is unavailable.
        """
        return (
            f"{character_name}, consistent character design, "
            f"same face same clothes, {style} style"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_clip(self) -> None:
        if self._clip_model is not None:
            return
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

            self._clip_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self._clip_model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=self._dtype,
            ).to(self._device)
            self._clip_model.eval()
            logger.info("CLIP vision encoder loaded for character consistency")
        except Exception as exc:
            logger.warning(f"CLIP load failed: {exc}")

    def _encode_single(self, image: Image.Image) -> Optional[torch.Tensor]:
        try:
            inputs = self._clip_processor(images=image, return_tensors="pt")  # type: ignore[operator]
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                output = self._clip_model(**inputs)  # type: ignore[operator]
            return output.image_embeds.to(self._dtype)
        except Exception as exc:
            logger.warning(f"CLIP encode failed: {exc}")
            return None


class AssetLibrary:
    """
    Manages uploaded character / object / background assets.
    Stores original images and their CLIP embeddings for reuse.
    """

    def __init__(self) -> None:
        self._assets: dict[str, dict] = {}
        self._encoder = CharacterConsistencyEncoder()

    def register(
        self,
        label: str,
        images: list[Image.Image],
        role: str = "character",
        tags: Optional[list[str]] = None,
    ) -> None:
        embedding = self._encoder.encode(images)
        self._assets[label] = {
            "images": images,
            "embedding": embedding,
            "role": role,
            "tags": tags or [],
        }
        logger.info(f"Asset registered: '{label}' ({role}, {len(images)} images)")

    def get_embedding(self, label: str) -> Optional[torch.Tensor]:
        asset = self._assets.get(label)
        return asset["embedding"] if asset else None

    def get_images(self, label: str) -> list[Image.Image]:
        asset = self._assets.get(label)
        return asset["images"] if asset else []

    def list_assets(self, role: Optional[str] = None) -> list[str]:
        if role is None:
            return list(self._assets.keys())
        return [k for k, v in self._assets.items() if v["role"] == role]

    def get_reference_images_for_prompt(
        self, character_labels: list[str]
    ) -> list[Image.Image]:
        """Collect one representative image per character for IP-Adapter."""
        images: list[Image.Image] = []
        for label in character_labels:
            imgs = self.get_images(label)
            if imgs:
                images.append(imgs[0])  # front-facing first
        return images
