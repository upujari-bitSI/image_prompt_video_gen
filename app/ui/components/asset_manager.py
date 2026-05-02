"""
Asset Manager UI Component.
Provides upload + labelling widgets for characters, objects, and backgrounds.
Returns registered data to the parent app via shared AssetLibrary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
from loguru import logger
from PIL import Image

if TYPE_CHECKING:
    from app.backend.diffusion_engine.character_consistency import AssetLibrary


def build_asset_manager(asset_library: "AssetLibrary") -> gr.Column:
    """Return a Gradio Column containing the full asset manager UI."""

    with gr.Column() as col:
        gr.Markdown("## Asset Library")
        gr.Markdown(
            "Upload reference images for your characters, objects, and "
            "backgrounds. Add a unique label so the AI can find them."
        )

        with gr.Tabs():
            # ── Characters ────────────────────────────────────────────
            with gr.Tab("Characters"):
                gr.Markdown(
                    "Upload 1–4 images per character (front, side, back, ¾) "
                    "for best identity consistency."
                )
                char_images = gr.File(
                    label="Character Images",
                    file_count="multiple",
                    file_types=["image"],
                )
                char_label = gr.Textbox(
                    label="Character Name / Label",
                    placeholder="e.g. Hero, Villain, Sidekick",
                )
                char_tags = gr.Textbox(
                    label="Optional Tags (comma-separated)",
                    placeholder="e.g. futuristic, armoured, young",
                )
                char_register_btn = gr.Button("Register Character", variant="primary")
                char_status = gr.Textbox(label="Status", interactive=False)

            # ── Objects ───────────────────────────────────────────────
            with gr.Tab("Objects"):
                obj_images = gr.File(
                    label="Object Images",
                    file_count="multiple",
                    file_types=["image"],
                )
                obj_label = gr.Textbox(
                    label="Object Label",
                    placeholder="e.g. Red Sports Car, Ancient Sword",
                )
                obj_register_btn = gr.Button("Register Object", variant="primary")
                obj_status = gr.Textbox(label="Status", interactive=False)

            # ── Backgrounds ───────────────────────────────────────────
            with gr.Tab("Backgrounds"):
                bg_images = gr.File(
                    label="Background / Environment Images",
                    file_count="multiple",
                    file_types=["image"],
                )
                bg_label = gr.Textbox(
                    label="Background Label",
                    placeholder="e.g. Neon City, Forest, Space Station",
                )
                bg_register_btn = gr.Button("Register Background", variant="primary")
                bg_status = gr.Textbox(label="Status", interactive=False)

            # ── Asset Library View ────────────────────────────────────
            with gr.Tab("Library"):
                gr.Markdown("All registered assets:")
                library_view = gr.Dataframe(
                    headers=["Label", "Role", "# Images", "Tags"],
                    label="Registered Assets",
                    interactive=False,
                )
                refresh_btn = gr.Button("Refresh")

        # ── Event handlers ────────────────────────────────────────────

        def register_asset(
            files, label: str, tags: str, role: str
        ) -> tuple[str, list]:
            if not files or not label.strip():
                return "Please upload at least one image and enter a label.", []
            try:
                images = [Image.open(f.name).convert("RGB") for f in files]
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                asset_library.register(label.strip(), images, role=role, tags=tag_list)
                return f"Registered '{label}' ({role}) — {len(images)} image(s).", _library_rows(asset_library)
            except Exception as exc:
                logger.error(f"Asset registration failed: {exc}")
                return f"Error: {exc}", []

        def _library_rows(lib: "AssetLibrary") -> list:
            rows = []
            for lbl in lib.list_assets():
                asset = lib._assets[lbl]  # type: ignore[attr-defined]
                rows.append([lbl, asset["role"], len(asset["images"]), ", ".join(asset["tags"])])
            return rows

        char_register_btn.click(
            fn=lambda f, l, t: register_asset(f, l, t, "character"),
            inputs=[char_images, char_label, char_tags],
            outputs=[char_status, library_view],
        )
        obj_register_btn.click(
            fn=lambda f, l: register_asset(f, l, "", "object"),
            inputs=[obj_images, obj_label],
            outputs=[obj_status, library_view],
        )
        bg_register_btn.click(
            fn=lambda f, l: register_asset(f, l, "", "background"),
            inputs=[bg_images, bg_label],
            outputs=[bg_status, library_view],
        )
        refresh_btn.click(
            fn=lambda: _library_rows(asset_library),
            outputs=[library_view],
        )

    return col
