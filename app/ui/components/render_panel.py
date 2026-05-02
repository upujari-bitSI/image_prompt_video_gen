"""
Render Panel UI Component.
Shows progress, per-stage status, frame preview grid, quality score,
agent logs, and the final video player.
"""

from __future__ import annotations

import gradio as gr


def build_render_panel() -> tuple[gr.Column, dict[str, gr.components.Component]]:
    """Returns (column, components_dict)."""
    components: dict[str, gr.components.Component] = {}

    with gr.Column() as col:
        gr.Markdown("## Render Queue & Output")

        # ── Action buttons ────────────────────────────────────────────
        with gr.Row():
            generate_btn = gr.Button("🎬 Generate Video", variant="primary", scale=3)
            cancel_btn = gr.Button("⛔ Cancel", variant="stop", scale=1)

        # ── Overall progress ──────────────────────────────────────────
        progress_bar = gr.Slider(
            label="Overall Progress",
            minimum=0, maximum=100, value=0,
            interactive=False,
        )
        stage_label = gr.Textbox(
            label="Current Stage",
            value="Idle",
            interactive=False,
        )

        # ── Stage indicators ──────────────────────────────────────────
        gr.Markdown("### Pipeline Stages")
        with gr.Row():
            stage_scene = gr.Textbox(label="Scene Planner", value="⏳ Waiting", interactive=False)
            stage_char = gr.Textbox(label="Character Consistency", value="⏳ Waiting", interactive=False)
            stage_motion = gr.Textbox(label="Motion Director", value="⏳ Waiting", interactive=False)
            stage_render = gr.Textbox(label="Rendering", value="⏳ Waiting", interactive=False)
            stage_quality = gr.Textbox(label="Quality Check", value="⏳ Waiting", interactive=False)

        # ── Frame preview grid ────────────────────────────────────────
        with gr.Accordion("🖼️ Frame Preview Grid", open=True):
            frame_gallery = gr.Gallery(
                label="Generated Frames (sampled)",
                columns=6,
                height=300,
                object_fit="contain",
            )

        # ── Quality metrics ───────────────────────────────────────────
        with gr.Row():
            quality_score = gr.Number(
                label="Quality Score / 100",
                value=0,
                interactive=False,
            )
            quality_notes = gr.Textbox(
                label="Quality Notes",
                interactive=False,
                lines=2,
            )

        # ── Video output ──────────────────────────────────────────────
        gr.Markdown("### Output Video")
        video_output = gr.Video(label="Generated Video", interactive=False)
        download_btn = gr.File(label="Download MP4", interactive=False)

        # ── Logs ──────────────────────────────────────────────────────
        with gr.Accordion("📋 Agent Logs", open=False):
            log_box = gr.Textbox(
                label="Pipeline Logs",
                lines=12,
                interactive=False,
                autoscroll=True,
            )

        components = {
            "generate_btn": generate_btn,
            "cancel_btn": cancel_btn,
            "progress_bar": progress_bar,
            "stage_label": stage_label,
            "stage_scene": stage_scene,
            "stage_char": stage_char,
            "stage_motion": stage_motion,
            "stage_render": stage_render,
            "stage_quality": stage_quality,
            "frame_gallery": frame_gallery,
            "quality_score": quality_score,
            "quality_notes": quality_notes,
            "video_output": video_output,
            "download_btn": download_btn,
            "log_box": log_box,
        }

    return col, components
