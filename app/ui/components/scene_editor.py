"""
Scene Editor UI Component.
Scene prompt, motion controls, style presets, and advanced settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from config import settings


@dataclass
class SceneEditorState:
    scene_prompt: str = ""
    motion_description: str = ""
    style: str = "cinematic"
    duration_s: float = 10.0
    fps: int = 24
    width: int = 1280
    height: int = 720
    seed: int = 42
    motion_intensity: float = 1.0
    use_upscale: bool = False
    use_interpolation: bool = False
    save_frames: bool = False
    reference_video_path: str = ""


def build_scene_editor() -> tuple[gr.Column, dict[str, gr.components.Component]]:
    """
    Returns (column, components_dict).
    components_dict maps string keys to Gradio component objects for
    event wiring in the parent app.
    """
    components: dict[str, gr.components.Component] = {}

    with gr.Column() as col:
        gr.Markdown("## Scene Editor")

        # ── Scene Prompt ──────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Scene Prompt")
            scene_prompt = gr.Textbox(
                label="Scene Description",
                placeholder=(
                    "A young woman runs through a rain-soaked neon city at night, "
                    "camera tracking her movement, cinematic..."
                ),
                lines=4,
            )
            enhance_btn = gr.Button("✨ Auto-Enhance Prompt (LLM)", size="sm")
            enhanced_display = gr.Textbox(
                label="Enhanced Prompt (editable)",
                lines=3,
                interactive=True,
            )

        # ── Motion Controls ───────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Motion & Camera")
            motion_desc = gr.Textbox(
                label="Motion Description",
                placeholder=(
                    "Camera slowly pans left, character walks towards the viewer, "
                    "subtle ambient movement..."
                ),
                lines=2,
            )
            with gr.Row():
                camera_move = gr.Dropdown(
                    label="Camera Movement",
                    choices=[
                        "static", "pan_left", "pan_right", "zoom_in", "zoom_out",
                        "orbit", "dolly_in", "dolly_out", "tilt_up", "tilt_down",
                        "handheld",
                    ],
                    value="static",
                )
                motion_intensity = gr.Slider(
                    label="Motion Intensity",
                    minimum=0.1, maximum=2.5, value=1.0, step=0.1,
                )

        # ── Style ─────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Visual Style")
            style_preset = gr.Dropdown(
                label="Style Preset",
                choices=list(settings.style_presets.keys()),
                value="cinematic",
            )
            gr.Markdown(
                "_Each preset appends a carefully crafted suffix to the diffusion prompt._"
            )

        # ── Reference Video ───────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Optional Reference Video (motion/style transfer)")
            ref_video = gr.File(
                label="Reference Video",
                file_types=["video"],
                file_count="single",
            )

        # ── Advanced Settings ─────────────────────────────────────────
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                duration = gr.Slider(
                    label="Duration (seconds)",
                    minimum=3, maximum=120, value=10, step=1,
                )
                fps_ctrl = gr.Slider(
                    label="FPS",
                    minimum=12, maximum=60, value=24, step=1,
                )
            with gr.Row():
                resolution = gr.Dropdown(
                    label="Resolution",
                    choices=["1280x720 (HD)", "1920x1080 (FHD)", "3840x2160 (4K)", "512x512 (preview)"],
                    value="1280x720 (HD)",
                )
                seed_ctrl = gr.Number(label="Seed", value=42, precision=0)
            with gr.Row():
                use_upscale = gr.Checkbox(label="Upscale ×2 (Real-ESRGAN)", value=False)
                use_interp = gr.Checkbox(label="Frame Interpolation ×2 (RIFE)", value=False)
                save_frames = gr.Checkbox(label="Save Frame Sequence", value=False)

        # ── Collect into dict ─────────────────────────────────────────
        components = {
            "scene_prompt": scene_prompt,
            "enhance_btn": enhance_btn,
            "enhanced_display": enhanced_display,
            "motion_desc": motion_desc,
            "camera_move": camera_move,
            "motion_intensity": motion_intensity,
            "style_preset": style_preset,
            "ref_video": ref_video,
            "duration": duration,
            "fps": fps_ctrl,
            "resolution": resolution,
            "seed": seed_ctrl,
            "use_upscale": use_upscale,
            "use_interp": use_interp,
            "save_frames": save_frames,
        }

    return col, components


def parse_resolution(res_str: str) -> tuple[int, int]:
    try:
        part = res_str.split("(")[0].strip()
        w, h = part.split("x")
        return int(w), int(h)
    except Exception:
        return 1280, 720
