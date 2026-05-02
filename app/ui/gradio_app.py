"""
Main Gradio Application.
Wires together the Asset Manager, Scene Editor, and Render Panel into a
single-page production UI for the AI Video Generation Studio.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Generator, Optional

import gradio as gr
from loguru import logger
from PIL import Image

from config import settings
from app.backend.agents.base_agent import AgentContext
from app.backend.agents.orchestrator import PipelineOrchestrator, RenderJob, JobStatus
from app.ui.components.asset_manager import build_asset_manager
from app.ui.components.scene_editor import build_scene_editor, parse_resolution
from app.ui.components.render_panel import build_render_panel


# Shared orchestrator (singleton)
_orchestrator = PipelineOrchestrator()
_active_job: Optional[RenderJob] = None
_cancel_event = threading.Event()


# ---------------------------------------------------------------------------
# Prompt enhancer (calls scene parser's LLM)
# ---------------------------------------------------------------------------

def enhance_prompt(raw_prompt: str) -> str:
    if not raw_prompt.strip():
        return ""
    try:
        from app.backend.scene_parser import SceneParser
        parser = SceneParser()
        scene = parser.parse(raw_prompt)
        return scene.enhanced_prompt or raw_prompt
    except Exception as exc:
        logger.warning(f"Prompt enhancement failed: {exc}")
        return raw_prompt


# ---------------------------------------------------------------------------
# Main generate function (streaming updates via Gradio generator)
# ---------------------------------------------------------------------------

def generate_video(
    scene_prompt: str,
    enhanced_prompt: str,
    motion_desc: str,
    camera_move: str,
    motion_intensity: float,
    style_preset: str,
    ref_video,
    duration: float,
    fps: int,
    resolution: str,
    seed: int,
    use_upscale: bool,
    use_interp: bool,
    save_frames: bool,
) -> Generator:
    global _active_job, _cancel_event
    _cancel_event.clear()

    final_prompt = enhanced_prompt.strip() or scene_prompt.strip()
    if not final_prompt:
        yield (
            0, "Error: please enter a scene prompt", "❌", "❌", "❌", "❌", "❌",
            [], 0, "No prompt provided", None, None, "No prompt provided"
        )
        return

    w, h = parse_resolution(resolution)

    # Inject camera movement into motion text
    motion_text = motion_desc
    if camera_move != "static":
        motion_text = f"{camera_move} camera, {motion_text}"

    ctx = AgentContext(
        raw_prompt=final_prompt,
        motion_text=motion_text,
        duration_s=float(duration),
        fps=int(fps),
        width=w,
        height=h,
        seed=int(seed),
        style=style_preset,
        use_upscale=use_upscale,
        use_interpolation=use_interp,
        save_frames=save_frames,
    )

    stage_status = {
        "ScenePlanner": "⏳ Waiting",
        "CharacterConsistency": "⏳ Waiting",
        "MotionDirector": "⏳ Waiting",
        "Rendering": "⏳ Waiting",
        "QualityChecker": "⏳ Waiting",
    }

    _STAGE_KEYS = [
        "ScenePlanner", "CharacterConsistency",
        "MotionDirector", "Rendering", "QualityChecker",
    ]

    def _pack(job: RenderJob) -> tuple:
        """Pack current state into the expected output tuple."""
        prog = job.progress
        stage_msg = job.stage

        # Update stage indicators
        for k in _STAGE_KEYS:
            matched = [r for r in job.results if r.agent_name == k]
            if matched:
                stage_status[k] = "✅ Done" if matched[-1].success else "❌ Failed"
            elif job.stage == k:
                stage_status[k] = "🔄 Running…"

        # Sample frames for gallery
        frames = []
        result = job.context.generation_result if job.context else None
        if result and result.frames:
            step = max(1, len(result.frames) // 12)
            frames = result.frames[::step][:12]

        quality = job.context.quality_score if job.context else 0
        notes = "\n".join(job.context.quality_notes) if job.context else ""
        logs = "\n".join(job.context.agent_logs) if job.context else ""

        video_path = str(job.output_path) if job.output_path and job.output_path.exists() else None

        return (
            prog, stage_msg,
            stage_status["ScenePlanner"],
            stage_status["CharacterConsistency"],
            stage_status["MotionDirector"],
            stage_status["Rendering"],
            stage_status["QualityChecker"],
            frames, quality, notes,
            video_path, video_path,
            logs,
        )

    # Submit job (async thread)
    completed_event = threading.Event()
    last_update: list[tuple] = []

    def on_progress(job: RenderJob) -> None:
        last_update.clear()
        last_update.append(_pack(job))
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            completed_event.set()

    _active_job = _orchestrator.submit(ctx, on_progress=on_progress)

    # Poll for updates
    import time
    while not completed_event.is_set():
        if _cancel_event.is_set():
            yield (
                _active_job.progress, "Cancelled",
                *list(stage_status.values()),
                [], 0, "Job cancelled", None, None, "Cancelled by user"
            )
            return
        if last_update:
            yield last_update[-1]
        time.sleep(0.5)

    if last_update:
        yield last_update[-1]


def cancel_generation() -> str:
    global _cancel_event
    _cancel_event.set()
    return "Cancellation requested…"


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

_THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
)

_CSS = """
.gradio-container { max-width: 1400px !important; }
.stage-ok { color: #22c55e; }
.stage-fail { color: #ef4444; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title=settings.app_name) as demo:

        gr.Markdown(f"# 🎬 {settings.app_name} v{settings.version}")
        gr.Markdown(
            "**Production-grade AI video generation** — "
            "RTX 5090 optimised · AnimateDiff + SVD · IP-Adapter · ControlNet"
        )

        with gr.Tabs():
            # ── Tab 1: Assets ─────────────────────────────────────────
            with gr.Tab("📁 Assets"):
                build_asset_manager(_orchestrator.asset_library)

            # ── Tab 2: Scene + Generate ───────────────────────────────
            with gr.Tab("🎬 Create Video"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        _, scene_comps = build_scene_editor()
                    with gr.Column(scale=1):
                        _, render_comps = build_render_panel()

                # Wire enhance button
                scene_comps["enhance_btn"].click(
                    fn=enhance_prompt,
                    inputs=[scene_comps["scene_prompt"]],
                    outputs=[scene_comps["enhanced_display"]],
                )

                # Wire cancel
                render_comps["cancel_btn"].click(
                    fn=cancel_generation,
                    outputs=[render_comps["stage_label"]],
                )

                # Wire generate
                generate_outputs = [
                    render_comps["progress_bar"],
                    render_comps["stage_label"],
                    render_comps["stage_scene"],
                    render_comps["stage_char"],
                    render_comps["stage_motion"],
                    render_comps["stage_render"],
                    render_comps["stage_quality"],
                    render_comps["frame_gallery"],
                    render_comps["quality_score"],
                    render_comps["quality_notes"],
                    render_comps["video_output"],
                    render_comps["download_btn"],
                    render_comps["log_box"],
                ]

                render_comps["generate_btn"].click(
                    fn=generate_video,
                    inputs=[
                        scene_comps["scene_prompt"],
                        scene_comps["enhanced_display"],
                        scene_comps["motion_desc"],
                        scene_comps["camera_move"],
                        scene_comps["motion_intensity"],
                        scene_comps["style_preset"],
                        scene_comps["ref_video"],
                        scene_comps["duration"],
                        scene_comps["fps"],
                        scene_comps["resolution"],
                        scene_comps["seed"],
                        scene_comps["use_upscale"],
                        scene_comps["use_interp"],
                        scene_comps["save_frames"],
                    ],
                    outputs=generate_outputs,
                )

            # ── Tab 3: GPU Monitor ────────────────────────────────────
            with gr.Tab("🖥️ System"):
                gr.Markdown("### GPU & System Status")
                sys_info = gr.Textbox(
                    label="System Info",
                    lines=10,
                    interactive=False,
                )
                refresh_sys = gr.Button("Refresh")

                def get_system_info() -> str:
                    lines = []
                    try:
                        import torch
                        lines.append(f"PyTorch: {torch.__version__}")
                        if torch.cuda.is_available():
                            dev = torch.cuda.get_device_name(0)
                            mem_alloc = torch.cuda.memory_allocated(0) / 1e9
                            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                            lines.append(f"GPU: {dev}")
                            lines.append(f"VRAM: {mem_alloc:.1f} GB / {mem_total:.1f} GB used")
                            lines.append(f"CUDA: {torch.version.cuda}")
                        else:
                            lines.append("CUDA: Not available")
                    except ImportError:
                        lines.append("PyTorch not installed")
                    try:
                        import psutil
                        ram = psutil.virtual_memory()
                        lines.append(f"RAM: {ram.used/1e9:.1f} GB / {ram.total/1e9:.1f} GB")
                        lines.append(f"CPU: {psutil.cpu_percent()}%")
                    except ImportError:
                        pass
                    from config import settings as cfg
                    lines.append(f"\nModel: {cfg.diffusion.base_model_id}")
                    lines.append(f"Motion Adapter: {cfg.diffusion.motion_adapter_id}")
                    lines.append(f"Precision: {cfg.gpu.dtype}")
                    lines.append(f"xformers: {cfg.gpu.enable_xformers}")
                    return "\n".join(lines)

                refresh_sys.click(fn=get_system_info, outputs=[sys_info])
                demo.load(fn=get_system_info, outputs=[sys_info])

    return demo
