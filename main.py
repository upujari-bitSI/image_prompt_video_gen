"""
AI Video Generation Studio — Entry Point.

Usage:
  python main.py ui                  # launch Gradio UI (default)
  python main.py api                 # launch FastAPI only
  python main.py both                # Gradio + FastAPI side-by-side
  python main.py generate "prompt"   # headless CLI render
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import settings

app = typer.Typer(name="aivideo", add_completion=False)
console = Console()


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    console.print(Panel.fit(
        f"[bold violet]{settings.app_name}[/bold violet] v{settings.version}\n"
        "[dim]RTX 5090 · AnimateDiff · IP-Adapter · ControlNet · Real-ESRGAN[/dim]",
        border_style="violet",
    ))


def _print_gpu_info() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"[green]GPU:[/green] {name} ({total:.0f} GB)")
            console.print(f"[green]CUDA:[/green] {torch.version.cuda}")
            console.print(f"[green]dtype:[/green] {settings.gpu.dtype}")
        else:
            console.print("[yellow]WARNING: CUDA not available — CPU fallback active[/yellow]")
    except ImportError:
        console.print("[red]PyTorch not installed[/red]")


# ---------------------------------------------------------------------------
# UI command
# ---------------------------------------------------------------------------

@app.command()
def ui(
    host: str = typer.Option(settings.host, help="Bind host"),
    port: int = typer.Option(settings.port, help="Bind port"),
    share: bool = typer.Option(False, "--share", help="Create public Gradio share link"),
    debug: bool = typer.Option(settings.debug, "--debug"),
) -> None:
    """Launch the Gradio web UI."""
    _print_banner()
    _print_gpu_info()
    console.print(f"\n[bold]Starting Gradio UI[/bold] at http://{host}:{port}")

    from app.ui.gradio_app import build_app, _THEME, _CSS
    demo = build_app()
    demo.queue(max_size=10).launch(
        server_name=host,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True,
        theme=_THEME,
        css=_CSS,
    )


# ---------------------------------------------------------------------------
# API command
# ---------------------------------------------------------------------------

@app.command()
def api(
    host: str = typer.Option(settings.host, help="Bind host"),
    port: int = typer.Option(8000, help="FastAPI port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Launch the FastAPI REST backend only."""
    _print_banner()
    _print_gpu_info()
    console.print(f"\n[bold]Starting FastAPI[/bold] at http://{host}:{port}/api/docs")

    import uvicorn
    from app.backend.agents.orchestrator import PipelineOrchestrator
    from app.backend.api import create_app

    orch = PipelineOrchestrator()
    fastapi_app = create_app(orch)
    uvicorn.run(fastapi_app, host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# Both (Gradio + FastAPI)
# ---------------------------------------------------------------------------

@app.command()
def both(
    ui_port: int = typer.Option(settings.port, help="Gradio port"),
    api_port: int = typer.Option(8000, help="FastAPI port"),
    host: str = typer.Option(settings.host),
) -> None:
    """Run Gradio UI and FastAPI simultaneously (shared orchestrator)."""
    _print_banner()
    _print_gpu_info()

    from app.backend.agents.orchestrator import PipelineOrchestrator
    from app.backend.api import create_app
    from app.ui.gradio_app import build_app, _orchestrator as _ui_orch, _THEME, _CSS
    import uvicorn

    # FastAPI in background thread
    orch = _ui_orch  # reuse the UI's singleton
    fastapi_app = create_app(orch)

    def _run_api() -> None:
        uvicorn.run(fastapi_app, host=host, port=api_port, log_level="warning")

    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()
    console.print(f"[green]FastAPI[/green] running at http://{host}:{api_port}/api/docs")

    demo = build_app()
    demo.queue(max_size=10).launch(
        server_name=host, server_port=ui_port, show_error=True,
        theme=_THEME, css=_CSS,
    )


# ---------------------------------------------------------------------------
# Headless CLI generate
# ---------------------------------------------------------------------------

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Scene description"),
    motion: str = typer.Option("", "--motion", "-m", help="Motion description"),
    style: str = typer.Option("cinematic", "--style", "-s"),
    duration: float = typer.Option(10.0, "--duration", "-d", min=3, max=120),
    fps: int = typer.Option(24, "--fps"),
    width: int = typer.Option(1280, "--width"),
    height: int = typer.Option(720, "--height"),
    seed: int = typer.Option(42, "--seed"),
    upscale: bool = typer.Option(False, "--upscale"),
    interpolate: bool = typer.Option(False, "--interpolate"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    mock: bool = typer.Option(False, "--mock", help="Skip diffusion: use gradient frames (fast pipeline test)"),
) -> None:
    """Generate a video from the command line (no UI)."""
    _print_banner()
    _print_gpu_info()

    from app.backend.agents.base_agent import AgentContext
    from app.backend.agents.orchestrator import PipelineOrchestrator, JobStatus
    from rich.progress import Progress, SpinnerColumn, TextColumn

    if mock:
        console.print("[yellow]MOCK MODE[/yellow] — skipping diffusion, using synthetic frames")
        _run_mock(prompt, duration, fps, width, height, seed, output)
        return

    ctx = AgentContext(
        raw_prompt=prompt,
        motion_text=motion,
        duration_s=duration,
        fps=fps,
        width=width,
        height=height,
        seed=seed,
        style=style,
        use_upscale=upscale,
        use_interpolation=interpolate,
    )

    orch = PipelineOrchestrator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating video…", total=None)

        def on_progress(job) -> None:
            progress.update(task, description=f"[{job.progress}%] {job.stage}")

        job = orch.run_sync(ctx, on_progress=on_progress)

    if job.status == JobStatus.COMPLETED:
        out = job.output_path
        if output and out:
            import shutil
            shutil.copy(out, output)
            out = output
        console.print(Panel.fit(
            f"[green]Done![/green]\n"
            f"Output: [bold]{out}[/bold]\n"
            f"Quality: {job.context.quality_score:.0f}/100\n"
            f"Duration: {job.elapsed_s:.1f}s",
            border_style="green",
        ))
        _print_quality_table(job)
    else:
        console.print(f"[red]Generation failed:[/red] {job.error}")
        raise typer.Exit(1)


@app.command()
def quicktest(
    duration: float = typer.Option(5.0, "--duration", "-d"),
    fps: int = typer.Option(24, "--fps"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Instant pipeline test — no models needed. Generates a real MP4 in seconds."""
    _print_banner()
    console.print("[cyan]Quick pipeline test — synthetic frames, no GPU required[/cyan]\n")
    _run_mock(
        prompt="Quick test: gradient frames cycling through colors",
        duration=duration,
        fps=fps,
        width=640,
        height=360,
        seed=42,
        output=output,
    )


def _run_mock(
    prompt: str,
    duration: float,
    fps: int,
    width: int,
    height: int,
    seed: int,
    output: Optional[Path],
) -> None:
    """
    Full pipeline test without diffusion.
    Generates synthetic gradient frames so every other stage can be validated
    (scene parser, motion planner, assembler, quality checker) in seconds.
    """
    import time
    import numpy as np
    from PIL import Image
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from app.backend.scene_parser import SceneParser
    from app.backend.motion_engine import MotionPlanner
    from app.backend.motion_engine.keyframe_generator import KeyframeGenerator
    from app.backend.video_builder.assembler import VideoAssembler, AssemblyConfig
    from app.backend.agents.base_agent import AgentContext
    from config import OUTPUTS_DIR

    t0 = time.perf_counter()
    total_frames = int(duration * fps)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:

        # Stage 1: Scene parsing
        task = prog.add_task("[violet]Scene Parser[/violet]", total=1)
        parser = SceneParser()
        scene = parser.parse(prompt)
        prog.update(task, completed=1)
        console.print(f"  ✓ Camera: [cyan]{scene.camera.movement}[/cyan]  "
                      f"Subjects: [cyan]{len(scene.subjects)}[/cyan]  "
                      f"Style: [cyan]{scene.style_hint}[/cyan]")

        # Stage 2: Motion plan
        task = prog.add_task("[violet]Motion Planner[/violet]", total=1)
        planner = MotionPlanner()
        plan = planner.plan(scene, prompt, duration_s=duration, fps=fps)
        kf_gen = KeyframeGenerator()
        keyframes = kf_gen.generate(plan, list(range(0, total_frames, max(1, fps // 6))))
        prog.update(task, completed=1)
        console.print(f"  ✓ {total_frames} frames  "
                      f"{len(keyframes)} keyframes  "
                      f"intensity={plan.intensity:.1f}")

        # Stage 3: Synthetic frame generation
        rng = np.random.default_rng(seed)
        task = prog.add_task("[violet]Frame Synthesis[/violet]", total=total_frames)
        frames = []
        # Pick 3 random hue stops and lerp between them for smooth colour journey
        h1 = rng.integers(0, 255, 3).tolist()
        h2 = rng.integers(0, 255, 3).tolist()
        h3 = rng.integers(0, 255, 3).tolist()
        for i in range(total_frames):
            t = i / max(total_frames - 1, 1)
            if t < 0.5:
                alpha = t * 2
                colour = [int(h1[c] * (1 - alpha) + h2[c] * alpha) for c in range(3)]
            else:
                alpha = (t - 0.5) * 2
                colour = [int(h2[c] * (1 - alpha) + h3[c] * alpha) for c in range(3)]

            # Gradient frame with noise
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            grad_x = np.linspace(0, 1, width)
            grad_y = np.linspace(0, 1, height)
            gx, gy = np.meshgrid(grad_x, grad_y)
            for c in range(3):
                base = colour[c] / 255.0
                arr[:, :, c] = np.clip(
                    (base * 0.7 + gx * 0.15 + gy * 0.15 +
                     rng.random((height, width)) * 0.04) * 255,
                    0, 255,
                ).astype(np.uint8)

            # Stamp frame number onto frame
            from PIL import ImageDraw
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Frame {i+1}/{total_frames}", fill=(255, 255, 255))
            draw.text((10, 30), prompt[:60], fill=(200, 200, 200))
            frames.append(img)
            prog.update(task, completed=i + 1)

        # Stage 4: Assemble video
        task = prog.add_task("[violet]Video Assembly[/violet]", total=1)
        out_path = output or (OUTPUTS_DIR / f"mock_{int(time.time())}.mp4")
        assembler = VideoAssembler()
        config = AssemblyConfig(fps=fps, width=width, height=height, crf=23, preset="fast")
        final_path = assembler.assemble(frames, output_path=out_path, config=config)
        prog.update(task, completed=1)

        # Stage 5: Quality check (import directly to avoid triggering torch)
        task = prog.add_task("[violet]Quality Check[/violet]", total=1)
        import importlib.util, sys as _sys
        _qc_spec = importlib.util.spec_from_file_location(
            "quality_checker",
            Path(__file__).parent / "app/backend/agents/quality_checker.py",
        )
        _qc_mod = importlib.util.module_from_spec(_qc_spec)  # type: ignore[arg-type]
        _qc_spec.loader.exec_module(_qc_mod)  # type: ignore[union-attr]

        class _MockResult:
            pass
        mock_result = _MockResult()
        mock_result.frames = frames  # type: ignore[attr-defined]

        ctx = AgentContext(raw_prompt=prompt)
        ctx.generation_result = mock_result  # type: ignore[assignment]
        checker = _qc_mod.QualityCheckerAgent()
        checker.process(ctx)
        prog.update(task, completed=1)

    elapsed = time.perf_counter() - t0
    console.print(Panel.fit(
        f"[green]Pipeline test passed![/green]\n\n"
        f"Output : [bold]{final_path}[/bold]\n"
        f"Frames : {total_frames}  ({duration}s @ {fps}fps)\n"
        f"Quality: {ctx.quality_score:.0f}/100\n"
        f"Time   : {elapsed:.1f}s\n\n"
        f"[dim]Notes: {', '.join(ctx.quality_notes) or 'None'}[/dim]",
        border_style="green",
        title="Mock Generation Complete",
    ))


def _print_quality_table(job) -> None:
    table = Table(title="Agent Results", show_header=True, header_style="bold")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Time (s)")
    table.add_column("Message")
    for r in job.results:
        status = "[green]✓[/green]" if r.success else "[red]✗[/red]"
        table.add_row(r.agent_name, status, f"{r.elapsed_s:.1f}", r.message[:60])
    console.print(table)


# ---------------------------------------------------------------------------
# Default: launch UI when called without subcommand
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No subcommand → default to UI
        sys.argv.append("ui")
    app()
