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

    from app.ui.gradio_app import build_app
    demo = build_app()
    demo.queue(max_size=10).launch(
        server_name=host,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True,
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
    from app.ui.gradio_app import build_app, _orchestrator as _ui_orch
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
        server_name=host, server_port=ui_port, show_error=True
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
) -> None:
    """Generate a video from the command line (no UI)."""
    _print_banner()
    _print_gpu_info()

    from app.backend.agents.base_agent import AgentContext
    from app.backend.agents.orchestrator import PipelineOrchestrator, JobStatus
    from rich.progress import Progress, SpinnerColumn, TextColumn

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
