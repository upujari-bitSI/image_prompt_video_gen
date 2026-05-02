# AI Video Generation Studio — CLAUDE.md

## Project Overview
Production-grade Python AI video generation tool optimised for RTX 5090.
Generates 10s–2min videos from labelled images + scene prompts.

## Architecture
```
main.py                        # CLI entry (typer): ui | api | both | generate
config.py                      # All config via pydantic-settings + .env
app/
  ui/
    gradio_app.py              # Main Gradio Blocks app
    components/
      asset_manager.py         # Upload + label characters/objects/backgrounds
      scene_editor.py          # Prompt, motion, style, advanced settings
      render_panel.py          # Progress, frame grid, video output
  backend/
    scene_parser/parser.py     # LLM → SceneDescription (Anthropic/OpenAI/regex)
    motion_engine/
      motion_planner.py        # Text → MotionPlan (camera + character paths)
      keyframe_generator.py    # MotionPlan → Keyframe list
    diffusion_engine/
      pipeline.py              # AnimateDiff + SVD chunked generation
      character_consistency.py # CLIP encoder + AssetLibrary
      controlnet_guide.py      # Pose / depth / canny conditioning
    video_builder/
      assembler.py             # PIL frames → MP4 via ffmpeg
      upscaler.py              # Real-ESRGAN ×2/×4
      interpolator.py          # RIFE or linear ×2/×4 frame interpolation
    agents/
      base_agent.py            # BaseAgent ABC + AgentContext + AgentResult
      scene_planner.py         # Agent 1: parse scene
      character_consistency_agent.py  # Agent 2: match assets
      motion_director.py       # Agent 3: generate motion plan
      rendering_agent.py       # Agent 4: run diffusion + assemble
      quality_checker.py       # Agent 5: flicker/blur/identity metrics
      orchestrator.py          # Pipeline coordinator + job queue
    api.py                     # FastAPI REST endpoints
```

## Key Design Decisions
- **Chunked generation**: AnimateDiff uses 16-frame windows with 4-frame overlap blending to avoid seam artefacts at arbitrary durations.
- **bf16 by default**: RTX 5090 has native bf16 tensor cores; avoids fp16 NaN overflow on long prompts.
- **Lazy model loading**: Pipeline loads on first generate() call, not at import, so UI startup is instant.
- **Single shared orchestrator**: Both Gradio and FastAPI share one `PipelineOrchestrator` to avoid double-loading 24 GB of models.
- **Agent pattern**: Each stage is an independent agent with its own error handling; a failure in one stage stops the chain cleanly.

## Running Locally
```bash
cp .env.example .env        # add ANTHROPIC_API_KEY
pip install -r requirements.txt
python main.py ui            # Gradio at http://localhost:7860
python main.py api           # FastAPI at http://localhost:8000/api/docs
python main.py generate "A knight rides through a stormy forest" --duration 10
```

## Common Dev Commands
```bash
# Verify imports without GPU
python -c "from app.backend.agents.orchestrator import PipelineOrchestrator; print('OK')"

# Run scene parser standalone
python -c "
from app.backend.scene_parser import SceneParser
s = SceneParser()
d = s.parse('A robot walks through a neon city at night, camera pans right')
print(d)
"
```

## Environment Variables
See `.env.example` for all tuneable knobs.
Critical: `ANTHROPIC_API_KEY` for LLM scene parsing (falls back to regex if absent).

## Adding a New Style Preset
Edit `config.py → AppConfig.style_presets` dict — the suffix is appended to every diffusion prompt for that style.

## Adding a New Agent
1. Create `app/backend/agents/my_agent.py` subclassing `BaseAgent`
2. Implement `process(ctx) → Optional[str]`
3. Insert into `orchestrator.py → _run_job agents list`
4. Add weight to `_STAGE_WEIGHTS`
