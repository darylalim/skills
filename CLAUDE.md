# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A collection of Claude Code skills. Each skill is a `SKILL.md` file in its own subdirectory, invoked at runtime via the `Skill` tool.

## Skill File Format

Each skill lives at `<skill-name>/SKILL.md`:

```
---
name: <skill-name>
description: >
  Trigger description with concrete phrases — users' natural phrasings
  ("turn a notebook into X", "generate a UI for Y"), file extensions
  (`.ipynb`), and URL patterns (`huggingface.co/...`) the skill should match.
---

# Skill Title
...
```

The `description` field drives invocation — make it concrete and trigger-driven.

Long lookup tables (catalogs, mappings) can live in a `<skill-name>/references/` subdirectory and be referenced by path from `SKILL.md`.

## App-Builder Skills

`dash-app-builder` and `gradio-app-builder` share a single-file-app workflow. `streamlit-app-builder` produces a production-structured package instead. `dash-app-builder` additionally supports an Analytics Dashboard pattern with `dash-bootstrap-components`.

### `dash-app-builder` and `gradio-app-builder`

**Workflow:** Analyze source (including notebook URL fetching) → Classify pattern → Generate app → Code quality → Testing

**Outputs:**
- `dash_app.py` / `gradio_app.py` — single-file app with type annotations and inline comments
- `test_dash_app.py` / `test_gradio_app.py` — pytest unit tests for non-UI functions only
- `pyproject.toml` — uv-managed project
- `.env.example` — all configurable env vars with placeholder defaults

**Toolchain:**
```bash
pip install uv --break-system-packages  # if uv is not already available

uv init --name <app-name>
uv add <framework> python-dotenv  # plus dependencies identified in Step 1
uv add --dev ruff ty pytest

uv run ruff check --fix <app>.py && uv run ruff format <app>.py
uv run ty check <app>.py
uv run pytest test_<app>.py -v
```

### `streamlit-app-builder`

**Workflow:** Analyze source (script / notebook / HF model card URL) → Fetch live Streamlit docs → Classify UI pattern → Scaffold production package → Code quality → Testing

**Outputs:**
- `streamlit_app.py` — `st.navigation` router entrypoint
- `src/<app_name>/` — package with `config.py`, `inference.py` (MLX/transformers dispatch), `data.py`, `viz.py`, and `pages/`
- `tests/` — pytest unit tests plus a `streamlit.testing.v1.AppTest` smoke test
- `.streamlit/config.toml` — Streamlit server and theme config
- `pyproject.toml` — uv-managed, platform-conditional deps: `mlx-lm` / `mlx-vlm` / `mlx-audio` on Apple Silicon, `transformers` elsewhere (`audio-to-audio` is Apple-Silicon-only)
- `.env.example` — documents every env var the app reads

See `streamlit-app-builder/SKILL.md` for the full workflow.

### Shared code principles

- Wrap original logic — don't rewrite it (streamlit distributes by concern across `src/<app_name>/` modules)
- Load config from `.env` via `python-dotenv`; always generate `.env.example`
- Cache model loading with `@lru_cache(maxsize=1)`
- Test underlying Python functions only, not UI callbacks or wiring
