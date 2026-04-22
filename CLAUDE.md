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
  Trigger description. Include explicit trigger phrases (e.g., "Triggers: turn a notebook into a Dash app...").
---

# Skill Title
...
```

The `description` field drives invocation — make it concrete and trigger-driven.

## App-Builder Skills

`dash-app-builder` and `gradio-app-builder` share a single-file-app workflow. `streamlit-app-builder` diverged in the 2026-04-21 production rewrite — it produces a full modular package instead of a single file. `dash-app-builder` additionally supports an Analytics Dashboard pattern with `dash-bootstrap-components`.

### `dash-app-builder` and `gradio-app-builder`

**Workflow:** Analyze source (including notebook URL fetching) → Classify pattern → Generate app → Code quality → Testing

**Outputs:**
- `dash_app.py` / `gradio_app.py` — single-file app with type annotations and inline comments
- `test_dash_app.py` / `test_gradio_app.py` — pytest unit tests for non-UI functions only
- `pyproject.toml` — uv-managed project
- `.env.example` — all configurable env vars with placeholder defaults

### `streamlit-app-builder`

Produces a production-structured app package (see `streamlit-app-builder/SKILL.md` for the full workflow). Outputs include a `src/<app_name>/` package, a multipage `st.navigation` router at `streamlit_app.py`, `src/<app_name>/config.py` with env-based fail-fast validation, a `tests/` directory (including a `streamlit.testing.v1.AppTest` smoke test), and platform-conditional dependencies for MLX on Apple Silicon. Accepts notebooks, scripts, and HuggingFace model card URLs as input.

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

**Code principles:**
- Wrap original logic — don't rewrite it
- Load config from `.env` via `python-dotenv`; always generate `.env.example`
- Cache model loading with `@st.cache_resource` (Streamlit) or `@lru_cache(maxsize=1)` (Dash, Gradio)
- Test underlying Python functions only, not UI callbacks or wiring
