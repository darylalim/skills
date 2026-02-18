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

All three skills (`dash-app-builder`, `gradio-app-builder`, `streamlit-app-builder`) share the same workflow and outputs. `dash-app-builder` additionally supports an Analytics Dashboard pattern with `dash-bootstrap-components`.

**Workflow:** Analyze source (including notebook URL fetching) → Classify pattern → Generate app → Code quality → Testing

**Outputs:**
- `dash_app.py` / `gradio_app.py` / `streamlit_app.py` — single-file app with type annotations and inline comments
- `test_dash_app.py` / `test_gradio_app.py` / `test_streamlit_app.py` — pytest unit tests for non-UI functions only
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

**Code principles:**
- Wrap original logic — don't rewrite it
- Load config from `.env` via `python-dotenv`; always generate `.env.example`
- Cache model loading with `@st.cache_resource` (Streamlit) or `@lru_cache(maxsize=1)` (Dash, Gradio)
- Test underlying Python functions only, not UI callbacks or wiring
