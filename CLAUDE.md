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

All three app-builders share a single-file-app workflow that produces a `<framework>_app.py` plus tests, `pyproject.toml`, and `.env.example`. `dash-app-builder` additionally supports an Analytics Dashboard pattern with `dash-bootstrap-components`.

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

**Workflow:** Identify HF model card URL → Classify UI pattern by `pipeline_tag` → Scaffold single-file app → Code quality → Testing

**Inputs:** HuggingFace model card URL only (`https://huggingface.co/<org>/<model>`). Other inputs (scripts, notebooks, GitHub URLs) are rejected with a redirect to `dash-app-builder` or `gradio-app-builder`.

**Outputs:**
- `streamlit_app.py` — single-file app: env loading, `MODEL_ID` const, gated-model gate, `@st.cache_resource`-decorated `load_model`, inference function, top-level UI body
- `test_streamlit_app.py` — pytest unit test for the inference function (mocked model)
- `pyproject.toml` — uv-managed
- `.env.example` — documents `HF_TOKEN` for gated models

**Toolchain:**
```bash
pip install uv --break-system-packages  # if uv is not already available

uv init --name <app-name>
uv add streamlit python-dotenv huggingface_hub  # plus library-specific deps from SKILL.md routing table
uv add --dev ruff ty pytest

uv run ruff check --fix streamlit_app.py test_streamlit_app.py
uv run ruff format streamlit_app.py test_streamlit_app.py
uv run ty check streamlit_app.py
uv run pytest test_streamlit_app.py -v
```

**Repository structure:**
- `streamlit-app-builder/references/scaffolding-templates.md` — `load_model` + inference-function templates (T1-T5) plus the `test_streamlit_app.py` skeleton (T6).
- `streamlit-app-builder/references/pipeline-tag-patterns.md` — UI body templates indexed by `pipeline_tag`.
- `streamlit-app-builder/tests/` — static validator (`ast.parse` + `ruff check --select E,F,I`) for embedded Python blocks, plus structural-consistency tests (routing-table coverage, template-name references, rejection-message sync, skip-validate marker count). Run `uv run pytest` from that directory before committing changes to skill templates.

See `streamlit-app-builder/SKILL.md` for the full workflow.

### Shared code principles

- Wrap original logic — don't rewrite it
- Load config from `.env` via `python-dotenv`; always generate `.env.example`
- Cache model loading: `@lru_cache(maxsize=1)` (dash, gradio) or `@st.cache_resource` (streamlit)
- Test underlying Python functions only, not UI callbacks or wiring
