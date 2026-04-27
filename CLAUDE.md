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

The three app-builders split into one broad-scope skill and two narrow HF-card skills.

### `dash-app-builder` (broad scope)

**Workflow:** Analyze source (including notebook URL fetching) → Classify pattern → Generate app → Code quality → Testing.

Accepts Python scripts, Jupyter notebooks (local or notebook URLs), and GitHub URLs.

**Outputs:**
- `dash_app.py` — single-file app with type annotations and inline comments
- `test_dash_app.py` — pytest unit tests for non-UI functions only
- `pyproject.toml` — uv-managed project
- `.env.example` — all configurable env vars with placeholder defaults

**Toolchain:**
```bash
pip install uv --break-system-packages  # if uv is not already available

uv init --name <app-name>
uv add dash python-dotenv  # plus dependencies identified in Step 1
uv add --dev ruff ty pytest

uv run ruff check --fix dash_app.py && uv run ruff format dash_app.py
uv run ty check dash_app.py
uv run pytest test_dash_app.py -v
```

`dash-app-builder` additionally supports an Analytics Dashboard pattern with `dash-bootstrap-components`.

### `streamlit-app-builder` (HF model card → local prototype)

**Workflow:** Identify HF model card URL → Classify UI pattern by `pipeline_tag` → Scaffold single-file app → Code quality → Testing.

**Inputs:** HuggingFace model card URL only (`https://huggingface.co/<org>/<model>`). Other inputs (scripts, notebooks, GitHub URLs) are rejected with a redirect to `dash-app-builder`.

**Outputs:**
- `streamlit_app.py` — single-file app: env loading, `MODEL_ID` const, gated-model gate, `@st.cache_resource`-decorated `load_model`, inference function, top-level UI body
- `test_streamlit_app.py` — pytest unit test for the inference function (mocked model)
- `pyproject.toml` — uv-managed
- `.env.example` — documents `HF_TOKEN` for gated models

**Toolchain:**
```bash
pip install uv --break-system-packages  # if uv is not already available

uv init --name <app-name>
uv add streamlit python-dotenv  # plus library-specific deps from SKILL.md routing table
uv add --dev ruff ty pytest

uv run ruff check --fix streamlit_app.py test_streamlit_app.py
uv run ruff format streamlit_app.py test_streamlit_app.py
uv run ty check streamlit_app.py
uv run pytest test_streamlit_app.py -v
```

**Repository structure:**
- `streamlit-app-builder/references/scaffolding-templates.md` — `load_model` + inference-function templates (T1-T5) plus the `test_streamlit_app.py` skeleton (T6).
- `streamlit-app-builder/references/pipeline-tag-patterns.md` — UI body templates indexed by `pipeline_tag`.
- `streamlit-app-builder/tests/` — static validator (`ast.parse` + `ruff check --select E,F,I`) for embedded Python blocks, plus structural-consistency tests (routing-table coverage, template-name references, rejection-message sync, skip-validate marker count, file-list ↔ checklist parity, inference-function name resolution). The directory has its own `pyproject.toml` and `.venv`, independent of any scaffolded app. Run `uv run pytest` from that directory before committing changes to skill templates.

See `streamlit-app-builder/SKILL.md` for the full workflow.

### `gradio-app-builder` (HF model card → Hugging Face Space)

**Workflow:** Identify HF model card URL → Classify UI pattern by `pipeline_tag` → Scaffold five-file Spaces project → Code quality → Testing.

**Inputs:** HuggingFace model card URL only (`https://huggingface.co/<org>/<model>`). Other inputs are rejected with a redirect to `dash-app-builder`.

**Outputs (canonical Spaces format):**
- `app.py` — Spaces entry point: env loading, `MODEL_ID` const, gated-model gate, `@lru_cache(maxsize=1)`-decorated `load_model`, inference function, top-level `demo = gr.<Interface|ChatInterface>(...)`, `if __name__ == "__main__": demo.launch()`
- `requirements.txt` — Spaces installs from this; pins `gradio==<version>` plus library-specific runtime deps
- `README.md` — YAML frontmatter (`sdk: gradio`, `sdk_version` matching `requirements.txt`) + brief description + local-run snippet
- `.env.example` — documents `HF_TOKEN` for gated models with a comment pointing at the Spaces Settings → Variables and secrets UI
- `test_app.py` — pytest unit test for the inference function (mocked model); local-only, not run by Spaces

**Toolchain:**
```bash
pip install -r requirements.txt
pip install ruff ty pytest

ruff check --fix app.py test_app.py
ruff format app.py test_app.py
ty check app.py
pytest test_app.py -v
```

(No `uv init` / `pyproject.toml` for the generated app — Spaces reads `requirements.txt`.)

**Repository structure:**
- `gradio-app-builder/references/scaffolding-templates.md` — `load_model` + inference-function templates (T1-T5) plus the `test_app.py` skeleton (T6).
- `gradio-app-builder/references/pipeline-tag-patterns.md` — UI body templates (`gr.Interface` / `gr.ChatInterface`) indexed by `pipeline_tag`.
- `gradio-app-builder/tests/` — static validator + structural-consistency tests + Gradio-specific checks (Spaces frontmatter validity, `gr.ChatInterface` used in T2). Has its own `pyproject.toml` and `.venv`. Run `uv run pytest` from that directory before committing changes to skill templates.

See `gradio-app-builder/SKILL.md` for the full workflow.

### Shared code principles

- Wrap original logic — don't rewrite it
- Load config from `.env` via `python-dotenv`; always generate `.env.example`
- Cache model loading: `@lru_cache(maxsize=1)` (dash, gradio) or `@st.cache_resource` (streamlit)
- Test underlying Python functions only, not UI callbacks or wiring
