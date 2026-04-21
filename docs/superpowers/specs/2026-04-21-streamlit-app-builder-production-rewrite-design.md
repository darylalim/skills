# Streamlit App Builder — Production Rewrite

**Date:** 2026-04-21
**Skill affected:** `streamlit-app-builder`
**Status:** Design approved, ready for implementation plan

## Context

The existing `streamlit-app-builder/SKILL.md` produces a flat single-file Streamlit app (`streamlit_app.py`) that wraps an existing Python script or Jupyter notebook. Config is loaded from a `.env` file via `python-dotenv`. The output is suitable for demos and prototypes but not for an application intended for release to paying customers.

This rewrite re-scopes the skill to produce **production-structured Streamlit apps** from an existing artifact, while preserving the "existing-artifact-in, runnable-app-out" value proposition.

## Scope

### In scope

Given one of three existing artifacts, generate a production-structured Streamlit app package:

- Local Python script (`.py`)
- Jupyter notebook (local `.ipynb` or URL — GitHub, Colab, GitLab)
- HuggingFace model card URL (e.g., `https://huggingface.co/<org>/<model>`)

The generated app has: modular code in a `src/<app_name>/` Python package, multipage routing via `st.navigation`, `.env`-based config with fail-fast validation via `os.getenv`, linted/typed/tested code, and a `pyproject.toml` declaring runtime and dev dependencies with environment markers for platform-conditional installs.

### Explicit non-goals

The skill does **not** produce, and its generated `README.md` documents as the team's responsibility:

- Authentication / authorization
- Deployment artifacts (Dockerfile, Kubernetes manifests, CI workflows)
- Database or storage layers
- Observability (structured logging infrastructure, metrics, tracing)
- Secrets management at deploy time — the skill covers only the local `.env` source; production secrets come from the deployment platform

## Cross-cutting principles

### 1. Always verify against live `docs.streamlit.io`

Streamlit's API evolves (e.g., `st.navigation` added in 1.36, `st.fragment` in 1.37, caching decorators renamed). The SKILL.md freezes at write time and drifts. To counter drift, the skill includes:

- **A reference table** appendix mapping topics (multipage, caching, widgets, testing, secrets, file organization) to canonical `docs.streamlit.io` URLs.
- **Mandatory live fetches** for high-churn topics before code generation: multipage, caching, testing, secrets, file organization, plus the widget page(s) for the classified UI pattern.

### 2. Prefer MLX on Apple Silicon

Whenever the source artifact references a model with an MLX-converted equivalent at `huggingface.co/mlx-community/...`, the generated app uses the MLX backend (`mlx-lm` for text generation, `mlx-vlm` for vision-language, `mlx-whisper` for ASR) when running on `arm64-darwin`, and falls back to `transformers` / `diffusers` / `huggingface_hub` elsewhere.

The lookup and dispatch are independent of where the skill itself runs. A Linux developer scaffolding from an HF model card URL still produces an app with MLX support wired in — the runtime dispatch activates MLX only when a user later runs the app on a Mac.

MLX support is encoded in the generated app as follows:
- `pyproject.toml` declares MLX and transformers with environment markers so `uv sync` installs the right backend per host.
- `src/<app_name>/inference.py` reads `config.IS_APPLE_SILICON` and dispatches to the appropriate backend at runtime.
- Same code runs on a Mac developer machine (MLX) and a Linux production host (transformers).

MLX model resolution uses exact-match lookup on `mlx-community/<base-name>*`, picking the highest-download-count variant. The chosen variant is commented in `inference.py` with override instructions.

## Input modes and analysis

Each input type is analyzed into a common internal representation before scaffold generation:

```
{
  pattern: <UI pattern>,
  inference_fn: <optional function signature>,
  data_fns: [<function signatures>],
  viz_fns: [<function signatures>],
  deps: [<pypi deps>],
  is_gated: bool,
  license: <SPDX or free-text>,
  mlx_equivalent: <model id or None>,
}
```

### Local Python script (`.py`)

AST-parse the file for top-level functions (signatures, return types), imports, visualization libraries, data-IO patterns, and model-loading. Distribute functions across `src/<app>/` modules by concern.

### Jupyter notebook (local `.ipynb` or URL)

Resolve URL (current skill's rules for GitHub/Colab/GitLab), fetch, extract code cells, concatenate, then apply the same analysis as a script. Markdown cells are preserved only as docstrings on the scaffolded home page.

### HuggingFace model card URL

Resolve model ID from the URL. Fetch:

- `https://huggingface.co/api/models/<id>` → JSON with `pipeline_tag`, `library_name`, `tags`, `gated`, `license`, `downloads`.
- `README.md` from the repo → YAML frontmatter fallback for above fields, first library-idiomatic code snippet (seeds the inference function), license text.
- `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` → MLX equivalent lookup. Always performed for HF-card inputs, regardless of the skill's host platform; the generated app's runtime dispatch decides whether to use it.

### `pipeline_tag` → UI pattern mapping

| `pipeline_tag`                                      | UI pattern                                          |
|-----------------------------------------------------|-----------------------------------------------------|
| `text-generation`, `conversational`                 | Chat page (`st.chat_input` / `st.chat_message`)     |
| `text-classification`, `zero-shot-classification`   | Text input → label + scores                         |
| `token-classification`                              | Text input → highlighted entities                   |
| `question-answering`                                | Context + question inputs → answer                  |
| `summarization`, `translation`                      | Text area → text area                               |
| `feature-extraction`                                | Text input → embedding / similarity search          |
| `automatic-speech-recognition`                      | `st.audio_input` or file upload → transcript        |
| `text-to-speech`                                    | Text → `st.audio` playback                          |
| `image-classification`, `object-detection`          | `st.file_uploader` → labels / annotated image       |
| `image-to-text`                                     | Image upload → caption                              |
| `image-to-image`, `text-to-image`                   | Prompt / image input → image output                 |
| missing / unrecognized                              | General Script template with a TODO                 |

### Gated models

When `gated: true`, scaffold generation proceeds but the generated `config.py` adds `HF_TOKEN = _require("HF_TOKEN")` so the app fails fast with a clear error instead of 401-ing mid-inference. `README.md` includes `huggingface-cli login` instructions alongside the env-var path.

### License

License text is surfaced in the generated `README.md` under a "License & Commercial Use" section, with an explicit flag when the license is known to restrict commercial use (Llama community license, Gemma terms, `cc-by-nc-*` variants).

## Output directory structure

```
<app-name>/
├── .streamlit/
│   └── config.toml                 # theme, server settings (port, headless, etc.)
├── src/
│   └── <app_name>/
│       ├── __init__.py
│       ├── config.py                # env-based config + fail-fast validation
│       ├── inference.py             # model loading (MLX/transformers dispatch) + predict wrappers
│       ├── data.py                  # file IO, DataFrame transforms, validation
│       ├── viz.py                   # chart builders (if source uses plotting)
│       └── pages/
│           ├── __init__.py
│           ├── home.py              # default landing page — each module exposes render()
│           └── <feature>.py         # additional pages when source has >1 flow
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # sets required env vars; mocked-model fixture
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_inference.py
│   ├── test_viz.py
│   └── test_app_smoke.py            # streamlit.testing.v1.AppTest smoke test
├── streamlit_app.py                 # entrypoint: st.navigation router
├── pyproject.toml                   # uv-managed; platform env markers for MLX / transformers
├── uv.lock
├── .env.example                     # documents every env var the app reads
├── .gitignore                       # .env, .venv, __pycache__, .streamlit/secrets.toml
└── README.md                        # setup, config, license, gated-model notes
```

### Layout decisions

1. **`src/` layout** (not flat): standard Python convention; clean separation of app code, tests, and tooling.
2. **Pages live inside the package** (`src/<app_name>/pages/`), not at the repo root. Streamlit's docs state the two multipage methods (`pages/` auto-discovery and `st.navigation`) cannot be mixed; putting pages inside the package eliminates ambiguity and lets them import sibling modules (`config`, `inference`) cleanly.
3. **Each page is a module with a `render()` function.** The entrypoint registers them via `st.navigation([st.Page(home.render, title="Home"), ...])`.
4. **Multipage even when only one page is needed.** A single `home.py` costs almost nothing and makes adding pages later zero-friction.
5. **`.streamlit/` contains only `config.toml`.** No `secrets.toml` or `.example` variant — the single source of truth for environment config is `.env.example`. Adding a `secrets.toml.example` would re-introduce two-source-of-truth confusion.
6. **`.gitignore` preemptively excludes `.streamlit/secrets.toml`** to prevent accidental commits if the team later adds one for Streamlit Community Cloud deployment.

## Config loading pattern

`src/<app_name>/config.py` is the single module that loads and validates all environment configuration. Every other module imports from it.

```python
"""Environment config. Fail fast at import if required vars are missing."""
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[2]
_env_file = _repo_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=False)  # shell env always wins


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Required env var {key!r} not set. Copy .env.example to .env "
            f"(local dev) or set it in your platform's secret store (prod)."
        )
    return value


def _get(key: str, fallback: str) -> str:
    return os.getenv(key, fallback)


# Required
MODEL_ID: str = _require("MODEL_ID")

# Optional with defaults
MODEL_REVISION: str = _get("MODEL_REVISION", "main")
DEVICE: str = _get("DEVICE", "auto")                      # "auto" | "cpu" | "cuda" | "mps"
MAX_NEW_TOKENS: int = int(_get("MAX_NEW_TOKENS", "512"))

# Optional secrets (HF libs auto-detect HF_TOKEN from env; exposed here so the
# app can fail fast for gated models instead of 401-ing mid-inference)
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# Runtime-detected (not env)
IS_APPLE_SILICON: bool = (
    sys.platform == "darwin" and platform.machine() == "arm64"
)
```

### Design decisions

1. **Plain module-level constants, not `pydantic-settings`.** Matches the thin scaffold tier and adds no dependency. Fail-fast achieved with `_require`. Graduating to pydantic later is a single-file refactor.
2. **`load_dotenv(..., override=False)`** — shell env always wins over `.env`. The same `os.getenv`-based code works locally (values from `.env`) and in production (values from the platform's env injection).
3. **`IS_APPLE_SILICON` lives in `config.py`.** Single source of truth for platform detection, consumed by `inference.py`'s MLX / transformers dispatch.
4. **Gated-model handling.** When the scaffold is produced from an HF model card with `gated: true`, `HF_TOKEN = _require("HF_TOKEN")` is emitted instead of the optional form.
5. **`.env.example` is generated in lockstep with `config.py`.** Every `_require` / `_get` key appears in `.env.example` with a placeholder default and a one-line comment.

### Testing implication

Because `config.py` raises at import, `tests/conftest.py` sets required env vars before any test imports the package:

```python
# tests/conftest.py
import os
os.environ.setdefault("MODEL_ID", "test-model")
os.environ.setdefault("HF_TOKEN", "test-token")  # only when gated
```

## Code quality and testing

### Toolchain (unchanged from current skill)

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ty check src/ tests/
uv run pytest -v
```

All four commands must pass clean before the skill reports the scaffold as complete. Tool configs in `pyproject.toml`:

- `[tool.ruff]` — sensible defaults plus import-sorting (`I` rule group).
- `[tool.ty]` — strict mode on `src/`, non-strict on `tests/`.
- `[tool.pytest.ini_options]` — `testpaths = ["tests"]`, `pythonpath = ["src"]`.

### What gets tested

1. **Pure Python functions** in every `src/<app>/` module — matches the current skill's principle.
2. **Config module** — `_require` raises when missing, `_get` returns default, platform detection with mocked `sys.platform`.
3. **Inference wrappers with mocked model** — `conftest.py` monkeypatches `load_model()` to return a stub with the expected interface. Tests the pre/post-processing code around the model, not the model itself.
4. **Smoke test with `streamlit.testing.v1.AppTest`** — one test boots `streamlit_app.py`, runs the default page, asserts no exceptions. Catches import errors, missing config references, wiring mistakes.

### What does NOT get tested

- Real model loading or real HF Hub calls (too slow, network-dependent, requires tokens).
- Streamlit widget interactions beyond the smoke test (too brittle; manual QA or E2E tests outside this scaffold).
- Page layout / visual regressions.

## Workflow steps (becomes the SKILL.md body)

### Step 1 — Identify and load the input

Determine input type: local `.py`, local `.ipynb`, notebook URL, or HF model card URL.

- **Notebook URL:** resolve raw URL (GitHub/Colab/GitLab as in current skill) and extract code cells.
- **HF model card URL:** resolve model ID; fetch `https://huggingface.co/api/models/<id>` for metadata; fetch `README.md` for YAML frontmatter + example snippet; query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` for an MLX equivalent (always, regardless of the skill's host platform).

### Step 2 — Build the internal representation

For code inputs: AST-parse for functions, signatures, imports, visualization libraries, data-IO, model-loading. For HF inputs: extract `pipeline_tag`, `library_name`, `gated`, `license`, `downloads`, example snippet. Merge into the common representation described above.

### Step 3 — Classify UI pattern

For HF inputs: `pipeline_tag` → pattern via the mapping table. For code inputs: the current skill's heuristic (imports + function shapes). Fall through to "General Script" when ambiguous.

### Step 4 — Fetch live Streamlit docs

Mandatory fetches before code generation:

- `docs.streamlit.io/.../multipage` — confirm `st.navigation` + `st.Page` API.
- `docs.streamlit.io/.../caching` — confirm `@st.cache_resource` / `@st.cache_data` signatures.
- `docs.streamlit.io/.../testing` — confirm `streamlit.testing.v1.AppTest` API.
- `docs.streamlit.io/.../secrets` — inform `.gitignore` and README.
- `docs.streamlit.io/.../file-organization` — confirm directory conventions.
- Widget page(s) for the chosen UI pattern (e.g., `st.chat_input` for chat, `st.audio_input` for ASR).

The skill's reference-table appendix lists the canonical URLs for lookup beyond these mandatory topics.

### Step 5 — Scaffold files

Create the directory tree documented above. Write `streamlit_app.py` (navigation router), `src/<app>/config.py`, `inference.py` (with platform dispatch when model-based), `data.py`, `viz.py` (if applicable), `pages/home.py` plus additional pages, `tests/conftest.py` plus per-module test files. Write `.streamlit/config.toml`, `pyproject.toml`, `.env.example`, `.gitignore`, `README.md`.

### Step 6 — Initialize and install via `uv`

```bash
uv init --name <app-name> --package
uv add streamlit python-dotenv huggingface_hub
uv add "mlx-lm;platform_machine=='arm64' and sys_platform=='darwin'"
uv add "transformers;platform_machine!='arm64' or sys_platform!='darwin'"
uv add --dev ruff ty pytest
```

Versions are not pinned on the command line — `uv add` resolves the current latest at skill-run time and writes the resolved specifier to `pyproject.toml`. Additional runtime deps (`torch`, `diffusers`, `sentence-transformers`, etc.) are added per pattern and detected `library_name`.

### Step 7 — Code-quality gate

Run, in order:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ty check src/ tests/
uv run pytest -v
```

Fix until all four pass. If a mocked-model test fails because the fixture diverged from the real signature, adjust the fixture — do not weaken the test.

### Step 8 — Report to user

Surface: files created (grouped by purpose); chosen model variant (MLX vs. base, with the resolution shown); license plus any commercial-use flags from the HF card; gated-model setup steps if applicable; the exact local-run command (`uv sync && streamlit run streamlit_app.py`).

## Output checklist (updated)

- [ ] Directory tree above is fully populated
- [ ] `pyproject.toml` declares MLX and transformers with correct environment markers
- [ ] `.env.example` covers every `_require` and `_get` key in `config.py`
- [ ] ruff check, ruff format, ty check, and pytest all pass clean
- [ ] `README.md` documents setup, env vars, license, gated-model instructions
- [ ] Report to user surfaces MLX resolution + license flags
