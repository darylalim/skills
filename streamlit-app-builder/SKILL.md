---
name: streamlit-app-builder
description: >
  Generate a production-structured Streamlit app from an existing artifact: a
  local Python script, a Jupyter notebook (local or URL), or a HuggingFace
  model card URL. Triggers: build a Streamlit app for production, wrap a
  notebook into a Streamlit app, generate a UI for a HuggingFace model, any
  link to a `.ipynb` or to `huggingface.co/<org>/<model>`, turn a script into
  a multipage Streamlit app, scaffold a Streamlit app intended for paying
  customers.
---

# Streamlit App Builder

Generate a production-structured Streamlit app package from an existing artifact. Output is a `src/<app_name>/` Python package with a multipage `st.navigation` router, env-based config with fail-fast validation, and linted/typed/tested code — ready to plug into the team's own production infrastructure.

## Non-goals

The skill **does not** produce, and the generated `README.md` documents these as the team's responsibility:

- Authentication / authorization
- Deployment artifacts (Dockerfile, Kubernetes manifests, CI workflows)
- Database or storage layers
- Observability (structured logging infra, metrics, tracing)
- Secrets management at deploy time — the skill covers only the local `.env` source; production secrets come from the deployment platform

## Cross-cutting principles

### 1. Always verify against live `docs.streamlit.io` and `huggingface.co/docs`

Streamlit and HuggingFace APIs evolve. Before generating code, fetch the current docs for high-churn topics and confirm API shapes. Canonical URLs live in `references/streamlit-docs-index.md` and `references/huggingface-docs-index.md`. See Step 4 for the full fetch rules, including which HF pages are library-conditional.

**Mandatory Streamlit fetches** before code generation:
- Multipage + `st.navigation` + `st.Page`
- Caching (`@st.cache_resource`, `@st.cache_data`)
- App testing (`streamlit.testing.v1.AppTest`)
- Secrets
- File organization
- Plus the widget page(s) for the classified UI pattern (e.g., `st.chat_input` for chat, `st.audio_input` for ASR)

### 2. Prefer MLX on Apple Silicon

When the source artifact references a model with an MLX-converted equivalent on HuggingFace, the generated app uses an MLX backend on `arm64-darwin` and falls back to `transformers` / `diffusers` / `huggingface_hub` elsewhere.

**MLX backend index:**

| `pipeline_tag` | MLX module | PyPI | Apple-only? | Transformers fallback |
|---|---|---|---|---|
| `text-generation`, `conversational` | `mlx_lm` | `mlx-lm` | no | `transformers` |
| `image-to-text`, `image-text-to-text` | `mlx_vlm` | `mlx-vlm` | no | `transformers` |
| `automatic-speech-recognition` | `mlx_audio.stt` | `mlx-audio` | no | `transformers[audio]` (via `pipeline("automatic-speech-recognition")`) |
| `text-to-speech` | `mlx_audio.tts` | `mlx-audio` | no | `transformers[audio]` (SpeechT5 / Bark / Parler-TTS) |
| `audio-to-audio` | `mlx_audio.sts` | `mlx-audio` | **yes** | — (`RuntimeError` at model load off Apple Silicon) |

Apple-only rows install no transformers fallback; `inference.py` raises a clear `RuntimeError` at model load on non-Apple-Silicon hosts, and the generated `README.md` notes the platform requirement.

The MLX lookup is independent of where the skill itself runs. A Linux developer scaffolding from an HF model card still produces an app with MLX support wired in — runtime dispatch activates MLX only when a user later runs the app on a Mac. (Exception: `audio-to-audio` apps run on Apple Silicon only, by design.)

MLX support is encoded in the generated app as follows:
- `pyproject.toml` declares MLX and transformers with environment markers so `uv sync` installs the right backend per host. Audio-to-audio apps declare `mlx-audio` with an Apple-only marker and omit the fallback dep.
- `src/<app_name>/inference.py` reads `config.IS_APPLE_SILICON` and dispatches at runtime.

**MLX model resolution:** query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` and pick the highest-download-count variant. If the base name has no `mlx-community` match, note "no MLX equivalent found" in the final report and generate the app with `transformers` only. Audio-to-audio inputs without an `mlx-community` match fail at scaffold time — there is no transformers fallback to generate toward. (A separate failure mode, where the model exists but no `mlx_audio.sts` class mapping is known, is handled by the dispatch table in the `inference.py` template.)

## Step 1: Identify and load the input

Classify the input into one of four types, then load it.

### Local Python script (`.py`)

Read the file. Apply Step 2's AST analysis directly.

### Jupyter notebook (`.ipynb`, local or URL)

Resolve the URL to a raw `.ipynb` source:

- GitHub: `github.com/<owner>/<repo>/blob/<ref>/<path>.ipynb` → `raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>.ipynb`
- Colab: `colab.research.google.com/drive/<id>` → Colab export endpoint
- GitLab: raw endpoint, or append `?ref=main&format=json` to the web URL
- Other: use directly if the URL serves `.ipynb` JSON

Download and extract code cells:

```bash
curl -L -o notebook.ipynb "<resolved_raw_url>"
```

```python
import json
with open("notebook.ipynb") as f:
    nb = json.load(f)
code_cells = [
    "".join(cell["source"])
    for cell in nb["cells"]
    if cell["cell_type"] == "code"
]
```

Markdown cells are preserved only as docstrings on the scaffolded home page.

### HuggingFace model card URL

Resolve the model ID from the URL (strip `https://huggingface.co/` prefix; keep `<org>/<model>`). Fetch:

- **Metadata:** `https://huggingface.co/api/models/<id>` → JSON with `pipeline_tag`, `library_name`, `tags`, `gated`, `license`, `license_name`, `downloads`.
- **README:** `https://huggingface.co/<id>/raw/main/README.md` → YAML frontmatter (fallback for metadata fields), first library-idiomatic code snippet (seeds the inference function), license text.
- **MLX equivalent:** `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` — always performed for HF-card inputs, regardless of the skill's host platform. The generated app's runtime dispatch decides whether to use it.
- **Org freshness (conditional):** When `<org>` ∈ `{mlx-community, google, ibm-granite}`, WebFetch `https://huggingface.co/<org>` with the prompt:

  > "List the models shown prominently on this organization's page. For each, give the model ID (`<org>/<name>`) and the pipeline_tag if visible (e.g., `text-generation`, `image-to-text`). Return up to 10 models. Skip datasets and Spaces."

  Filter the returned models: keep those with the same `pipeline_tag` as the input and a different model ID. Take up to 2 as *siblings*. Store the list as `siblings` in the Step 2 IR dict; Step 8 reads it to render the conditional "Sibling models" sub-section. If the fetch fails or the filter returns nothing, the check is silent — no report line, no error.

### GitHub URL

Classified **after** the `.ipynb` and HuggingFace branches — those already claim their URL shapes; this branch only sees GitHub `.py` blob URLs, repo-root URLs, or unsupported variants.

**URL normalization (required before pattern matching):** strip any query string (`?foo=bar`) and fragment (`#L12`) from the input URL, and apply the classification patterns below to the cleaned URL only. Trailing-slash tolerance is preserved by the patterns themselves. Example: `…/inference.py?ref=main#L42` → `…/inference.py`.

**Classification patterns, in order:**

| Match | Mode |
|---|---|
| `^https?://github\.com/([\w.-]+)/([\w.-]+)/blob/([^/]+)/(.+\.py)/?$` | **blob-`.py`** |
| `^https?://github\.com/([\w.-]+)/([\w.-]+)/?$` | **repo-root** |
| Other `github.com/...` URL (`tree/`, `pulls/`, `issues/`, `wiki/`, `commit/`, …) | **reject** |
| Non-github.com host (`gist.github.com`, `gitlab.com`, `bitbucket.org`, raw pastebins) | **reject** |

Branch names, tags, and 7–40-char commit SHAs all match the `[^/]+` ref class. **Limitation:** slash-containing branch names (e.g., `feature/foo-bar`) are not supported — the regex captures the first segment as `<ref>` and the rest as `<path>`, which can mis-parse. Ask users to pass a blob URL using the default branch, a tag, or a commit SHA instead.

**Rejection messages (exact text — no silent fallbacks):**
- Unsupported `github.com` variant: *"Pass a blob URL to a `.py` file (`github.com/<o>/<r>/blob/<ref>/<path>.py`), a `.ipynb` file (handled by the notebook branch), or the repo root URL (`github.com/<o>/<r>`). `tree/` / `pulls/` / `commit/` / etc. are not supported."*
- Non-github.com host: *"This skill accepts github.com URLs only. Clone locally and re-run with a file path."*

**Blob-`.py` mode:**

1. Resolve `blob/<ref>/<path>.py` → `https://raw.githubusercontent.com/<o>/<r>/<ref>/<path>.py`.
2. Download with `curl -L`. On non-200 response: fail with the HTTP status code and resolved URL.
3. Feed the downloaded source into Step 2's AST walker, unchanged.
4. Populate IR: `source_url = <original input URL>`, `source_ref = <ref parsed from URL>`.

**Repo-root mode:**

1. `GET https://api.github.com/repos/<o>/<r>`. Capture `default_branch` and `license.spdx_id` from the response.
   - HTTP 404 → fail: *"Repo not found or private: `<o>/<r>`."*
   - HTTP 403 (rate limit) → fail: *"GitHub API rate limit hit. Retry later or set `GH_TOKEN` in the environment."* (Authenticated calls are a separate change.)
   - Other non-200 → fail with status code and URL.
2. Fetch `https://raw.githubusercontent.com/<o>/<r>/<default_branch>/README.md`. On 404: fail — *"Repo has no README.md at the default branch root."*
3. Extract fenced code blocks matching `` ```(?:python|py)\n(.*?)``` `` (DOTALL). Take the **first match**. On zero matches: fail with *"No `python`-tagged code block found in `<o>/<r>`'s `README.md`. Pass a blob URL to the specific file to wrap — e.g., `github.com/<o>/<r>/blob/<default_branch>/inference.py`."*
4. Run `ast.parse` on the extracted snippet. On `SyntaxError`: fail — *"First `python` block in README has syntax errors: `<msg>`. Pass a blob URL instead."*
5. **Local-import guard.** Walk the AST for `Import` / `ImportFrom` nodes and reject **relative imports only** (`from . import x`, `from .foo import y`, `from .. import z`) with *"README snippet uses a relative import, which isn't resolvable from a standalone scaffold. Pass a blob URL to the source file instead."* All absolute imports pass through. If an absolute import turns out to be local-only (no PyPI counterpart), Step 6's `uv add <name>` fails there with a clear error — late-binding but reliable, and it avoids the false-positive/negative traps of heuristic local-import detection at this step.
6. Feed the snippet into Step 2's AST walker.
7. Populate IR: `source_url = <original input URL>`, `source_ref = <default_branch>`, `license = <license.spdx_id from API>`.

## Step 2: Build the internal representation

Produce this structure in memory, consumed by all subsequent steps:

```python
{
    "pattern": "<UI pattern key from pipeline-tag-patterns.md>",
    "inference_fn": {"name": "...", "params": [...], "returns": "..."} or None,
    "data_fns": [...],
    "viz_fns": [...],
    "deps": ["pypi-name", ...],
    "is_gated": bool,
    "license": "<SPDX or license_name>",
    "mlx_equivalent": "<mlx-community/...>" or None,
    "siblings": ["<org>/<model>", ...],   # from Step 1's org-freshness check; [] when the org is not a priority org or no same-task siblings were found
    "source_url": "<original input URL>" or None,   # populated by GitHub URL branch only
    "source_ref": "<resolved git ref>" or None,     # populated by GitHub URL branch only
}
```

**Absent-value convention:** scalar fields (`inference_fn`, `mlx_equivalent`, `source_url`, `source_ref`) use `None` when absent; list fields (`data_fns`, `viz_fns`, `deps`, `siblings`) use `[]`. New fields follow the same pattern. Existing `.py` / `.ipynb` / HF-card branches leave `source_url` / `source_ref` as `None` — only the GitHub URL branch populates them.

**For code inputs (script or notebook):** AST-parse the code (`ast.parse` + walk `FunctionDef`) to extract top-level function signatures with type annotations. Classify each function as inference (calls `.predict`, `.generate`, `.forward`, `.__call__` on a model), data (reads/writes files, manipulates DataFrames), or viz (returns a matplotlib/plotly figure). Collect imports for dependency inference. **MLX resolution is not performed for code inputs** — `mlx_equivalent` stays `None` and `siblings` stays `[]` even when the snippet contains `from_pretrained("<org>/<model>")` string literals. MLX lookup fires only for HF-card URL inputs, per Step 1.

**For HF model card inputs:** Map fields directly from the API JSON. Derive `deps` from `library_name` + `tags` (e.g., `transformers` → `transformers` + `torch`; `diffusers` → `diffusers` + `transformers` + `torch`; `sentence-transformers` → `sentence-transformers` + `torch`). Extract the first library-idiomatic snippet from the README as the seed for `inference.py`'s transformers branch.

## Step 3: Classify the UI pattern

**HF input:** look up `pipeline_tag` in `references/pipeline-tag-patterns.md`. Use the matching page body template.

**Code input:** use the heuristic below. When multiple indicators match, classify on the most specific:

| Indicators                                                                       | Pattern                               |
|----------------------------------------------------------------------------------|---------------------------------------|
| `sklearn`, `torch`, `keras`, `.predict()`, loads a model                         | Inference (match to `pipeline_tag` if recognizable, else General Script) |
| `pandas` I/O + DataFrame transforms, no model                                    | Data processing page (file upload → transform → download) |
| `matplotlib`, `plotly`, `seaborn`, `.plot()`                                     | Visualization page (interactive chart controls) |
| Functions with scalar / text parameters, no model, no I/O                        | General Script                        |

Fall through to "General Script" when ambiguous. The corresponding template is in `references/pipeline-tag-patterns.md` under the Fallback section.

## Step 4: Fetch live Streamlit and HuggingFace docs

Before writing code, fetch the relevant pages from `docs.streamlit.io` and `huggingface.co/docs` and verify APIs match the templates below. Canonical URLs live in `references/streamlit-docs-index.md` and `references/huggingface-docs-index.md`.

### Streamlit docs

Fetched for every run — Streamlit is the output framework regardless of input type.

**Always:**
- Multipage + `st.navigation` + `st.Page`
- `@st.cache_resource` + `@st.cache_data`
- `streamlit.testing.v1.AppTest`
- Secrets (informs `.gitignore` entry and README guidance)
- File organization

**Conditional — based on classified pattern:**
- Chat: `st.chat_input`, `st.chat_message`
- ASR / audio: `st.audio_input`, `st.audio`
- Image: `st.file_uploader`, `st.image`
- Visualization: `st.plotly_chart`, `st.line_chart`, `st.dataframe`

### HuggingFace docs

**When to fetch the baseline set** (Hub security tokens, `huggingface-cli` login):
- **HF-card input:** always.
- **Code / notebook input:** when Step 2's AST walk detected an import of `huggingface_hub`, `transformers`, or `diffusers`. If none of those is imported, skip the HF docs subsection entirely.

**Library-conditional fetches:** include a row from `references/huggingface-docs-index.md` when its trigger matches:
- `library_name == "transformers"` (HF-card) or `transformers` imported (code input) → Pipelines + Auto classes.
- Additionally, when `pipeline_tag == "automatic-speech-recognition"` → ASR task guide.
- Additionally, when `pipeline_tag == "text-to-speech"` → TTS task guide.
- `library_name == "diffusers"` (HF-card) or `diffusers` imported (code input) → Loading pipelines + Quick tour.

### Rules for both sources

If any fetched page shows an API that differs from the template in this file, prefer the fetched docs. Update the template accordingly before generating the app. When a page is unreachable, proceed with the templates here and note in the final report: "live verification skipped for <URL>".

## Step 5: Scaffold files

Create the following directory tree (substitute `<app-name>` / `<app_name>`):

```
<app-name>/
├── .streamlit/
│   └── config.toml
├── src/
│   └── <app_name>/
│       ├── __init__.py
│       ├── config.py
│       ├── inference.py
│       ├── data.py
│       ├── viz.py
│       └── pages/
│           ├── __init__.py
│           ├── home.py
│           └── <feature>.py           # only when source has multiple flows
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_inference.py
│   ├── test_data.py
│   ├── test_viz.py                    # only when viz.py exists
│   └── test_app_smoke.py
├── streamlit_app.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
└── README.md
```

`<app-name>` is hyphenated for the outer directory / `uv init` project name. `<app_name>` is its Python-safe equivalent (underscores) for the importable package — `uv init --package` performs this normalization automatically.

**Always generate multipage** even when the source has a single flow. A single `home.py` costs almost nothing and makes adding pages later frictionless.

### `streamlit_app.py` (entrypoint — navigation router)

```python
"""Streamlit entrypoint. Registers pages with st.navigation."""
import streamlit as st

from <app_name>.pages import home


def main() -> None:
    pages = [
        st.Page(home.render, title="Home", icon=":material/home:"),
        # Add st.Page(<module>.render, title="...") as the app grows.
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
```

### `.streamlit/config.toml`

```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
# Override as the team's branding requires; defaults are sensible.
```

### `src/<app_name>/config.py`

Single source of truth for environment config. Imported by every other module; exposes module-level constants loaded from the environment at import time. Raises `RuntimeError` on missing required vars.

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
DEVICE: str = _get("DEVICE", "auto")          # "auto" | "cpu" | "cuda" | "mps"
MAX_NEW_TOKENS: int = int(_get("MAX_NEW_TOKENS", "512"))

# Optional secrets. For gated models, swap this line for:
#   HF_TOKEN: str = _require("HF_TOKEN")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# Runtime-detected
IS_APPLE_SILICON: bool = (
    sys.platform == "darwin" and platform.machine() == "arm64"
)
```

**Gated models:** if the source HF card has `gated: true`, emit `HF_TOKEN = _require("HF_TOKEN")` instead of the optional form.

### `.env.example`

Every `_require` and `_get` key in `config.py` gets a line here with a placeholder default and a comment. Example:

```
# Required
MODEL_ID=<org>/<model>

# Optional
MODEL_REVISION=main
DEVICE=auto                  # auto | cpu | cuda | mps
MAX_NEW_TOKENS=512

# Optional unless the model is gated
HF_TOKEN=
```

### `src/<app_name>/inference.py` (with platform dispatch)

When the source is a model-based artifact, `inference.py` loads the model and wraps calls. It dispatches between MLX and transformers by reading `config.IS_APPLE_SILICON`. The template below is for `text-generation`; other pipeline tags use the equivalent library calls (see `references/pipeline-tag-patterns.md` for signatures).

```python
"""Model loading and inference. Dispatches MLX <-> transformers by platform."""
from functools import lru_cache
from typing import Any

from <app_name> import config

# MLX model ID chosen at scaffold time (highest downloads under mlx-community).
# Override by setting MLX_MODEL_ID in .env.
MLX_MODEL_ID_DEFAULT: str | None = "<mlx-community/...>"  # set when Step 1 found a match; else None


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Lazy-load the model once per process."""
    if config.IS_APPLE_SILICON and MLX_MODEL_ID_DEFAULT:
        return _load_mlx()
    return _load_transformers()


def _load_mlx():
    from mlx_lm import load

    import os as _os
    mlx_id = _os.getenv("MLX_MODEL_ID", MLX_MODEL_ID_DEFAULT)
    model, tokenizer = load(mlx_id)
    return ("mlx", model, tokenizer)


def _load_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, token=config.HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION, token=config.HF_TOKEN
    )
    return ("transformers", model, tokenizer)


def generate_response(prompt: str, max_new_tokens: int | None = None) -> str:
    backend, model, tokenizer = load_model()
    max_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    if backend == "mlx":
        from mlx_lm import generate
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)
```

For non-text-generation pipelines, each variant still dispatches via `config.IS_APPLE_SILICON` and exposes a function named per the page template (`transcribe`, `synthesize`, `transform_audio`, `caption`, `classify`, etc.). Backend call shapes:

| Pipeline | MLX branch | Transformers branch |
|---|---|---|
| `image-to-text`, `image-text-to-text` | `from mlx_vlm import load, generate` → `generate(model, processor, formatted_prompt, image)` | `pipeline("image-to-text", model=config.MODEL_ID)` |
| `automatic-speech-recognition` | `from mlx_audio.stt.utils import load` → `load(id).generate(audio).text` | `pipeline("automatic-speech-recognition", model=config.MODEL_ID)` |
| `text-to-speech` | `from mlx_audio.tts.utils import load_model`; iterate `load_model(id).generate(text=..., voice=...)` and concatenate each result's `.audio` | `pipeline("text-to-speech", model=config.MODEL_ID)` |
| `audio-to-audio` | `mlx_audio.sts.<ModelClass>.from_pretrained(id)` + model-specific method (e.g. `.enhance(audio)`, `.separate_long(...)`). **Apple-only.** | — (`RuntimeError` on non-Apple-Silicon hosts) |

For `audio-to-audio`, the exact `mlx_audio.sts` class and method depend on the model (SAM-Audio → `separate_long`, MossFormer2 → `enhance`, DeepFilterNet → `enhance`). Step 2 maps the HF card's tags/name to a known `mlx_audio.sts` class; if no mapping exists, the skill reports "no supported STS backend" and emits a General Script page with a manual-wiring TODO instead of scaffolding broken inference code.

### `src/<app_name>/data.py` and `viz.py`

Only generated when the source contains data transforms or visualizations. Each file holds pure functions (no Streamlit imports) extracted from the source — preserve the original logic verbatim; do not rewrite. Add type annotations on all signatures.

### `src/<app_name>/pages/home.py`

Uses the page body from `references/pipeline-tag-patterns.md` that matches the classified pattern. Every page module exposes `render() -> None`:

```python
"""Home page."""
import streamlit as st


def render() -> None:
    st.title("...")
    # Body per references/pipeline-tag-patterns.md for the classified pattern.
```

When the source has multiple independent flows, generate one page module per flow (`home.py`, `<feature>.py`) and register them all in `streamlit_app.py`'s navigation list.

### `tests/conftest.py`

Sets required env vars before any test imports the package. Provides a mocked-model fixture so inference tests do not touch the network.

```python
"""Test fixtures. Set required env vars before package import."""
import os

os.environ.setdefault("MODEL_ID", "test-model")
# os.environ.setdefault("HF_TOKEN", "test-token")  # enable for gated models

import pytest


class _StubModel:
    """Minimal interface used by inference.py."""
    def generate(self, *args, **kwargs):
        return [[0, 1, 2]]

    def predict(self, x):
        return [0] * len(x)


@pytest.fixture
def mock_model(monkeypatch):
    from <app_name> import inference
    monkeypatch.setattr(inference, "load_model", lambda: ("stub", _StubModel(), None))
    return _StubModel()
```

### `tests/test_config.py`

```python
"""Tests for src/<app_name>/config.py."""
import importlib
import os

import pytest


def test_require_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MODEL_ID", raising=False)
    # Re-import to trigger fail-fast with the missing var.
    import <app_name>.config as cfg
    importlib.reload(cfg)  # will raise

# Note: because config.py raises at import, the assertion happens during reload.
# Surround with pytest.raises to capture:
def test_require_raises_explicit(monkeypatch):
    monkeypatch.delenv("MODEL_ID", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_ID"):
        import <app_name>.config as cfg
        importlib.reload(cfg)


def test_get_returns_default(monkeypatch):
    monkeypatch.delenv("DEVICE", raising=False)
    import <app_name>.config as cfg
    importlib.reload(cfg)
    assert cfg.DEVICE == "auto"


def test_is_apple_silicon_detection(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    import <app_name>.config as cfg
    importlib.reload(cfg)
    assert cfg.IS_APPLE_SILICON is True
```

### `tests/test_inference.py`

```python
"""Tests for src/<app_name>/inference.py with mocked model."""
from <app_name> import inference


def test_generate_response_uses_loaded_model(mock_model):
    out = inference.generate_response("hello", max_new_tokens=5)
    assert isinstance(out, str)
```

### `tests/test_app_smoke.py`

```python
"""Smoke test: boots the app and runs the default page, asserts no exceptions."""
from streamlit.testing.v1 import AppTest


def test_app_boots():
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    assert not at.exception
```

### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
.venv/
venv/
.env

# Streamlit
.streamlit/secrets.toml

# Tooling
.ruff_cache/
.pytest_cache/

# OS
.DS_Store
```

`.streamlit/secrets.toml` is ignored preemptively even though the skill does not create one — prevents accidental commits if the team later adds one for Streamlit Community Cloud deployment.

### `README.md`

The generated README covers setup, env vars, license, and gated-model instructions. Template:

````markdown
# <App Name>

<One-line description extracted from the source>

<When `source_url` is non-None, insert: **Source:** [<source_url>](<source_url>) (`<source_ref>`). Omit the entire line otherwise.>

## Setup

```bash
uv sync
cp .env.example .env      # then edit .env with your values
streamlit run streamlit_app.py
```

## Environment variables

| Name             | Required | Default | Description                                       |
|------------------|----------|---------|---------------------------------------------------|
| `MODEL_ID`       | yes      | —       | HuggingFace model identifier                      |
| `MODEL_REVISION` | no       | `main`  | Model revision / branch / tag                     |
| `DEVICE`         | no       | `auto`  | `auto` \| `cpu` \| `cuda` \| `mps`                |
| `MAX_NEW_TOKENS` | no       | `512`   | Max tokens generated per call                     |
| `HF_TOKEN`       | see notes | —      | Required for gated / private models; otherwise optional |

## License & Commercial Use

**Model:** `<org>/<model>` — license: `<license-identifier>`

<When the license is in license-flags.md as restrictive, insert the flag text here>

## Gated model

<Included when the source model card has gated: true>
Run `huggingface-cli login` on the host before first use, OR set `HF_TOKEN` in `.env` / your platform's secret store. Without a token, the app will fail fast at startup with a clear error.
````

## Step 6: Initialize and install via `uv`

```bash
pip install uv --break-system-packages   # if uv is not already available

uv init --name <app-name> --package
uv add streamlit python-dotenv huggingface_hub
uv add "mlx-lm;platform_machine=='arm64' and sys_platform=='darwin'"
uv add "transformers;platform_machine!='arm64' or sys_platform!='darwin'"
uv add --dev ruff ty pytest
```

Versions are not pinned on the command line — `uv add` resolves the current latest at skill-run time and writes the resolved specifier to `pyproject.toml`.

**Pattern-specific additional deps:**

| Pattern                              | Add                                    |
|--------------------------------------|----------------------------------------|
| Text generation (non-transformers)   | `accelerate` (transformers branch)     |
| Image / vision / diffusion           | `diffusers`, `accelerate`, `pillow`    |
| Automatic speech recognition         | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"`, `"transformers[audio];platform_machine!='arm64' or sys_platform!='darwin'"` |
| Text to speech                       | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"`, `"transformers[audio];platform_machine!='arm64' or sys_platform!='darwin'"` |
| Audio to audio (STS)                 | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"` — **no fallback** (Apple-only) |
| Vision-language                      | `"mlx-vlm;platform_machine=='arm64' and sys_platform=='darwin'"` |
| Embeddings                           | `sentence-transformers`                |
| Data processing                      | `pandas`, `pyarrow`                    |
| Visualization                        | `plotly`                               |

Add `[tool.pytest.ini_options]` to `pyproject.toml` with `testpaths = ["tests"]` and `pythonpath = ["src"]`. Configure `[tool.ruff]` with import-sorting (`select = ["E", "F", "I"]`). Configure `[tool.ty]` with `strict = ["src/"]`.

## Step 7: Code-quality gate

Run these four commands, in order. All must pass before reporting the scaffold as complete.

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ty check src/ tests/
uv run pytest -v
```

Fix failures by adjusting the generated code or fixtures. Do not weaken tests to make them pass. If the smoke test fails because a required env var is missing, add the var to `tests/conftest.py`'s startup block.

## Step 8: Report to user

**Source preamble (conditional):** when `source_url` is non-None, emit the following before the numbered items below, followed by a blank line:

```
Source: <source_url>
Ref: <source_ref>
```

The `Ref:` line is included only when `source_ref` is non-None. Omit the entire preamble when `source_url` is None (HF-card / script / notebook paths) — no empty header.

Surface:

1. **Files created**, grouped by purpose: app code, config, tests, project files.
2. **Chosen model variant** — if MLX resolution returned a match, show `mlx-community/<variant>` alongside the original `<org>/<model>`; otherwise note "no MLX equivalent found, app uses transformers on all platforms."

   **Sibling models (conditional)** — when Step 2's `siblings` list is non-empty, append:

   ```
   Other <org> models for this task (consider if you want something newer/different):
     - <org>/<sibling1>
     - <org>/<sibling2>
   ```

   Omit entirely when `siblings` is empty — no empty header.
3. **Apple-Silicon-only warning (when applicable)** — if the classified pipeline is `audio-to-audio`, state: "This scaffold requires Apple Silicon at runtime. On non-Apple-Silicon hosts (including Intel Macs), `uv sync` will not install `mlx-audio` and the app will error at model load."
4. **License + commercial-use flag** — from `references/license-flags.md`, if the model's license matches a flagged entry. Quote the flag text inline.
5. **Gated-model setup** — when the source card had `gated: true`, show the `huggingface-cli login` command and the alternative `HF_TOKEN` path.
6. **Exact local-run command:**

   ```bash
   uv sync
   cp .env.example .env
   streamlit run streamlit_app.py
   ```

7. **Non-goals reminder** — a short list of things the scaffold does NOT include (auth, Docker, CI, DB, observability), explicitly marked as the team's responsibility.

## Output checklist

- [ ] Full directory tree populated (see Step 5)
- [ ] `pyproject.toml` declares MLX and transformers with correct environment markers
- [ ] `.env.example` covers every `_require` and `_get` key in `config.py`
- [ ] `ruff check --fix`, `ruff format`, `ty check`, and `pytest` all pass clean
- [ ] `README.md` documents setup, env vars, license, gated-model instructions
- [ ] Report to user surfaces MLX resolution + sibling models (when applicable) + license flags + non-goals reminder

## References

- `references/streamlit-docs-index.md` — canonical Streamlit docs URLs for live fetches
- `references/huggingface-docs-index.md` — canonical HuggingFace docs URLs (under `huggingface.co/docs`) for live fetches
- `references/pipeline-tag-patterns.md` — HF pipeline_tag → UI pattern catalog
- `references/license-flags.md` — commercial-use flags for model licenses
