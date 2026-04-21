# Streamlit App Builder Production Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the `streamlit-app-builder` skill to produce production-structured Streamlit apps (modular package, multipage, env-based config, tests) from notebooks, scripts, or HuggingFace model card URLs, with live docs verification and MLX-on-Apple-Silicon preference.

**Architecture:** Replace the single-file-app skill with a scaffold generator that writes a `src/<app_name>/` Python package, a multipage `st.navigation` router, a fail-fast `config.py`, and tests (including a smoke test via `streamlit.testing.v1.AppTest`). Add three reference documents for long lookups (Streamlit docs index, pipeline-tag pattern catalog, license flags) so `SKILL.md` stays navigable. Update root `README.md` and `CLAUDE.md` to reflect the new scope.

**Tech Stack:** Markdown (`SKILL.md` + references), Python 3.11+ target for generated apps, `uv` + `ruff` + `ty` + `pytest` toolchain (unchanged), `streamlit.testing.v1.AppTest` for the smoke test, `huggingface_hub` + (`transformers` | `mlx-lm` | `mlx-vlm` | `mlx-whisper`) in generated `pyproject.toml` via environment markers.

**Spec reference:** `docs/superpowers/specs/2026-04-21-streamlit-app-builder-production-rewrite-design.md`

---

## File Structure

Files created or modified by this plan:

**Created:**
- `streamlit-app-builder/references/streamlit-docs-index.md` — canonical `docs.streamlit.io` URLs by topic
- `streamlit-app-builder/references/pipeline-tag-patterns.md` — full `pipeline_tag` → UI pattern catalog with widget snippets
- `streamlit-app-builder/references/license-flags.md` — SPDX identifiers and commercial-use flags

**Modified:**
- `streamlit-app-builder/SKILL.md` — full rewrite (current 262 lines → ~new content)
- `README.md` — update the `streamlit-app-builder` row
- `CLAUDE.md` — carve out `streamlit-app-builder` from the shared-workflow section

The three reference files keep `SKILL.md` focused on workflow, since each reference is a large lookup table that would bloat the primary file.

---

## Task 1: Create `references/` directory and stub the docs index

**Files:**
- Create: `streamlit-app-builder/references/streamlit-docs-index.md`

- [ ] **Step 1: Create the directory**

```bash
mkdir -p streamlit-app-builder/references
```

- [ ] **Step 2: Write `streamlit-docs-index.md`**

Write the file with content:

````markdown
# Streamlit Docs Index

Canonical `docs.streamlit.io` URLs used by the `streamlit-app-builder` skill. Fetch these at skill-run time to verify current APIs before generating code.

## High-churn topics (mandatory fetch before generation)

| Topic              | URL                                                                     |
|--------------------|-------------------------------------------------------------------------|
| Multipage apps     | https://docs.streamlit.io/develop/concepts/multipage-apps/overview      |
| `st.navigation`    | https://docs.streamlit.io/develop/api-reference/navigation/st.navigation|
| `st.Page`          | https://docs.streamlit.io/develop/api-reference/navigation/st.page      |
| Caching            | https://docs.streamlit.io/develop/concepts/architecture/caching         |
| `@st.cache_resource` | https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource |
| `@st.cache_data`   | https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data |
| App testing        | https://docs.streamlit.io/develop/concepts/app-testing                  |
| `AppTest`          | https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest |
| Secrets            | https://docs.streamlit.io/develop/concepts/connections/secrets-management |
| File organization  | https://docs.streamlit.io/develop/concepts/multipage-apps/overview#file-organization |
| Config             | https://docs.streamlit.io/develop/api-reference/configuration/config.toml |

## Widget APIs (fetch the subset relevant to the classified UI pattern)

| Widget              | URL                                                                     |
|---------------------|-------------------------------------------------------------------------|
| `st.chat_input`     | https://docs.streamlit.io/develop/api-reference/chat/st.chat_input      |
| `st.chat_message`   | https://docs.streamlit.io/develop/api-reference/chat/st.chat_message    |
| `st.audio_input`    | https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input  |
| `st.file_uploader`  | https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader|
| `st.text_input`     | https://docs.streamlit.io/develop/api-reference/widgets/st.text_input   |
| `st.text_area`      | https://docs.streamlit.io/develop/api-reference/widgets/st.text_area    |
| `st.selectbox`      | https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox    |
| `st.number_input`   | https://docs.streamlit.io/develop/api-reference/widgets/st.number_input |
| `st.slider`         | https://docs.streamlit.io/develop/api-reference/widgets/st.slider       |
| `st.image`          | https://docs.streamlit.io/develop/api-reference/media/st.image          |
| `st.audio`          | https://docs.streamlit.io/develop/api-reference/media/st.audio          |
| `st.dataframe`      | https://docs.streamlit.io/develop/api-reference/data/st.dataframe       |
| `st.plotly_chart`   | https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart  |

## Root

| Area                | URL                                   |
|---------------------|---------------------------------------|
| Docs root           | https://docs.streamlit.io/            |
| API reference index | https://docs.streamlit.io/develop/api-reference |
````

- [ ] **Step 3: Verify**

Run: `ls streamlit-app-builder/references/streamlit-docs-index.md && grep -c "https://docs.streamlit.io" streamlit-app-builder/references/streamlit-docs-index.md`
Expected: file exists; grep count ≥ 20.

- [ ] **Step 4: Commit**

```bash
git add streamlit-app-builder/references/streamlit-docs-index.md
git commit -m "Add Streamlit docs URL reference for streamlit-app-builder skill"
```

---

## Task 2: Write `pipeline-tag-patterns.md`

**Files:**
- Create: `streamlit-app-builder/references/pipeline-tag-patterns.md`

- [ ] **Step 1: Write the file**

Content:

````markdown
# HuggingFace `pipeline_tag` → UI Pattern Catalog

When the input is an HF model card, the skill reads `pipeline_tag` from `https://huggingface.co/api/models/<id>` (or the YAML frontmatter of `README.md` as fallback) and selects a UI pattern from this catalog. Each entry specifies the primary Streamlit widgets and a minimal page body.

## Text generation / chat

`pipeline_tag`: `text-generation`, `conversational`

```python
# Chat page body
import streamlit as st
from <app_name>.inference import generate_response

st.title("Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

## Classification (text)

`pipeline_tag`: `text-classification`, `zero-shot-classification`

```python
import streamlit as st
from <app_name>.inference import classify

st.title("Classify Text")
text = st.text_area("Input", height=200)
if st.button("Classify") and text:
    result = classify(text)
    st.json(result)
```

## Token classification (NER)

`pipeline_tag`: `token-classification`

```python
import streamlit as st
from <app_name>.inference import extract_entities

st.title("Entity Recognition")
text = st.text_area("Input", height=200)
if st.button("Extract") and text:
    entities = extract_entities(text)
    st.write(entities)
    # Render highlighted spans via markdown with inline HTML,
    # or use a third-party component if preferred.
```

## Question answering

`pipeline_tag`: `question-answering`

```python
import streamlit as st
from <app_name>.inference import answer

st.title("Question Answering")
context = st.text_area("Context", height=250)
question = st.text_input("Question")
if st.button("Answer") and context and question:
    st.write(answer(context=context, question=question))
```

## Summarization / translation

`pipeline_tag`: `summarization`, `translation`

```python
import streamlit as st
from <app_name>.inference import transform_text

st.title("Transform Text")
src = st.text_area("Input", height=250)
if st.button("Run") and src:
    st.text_area("Output", transform_text(src), height=250)
```

## Feature extraction (embeddings)

`pipeline_tag`: `feature-extraction`

```python
import streamlit as st
from <app_name>.inference import embed

st.title("Embeddings")
text = st.text_area("Input", height=200)
if st.button("Embed") and text:
    vec = embed(text)
    st.write(f"Dim: {len(vec)}")
    st.line_chart(vec)  # quick visualization
```

## Automatic speech recognition (ASR)

`pipeline_tag`: `automatic-speech-recognition`

```python
import streamlit as st
from <app_name>.inference import transcribe

st.title("Transcribe Audio")
audio = st.audio_input("Record") or st.file_uploader("Upload", type=["wav", "mp3", "m4a", "flac"])
if audio and st.button("Transcribe"):
    text = transcribe(audio)
    st.text_area("Transcript", text, height=200)
```

## Text to speech

`pipeline_tag`: `text-to-speech`

```python
import streamlit as st
from <app_name>.inference import synthesize

st.title("Text to Speech")
text = st.text_area("Input", height=150)
if st.button("Speak") and text:
    audio_bytes = synthesize(text)
    st.audio(audio_bytes, format="audio/wav")
```

## Image classification / object detection

`pipeline_tag`: `image-classification`, `object-detection`

```python
import streamlit as st
from <app_name>.inference import classify_image

st.title("Classify Image")
img = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"])
if img and st.button("Classify"):
    st.image(img)
    st.json(classify_image(img))
```

## Image to text (captioning, VQA)

`pipeline_tag`: `image-to-text`

```python
import streamlit as st
from <app_name>.inference import caption

st.title("Image Captioning")
img = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"])
if img and st.button("Caption"):
    st.image(img)
    st.write(caption(img))
```

## Image-to-image / text-to-image

`pipeline_tag`: `image-to-image`, `text-to-image`

```python
import streamlit as st
from <app_name>.inference import generate_image

st.title("Generate Image")
prompt = st.text_area("Prompt", height=100)
if st.button("Generate") and prompt:
    img = generate_image(prompt)
    st.image(img)
```

## Fallback: General Script

`pipeline_tag`: missing / unrecognized / not applicable (code-based input without a clear pattern)

```python
import streamlit as st
import pandas as pd
from <app_name>.inference import run  # or equivalent entry from source

st.title("Run")
# TODO: replace with widgets that expose run()'s parameters
param = st.text_input("Parameter")
if st.button("Run") and param:
    result = run(param)
    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif isinstance(result, (dict, list)):
        st.json(result)
    else:
        st.write(result)
```
````

- [ ] **Step 2: Verify**

Run: `grep -c "pipeline_tag" streamlit-app-builder/references/pipeline-tag-patterns.md && grep -c "^## " streamlit-app-builder/references/pipeline-tag-patterns.md`
Expected: ≥ 11 for both (11+ pipeline_tag entries, 11+ ## headings).

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/references/pipeline-tag-patterns.md
git commit -m "Add pipeline_tag -> UI pattern catalog for streamlit-app-builder"
```

---

## Task 3: Write `license-flags.md`

**Files:**
- Create: `streamlit-app-builder/references/license-flags.md`

- [ ] **Step 1: Write the file**

Content:

````markdown
# License Flags

Licenses that the `streamlit-app-builder` skill flags in generated `README.md` under "License & Commercial Use". When the model card's `license` field matches (or `license_name` for custom licenses), surface the flag prominently — teams deploying to paying customers need to know commercial restrictions upfront.

## Licenses that restrict commercial use

| License identifier / name         | Commercial use                                                   | Notes                                                                 |
|-----------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------|
| `cc-by-nc-2.0`, `-3.0`, `-4.0`    | ❌ Not allowed                                                    | Non-commercial only                                                   |
| `cc-by-nc-sa-*`                   | ❌ Not allowed                                                    | Non-commercial + share-alike                                          |
| `cc-by-nc-nd-*`                   | ❌ Not allowed                                                    | Non-commercial + no-derivatives                                       |
| `llama2`                          | ⚠️ Restricted                                                    | Llama 2 Community License — free under 700M MAU threshold             |
| `llama3`, `llama3.1`, `llama3.2`, `llama3.3` | ⚠️ Restricted                                         | Llama Community License — 700M MAU threshold                          |
| `gemma`                           | ⚠️ Restricted                                                    | Gemma Terms of Use — use-case restrictions apply                      |
| `other` with `license_name: mistral-ai-research-license` | ❌ Research-only                           | Mistral Research License                                              |
| `openrail`, `bigscience-openrail-m`, `creativeml-openrail-m` | ⚠️ Use-case restricted                | RAIL licenses — enumerated prohibited uses                            |
| `deepfloyd-if-license`            | ⚠️ Research-only by default                                      | Commercial permitted after separate application                       |
| `apple-ascl`                      | ⚠️ Restricted                                                    | Apple Sample Code License — internal/research                         |

## Licenses that permit commercial use (no flag needed)

Surface the license name in the README but do not add a warning:

- `mit`
- `apache-2.0`
- `bsd-2-clause`, `bsd-3-clause`
- `cc0-1.0`
- `cc-by-2.0`, `cc-by-3.0`, `cc-by-4.0`
- `cc-by-sa-*` (share-alike obligations apply)
- `mpl-2.0`
- `unlicense`

## Unknown / missing license

When `license` is missing or set to `unknown`: surface as a warning — "License not specified on model card. Verify licensing before commercial deployment."
````

- [ ] **Step 2: Verify**

Run: `grep -c "cc-by-nc" streamlit-app-builder/references/license-flags.md && grep -c "llama" streamlit-app-builder/references/license-flags.md`
Expected: ≥ 3 and ≥ 4 respectively.

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/references/license-flags.md
git commit -m "Add license commercial-use flags reference for streamlit-app-builder"
```

---

## Task 4: Rewrite `SKILL.md` — frontmatter, intro, cross-cutting principles

**Files:**
- Modify (full rewrite starting here): `streamlit-app-builder/SKILL.md`

This task replaces the current `SKILL.md` in full with the new content, section by section. Subsequent tasks append more sections via `Edit` (matching anchor strings). This task writes the file header plus the first two sections.

- [ ] **Step 1: Replace the file**

Use the `Write` tool to fully replace the current contents with:

````markdown
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

### 1. Always verify against live `docs.streamlit.io`

Streamlit's API evolves. Before generating code, fetch the current Streamlit docs for high-churn topics and confirm API shapes. Canonical URLs live in `references/streamlit-docs-index.md`.

**Mandatory fetches** before code generation:
- Multipage + `st.navigation` + `st.Page`
- Caching (`@st.cache_resource`, `@st.cache_data`)
- App testing (`streamlit.testing.v1.AppTest`)
- Secrets
- File organization
- Plus the widget page(s) for the classified UI pattern (e.g., `st.chat_input` for chat, `st.audio_input` for ASR)

### 2. Prefer MLX on Apple Silicon

When the source artifact references a model with an MLX-converted equivalent at `huggingface.co/mlx-community/...`, the generated app uses the MLX backend (`mlx-lm` for text, `mlx-vlm` for vision-language, `mlx-whisper` for ASR) when running on `arm64-darwin`, and falls back to `transformers` / `diffusers` / `huggingface_hub` elsewhere.

The MLX lookup is independent of where the skill itself runs. A Linux developer scaffolding from an HF model card URL still produces an app with MLX support wired in — runtime dispatch activates MLX only when a user later runs the app on a Mac.

MLX support is encoded in the generated app as follows:
- `pyproject.toml` declares MLX and transformers with environment markers so `uv sync` installs the right backend per host.
- `src/<app_name>/inference.py` reads `config.IS_APPLE_SILICON` and dispatches to the appropriate backend at runtime.

**MLX model resolution:** query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` and pick the highest-download-count variant. The chosen variant is noted in `inference.py` with override instructions.
````

- [ ] **Step 2: Verify**

Run: `grep -c "^##" streamlit-app-builder/SKILL.md && grep -q "mlx-community" streamlit-app-builder/SKILL.md && grep -q "docs.streamlit.io" streamlit-app-builder/SKILL.md`
Expected: ≥ 3 and exit 0.

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/SKILL.md
git commit -m "Rewrite streamlit-app-builder SKILL.md: frontmatter, intro, cross-cutting principles"
```

---

## Task 5: Append Step 1–3 (input identification, representation, classification)

**Files:**
- Modify: `streamlit-app-builder/SKILL.md` (append sections)

- [ ] **Step 1: Append sections**

Use the `Edit` tool. `old_string` is the final line of Task 4's content (the end of Principle 2). `new_string` is that same line followed by the new sections.

`old_string`:
```
**MLX model resolution:** query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` and pick the highest-download-count variant. The chosen variant is noted in `inference.py` with override instructions.
```

`new_string`:
````
**MLX model resolution:** query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` and pick the highest-download-count variant. The chosen variant is noted in `inference.py` with override instructions.

## Step 1: Identify and load the input

Classify the input into one of three types, then load it.

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
}
```

**For code inputs (script or notebook):** AST-parse the code (`ast.parse` + walk `FunctionDef`) to extract top-level function signatures with type annotations. Classify each function as inference (calls `.predict`, `.generate`, `.forward`, `.__call__` on a model), data (reads/writes files, manipulates DataFrames), or viz (returns a matplotlib/plotly figure). Collect imports for dependency inference.

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
````

- [ ] **Step 2: Verify**

Run: `grep -c "^## Step " streamlit-app-builder/SKILL.md`
Expected: 3.

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/SKILL.md
git commit -m "Add Steps 1-3 to streamlit-app-builder SKILL.md: input, representation, classification"
```

---

## Task 6: Append Step 4 (live docs fetch) and Step 5a (directory + entrypoint)

**Files:**
- Modify: `streamlit-app-builder/SKILL.md`

- [ ] **Step 1: Append sections**

`old_string`:
```
Fall through to "General Script" when ambiguous. The corresponding template is in `references/pipeline-tag-patterns.md` under the Fallback section.
```

`new_string`:
````
Fall through to "General Script" when ambiguous. The corresponding template is in `references/pipeline-tag-patterns.md` under the Fallback section.

## Step 4: Fetch live Streamlit docs

Before writing code, fetch the following pages from `docs.streamlit.io` and verify APIs match the templates below. See `references/streamlit-docs-index.md` for full URLs.

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

If any fetched page shows an API that differs from the template in this file, prefer the fetched docs. Update the template accordingly before generating the app. When docs are unreachable, proceed with the templates here and note in the final report that live verification was skipped.

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
````

- [ ] **Step 2: Verify**

Run: `grep -c "^## Step " streamlit-app-builder/SKILL.md && grep -q "st.navigation" streamlit-app-builder/SKILL.md`
Expected: 5 and exit 0.

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/SKILL.md
git commit -m "Add Step 4 (docs fetch) and Step 5 start (dir tree + entrypoint) to SKILL.md"
```

---

## Task 7: Append Step 5b (`config.py`, `.env.example`, `inference.py` with dispatch)

**Files:**
- Modify: `streamlit-app-builder/SKILL.md`

- [ ] **Step 1: Append sections**

`old_string`:
```
[theme]
# Override as the team's branding requires; defaults are sensible.
```
```

(Note: that `old_string` includes the closing triple backticks of the previous TOML fence.)

`new_string`:
````
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

For non-text-generation pipelines, substitute the library calls: `mlx_vlm.load`/`generate` for vision-language, `mlx_whisper.transcribe` for ASR, `transformers.pipeline(<task>, ...)` for the transformers branch. Each variant still dispatches via `config.IS_APPLE_SILICON` and exposes a function named per the pipeline (`transcribe`, `caption`, `classify`, etc.) matching the page template.
````

- [ ] **Step 2: Verify**

Run: `grep -c "IS_APPLE_SILICON" streamlit-app-builder/SKILL.md && grep -q "load_dotenv" streamlit-app-builder/SKILL.md`
Expected: ≥ 2 and exit 0.

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/SKILL.md
git commit -m "Add config.py, .env.example, and inference.py dispatch templates to SKILL.md"
```

---

## Task 8: Append Step 5c (pages, data/viz, tests, .gitignore, README template)

**Files:**
- Modify: `streamlit-app-builder/SKILL.md`

- [ ] **Step 1: Append sections**

`old_string`:
```
For non-text-generation pipelines, substitute the library calls: `mlx_vlm.load`/`generate` for vision-language, `mlx_whisper.transcribe` for ASR, `transformers.pipeline(<task>, ...)` for the transformers branch. Each variant still dispatches via `config.IS_APPLE_SILICON` and exposes a function named per the pipeline (`transcribe`, `caption`, `classify`, etc.) matching the page template.
```

`new_string`:
````
For non-text-generation pipelines, substitute the library calls: `mlx_vlm.load`/`generate` for vision-language, `mlx_whisper.transcribe` for ASR, `transformers.pipeline(<task>, ...)` for the transformers branch. Each variant still dispatches via `config.IS_APPLE_SILICON` and exposes a function named per the pipeline (`transcribe`, `caption`, `classify`, etc.) matching the page template.

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
````

- [ ] **Step 2: Verify**

Run: `grep -c "AppTest" streamlit-app-builder/SKILL.md && grep -q "mock_model" streamlit-app-builder/SKILL.md && grep -q "\.gitignore" streamlit-app-builder/SKILL.md`
Expected: ≥ 1 and exit 0.

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/SKILL.md
git commit -m "Add pages, tests, .gitignore, README templates to SKILL.md"
```

---

## Task 9: Append Steps 6–8 (uv init, quality gate, report)

**Files:**
- Modify: `streamlit-app-builder/SKILL.md`

- [ ] **Step 1: Append sections**

`old_string`:
```
Run `huggingface-cli login` on the host before first use, OR set `HF_TOKEN` in `.env` / your platform's secret store. Without a token, the app will fail fast at startup with a clear error.
```

`new_string`:
````
Run `huggingface-cli login` on the host before first use, OR set `HF_TOKEN` in `.env` / your platform's secret store. Without a token, the app will fail fast at startup with a clear error.

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
| ASR                                  | `"mlx-whisper;platform_machine=='arm64' and sys_platform=='darwin'"`, `"openai-whisper;platform_machine!='arm64' or sys_platform!='darwin'"` — or substitute `transformers[audio]` |
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

Surface:

1. **Files created**, grouped by purpose: app code, config, tests, project files.
2. **Chosen model variant** — if MLX resolution returned a match, show `mlx-community/<variant>` alongside the original `<org>/<model>`; otherwise note "no MLX equivalent found, app uses transformers on all platforms."
3. **License + commercial-use flag** — from `references/license-flags.md`, if the model's license matches a flagged entry. Quote the flag text inline.
4. **Gated-model setup** — when the source card had `gated: true`, show the `huggingface-cli login` command and the alternative `HF_TOKEN` path.
5. **Exact local-run command:**

   ```bash
   uv sync
   cp .env.example .env
   streamlit run streamlit_app.py
   ```

6. **Non-goals reminder** — a short list of things the scaffold does NOT include (auth, Docker, CI, DB, observability), explicitly marked as the team's responsibility.
````

- [ ] **Step 2: Verify**

Run: `grep -c "^## Step " streamlit-app-builder/SKILL.md`
Expected: 8.

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/SKILL.md
git commit -m "Add Steps 6-8 (uv init, quality gate, report) to SKILL.md"
```

---

## Task 10: Append output checklist and appendix

**Files:**
- Modify: `streamlit-app-builder/SKILL.md`

- [ ] **Step 1: Append sections**

`old_string`:
```
6. **Non-goals reminder** — a short list of things the scaffold does NOT include (auth, Docker, CI, DB, observability), explicitly marked as the team's responsibility.
```

`new_string`:
````
6. **Non-goals reminder** — a short list of things the scaffold does NOT include (auth, Docker, CI, DB, observability), explicitly marked as the team's responsibility.

## Output checklist

- [ ] Full directory tree populated (see Step 5)
- [ ] `pyproject.toml` declares MLX and transformers with correct environment markers
- [ ] `.env.example` covers every `_require` and `_get` key in `config.py`
- [ ] `ruff check --fix`, `ruff format`, `ty check`, and `pytest` all pass clean
- [ ] `README.md` documents setup, env vars, license, gated-model instructions
- [ ] Report to user surfaces MLX resolution + license flags + non-goals reminder

## References

- `references/streamlit-docs-index.md` — canonical docs URLs for live fetches
- `references/pipeline-tag-patterns.md` — HF pipeline_tag → UI pattern catalog
- `references/license-flags.md` — commercial-use flags for model licenses
````

- [ ] **Step 2: Verify**

Run: `grep -q "Output checklist" streamlit-app-builder/SKILL.md && grep -q "References" streamlit-app-builder/SKILL.md && grep -c "references/" streamlit-app-builder/SKILL.md`
Expected: exit 0 and ≥ 3.

- [ ] **Step 3: Commit**

```bash
git add streamlit-app-builder/SKILL.md
git commit -m "Add output checklist and references appendix to SKILL.md"
```

---

## Task 11: Update root `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the streamlit-app-builder row**

`old_string`:
```
| `streamlit-app-builder` | Generate Streamlit apps from Python scripts or Jupyter notebooks |
```

`new_string`:
```
| `streamlit-app-builder` | Generate production-structured Streamlit apps from Python scripts, Jupyter notebooks, or HuggingFace model card URLs |
```

- [ ] **Step 2: Verify**

Run: `grep -q "HuggingFace model card" README.md`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Update README streamlit-app-builder description to cover production scope"
```

---

## Task 12: Update root `CLAUDE.md` to carve out `streamlit-app-builder`

**Files:**
- Modify: `CLAUDE.md`

Current `CLAUDE.md` asserts all three app-builder skills share the same workflow and outputs. After this rewrite, `streamlit-app-builder` diverges — it produces a `src/<app_name>/` package, multipage router, and tests directory, not a single-file app.

- [ ] **Step 1: Narrow the shared-workflow statement**

`old_string`:
```
## App-Builder Skills

All three skills (`dash-app-builder`, `gradio-app-builder`, `streamlit-app-builder`) share the same workflow and outputs. `dash-app-builder` additionally supports an Analytics Dashboard pattern with `dash-bootstrap-components`.

**Workflow:** Analyze source (including notebook URL fetching) → Classify pattern → Generate app → Code quality → Testing

**Outputs:**
- `dash_app.py` / `gradio_app.py` / `streamlit_app.py` — single-file app with type annotations and inline comments
- `test_dash_app.py` / `test_gradio_app.py` / `test_streamlit_app.py` — pytest unit tests for non-UI functions only
- `pyproject.toml` — uv-managed project
- `.env.example` — all configurable env vars with placeholder defaults
```

`new_string`:
```
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
```

- [ ] **Step 2: Verify**

Run: `grep -q "dash-app-builder.*gradio-app-builder" CLAUDE.md && grep -q "production rewrite" CLAUDE.md && grep -q "AppTest" CLAUDE.md`
Expected: exit 0 for all.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Carve out streamlit-app-builder from shared app-builder workflow in CLAUDE.md"
```

---

## Task 13: Structural audit of the rewritten `SKILL.md`

**Files:**
- Read-only: `streamlit-app-builder/SKILL.md`

A fast non-executional check that the file is complete, the frontmatter is valid, and all eight steps plus appendices exist.

- [ ] **Step 1: Verify frontmatter**

Run:
```bash
head -n 15 streamlit-app-builder/SKILL.md
```

Expected output starts with `---`, contains `name: streamlit-app-builder`, contains `description: >`, mentions `HuggingFace model card`, ends with `---` before the body.

- [ ] **Step 2: Verify all 8 workflow steps exist**

Run:
```bash
grep "^## Step " streamlit-app-builder/SKILL.md
```

Expected (exactly eight lines, in order):
```
## Step 1: Identify and load the input
## Step 2: Build the internal representation
## Step 3: Classify the UI pattern
## Step 4: Fetch live Streamlit docs
## Step 5: Scaffold files
## Step 6: Initialize and install via `uv`
## Step 7: Code-quality gate
## Step 8: Report to user
```

- [ ] **Step 3: Verify references section**

Run:
```bash
grep "^- \`references/" streamlit-app-builder/SKILL.md
```

Expected:
```
- `references/streamlit-docs-index.md` — canonical docs URLs for live fetches
- `references/pipeline-tag-patterns.md` — HF pipeline_tag → UI pattern catalog
- `references/license-flags.md` — commercial-use flags for model licenses
```

- [ ] **Step 4: Verify all three reference files exist**

Run:
```bash
ls streamlit-app-builder/references/
```

Expected:
```
license-flags.md
pipeline-tag-patterns.md
streamlit-docs-index.md
```

- [ ] **Step 5: If anything is missing, fix before continuing**

Use `Edit` to add missing content to `SKILL.md` or the reference files. Re-run the checks above until all pass.

- [ ] **Step 6: Commit any fixes**

If fixes were needed, commit:

```bash
git add streamlit-app-builder/
git commit -m "Fix structural gaps in SKILL.md / references"
```

If nothing needed fixing, skip this step.

---

## Task 14: End-to-end smoke test — invoke the updated skill on a sample HF model card

**Files:**
- Temporarily scaffolds into a throwaway directory under `/tmp/`

The final validation: confirm the rewritten skill actually produces the expected scaffold on a realistic input. Pick a small, public, non-gated model card with a clear `pipeline_tag`. The default recommendation is `mlx-community/Llama-3.2-1B-Instruct-4bit` (text-generation, already MLX-native on Mac; or equivalent on Linux).

- [ ] **Step 1: Prepare a throwaway target directory**

```bash
rm -rf /tmp/streamlit-skill-smoketest
mkdir -p /tmp/streamlit-skill-smoketest
cd /tmp/streamlit-skill-smoketest
```

- [ ] **Step 2: Invoke the skill in a fresh Claude Code session**

In a **fresh** Claude Code session (new chat), in the target directory, prompt:

```
Use the streamlit-app-builder skill to scaffold a production Streamlit app
from this HuggingFace model card: https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit
The app name should be "llama-chat".
```

The skill should run through all 8 steps and write the scaffold into the current directory.

- [ ] **Step 3: Verify the directory tree matches the spec**

Run in the target directory:

```bash
find . -type f -not -path './.venv/*' -not -path './.git/*' -not -name '*.pyc' | sort
```

Expected (or very close, with file names possibly adjusted for the chosen pattern):

```
./.env.example
./.gitignore
./.streamlit/config.toml
./README.md
./pyproject.toml
./src/llama_chat/__init__.py
./src/llama_chat/config.py
./src/llama_chat/data.py         # possibly omitted if source had no data funcs
./src/llama_chat/inference.py
./src/llama_chat/pages/__init__.py
./src/llama_chat/pages/home.py
./src/llama_chat/viz.py          # possibly omitted
./streamlit_app.py
./tests/__init__.py
./tests/conftest.py
./tests/test_app_smoke.py
./tests/test_config.py
./tests/test_inference.py
./uv.lock
```

- [ ] **Step 4: Verify code quality passed**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check src/ tests/
uv run pytest -v
```

Expected: all pass (pytest may skip the smoke test if model weights are unavailable, but should not fail).

- [ ] **Step 5: Verify content markers**

Run:

```bash
grep -l "IS_APPLE_SILICON" src/llama_chat/config.py
grep -l "mlx-community/Llama-3.2-1B-Instruct-4bit" src/llama_chat/inference.py
grep -l "st.navigation" streamlit_app.py
grep -l "license" README.md
grep -l "mlx-lm" pyproject.toml
grep -l "transformers" pyproject.toml
```

Expected: each command prints its matching file path (exit 0).

- [ ] **Step 6: Record the result in a brief note**

In the repo, append a line to a new file `docs/superpowers/plans/2026-04-21-streamlit-app-builder-production-rewrite.log.md` with the outcome:

```
# Smoke test — 2026-04-21-streamlit-app-builder-production-rewrite

- Model used: mlx-community/Llama-3.2-1B-Instruct-4bit
- Skill invocation: <PASS | FAIL with details>
- ruff check: <PASS | FAIL>
- ruff format: <PASS | FAIL>
- ty check: <PASS | FAIL>
- pytest: <PASS | FAIL>
- Structural markers (Step 5): <PASS | FAIL>
- Notes: <anything surprising>
```

- [ ] **Step 7: Commit the log**

```bash
git add docs/superpowers/plans/2026-04-21-streamlit-app-builder-production-rewrite.log.md
git commit -m "Log smoke-test result for streamlit-app-builder production rewrite"
```

- [ ] **Step 8: Clean up the throwaway directory**

```bash
rm -rf /tmp/streamlit-skill-smoketest
```

---

## Plan Self-Review

**Spec coverage check:**

- ✅ Three input modes (script, notebook, HF model card URL) — Task 5 Step 1 covers all three
- ✅ `os.getenv` interface + `.env` + `python-dotenv` — Task 7 `config.py` template
- ✅ `.env.example` generated in lockstep with `config.py` — Task 7 + Output checklist
- ✅ No `.streamlit/secrets.toml` written by the skill — Task 6 `.streamlit/config.toml` only
- ✅ `src/<app_name>/` package layout — Task 6 directory tree + Task 7/8 templates
- ✅ Multipage via `st.navigation`, pages inside the package — Task 6 entrypoint, Task 8 pages
- ✅ Always multipage (single `home.py` minimum) — Task 6 directive
- ✅ MLX lookup on HF card inputs, always (not conditional on skill's host) — Task 5 Step 1 (HF card section), Task 4 principle 2
- ✅ MLX + transformers env markers in `pyproject.toml` — Task 9 Step 6
- ✅ `IS_APPLE_SILICON` in `config.py`, dispatch in `inference.py` — Task 7
- ✅ Gated-model `_require("HF_TOKEN")` switch — Task 7 note
- ✅ License surfaced with commercial-use flags — Task 3 reference + Task 8 README template + Task 9 report
- ✅ Live `docs.streamlit.io` fetches (mandatory list) — Task 1 reference + Task 6 Step 4
- ✅ Toolchain: ruff + ty + pytest unchanged — Task 9 Step 7
- ✅ `AppTest` smoke test — Task 8 `test_app_smoke.py`
- ✅ Non-goals documented in generated README and surfaced in report — Task 4 Non-goals, Task 9 Step 8 item 6
- ✅ Output checklist — Task 10
- ✅ Parent `README.md` + `CLAUDE.md` updates — Tasks 11 & 12
- ✅ End-to-end smoke test to validate the skill actually works — Task 14

**Placeholder scan:** Reviewed all tasks; every code block shows literal code, every command is exact, every file path is absolute. `<app-name>` / `<app_name>` / `<org>/<model>` / `<feature>` / `<mlx-community/...>` are intentional template placeholders that the skill itself fills in at run time — they are documented as such in Task 6 and Task 7.

**Type consistency:** Function names used across tasks:
- `load_model()` returns `tuple[str, model, tokenizer]` — same signature in Task 7 and Task 8's `mock_model` fixture.
- `generate_response(prompt, max_new_tokens)` — consistent between Task 7 (`inference.py`) and Task 2 (pipeline-tag-patterns chat snippet).
- `render() -> None` — consistent between Task 6 entrypoint import and Task 8 `home.py` template.
- `IS_APPLE_SILICON` — consistent name between Task 4 (principle), Task 7 (`config.py`), and Task 8 (`test_config.py`).

---
