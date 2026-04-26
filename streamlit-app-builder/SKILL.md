---
name: streamlit-app-builder
description: >
  Use when the user wants to scaffold a Streamlit prototype for a HuggingFace
  model. Triggers: "build a Streamlit demo for an HF model", "generate a
  Streamlit UI for huggingface.co/<org>/<model>", "prototype a Streamlit app
  for an HF model", "turn an HF model card into a Streamlit demo", "scaffold a
  Streamlit app for an HF inference model", any URL of the shape
  `huggingface.co/<org>/<model>`. Skip when the input is a Python script, a
  Jupyter notebook, a GitHub URL, or any non-HF artifact — the
  dash-app-builder and gradio-app-builder skills are the right alternatives.
---

# Streamlit App Builder (HF model card → prototype)

Generate a single-file Streamlit prototype from a HuggingFace model card URL. Output is a flat four-file project (`streamlit_app.py`, `pyproject.toml`, `.env.example`, `test_streamlit_app.py`) ready to `uv sync && streamlit run`.

## When to use

**Use:** the user passes a `huggingface.co/<org>/<model>` URL and wants a working Streamlit demo.

**Don't use** (redirect explicitly):
- Python script (`.py` file path or `github.com/.../*.py` blob URL) → `gradio-app-builder` or `dash-app-builder`
- Jupyter notebook (`.ipynb`) → `gradio-app-builder` or `dash-app-builder`
- GitHub repo root URL → `dash-app-builder` or `gradio-app-builder`
- `huggingface.co/datasets/...`, `huggingface.co/spaces/...` — not supported

When rejecting, emit a one-line message naming the right alternative skill.

## Non-goals (silent — no need to surface in the report)

- Production-grade structure (multipage routing, fail-fast config, src layout) — this is a prototype generator
- Authentication / authorization
- Deployment artifacts (Dockerfile, K8s, CI)
- Database / storage layers
- Observability
- MLX / Apple-Silicon-specific acceleration (transformers / diffusers run on Macs, just slower)
- License or commercial-use surfacing
- Sibling-model suggestions

## Step 1: Identify and load the input

The input MUST be an HF model card URL of the shape `https://huggingface.co/<org>/<model>` (with or without trailing slash; query strings and fragments are stripped before classification).

**Reject early** with a clear message if the input is any other URL shape:
- `github.com/...` (any path) → *"Pass an HF model card URL (`huggingface.co/<org>/<model>`). For Python scripts or notebooks on GitHub, use the `dash-app-builder` or `gradio-app-builder` skills."*
- `*.ipynb` URL → same redirect to dash/gradio.
- `huggingface.co/datasets/...` or `huggingface.co/spaces/...` → *"This skill scaffolds prototypes from model cards only. Datasets and Spaces are not supported."*
- File path or any other shape → *"Pass an HF model card URL (`huggingface.co/<org>/<model>`)."*

**Load the model card:**

1. Strip query string and fragment from the URL. Resolve `<org>/<model>` from the path.
2. Fetch metadata: `https://huggingface.co/api/models/<org>/<model>` → JSON. Capture `pipeline_tag`, `library_name`, `tags`, `gated`, `downloads`.
3. Fetch the README: `https://huggingface.co/<org>/<model>/raw/main/README.md`. Use the YAML frontmatter as a fallback for any metadata fields the API didn't return; capture the first library-idiomatic Python code snippet for use as the `inference_snippet` in the IR.
4. On HTTP 404 from the metadata API: fail with *"Model `<org>/<model>` not found on HuggingFace, or it's a private repo without access. Verify the URL."*
5. On HTTP 401/403: *"Access denied to `<org>/<model>`. If gated, run `huggingface-cli login` first; if private, ensure your `HF_TOKEN` has access."*

## Step 2: Build the IR

<!-- skip-validate -->
```python
{
    "model_id": "<org>/<model>",
    "pipeline_tag": "<tag>",                              # from API metadata
    "library_name": "transformers" | "diffusers" | "sentence-transformers" | None,
    "is_gated": bool,                                     # api.gated == True
    "deps": ["streamlit", "python-dotenv", ...],          # see Step 4
    "inference_snippet": "<first python block from README>" or None,
}
```

`library_name` resolution: prefer the API's `library_name` field. If the field is missing, default to `transformers`. If the API's `library_name == "sentence-transformers"` OR the README explicitly recommends `sentence-transformers` (look for `from sentence_transformers import` in the README's code snippets), set `library_name = "sentence-transformers"` regardless of the API value.

`pipeline_tag` normalization: when `library_name` resolves to `sentence-transformers`, set `pipeline_tag = "feature-extraction"` regardless of the API's value. HF embedding models commonly report `"sentence-similarity"` or `"feature-extraction"` interchangeably, and both route to T5 (`embed`) in Step 4.

## Step 3: Classify the UI pattern

Look up `pipeline_tag` in `references/pipeline-tag-patterns.md`. The matching section's UI body is what gets pasted into `streamlit_app.py` after the inference function.

**Rejected pipeline tag:**
- `pipeline_tag == "audio-to-audio"` → fail at Step 3 with: *"audio-to-audio has no clean transformers pipeline. This skill can't scaffold a working prototype for audio-to-audio models. For source separation or speech enhancement, use the model's reference implementation directly."*

**Unrecognized pipeline tag** → fall through to the **Fallback: General Script** section in `references/pipeline-tag-patterns.md`.

## Step 4: Scaffold the four files

### Routing table: `library_name` × `pipeline_tag` → scaffolding template

| `library_name` | `pipeline_tag` | Template (in `references/scaffolding-templates.md`) |
|---|---|---|
| `transformers` | `text-generation`, `conversational` | T2 (`generate_response`) |
| `transformers` | `text-classification`, `zero-shot-classification`, `token-classification`, `question-answering`, `summarization`, `translation`, `automatic-speech-recognition`, `text-to-speech`, `image-classification`, `object-detection`, `image-to-text`, `image-text-to-text` | T1 (`run_inference`) |
| `transformers`, `sentence-transformers` | `feature-extraction` | T5 (`embed`) |
| `diffusers` | `text-to-image` | T3 (`generate_image`) |
| `diffusers` | `image-to-image` | T4 (`edit_image`) |
| any | unrecognized | T1 with the General Script UI body |

### File 1: `streamlit_app.py`

Assemble in this order:

1. **Module docstring** — one line describing the model and task.
2. **Imports** — `os`, `streamlit as st`, `dotenv.load_dotenv`, plus library-specific imports from the chosen scaffolding template.
3. **`load_dotenv()`** — call once at module level so `.env` values populate `os.environ`.
4. **`MODEL_ID` constant** — hard-coded from Step 1's input URL: `MODEL_ID = "<org>/<model>"`.
5. **Gated-model gate** — see the snippet below; emit only when `is_gated: true`.
6. **`load_model()`** — copy from the chosen scaffolding template, decorated with `@st.cache_resource`.
7. **Inference function** — copy from the chosen scaffolding template (`run_inference` / `generate_response` / `generate_image` / `edit_image` / `embed`).
8. **UI body** — paste at module scope after the inference function — no function wrapper.

#### Gated-model gate snippet

Insert this after `load_dotenv()` (between assembly step 4 and step 6) when the model card has `gated: true`:

<!-- skip-validate -->
```python
if not os.getenv("HF_TOKEN"):
    st.error(
        f"This model ({MODEL_ID}) is gated. Set HF_TOKEN in .env or run "
        "`huggingface-cli login` before launching."
    )
    st.stop()
```

### File 2: `pyproject.toml`

Generated by `uv init --name <app-name>`. After init:

```bash
uv add streamlit python-dotenv huggingface_hub
# Plus library-specific deps per the routing table:
# - transformers (T1, T2):
uv add transformers torch
# - For ASR / TTS (transformers pipeline):
uv add "transformers[audio]" scipy
# - For diffusers (T3, T4):
uv add diffusers transformers torch accelerate pillow
# - For sentence-transformers (T5):
uv add sentence-transformers
# Dev dependencies (always):
uv add --dev ruff ty pytest
```

The exact `uv add` calls depend on the matched template — emit only the rows that apply. Add `[tool.pytest.ini_options]` with `testpaths = ["."]` and `[tool.ruff]` with `select = ["E", "F", "I"]` to `pyproject.toml`.

### File 3: `.env.example`

Always emit this file. Template:

```
# Optional. Required for gated models (e.g. meta-llama/*, mistralai/*).
# Get a token at https://huggingface.co/settings/tokens.
HF_TOKEN=
```

### File 4: `test_streamlit_app.py`

Copy template T6 from `references/scaffolding-templates.md` and adapt the body of `test_inference_function_returns_expected_type` per the in-use scaffolding template — the comment hints inside T6 spell out each adaptation explicitly.

## Step 5: Code-quality gate

Run, in order. All four must pass before reporting the scaffold as complete.

```bash
uv run ruff check --fix streamlit_app.py test_streamlit_app.py
uv run ruff format streamlit_app.py test_streamlit_app.py
uv run ty check streamlit_app.py
uv run pytest test_streamlit_app.py -v
```

Fix failures by adjusting the generated code. Do not weaken the test to make it pass.

## Step 6: Report to the user

Emit exactly two items, plus a third when the model is gated.

**Always:**

1. **Files created.** List the four paths.
2. **Run command.**

   ```bash
   uv sync
   cp .env.example .env       # then add your HF_TOKEN if needed
   streamlit run streamlit_app.py
   ```

**Conditional (when `is_gated: true`):**

3. **Gated-model setup.** *"This model is gated. Either run `huggingface-cli login` on the host before first launch, OR set `HF_TOKEN=` in `.env`. Without a token, the app fails fast at startup with a clear error."*

## Output checklist

- [ ] Four files created: `streamlit_app.py`, `pyproject.toml`, `.env.example`, `test_streamlit_app.py`
- [ ] `streamlit_app.py` has `@st.cache_resource` on `load_model`
- [ ] `MODEL_ID` is hard-coded as a module-level constant (not env-driven)
- [ ] If `is_gated`, a `st.error` + `st.stop` gate runs before any model load
- [ ] `pyproject.toml` declares `streamlit`, `python-dotenv`, `huggingface_hub`, plus library-specific runtime deps and `ruff` / `ty` / `pytest` as dev deps
- [ ] `.env.example` always present, contains `HF_TOKEN=` line
- [ ] `ruff check --fix`, `ruff format`, `ty check`, and `pytest` all pass clean
- [ ] Step 6 report has 2 always-present items (or 3 when gated)

## References

- `references/pipeline-tag-patterns.md` — UI body templates indexed by `pipeline_tag`
- `references/scaffolding-templates.md` — `load_model` + inference-function templates indexed by `library_name` × `pipeline_tag`
