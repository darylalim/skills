---
name: mlx-app-converter
description: >
  Use when the user wants to convert an existing Streamlit or Gradio app
  to use MLX (Apple's machine learning framework for Apple Silicon) instead
  of HuggingFace transformers. Trigger phrases: "convert this app to MLX",
  "use mlx-lm for inference", "make this run on Apple Silicon with MLX",
  "swap transformers for MLX", "port this to MLX". Operates on an existing
  app file in the current working directory (`app.py`, `streamlit_app.py`,
  or `gradio_app.py`); rejects other inputs (notebooks, scripts, GitHub URLs,
  HF model card URLs). Scope (v1): LLMs only — text-generation models loaded
  via `AutoModelForCausalLM`. Other modalities (vision, audio, embeddings,
  diffusion) are out of scope.
---

# MLX App Converter

Converts an existing Streamlit or Gradio app from `transformers`-based inference to MLX-based inference on Apple Silicon.

## Workflow

1. **Identify the input.** This skill operates on an existing Streamlit or Gradio app file in the current working directory.
   - If the user references something this skill doesn't handle (a notebook, a model card URL, a script that isn't a Streamlit/Gradio app), reject with: `mlx-app-converter operates on an existing Streamlit or Gradio app file in the current working directory. For other inputs, use the appropriate skill (streamlit-app-builder, gradio-app-builder) or a general-purpose prompt.`
   - Otherwise, auto-discover the app file in CWD by looking for `app.py`, `streamlit_app.py`, or `gradio_app.py`.
   - **0 matches:** reject with: `No app file found. Expected one of app.py, streamlit_app.py, gradio_app.py in the current directory.`
   - **2+ matches:** ask: `Found multiple app files: <list>. Tell me which one to convert.`
   - **1 match:** proceed.

2. **Pre-flight gates.** All hard — failure exits without modifying anything.
   - **Hardware:** verify `platform.machine() == "arm64"` and `platform.system() == "Darwin"`. On failure: `MLX requires Apple Silicon (arm64 macOS). Detected: <machine>/<system>. Run this skill on an Apple Silicon Mac.`
   - **Framework:** parse the app file's top-level imports. Reject if neither `streamlit` nor `gradio` is imported: `<file> does not import streamlit or gradio at the top level. mlx-app-converter only supports Streamlit and Gradio apps.`
   - **Git-clean:** for each file the skill plans to modify (the app file, any test file alongside it, the dep manifest), check `git status --porcelain <file>`. If any has uncommitted changes: `Uncommitted changes in <file>. Commit or stash before running mlx-app-converter so the rewrite is reviewable via git diff.`

3. **Detect HF model IDs.** AST-scan the app file. Gate 6 below (no models found) is hard — exits the skill. Gate 8 (dynamic-arg per call site) is soft — skips that one model and proceeds with the rest.
   - Extract literal string args from any `<X>.from_pretrained("...")` call (`AutoTokenizer`, `AutoModelForCausalLM`, etc.).
   - Resolve simple variable indirection: top-level `MODEL_ID = "..."` (or any module-level `<NAME> = "<literal>"` constant) used as the arg.
   - Dynamic args (env var, UI input, function call) → soft per-model rejection: `Skipping <call site>: model ID is dynamic (env var or runtime input). v1 supports only statically-known model IDs.`
   - If zero literal model IDs remain after resolution: `No HF model IDs found in <file>. mlx-app-converter requires statically-known model IDs (string literal or simple constant). Dynamic IDs (env var, UI input) are not supported in v1.`
   - **Deduplicate by model ID string** before the next step. The canonical loader pattern uses two `from_pretrained` calls (one for the tokenizer, one for the model) referencing the same `MODEL_ID` — these collapse to one matrix prompt, not two.
   - **If every detected model hits a soft rejection** (all dynamic args, or all skipped via gate 7's no-match fallback), exit with: `Nothing to convert — every detected model was skipped. The app file was not modified.`

4. **Per-model variant resolution.** For each detected model, follow `references/variant-resolution.md`:
   - Query `huggingface_hub.list_models(author="mlx-community", search=<base_name>)`.
   - Filter to MLX quantization suffixes (`-bf16`, `-fp16`, `-8bit`, `-6bit`, `-4bit`).
   - Parse `(parameter_count, quantization)` per match; build the matrix.
   - Print the matrix with the highlighted default = `(original parameter count, max(bf16 > fp16 > 8bit > 6bit > 4bit))`.
   - Ask the user to reply with a cell (e.g., `8B@bf16`) or `default`.
   - **No-match fallback:** print up to 3 closest siblings via edit distance plus a "skip this model unchanged" option, with: `No MLX variants found for <model>. Closest siblings: <up to 3>. Pick one or reply "skip" to leave this model unchanged.`

5. **Rewrite.** Apply templates from `references/rewrite-templates.md`:
   - **T1 Loader** — replace transformers loader with `mlx_lm.load`. Preserve cache decorator and `MODEL_ID` constant name.
   - **T2 Inference** — replace transformers inference with `mlx_lm.generate`. Apply kwarg rewrites per T2's tables (`max_new_tokens` → `max_tokens` is a direct rename; sampling params `temperature`/`top_p`/`top_k`/`repetition_penalty` route through `make_sampler` / `make_logits_processors` helpers — they are NOT direct kwargs on `mlx_lm.generate`).
   - **T3 Apple Silicon runtime guard** — insert top-of-file check.
   - **T4 Test rewrite** — if a `test_*.py` exists alongside the app, swap mocks from `*.from_pretrained` to `mlx_lm.load`.
   - **T5 Dep manifest delta** — for `pyproject.toml` (Streamlit), print `uv add mlx-lm`. For `requirements.txt` (Gradio), append `mlx-lm`. Print the removal hint for `transformers` and `torch`.

6. **Verify.** Run quality tools on every touched file:
   - `ruff check --fix <files>`
   - `ruff format <files>`
   - `ty check <app file>`
   - `pytest <test file>` if a test file was modified.
   - **Note:** run `pytest` only after the user has executed Step 5's `uv add mlx-lm` (Streamlit) or `pip install -r requirements.txt` (Gradio). Otherwise pytest collection fails with `ModuleNotFoundError: No module named 'mlx_lm'` because the rewritten app imports `mlx_lm` at module top-level.

## Inputs

A Streamlit or Gradio app file in the current working directory. Canonical filenames: `app.py`, `streamlit_app.py`, `gradio_app.py`. The framework (Streamlit vs Gradio) is detected from top-level imports inside the file.

Other inputs (notebooks, scripts, GitHub URLs, HF model card URLs) are rejected. Use `streamlit-app-builder` or `gradio-app-builder` to scaffold a new app from an HF model card.

## Outputs (in-place edits)

- `<app file>` — loader + inference function rewritten for `mlx-lm`; imports updated; runtime Apple Silicon guard inserted at top of file.
- `test_*.py` (only if a test file already exists alongside the app) — mocks updated from `transformers.*.from_pretrained` to `mlx_lm.load`; inference test invocation updated.
- `requirements.txt` (Gradio) or `pyproject.toml` (Streamlit) — `mlx-lm` added. `transformers` and `torch` are left in place; the skill prints a removal hint instead of auto-removing.

## Toolchain

The skill operates on the user's existing toolchain — no new `uv init`, no new `pyproject.toml`. After rewrite:

```bash
# Streamlit (uv-managed)
uv add mlx-lm
uv run ruff check --fix <app file> [test file]
uv run ruff format <app file> [test file]
uv run ty check <app file>
uv run pytest <test file> -v   # if test file exists

# Gradio (pip + requirements.txt)
pip install -r requirements.txt
ruff check --fix <app file> [test file]
ruff format <app file> [test file]
ty check <app file>
pytest <test file> -v          # if test file exists
```

## Out of scope (v1)

- VLM and audio model conversion (planned for follow-up versions).
- Dynamic model IDs (env var, UI input) — soft-rejected per call site.
- Auto-removal of `transformers` / `torch` from dep manifests — skill prints a hint, user decides.
- Apps that aren't Streamlit or Gradio.
- Custom model conversion (calling `mlx-lm convert` to produce new MLX variants).
