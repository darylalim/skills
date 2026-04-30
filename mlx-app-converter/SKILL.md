---
name: mlx-app-converter
description: >
  Use when the user wants to convert an existing Streamlit or Gradio app
  to use MLX (Apple's machine learning framework for Apple Silicon) instead
  of HuggingFace transformers. Trigger phrases: "convert this app to MLX",
  "use mlx-lm for inference", "use mlx-vlm for inference", "make this run on
  Apple Silicon with MLX", "swap transformers for MLX", "port this to MLX",
  "convert this VLM app to MLX". Operates on an existing app file in the
  current working directory (`app.py`, `streamlit_app.py`, or `gradio_app.py`);
  rejects other inputs (notebooks, scripts, GitHub URLs, HF model card URLs).
  Scope (v2): LLMs (text-generation via `AutoModelForCausalLM`) and VLMs
  (vision-language via `AutoModelForVision2Seq`, `AutoModelForImageTextToText`,
  or a curated list of family-specific classes). Multi-modal apps (LLM + VLM
  in the same file) are supported. Audio modalities, streaming inference,
  and VLM family-specific classes outside the curated allowlist are out of
  scope (planned for follow-up versions).
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

3. **Detect HF model IDs.** AST-scan the app file. Each `<X>.from_pretrained("...")` call site is processed independently; per-call-site rejections below are soft and skip only that model.
   - Extract literal string args from any `<X>.from_pretrained("...")` call.
   - Resolve simple variable indirection: top-level `MODEL_ID = "..."` (or any module-level `<NAME> = "<literal>"` constant) used as the arg.
   - **Per-call-site modality tagging:** classify `<X>`:
     - LLM allowlist: `AutoModelForCausalLM` → tag `modality = "llm"`.
     - VLM allowlist (umbrella + curated families): `AutoModelForVision2Seq`, `AutoModelForImageTextToText`, `LlavaForConditionalGeneration`, `LlavaNextForConditionalGeneration`, `Qwen2VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration`, `Idefics3ForConditionalGeneration`, `PaliGemmaForConditionalGeneration`, `MllamaForConditionalGeneration` → tag `modality = "vlm"`.
     - Tokenizer/processor classes (`AutoTokenizer`, `AutoProcessor`, `AutoFeatureExtractor`, `AutoImageProcessor`) are not modality-bearing on their own. They fold into deduplication when paired with a model class on the same `MODEL_ID`.
     - **Class outside both allowlists** → soft per-call-site reject: `Skipping <call site>: model class <X> is not in v2's supported list. Supported VLM classes: AutoModelForVision2Seq, AutoModelForImageTextToText, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Idefics3ForConditionalGeneration, PaliGemmaForConditionalGeneration, MllamaForConditionalGeneration. LLM classes: AutoModelForCausalLM. Other classes are planned for follow-up versions.`
   - **Streaming gate** (modality-agnostic): the call is soft-rejected if either of these holds for the function containing it:
     - The function references `TextIteratorStreamer` (imported or used), OR
     - The function has a `yield` statement AND the corresponding `model.generate(...)` call passes a `streamer=` kwarg.
     - On match: `Streaming inference detected at <call site>. v2 supports non-streaming inference only; streaming is planned for a follow-up version.`
   - **Dynamic args** (env var, UI input, function call) → soft per-model rejection: `Skipping <call site>: model ID is dynamic (env var or runtime input). v1 supports only statically-known model IDs.`
   - If zero literal model IDs remain after resolution: `No HF model IDs found in <file>. mlx-app-converter requires statically-known model IDs (string literal or simple constant). Dynamic IDs (env var, UI input) are not supported in v1.`
   - **Deduplicate by (model ID, modality) pair** before the next step. The canonical loader pattern uses two `from_pretrained` calls referencing the same `MODEL_ID` — tokenizer+model for LLMs, processor+model for VLMs — these collapse to one matrix prompt, not two.
   - **If every detected model hits a soft rejection** (class-mismatch, streaming, dynamic-args, or all-skipped via Step 4's no-match fallback), exit with: `Nothing to convert — every detected model was skipped. The app file was not modified.`

4. **Per-model variant resolution.** For each detected model, invoke the helper from the skill root (`mlx-app-converter/`):

   ```bash
   python -m lib.variant_resolution query \
       --base-name "<base_name>" \
       --orig-param-count "<original_param_count>" \
       --model-id "<full_model_id>"
   ```

   The helper queries `mlx-community` on HuggingFace Hub, builds a (param-count × quantization) matrix, marks the default cell (original parameter count + highest precision per `bf16 > fp16 > 8bit > 6bit > 4bit`), and prints it. Present the matrix output to the user and ask them to reply with a cell address (e.g., `8B@bf16`) or `default` to accept the highlighted default.

   Use `parse_reply` from `lib/variant_resolution.py` to validate the user's response. Re-prompt on invalid input (missing `@`, unrecognized cell); exit after three consecutive invalid replies with: `Too many invalid replies for <model>. Skipping this model.`

   - **No-match fallback:** if `query` returns zero variants, run:
     ```bash
     python -m lib.variant_resolution siblings --base-name "<base_name>"
     ```
     Print up to 3 closest siblings plus a "skip this model unchanged" option, with: `No MLX variants found for <model>. Closest siblings: <up to 3>. Pick one or reply "skip" to leave this model unchanged.`

5. **Rewrite.** Apply templates from `references/rewrite-templates.md`, routing each detected (deduped) model to its modality-specific subsection:
   - **T1 Loader** — LLM models use the LLM subsection (`mlx_lm.load`); VLM models use the VLM subsection (`mlx_vlm.load`). Cache decorator and `MODEL_ID` constant name preserved verbatim regardless of modality.
   - **T2 Inference** — LLM models use T2-LLM (kwarg rewrites: `max_new_tokens` → `max_tokens` direct rename; `temperature`/`top_p`/`top_k`/`repetition_penalty` route through `make_sampler` / `make_logits_processors` helpers — they are NOT direct kwargs on `mlx_lm.generate`). VLM models use T2-VLM (kwarg rewrites: `max_new_tokens` → `max_tokens` direct rename; sampling kwargs `temperature`/`top_p`/`top_k`/`repetition_penalty` are direct kwargs on `mlx_vlm.generate` — no helper construction).
   - **T3 Apple Silicon runtime guard** — insert top-of-file check (modality-agnostic; one insertion per file).
   - **T4 Test rewrite** — if a `test_*.py` exists alongside the app, swap mocks per detected model: LLM models patch `mlx_lm.load`; VLM models patch `mlx_vlm.load`. Multi-modal test files patch both independently.
   - **T5 Dep manifest delta** — emit install command / appends covering the union of target packages used: `{mlx-lm}` for LLM-only files, `{mlx-vlm}` for VLM-only, `{mlx-lm, mlx-vlm}` for multi-modal. For `pyproject.toml` (Streamlit), print `uv add <space-separated set>`. For `requirements.txt` (Gradio), append each as its own line, idempotently. Print the removal hint for `transformers` and `torch`.
   - **Multi-modal import unioning:** when a file has both LLM and VLM models, top-level imports include both `import mlx_lm` and `import mlx_vlm`, deduped. Sampling helpers (`from mlx_lm.sample_utils import make_sampler, make_logits_processors`) are imported only if at least one LLM call site uses sampling kwargs.

6. **Verify.** Run quality tools on every touched file:
   - `ruff check --fix <files>`
   - `ruff format <files>`
   - `ty check <app file>`
   - `pytest <test file>` if a test file was modified.
   - **Note:** run `pytest` only after the user has executed Step 5's `uv add mlx-lm` (Streamlit) or `pip install -r requirements.txt` (Gradio). Otherwise pytest collection fails with `ModuleNotFoundError: No module named 'mlx_lm'` because the rewritten app imports `mlx_lm` at module top-level.

## Inputs

A Streamlit or Gradio app file in the current working directory. Canonical filenames: `app.py`, `streamlit_app.py`, `gradio_app.py`. The framework (Streamlit vs Gradio) is detected from top-level imports inside the file.

The app may load LLM models (via `AutoModelForCausalLM`), VLM models (via the umbrella `AutoModelForVision2Seq` / `AutoModelForImageTextToText`, or one of a curated list of family-specific classes), or both (multi-modal). Each detected model is routed to its modality-specific MLX target package (`mlx-lm` for LLM, `mlx-vlm` for VLM) independently.

Other inputs (notebooks, scripts, GitHub URLs, HF model card URLs) are rejected. Use `streamlit-app-builder` or `gradio-app-builder` to scaffold a new app from an HF model card.

## Outputs (in-place edits)

- `<app file>` — loader + inference function rewritten for the detected modality's MLX target package (`mlx-lm` for LLM, `mlx-vlm` for VLM); imports updated; runtime Apple Silicon guard inserted at top of file. Multi-modal apps include both `import mlx_lm` and `import mlx_vlm`, deduped.
- `test_*.py` (only if a test file already exists alongside the app) — mocks updated per detected model: `transformers.*.from_pretrained` → `mlx_lm.load` for LLM models, `mlx_vlm.load` for VLM models; inference test invocation updated.
- `requirements.txt` (Gradio) or `pyproject.toml` (Streamlit) — `mlx-lm` and/or `mlx-vlm` added (union of target packages used). `transformers` and `torch` are left in place; the skill prints a removal hint instead of auto-removing.

## Toolchain

The skill operates on the user's existing toolchain — no new `uv init`, no new `pyproject.toml`. After rewrite:

```bash
# Streamlit (uv-managed) — package set depends on detected modalities
uv add mlx-lm                 # LLM-only files
uv add mlx-vlm                # VLM-only files
uv add mlx-lm mlx-vlm         # multi-modal files
uv run ruff check --fix <app file> [test file]
uv run ruff format <app file> [test file]
uv run ty check <app file>
uv run pytest <test file> -v   # if test file exists

# Gradio (pip + requirements.txt) — append each target package per modality set
pip install -r requirements.txt
ruff check --fix <app file> [test file]
ruff format <app file> [test file]
ty check <app file>
pytest <test file> -v          # if test file exists
```

## Out of scope (v2)

- Audio modalities (ASR, TTS, classification, embedding) — planned for v3.
- **Streaming inference** for both LLM and VLM. v2 introduces an explicit soft-reject gate for source apps using `TextIteratorStreamer` or yielding from a `streamer=`-passing inference function. Streaming planned for a follow-up version.
- VLM family-specific classes outside the curated allowlist (the listed nine entries: two umbrella + seven family-specific). Soft-rejected with the supported-families list as the hint.
- Dynamic model IDs (env var, UI input) — soft-rejected per call site (preserved from v1).
- Auto-removal of `transformers` / `torch` from dep manifests — skill prints a hint, user decides (preserved from v1).
- Apps that aren't Streamlit or Gradio.
- Custom MLX model conversion (calling `mlx-lm convert` / `mlx-vlm convert` to produce new MLX variants).
