# Rewrite Templates

This document defines the code transformations `mlx-app-converter` applies to convert a Streamlit or Gradio app from `transformers`-based inference to MLX-based inference. Each template has a before/after pair plus a list of preservation rules.

Templates are referenced from `SKILL.md` by name (T1–T5).

## Template T1: Loader

Replace the `transformers`-based model-loading function with an `mlx_lm.load`-based equivalent. The cache decorator (`@st.cache_resource` for Streamlit, `@lru_cache(maxsize=1)` for Gradio) is preserved verbatim.

**Streamlit form (before):**

```python
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tokenizer
```

**Streamlit form (after):**

```python
import mlx_lm
import streamlit as st

MODEL_ID = "mlx-community/Llama-3.1-8B-Instruct-bf16"


@st.cache_resource
def load_model():
    return mlx_lm.load(MODEL_ID)
```

**Gradio form (before):**

```python
from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tokenizer
```

**Gradio form (after):**

```python
from functools import lru_cache

import mlx_lm

MODEL_ID = "mlx-community/Llama-3.1-8B-Instruct-bf16"


@lru_cache(maxsize=1)
def load_model():
    return mlx_lm.load(MODEL_ID)
```

**Preservation rules:**
- The cache decorator (`@st.cache_resource` or `@lru_cache(maxsize=1)`) is preserved verbatim — including any user-supplied arguments to it.
- The `MODEL_ID` constant name is preserved; only its value is rewritten to the user-selected MLX variant.
- The function name (`load_model` in the canonical case) and its return shape (`(model, tokenizer)` tuple) are preserved — `mlx_lm.load` returns the same tuple shape.

**Import handling:**
- Remove `from transformers import AutoTokenizer, AutoModelForCausalLM` (and any subset).
- Remove unused `from transformers import AutoTokenizer` / `AutoModelForCausalLM` standalone imports.
- Add `import mlx_lm` if not already present.
- Other `transformers` imports (e.g., `pipeline`, `AutoConfig` used elsewhere in the file) are left alone.

## Template T2: Inference

Replace the transformers-style inference body (`tokenizer(...)`, `model.generate(**inputs)`, `tokenizer.decode(...)`) with the single-call `mlx_lm.generate(...)` form. The function signature and name are preserved. Only `max_new_tokens` is a direct kwarg rename to `max_tokens`; sampling parameters (`temperature`, `top_p`, `top_k`, `repetition_penalty`) require constructing helper objects from `mlx_lm.sample_utils`.

**Before (no sampling kwargs):**

```python
def run_inference(prompt: str, model, tokenizer, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**After (no sampling kwargs):**

```python
import mlx_lm


def run_inference(prompt: str, model, tokenizer, max_tokens: int = 200) -> str:
    return mlx_lm.generate(model, tokenizer, prompt, max_tokens=max_tokens)
```

### Direct kwarg rename

| transformers kwarg | mlx_lm kwarg | Notes |
|---|---|---|
| `max_new_tokens` | `max_tokens` | Direct rename. Only kwarg that maps 1:1. |

### Sampling parameters (helper construction required)

`mlx_lm.generate` does not accept sampling kwargs directly. Wrap them via `mlx_lm.sample_utils.make_sampler` and `make_logits_processors`.

**Before (with sampling kwargs):**

```python
def run_inference(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**After (with sampling kwargs):**

```python
import mlx_lm
from mlx_lm.sample_utils import make_logits_processors, make_sampler


def run_inference(prompt: str, model, tokenizer) -> str:
    sampler = make_sampler(temp=0.7, top_p=0.9, top_k=50)
    logits_processors = make_logits_processors(repetition_penalty=1.1)
    return mlx_lm.generate(
        model,
        tokenizer,
        prompt,
        max_tokens=200,
        sampler=sampler,
        logits_processors=logits_processors,
    )
```

| transformers kwarg | mlx_lm placement | Notes |
|---|---|---|
| `temperature` | `make_sampler(temp=...)` | Pass to `make_sampler`; argname is `temp`, not `temperature`. |
| `top_p` | `make_sampler(top_p=...)` | Pass to `make_sampler`. |
| `top_k` | `make_sampler(top_k=...)` | Pass to `make_sampler`. |
| `repetition_penalty` | `make_logits_processors(repetition_penalty=...)` | Pass to `make_logits_processors`. |

### Dropped or transformed kwargs

| transformers kwarg | Action | Notes |
|---|---|---|
| `do_sample=True` | Drop (default behavior) | mlx-lm samples by default when `temp > 0`. |
| `do_sample=False` | Map to `make_sampler(temp=0.0)` | Forces greedy decoding. Add `temp=0.0` to the constructed sampler regardless of any `temperature` kwarg. |
| `pad_token_id` | Drop | Not used by `mlx_lm.generate`; pad handling is internal. |

### Unknown kwargs

For any transformers generation kwarg not in the tables above (e.g., `min_length`, `num_beams`, `eos_token_id`, `renormalize_logits`), **drop it from the generated call site** and add a `# TODO` comment immediately above the `mlx_lm.generate` call describing the dropped kwarg. Do not pass unknown kwargs through — `generate_step` rejects them with `TypeError`.

Example (original code used `min_length=20`):

```python
import mlx_lm


def run_inference(prompt: str, model, tokenizer) -> str:
    # TODO: min_length=20 not supported by mlx-lm — review manually
    return mlx_lm.generate(model, tokenizer, prompt, max_tokens=200)
```

### Preservation rules

- The function name and parameter list are preserved (renames apply to known kwargs only; unknown kwargs are dropped from the call but their original parameter, if any, may need to be removed too if it's only used at the dropped call site).
- The return type (`str`) is preserved — `mlx_lm.generate` returns the decoded text directly.
- Type hints are preserved verbatim.
- If the original function returned a list of dicts (e.g., from `transformers.pipeline`), wrap the `mlx_lm.generate` result to match: `return [{"generated_text": <result>}]`.

### Import handling

- Add `import mlx_lm` if not already added by T1.
- If sampling parameters are used in the original code, add `from mlx_lm.sample_utils import make_logits_processors, make_sampler` (only the helpers actually needed).
- Other `transformers` imports (e.g., `pipeline`, `AutoConfig` used elsewhere) are left alone.

## Template T3: Apple Silicon runtime guard

Insert a top-of-file runtime guard that fails fast with a clear error when the app runs on a non-Apple-Silicon host. Defends against deploys to x86 environments (e.g., Streamlit Cloud, generic Linux containers) where MLX cannot import.

**Insertion point:** immediately after the module docstring (if present) or as the first executable statement in the file. Above all imports of `mlx_lm` or app-framework imports.

**Code to insert:**

```python
import platform

if not (platform.machine() == "arm64" and platform.system() == "Darwin"):
    raise RuntimeError(
        f"This app uses MLX and requires Apple Silicon (arm64 macOS). "
        f"Detected: {platform.machine()}/{platform.system()}."
    )
```

**Preservation rules:**
- If the file already imports `platform` for another purpose, do not duplicate the import — reuse it and place the guard immediately after the existing `import platform` statement.
- The check is inserted exactly once per file; the skill is idempotent across re-runs.
- The literal string `platform.machine() == "arm64"` must appear in the inserted code (used by `tests/test_templates.py::test_t3_contains_apple_silicon_check`).

## Template T4: Test rewrite

If a test file (`test_app.py`, `test_streamlit_app.py`, `test_gradio_app.py`, or any `test_*.py` next to the app file) exists, update its mocks from `transformers.*.from_pretrained` to `mlx_lm.load`. The test file is otherwise left in place — the skill modifies only the mock targets and the inference invocation.

**Before:**

```python
from unittest.mock import MagicMock, patch

from streamlit_app import run_inference


@patch("streamlit_app.AutoTokenizer.from_pretrained")
@patch("streamlit_app.AutoModelForCausalLM.from_pretrained")
def test_run_inference_returns_string(mock_model_cls, mock_tokenizer_cls):
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "hello"
    mock_tokenizer_cls.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    mock_model_cls.return_value = mock_model

    result = run_inference("hi", mock_model, mock_tokenizer)

    assert result == "hello"
```

**After:**

```python
from unittest.mock import MagicMock, patch

from streamlit_app import run_inference


@patch("streamlit_app.mlx_lm.load")
def test_run_inference_returns_string(mock_load):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load.return_value = (mock_model, mock_tokenizer)

    with patch("streamlit_app.mlx_lm.generate", return_value="hello"):
        result = run_inference("hi", mock_model, mock_tokenizer)

    assert result == "hello"
```

**Preservation rules:**
- Test function names are preserved.
- Test assertions are preserved.
- The `mock_load.return_value` is set to a `(mock_model, mock_tokenizer)` tuple so callers of `load_model()` see the same shape as before.
- `mlx_lm.generate` is mocked separately when the test exercises the inference path; the return value is a string (matching mlx_lm's actual API).
- If the test file imports `from transformers import ...` directly (uncommon in tests), strip those imports.

## Template T5: Dep manifest delta

Update the project's dependency file to add `mlx-lm`. The skill detects the project type from which file exists in the working directory:

- `pyproject.toml` present → uv-managed (Streamlit convention).
- `requirements.txt` present → pip-installed (Gradio / Hugging Face Spaces convention).

If both files exist, the skill prefers `pyproject.toml` (uv) and prints a one-line warning that `requirements.txt` was not modified.

**Streamlit (uv-managed):**

The skill prints (does not auto-execute) the following command for the user to run:

```
uv add mlx-lm
```

The skill does not directly edit `pyproject.toml` — it lets `uv add` handle the toml manipulation, including version pinning and lockfile update.

**Gradio (`requirements.txt`):**

The skill appends a single line to `requirements.txt`:

```
mlx-lm
```

If `requirements.txt` already contains a pinned `mlx-lm` line, no change is made. The skill does not pin a version unless the user has pinned other deps in the same file (in which case it pins to the latest stable, queried via the HF Hub API alongside the variant resolution step).

**Removal hint (printed to user, both project types):**

```
Note: transformers and torch may now be unused in this app. If no other code
in your project imports them, you can remove them with:
  uv remove transformers torch        # Streamlit (uv)
  # or delete their lines from requirements.txt   # Gradio
The skill does not auto-remove these dependencies because they may be used by
other modules outside the converted file.
```

**Preservation rules:**
- The skill never auto-removes `transformers` or `torch` from dep files.
- Other dependencies are not modified.
- The version pin policy is: pin `mlx-lm` only if the surrounding file uses pins; otherwise leave unpinned.
