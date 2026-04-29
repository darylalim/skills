# Expected Output: Streamlit LLM Fixture

Post-rewrite invariants for `streamlit_app.py` and `test_streamlit_app.py`.
These are checked by a human against the EXPECTED.md checklist after running
`mlx-app-converter` against this fixture.

This is NOT a byte-for-byte diff. Claude varies whitespace and comment
placement; the invariants below describe the user-visible contract, not
formatting details.

---

## `streamlit_app.py` — post-rewrite checklist

### Imports

- [ ] `from transformers import AutoModelForCausalLM, AutoTokenizer` is **removed**.
- [ ] `import mlx_lm` is **added**.
- [ ] `from mlx_lm.sample_utils import make_logits_processors, make_sampler` is **added** (fixture uses `temperature` and `top_p`, which route through `make_sampler`).
- [ ] `import streamlit as st` is **preserved** verbatim.
- [ ] `from dotenv import load_dotenv` and `load_dotenv()` call are **preserved**.

### Apple Silicon runtime guard

- [ ] A runtime guard block is **inserted** near the top of the file (after the module docstring if present, before other imports).
- [ ] The guard contains `platform.machine() == "arm64"` and `platform.system() == "Darwin"`.
- [ ] The guard raises `RuntimeError` (not `SystemExit`) on non-arm64 macOS.
- [ ] `import platform` appears **before** `import mlx_lm` (so the guard fires before MLX import attempts on x86).

### `MODEL_ID`

- [ ] `MODEL_ID` constant **name** is unchanged.
- [ ] `MODEL_ID` **value** is updated to the user-selected MLX variant, e.g., `"mlx-community/Llama-3.1-8B-Instruct-bf16"`.

### `load_model()`

- [ ] `@st.cache_resource` decorator is **preserved verbatim** (not replaced, not removed).
- [ ] Body is replaced with a single `return mlx_lm.load(MODEL_ID)` call.
- [ ] `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` calls are **gone**.

### `run_inference()`

- [ ] Function **name** is unchanged: `run_inference`.
- [ ] `max_new_tokens` parameter renamed to `max_tokens`.
- [ ] `temperature` and `top_p` parameters are **preserved** in the signature.
- [ ] Body uses `make_sampler(temp=0.7, top_p=0.9)` to route sampling kwargs (note: `temp=`, not `temperature=`).
- [ ] Body calls `mlx_lm.generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler)`.
- [ ] Old three-call pattern (`tokenizer(...)`, `model.generate(...)`, `tokenizer.decode(...)`) is **gone**.
- [ ] `do_sample=True` kwarg is **dropped** (mlx-lm samples by default when temp > 0).
- [ ] Return type annotation `-> str` is **preserved**.

### Streamlit UI section

- [ ] All UI code below `run_inference` is **unchanged** — `st.title`, `st.caption`, `load_model()` call, `st.text_input`, `st.button`, `st.spinner`, `st.markdown`, `st.write`.

---

## `test_streamlit_app.py` — post-rewrite checklist

- [ ] `@patch("streamlit_app.AutoTokenizer.from_pretrained")` **removed**.
- [ ] `@patch("streamlit_app.AutoModelForCausalLM.from_pretrained")` **removed**.
- [ ] `@patch("streamlit_app.mlx_lm.load")` **added** as the outer decorator.
- [ ] `mock_load.return_value = (mock_model, mock_tokenizer)` present (tuple, matching `mlx_lm.load`'s return shape).
- [ ] `mlx_lm.generate` is mocked separately via a `with patch("streamlit_app.mlx_lm.generate", return_value="hello"):` context manager.
- [ ] Test function **name** `test_run_inference_returns_string` is unchanged.
- [ ] Assertion `assert isinstance(result, str)` (or equivalent) is preserved.

---

## `pyproject.toml` — post-rewrite checklist

- [ ] `mlx-lm` is **added** to `dependencies` (via `uv add mlx-lm`, printed by the skill — run it manually).
- [ ] Existing deps (`streamlit`, `transformers`, `torch`, `python-dotenv`) remain in the file — the skill never auto-removes them.
- [ ] `requires-python` and `[dependency-groups]` are unchanged.

---

## Things that should NOT change

- The file structure (no new files created, no files deleted).
- The Streamlit UI code block at the bottom of `streamlit_app.py`.
- The `pyproject.toml` order of existing dependencies (only `mlx-lm` appended/added).
- The test function name and its `assert isinstance(result, str)` assertion.
