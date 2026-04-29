# Expected Output: Gradio LLM Fixture

Post-rewrite invariants for `app.py` and `test_app.py`.
These are checked by a human against the EXPECTED.md checklist after running
`mlx-app-converter` against this fixture.

This is NOT a byte-for-byte diff. Claude varies whitespace and comment
placement; the invariants below describe the user-visible contract, not
formatting details.

---

## `app.py` — post-rewrite checklist

### Imports

- [ ] `from transformers import AutoModelForCausalLM, AutoTokenizer` is **removed**.
- [ ] `import mlx_lm` is **added**.
- [ ] `from mlx_lm.sample_utils import make_logits_processors, make_sampler` is **added** (fixture uses `temperature` and `top_p`, which route through `make_sampler`).
- [ ] `from functools import lru_cache` is **preserved** (still needed for the cache decorator).
- [ ] `import gradio as gr` is **preserved** verbatim.
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

- [ ] `@lru_cache(maxsize=1)` decorator is **preserved verbatim** (not replaced, not removed, not changed to `@st.cache_resource`).
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

### Gradio UI section

- [ ] `run_inference_for_chat` adapter function is **unchanged**.
- [ ] `demo = gr.ChatInterface(...)` block is **unchanged**.
- [ ] `if __name__ == "__main__": demo.launch()` is **unchanged**.

---

## `test_app.py` — post-rewrite checklist

- [ ] `@patch("app.AutoTokenizer.from_pretrained")` **removed**.
- [ ] `@patch("app.AutoModelForCausalLM.from_pretrained")` **removed**.
- [ ] `@patch("app.mlx_lm.load")` **added** as the outer decorator.
- [ ] `mock_load.return_value = (mock_model, mock_tokenizer)` present (tuple, matching `mlx_lm.load`'s return shape).
- [ ] `mlx_lm.generate` is mocked separately via a `with patch("app.mlx_lm.generate", return_value="hello"):` context manager.
- [ ] Test function **name** `test_run_inference_returns_string` is unchanged.
- [ ] Assertion `assert isinstance(result, str)` (or equivalent) is preserved.

---

## `requirements.txt` — post-rewrite checklist

- [ ] `mlx-lm` line is **appended** to `requirements.txt` by the skill directly (no `uv add` — Gradio uses `requirements.txt`, not `pyproject.toml`).
- [ ] Existing deps (`gradio==5.0.0`, `transformers`, `torch`, `python-dotenv`) remain — the skill never auto-removes them.
- [ ] Because the existing file uses `gradio==5.0.0` (pinned), `mlx-lm` should be pinned to the latest stable version too (queried alongside variant resolution).

---

## Things that should NOT change

- The file structure (no new files created, no files deleted).
- The Gradio UI code block at the bottom of `app.py` (`run_inference_for_chat`, `demo`, `if __name__ == "__main__"`).
- The `requirements.txt` order of existing dependencies (only `mlx-lm` appended at the end).
- The test function name and its `assert isinstance(result, str)` assertion.
- The `uv add` command is **not** printed for Gradio fixtures — the skill appends to `requirements.txt` directly and mentions that in its output.
