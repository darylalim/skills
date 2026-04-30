# Streamlit multi-modal fixture — expected post-rewrite state

This fixture exercises **multi-modal** features unique to mlx-app-converter v2:
two models in one file with different modalities, import unioning, and combined
dep manifest.

After running `mlx-app-converter` on this fixture, walk through each invariant.
Mark **✓** if it holds, **✗** if not.

## Step 4 — variant resolution

- [ ] Skill presents **two** variant matrices, one per detected (deduped) model:
  one for `Qwen/Qwen2-VL-7B-Instruct`, one for `meta-llama/Llama-3.1-8B-Instruct`.
- [ ] Each matrix has its own default-pick marker and accepts independent replies.

## T1 Loader

- [ ] Both `VLM_MODEL_ID` and `LLM_MODEL_ID` constant names preserved.
- [ ] `VLM_MODEL_ID` rewritten to a `mlx-community/Qwen2-VL-7B-Instruct-<quant>` variant.
- [ ] `LLM_MODEL_ID` rewritten to a `mlx-community/Llama-3.1-8B-Instruct-<quant>` variant.
- [ ] `load_vlm` body is `return mlx_vlm.load(VLM_MODEL_ID)`; `@st.cache_resource` preserved.
- [ ] `load_llm` body is `return mlx_lm.load(LLM_MODEL_ID)`; `@st.cache_resource` preserved.

## Imports (unioning)

- [ ] `import mlx_lm` present at top of file.
- [ ] `import mlx_vlm` present at top of file.
- [ ] No duplicate `import mlx_lm` or `import mlx_vlm` lines.
- [ ] `from transformers import (...)` removed entirely (all four imported names — `AutoModelForCausalLM`, `AutoModelForVision2Seq`, `AutoProcessor`, `AutoTokenizer` — now unused).

## T2 Inference

- [ ] `describe_image` body returns `mlx_vlm.generate(model, processor, prompt, image, max_tokens=max_tokens).text` (signature renames `max_new_tokens` → `max_tokens`; `.text` extracts from `GenerationResult`).
- [ ] `follow_up` body is `return mlx_lm.generate(model, tokenizer, prompt, max_tokens=max_tokens)`.

## T3 Apple Silicon guard

- [ ] Inserted **once** at top of file (above both `import mlx_lm` and `import mlx_vlm`).

## T4 Test rewrite

- [ ] `test_describe_image_returns_string` patches `streamlit_app.mlx_vlm.load` (returns `(mock_model, mock_processor)`).
- [ ] `test_follow_up_returns_string` patches `streamlit_app.mlx_lm.load` (returns `(mock_model, mock_tokenizer)`).
- [ ] Both tests exercise their inference path via inner `with patch(... .generate, return_value=...)` blocks.

## T5 Dep manifest (multi-modal)

- [ ] Skill prints a **single** `uv add mlx-lm mlx-vlm` command (not two separate commands).
- [ ] Removal hint printed for `transformers` and `torch`.
- [ ] `pyproject.toml` not directly edited.

## Verification

- [ ] `uv run ruff check streamlit_app.py test_streamlit_app.py` — no errors.
- [ ] `uv run ruff format --check streamlit_app.py test_streamlit_app.py` — no diffs.
- [ ] `uv run ty check streamlit_app.py` — no errors.
- [ ] `uv run pytest test_streamlit_app.py -v` — both tests pass after `uv add mlx-lm mlx-vlm`.
