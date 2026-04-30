# Streamlit VLM fixture — expected post-rewrite state

After running `mlx-app-converter` on this fixture, walk through each invariant
below. Mark **✓** if it holds, **✗** if not.

## T1 Loader (VLM)

- [ ] `MODEL_ID` constant name preserved.
- [ ] `MODEL_ID` value rewritten to a `mlx-community/Qwen2-VL-7B-Instruct-<quant>` variant (user-selected).
- [ ] `@st.cache_resource` decorator preserved verbatim on `load_model`.
- [ ] Function body is exactly `return mlx_vlm.load(MODEL_ID)`.
- [ ] `from transformers import AutoProcessor, AutoModelForVision2Seq` removed.
- [ ] `import mlx_vlm` added at module top.

## T2 Inference (VLM)

- [ ] `run_inference` signature preserved (prompt, image, model, processor, max_tokens).
- [ ] `max_new_tokens` parameter renamed to `max_tokens`.
- [ ] Function body returns `mlx_vlm.generate(model, processor, prompt, image, max_tokens=max_tokens).text` (`.text` extracts from `GenerationResult` dataclass).
- [ ] `processor(...)`, `model.generate(...)`, `processor.decode(...)` calls removed.
- [ ] No `make_sampler` / `make_logits_processors` imports added (mlx-vlm uses direct kwargs).

## T3 Apple Silicon guard

- [ ] `import platform` and the `arm64`/`Darwin` check inserted at top of file (before any `mlx_vlm` import).
- [ ] Single insertion (not duplicated).

## T4 Test rewrite

- [ ] `@patch` target switched to `streamlit_app.mlx_vlm.load`.
- [ ] `mock_load.return_value = (mock_model, mock_processor)`.
- [ ] Inner `mock_result = MagicMock(); mock_result.text = "..."` setup, then `with patch("streamlit_app.mlx_vlm.generate", return_value=mock_result)` (matches `GenerationResult` shape).
- [ ] Test function name and assertion preserved.

## T5 Dep manifest

- [ ] Skill prints `uv add mlx-vlm` (single-package set, VLM-only file).
- [ ] `pyproject.toml` not directly edited by the skill (uv add owns toml manipulation).
- [ ] Removal hint printed for `transformers` and `torch`.

## Verification

- [ ] `uv run ruff check streamlit_app.py test_streamlit_app.py` — no errors.
- [ ] `uv run ruff format --check streamlit_app.py test_streamlit_app.py` — no diffs.
- [ ] `uv run ty check streamlit_app.py` — no errors.
- [ ] `uv run pytest test_streamlit_app.py -v` — passes (after `uv add mlx-vlm`).
