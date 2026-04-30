# Gradio VLM fixture — expected post-rewrite state

After running `mlx-app-converter` on this fixture, walk through each invariant
below. Mark **✓** if it holds, **✗** if not.

## T1 Loader (VLM)

- [ ] `MODEL_ID` constant name preserved.
- [ ] `MODEL_ID` value rewritten to a `mlx-community/Qwen2-VL-7B-Instruct-<quant>` variant.
- [ ] `@lru_cache(maxsize=1)` decorator preserved verbatim.
- [ ] Function body is exactly `return mlx_vlm.load(MODEL_ID)`.
- [ ] `from functools import lru_cache` import preserved.
- [ ] `from transformers import AutoProcessor, AutoModelForVision2Seq` removed.
- [ ] `import mlx_vlm` added at module top.

## T2 Inference (VLM)

- [ ] `run_inference` signature preserved.
- [ ] `max_new_tokens` parameter renamed to `max_tokens`.
- [ ] Function body returns `mlx_vlm.generate(model, processor, prompt, image, max_tokens=max_tokens).text` (`.text` extracts from `GenerationResult`).

## T3 Apple Silicon guard

- [ ] Inserted at top of file (before any `mlx_vlm` import).
- [ ] Single insertion (not duplicated).

## T4 Test rewrite

- [ ] `@patch` target switched to `app.mlx_vlm.load`.
- [ ] `mock_load.return_value = (mock_model, mock_processor)`.
- [ ] Inner mock has `mock_result.text = "..."` and `with patch("app.mlx_vlm.generate", return_value=mock_result)`.

## T5 Dep manifest

- [ ] `mlx-vlm` line appended to `requirements.txt`.
- [ ] No duplicate `mlx-vlm` line if one was already present.
- [ ] Skill does NOT remove `transformers` or `torch` lines automatically.
- [ ] Removal hint printed.

## Verification

- [ ] `ruff check app.py test_app.py` — no errors.
- [ ] `ruff format --check app.py test_app.py` — no diffs.
- [ ] `ty check app.py` — no errors.
- [ ] `pytest test_app.py -v` — passes (after `pip install -r requirements.txt`).
