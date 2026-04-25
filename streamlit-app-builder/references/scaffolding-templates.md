# Scaffolding templates

Concrete template blocks consumed by `SKILL.md` Step 5. Each variant below lists the files it produces, the placeholder substitutions to perform at scaffold time, and the verbatim code body. Selection is per-pipeline-tag and per-`mflux_family`; SKILL.md Step 5 prose names which variant to use for each input.

The mflux family-specific blocks in Part B of `mflux-families.md` are inlined into Variant A and Variant B `inference.py` templates here at scaffold time. Where a template has multiple per-family forms (Variant B `image-to-image`), each family gets its own subsection.

## Variant: text-generation `inference.py`

Used for `pipeline_tag ∈ {text-generation, conversational}`.

**Files produced:** `src/<app_name>/inference.py`

**Placeholder substitutions at scaffold time:**
- `<app_name>` → app's importable Python name
- `MLX_MODEL_ID_DEFAULT` value → matched mlx-community model ID, or `None` if no MLX equivalent

**Body:**

```python
"""Model loading and inference. Dispatches MLX <-> transformers by platform."""
from functools import lru_cache
from typing import Any

from <app_name> import config

# MLX model ID chosen at scaffold time (highest downloads under mlx-community).
# Override by setting MLX_MODEL_ID in .env.
MLX_MODEL_ID_DEFAULT: str | None = "<mlx-community/...>"


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Lazy-load the model once per process."""
    if config.IS_APPLE_SILICON and MLX_MODEL_ID_DEFAULT:
        return _load_mlx()
    return _load_transformers()


def _load_mlx():
    import os as _os

    from mlx_lm import load

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

(The `generate_response_stream` function is added in Task 17.)

## Variant A: text-to-image `inference.py` (mflux + diffusers fallback)

Used for `pipeline_tag = text-to-image` with `mflux_family ∈ {flux}` (the only family with a diffusers fallback today).

(body added in Task 10)

## Variant A: image-to-image `inference.py` (flux family — diffusers fallback)

Used for `pipeline_tag = image-to-image` with `mflux_family = flux`.

(body added in Task 10)

## Variant B: text-to-image `inference.py` (Apple-Silicon-only)

Used for `pipeline_tag = text-to-image` with `mflux_family ∈ {flux2, qwen_image, fibo, z_image}`.

(body added in Task 11)

## Variant B: image-to-image `inference.py` — `flux2`

(body added in Task 12)

## Variant B: image-to-image `inference.py` — `qwen_image`

(body added in Task 12)

## Variant B: image-to-image `inference.py` — `fibo`

(body added in Task 12)

## Variant: diffusers-only fallback `inference.py`

Used for `pipeline_tag ∈ {text-to-image, image-to-image}` when `mflux_family = None`.

(body added in Task 13)

## Test fixtures: `tests/conftest.py`

(body added in Task 14)
