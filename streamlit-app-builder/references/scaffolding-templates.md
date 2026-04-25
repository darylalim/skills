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

**Files produced:** `src/<app_name>/inference.py`

**Placeholder substitutions at scaffold time:**
- `<app_name>` → app's importable name
- The `_load_mflux()` body's import + instantiation lines are inlined verbatim from `references/mflux-families.md` Part B (matched family — currently only `flux`)

**Body:**

```python
"""Image inference. Dispatches mflux <-> diffusers by platform."""
from functools import lru_cache
from typing import Any

from <app_name> import config
from PIL import Image


@lru_cache(maxsize=1)
def load_model() -> Any:
    if config.IS_APPLE_SILICON:
        return _load_mflux()
    return _load_diffusers()


def _load_mflux():
    # <inlined verbatim from mflux-families.md Part B — imports + instantiation>
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1

    model = Flux1(model_config=ModelConfig.schnell())
    return ("mflux", model)


def _load_diffusers():
    import torch
    from diffusers import FluxPipeline

    device = (
        config.DEVICE
        if config.DEVICE != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pipe = FluxPipeline.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION,
        torch_dtype=torch.bfloat16, token=config.HF_TOKEN,
    ).to(device)
    return ("diffusers", pipe)


def generate_image(prompt, width, height, num_inference_steps, seed) -> Image.Image:
    backend, model = load_model()
    if backend == "mflux":
        # Match kwargs to the Part B snippet (e.g. guidance=4.0 for flux).
        return model.generate_image(
            seed=seed, prompt=prompt, width=width, height=height,
            num_inference_steps=num_inference_steps,
        ).image
    import torch
    return model(
        prompt=prompt, width=width, height=height,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]
```

## Variant A: image-to-image `inference.py` (flux family — diffusers fallback)

Used for `pipeline_tag = image-to-image` with `mflux_family = flux`.

**Files produced:** `src/<app_name>/inference.py`

**Placeholder substitutions:** same as Variant A t2i.

**Body:**

```python
"""Image-to-image inference. Dispatches mflux <-> diffusers by platform."""
from functools import lru_cache
from typing import Any

from <app_name> import config
from PIL import Image


# Flux1Kontext.generate_image returns a GeneratedImage wrapper; call .image.
@lru_cache(maxsize=1)
def load_model() -> Any:
    if config.IS_APPLE_SILICON:
        return _load_mflux()
    return _load_diffusers()


def _load_mflux():
    # <inlined verbatim from mflux-families.md Part B, flux i2i subsection>
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

    model = Flux1Kontext(model_config=ModelConfig.dev_kontext())
    return ("mflux", model)


def _load_diffusers():
    import torch
    from diffusers import FluxImg2ImgPipeline

    device = (
        config.DEVICE
        if config.DEVICE != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    pipe = FluxImg2ImgPipeline.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION,
        torch_dtype=torch.bfloat16, token=config.HF_TOKEN,
    ).to(device)
    return ("diffusers", pipe)


def edit_image(prompt, image_paths, num_inference_steps, seed) -> Image.Image:
    backend, model = load_model()
    if backend == "mflux":
        # Flux1Kontext takes image_path (singular) and guidance=4.0.
        return model.generate_image(
            seed=seed, prompt=prompt,
            num_inference_steps=num_inference_steps,
            image_path=image_paths[0],
        ).image
    import torch
    reference = Image.open(image_paths[0])
    return model(
        prompt=prompt, image=reference,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]
```

## Variant B: text-to-image `inference.py` (Apple-Silicon-only)

Used for `pipeline_tag = text-to-image` with `mflux_family ∈ {flux2, qwen_image, fibo, z_image}`.

**Files produced:** `src/<app_name>/inference.py`

**Placeholder substitutions:**
- `<app_name>` → app's importable name
- `<family>` → matched `mflux_family` (used in the RuntimeError message)
- The `_load_mflux()` body's imports + instantiation are inlined from `references/mflux-families.md` Part B for the matched family. The example below uses `flux2`; substitute the per-family lines from Part B.

**Body:**

```python
"""Image inference. Apple-Silicon-only (no diffusers fallback for this family)."""
from functools import lru_cache
from typing import Any

from <app_name> import config
from PIL import Image


# generate_image returns a GeneratedImage wrapper; call .image for PIL.Image.
@lru_cache(maxsize=1)
def load_model() -> Any:
    if not config.IS_APPLE_SILICON:
        raise RuntimeError(
            "This app requires Apple Silicon. The <family> family has no "
            "diffusers fallback. Run on a Mac with Apple Silicon, or "
            "re-scaffold from an HF model card whose family has a diffusers "
            "fallback (e.g., black-forest-labs/FLUX.1-schnell)."
        )
    # <inlined verbatim from mflux-families.md Part B>
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein

    model = Flux2Klein(model_config=ModelConfig.flux2_klein_9b())
    return ("mflux", model)


def generate_image(prompt, width, height, num_inference_steps, seed) -> Image.Image:
    _, model = load_model()
    # Match kwargs to the Part B snippet (e.g. guidance=4.0 for flux).
    return model.generate_image(
        seed=seed, prompt=prompt, width=width, height=height,
        num_inference_steps=num_inference_steps,
    ).image
```

## Variant B: image-to-image `inference.py` — `flux2`

**Files produced:** `src/<app_name>/inference.py`

**Trigger:** `pipeline_tag = image-to-image` AND `mflux_family = flux2`.

**Body:**

```python
"""Image-to-image inference. Apple-Silicon-only (flux2, no fallback)."""
from functools import lru_cache
from typing import Any

from <app_name> import config
from PIL import Image


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not config.IS_APPLE_SILICON:
        raise RuntimeError(
            "This app requires Apple Silicon. The flux2 family has no "
            "diffusers fallback. Run on a Mac with Apple Silicon, or "
            "pick a model family with a generic diffusers backend "
            "(e.g., any black-forest-labs/FLUX.1-* model)."
        )
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2KleinEdit

    model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_9b())
    return ("mflux", model)


def edit_image(prompt, image_paths, num_inference_steps, seed) -> Image.Image:
    _, model = load_model()
    return model.generate_image(
        seed=seed, prompt=prompt,
        image_paths=image_paths,
        num_inference_steps=num_inference_steps,
    ).image
```

## Variant B: image-to-image `inference.py` — `qwen_image`

**Files produced:** `src/<app_name>/inference.py`

**Trigger:** `pipeline_tag = image-to-image` AND `mflux_family = qwen_image`.

**Body:**

```python
"""Image-to-image inference. Apple-Silicon-only (qwen_image, no fallback)."""
from functools import lru_cache
from typing import Any

from <app_name> import config
from PIL import Image


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not config.IS_APPLE_SILICON:
        raise RuntimeError(
            "This app requires Apple Silicon. The qwen_image family has no "
            "diffusers fallback. Run on a Mac with Apple Silicon, or "
            "pick a model family with a generic diffusers backend "
            "(e.g., any black-forest-labs/FLUX.1-* model)."
        )
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

    model = QwenImageEdit(model_config=ModelConfig.qwen_image_edit())
    return ("mflux", model)


def edit_image(prompt, image_paths, num_inference_steps, seed) -> Image.Image:
    _, model = load_model()
    return model.generate_image(
        seed=seed, prompt=prompt,
        image_paths=image_paths,
        num_inference_steps=num_inference_steps,
        guidance=2.5,
    ).image
```

## Variant B: image-to-image `inference.py` — `fibo`

**Files produced:** `src/<app_name>/inference.py`

**Trigger:** `pipeline_tag = image-to-image` AND `mflux_family = fibo`.

**Note:** `FIBOEdit.generate_image` takes a single `image_path` (not a list). The wrapper passes `image_paths[0]`.

**Body:**

```python
"""Image-to-image inference. Apple-Silicon-only (fibo, no fallback)."""
from functools import lru_cache
from typing import Any

from <app_name> import config
from PIL import Image


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not config.IS_APPLE_SILICON:
        raise RuntimeError(
            "This app requires Apple Silicon. The fibo family has no "
            "diffusers fallback. Run on a Mac with Apple Silicon, or "
            "pick a model family with a generic diffusers backend "
            "(e.g., any black-forest-labs/FLUX.1-* model)."
        )
    from mflux.models.common.config import ModelConfig
    from mflux.models.fibo.variants.edit import FIBOEdit

    model = FIBOEdit(model_config=ModelConfig.fibo_edit())
    return ("mflux", model)


def edit_image(prompt, image_paths, num_inference_steps, seed) -> Image.Image:
    _, model = load_model()
    return model.generate_image(
        seed=seed, prompt=prompt,
        image_path=image_paths[0],
        num_inference_steps=num_inference_steps,
        guidance=3.5,
    ).image
```

## Variant: diffusers-only fallback `inference.py`

Used for `pipeline_tag ∈ {text-to-image, image-to-image}` when `mflux_family = None`.

(body added in Task 13)

## Test fixtures: `tests/conftest.py`

(body added in Task 14)
