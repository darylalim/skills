# mflux Family Routing and Invocation

When the input is an HF model card with `pipeline_tag` in `{text-to-image, image-to-image}`, the skill matches the model ID against the routing table in Part A to select an `mflux_family`. On match, the family's Part B invocation block is inlined into the generated `inference.py` at scaffold time. On no match, `mflux_family = None` and the scaffold emits a diffusers-only app.

mflux is Apple-Silicon-only. Families flagged "Apple-Silicon-only" have no diffusers-compatible fallback and the generated `inference.py` raises `RuntimeError` on non-Apple-Silicon hosts. Families without that flag fall back to a `diffusers` pipeline off Apple Silicon.

## Part A — Routing table

First-row-wins matching of `<org>/<model>` against each regex. On match, the `Family key` value becomes the IR's `mflux_family`; the `Apple-Silicon-only` and `Diffusers fallback` columns drive Step 6 platform conditionals and the Step 7 no-Apple-Silicon path.

| Family key  | Regex                                                              | Apple-Silicon-only | Diffusers fallback                                                                |
|-------------|--------------------------------------------------------------------|--------------------|-----------------------------------------------------------------------------------|
| `flux`      | `^black-forest-labs/FLUX\.1-(schnell\|dev\|Kontext-dev)$`          | no                 | `diffusers.FluxPipeline` (t2i) / `diffusers.FluxImg2ImgPipeline` (i2i)            |
| `flux2`     | `^black-forest-labs/FLUX\.2-.*$`                                   | yes                | —                                                                                 |
| `qwen_image`| `^Qwen/Qwen-Image(-Edit(-\d+)?)?$`                                 | yes                | —                                                                                 |
| `fibo`      | `^briaai/(FIBO\|FIBO-lite\|Fibo-Edit\|Fibo-Edit-RMBG)$`            | yes                | —                                                                                 |
| `z_image`   | `^Tongyi-MAI/Z-Image$\|^filipstrand/Z-Image-Turbo-mflux-4bit$`     | yes                | —                                                                                 |

If no row matches, `mflux_family = None`.

## Part B — Per-family invocation blocks

Each block provides the canonical mflux snippet for text-to-image and (where applicable) image-to-image, plus default generation params. Scaffold-time consumers read two things from each block:

1. Import + instantiation + generate lines — inlined verbatim into `inference.py` at scaffold time.
2. Default generation params — substituted into the page template's slider `value=...` arguments.

### `flux`

Text-to-image:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

model = Flux1(model_config=ModelConfig.schnell())
image = model.generate_image(
    seed=seed, prompt=prompt,
    num_inference_steps=num_inference_steps,
    width=width, height=height,
    guidance=4.0,
)
```

Image-to-image (Kontext):

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

model = Flux1Kontext(model_config=ModelConfig.dev_kontext())
image = model.generate_image(
    seed=seed, prompt=prompt,
    num_inference_steps=num_inference_steps,
    width=width, height=height,
    guidance=4.0,
    image_path=image_paths[0],
)
```

`Flux1Kontext.generate_image` takes a single `image_path` (not an `image_paths` list); the scaffold inference wrapper passes `image_paths[0]`.

Defaults:

| Mode | steps | width | height |
|------|-------|-------|--------|
| t2i  | 4     | 1024  | 1024   |
| i2i  | 4     | 1024  | 1024   |

### `flux2`

Apple-Silicon-only — no diffusers fallback.

Text-to-image:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein

model = Flux2Klein(model_config=ModelConfig.flux2_klein_9b())
image = model.generate_image(
    seed=seed, prompt=prompt,
    num_inference_steps=num_inference_steps,
    width=width, height=height,
)
```

Image-to-image:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants import Flux2KleinEdit

model = Flux2KleinEdit(model_config=ModelConfig.flux2_klein_9b())
image = model.generate_image(
    seed=seed, prompt=prompt,
    image_paths=image_paths,
    num_inference_steps=num_inference_steps,
)
```

`Flux2KleinEdit.generate_image` derives output dimensions from the input reference image(s); no `width` / `height` kwargs are accepted.

Defaults:

| Mode | steps | width | height |
|------|-------|-------|--------|
| t2i  | 4     | 1024  | 560    |
| i2i  | 4     | —     | —      |

### `qwen_image`

Apple-Silicon-only — no diffusers fallback.

Text-to-image:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

model = QwenImage(model_config=ModelConfig.qwen_image())
image = model.generate_image(
    seed=seed, prompt=prompt,
    num_inference_steps=num_inference_steps,
    width=width, height=height,
)
```

Image-to-image (Edit):

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

model = QwenImageEdit(model_config=ModelConfig.qwen_image_edit())
image = model.generate_image(
    seed=seed, prompt=prompt,
    image_paths=image_paths,
    num_inference_steps=num_inference_steps,
    width=width, height=height,
    guidance=2.5,
)
```

Defaults:

| Mode | steps | width | height |
|------|-------|-------|--------|
| t2i  | 30    | 1920  | 816    |
| i2i  | 30    | 624   | 1024   |

### `fibo`

Apple-Silicon-only — no diffusers fallback.

Text-to-image:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO

model = FIBO(model_config=ModelConfig.fibo())
image = model.generate_image(
    seed=seed, prompt=prompt,
    num_inference_steps=num_inference_steps,
    width=width, height=height,
    guidance=4.0,
)
```

Image-to-image (Edit):

```python
from mflux.models.common.config import ModelConfig
from mflux.models.fibo.variants.edit import FIBOEdit

model = FIBOEdit(model_config=ModelConfig.fibo_edit())
image = model.generate_image(
    seed=seed, prompt=prompt,
    image_path=image_paths[0],
    num_inference_steps=num_inference_steps,
    width=width, height=height,
    guidance=3.5,
)
```

`FIBOEdit.generate_image` takes a single `image_path` (not an `image_paths` list); the scaffold inference wrapper passes `image_paths[0]`.

Defaults:

| Mode | steps | width | height |
|------|-------|-------|--------|
| t2i  | 50    | 1200  | 540    |
| i2i  | 30    | 640   | 384    |

### `z_image`

Apple-Silicon-only — no diffusers fallback.

No image-to-image variant for Z-Image — emit a text-to-image-only page.

The scaffold consumer picks between the two Z-Image variants based on the matched HF model ID, not on the family key alone:

| Matched HF model ID                     | Variant | `ModelConfig` factory         | `model_path` argument                          | steps | width | height |
|-----------------------------------------|---------|-------------------------------|------------------------------------------------|-------|-------|--------|
| `Tongyi-MAI/Z-Image`                    | base    | `ModelConfig.z_image()`       | `"Tongyi-MAI/Z-Image"`                         | 50    | 720   | 1280   |
| `filipstrand/Z-Image-Turbo-mflux-4bit`  | turbo   | `ModelConfig.z_image_turbo()` | `"filipstrand/Z-Image-Turbo-mflux-4bit"`       | 9     | 1280  | 720    |

Base text-to-image:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImage

model = ZImage(
    model_config=ModelConfig.z_image(),
    model_path="Tongyi-MAI/Z-Image",
)
image = model.generate_image(
    seed=seed, prompt=prompt,
    num_inference_steps=num_inference_steps,
    width=width, height=height,
    guidance=4.0,
)
```

Turbo text-to-image:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.z_image import ZImage

model = ZImage(
    model_config=ModelConfig.z_image_turbo(),
    model_path="filipstrand/Z-Image-Turbo-mflux-4bit",
)
image = model.generate_image(
    seed=seed, prompt=prompt,
    num_inference_steps=num_inference_steps,
    width=width, height=height,
    guidance=4.0,
)
```

## Adding a new family

1. Read `src/mflux/models/<family>/README.md` in the upstream mflux repo to identify the canonical text-to-image (and, if present, image-to-image) class, `ModelConfig` factory, and required `generate_image` arguments.
2. Copy the canonical snippet into a new Part B section, keeping the `seed` / `prompt` / `num_inference_steps` / `width` / `height` argument names consistent with existing sections so the scaffold template can substitute them uniformly.
3. Add a routing row to Part A with an anchored regex (`^...$`) covering the family's canonical HF model IDs. Verify the regex does not overlap any earlier row — first-row-wins means an overlapping earlier row would shadow the new family.
4. If the family has no diffusers-compatible fallback, set `Apple-Silicon-only` to `yes` and leave `Diffusers fallback` as `—`. Otherwise, name the concrete `diffusers` pipeline class(es) used off Apple Silicon.
