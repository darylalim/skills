# Scaffolding templates

Concrete template blocks consumed by `SKILL.md` Step 5. Each variant below lists the files it produces, the placeholder substitutions to perform at scaffold time, and the verbatim code body. Selection is per-pipeline-tag and per-`mflux_family`; SKILL.md Step 5 prose names which variant to use for each input.

The mflux family-specific blocks in Part B of `mflux-families.md` are inlined into Variant A and Variant B `inference.py` templates here at scaffold time. Where a template has multiple per-family forms (Variant B `image-to-image`), each family gets its own subsection.

## Variant: text-generation `inference.py`

Used for `pipeline_tag ∈ {text-generation, conversational}`.

(body added in Task 9)

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
