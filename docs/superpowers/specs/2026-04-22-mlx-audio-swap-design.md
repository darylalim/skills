# Streamlit App Builder — MLX Audio Backend Swap

**Date:** 2026-04-22
**Skill affected:** `streamlit-app-builder`
**Status:** Design approved, ready for implementation plan

## Context

The `streamlit-app-builder` skill currently wires three MLX libraries into generated apps: `mlx-lm` for text generation, `mlx-vlm` for vision-language, and `mlx-whisper` for ASR. The MLX ecosystem has since consolidated — `mlx-whisper` has been superseded by `mlx-audio` (https://github.com/Blaizzy/mlx-audio), which exposes a broader surface covering speech-to-text (STT), text-to-speech (TTS), and speech-to-speech / audio-to-audio (STS) from a single package.

This change retargets the skill's MLX policy to:

1. `mlx-lm` (https://github.com/ml-explore/mlx-lm) — text generation
2. `mlx-vlm` (https://github.com/Blaizzy/mlx-vlm) — vision-language
3. `mlx-audio` (https://github.com/Blaizzy/mlx-audio) — audio (STT, TTS, STS)

`mlx-lm` and `mlx-vlm` APIs continue to match what the skill already encodes (verified against current READMEs). The material change is the audio surface: swap `mlx-whisper` for `mlx-audio`, add MLX backing to the existing `text-to-speech` pattern (which today has no MLX branch), and add a new `audio-to-audio` pipeline pattern.

## Scope

### In scope

- Replace every `mlx-whisper` reference in `SKILL.md` with `mlx-audio`, using the correct `mlx_audio.stt` import path and model API (`load(...).generate(audio).text`).
- Extend `inference.py`'s per-pipeline dispatch guidance to cover `mlx_audio.stt` (ASR), `mlx_audio.tts` (TTS), and `mlx_audio.sts` (STS).
- Wire MLX into the existing TTS pattern (currently transformers-only on all platforms).
- Add a new `audio-to-audio` pipeline entry to `references/pipeline-tag-patterns.md` for speech-to-speech / separation / enhancement.
- Replace the current prose description of MLX backends in the cross-cutting principle with a compact pipeline → MLX backend index table.
- Update Step 6's pattern-specific deps table with new ASR, TTS, and audio-to-audio rows.
- Add a conditional "Apple-Silicon-only warning" bullet to Step 8 (Report to user) for audio-to-audio scaffolds.

### Explicit non-goals

- **No backend-catalog reference file.** The MLX backend mapping stays inline in `SKILL.md`; no new `references/mlx-backends.md` file.
- **No transformers fallback for STS.** Audio-to-audio apps are Apple-Silicon-only by design. `inference.py` raises `RuntimeError` at model load on non-Apple hosts; `pyproject.toml` does not declare a transformers fallback dep for this pipeline.
- **No MLX live-docs fetch rule.** The skill's principle 1 ("always verify against live `docs.streamlit.io`") does not gain a parallel rule for MLX READMEs. If MLX APIs drift, that is addressed in a separate change.
- **No `mlx-audio` REST server mode.** `mlx-audio`'s OpenAI-compatible server endpoints are not used; inference is in-process in the Streamlit app, consistent with the rest of the skill.
- **No expansion into adjacent HF pipeline tags** (`audio-classification`, `voice-activity-detection`, `audio-text-to-text` / forced alignment, `text-to-audio`). These remain unsupported and fall through to General Script.
- **No changes to `mlx-lm` or `mlx-vlm` wiring.** The existing text-generation `inference.py` template and vision-language references are correct and remain unchanged except for the paragraph restructured below.

## Design

### 1. MLX cross-cutting principle rewrite

Replace the existing cross-cutting principle 2 in `SKILL.md` (currently lines 41-52) with:

> ### 2. Prefer MLX on Apple Silicon
>
> When the source artifact references a model with an MLX-converted equivalent on HuggingFace, the generated app uses an MLX backend on `arm64-darwin` and falls back to `transformers` / `diffusers` / `huggingface_hub` elsewhere.
>
> **MLX backend index:**
>
> | `pipeline_tag` | MLX module | PyPI | Apple-only? | Transformers fallback |
> |---|---|---|---|---|
> | `text-generation`, `conversational` | `mlx_lm` | `mlx-lm` | no | `transformers` |
> | `image-to-text`, `image-text-to-text` | `mlx_vlm` | `mlx-vlm` | no | `transformers` |
> | `automatic-speech-recognition` | `mlx_audio.stt` | `mlx-audio` | no | `transformers[audio]` (via `pipeline("automatic-speech-recognition")`) |
> | `text-to-speech` | `mlx_audio.tts` | `mlx-audio` | no | `transformers[audio]` (SpeechT5 / Bark / Parler-TTS) |
> | `audio-to-audio` | `mlx_audio.sts` | `mlx-audio` | **yes** | — (`RuntimeError` at model load off Apple Silicon) |
>
> Apple-only rows install no transformers fallback; `inference.py` raises a clear `RuntimeError` at model load on non-Apple hosts, and the generated `README.md` notes the platform requirement.
>
> The MLX lookup is independent of where the skill itself runs. A Linux developer scaffolding from an HF model card still produces an app with MLX support wired in — runtime dispatch activates MLX only when a user later runs the app on a Mac. (Exception: `audio-to-audio` apps run on Apple Silicon only, by design.)
>
> MLX support is encoded in the generated app as follows:
> - `pyproject.toml` declares MLX and transformers with environment markers so `uv sync` installs the right backend per host. Audio-to-audio apps declare `mlx-audio` with an Apple-only marker and omit the fallback dep.
> - `src/<app_name>/inference.py` reads `config.IS_APPLE_SILICON` and dispatches at runtime.
>
> **MLX model resolution:** query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` and pick the highest-download-count variant. If the base name has no `mlx-community` match, note "no MLX equivalent found" in the final report and generate the app with `transformers` only. Audio-to-audio inputs without a match fail at scaffold time with a clear error — there is no fallback to generate toward.

### 2. `inference.py` template — per-pipeline dispatch table

The text-generation template block (currently `SKILL.md` lines 300-350) is unchanged — it remains the canonical worked example. The single prose paragraph that follows it (currently lines 351-352, starting "For non-text-generation pipelines, substitute the library calls…") is replaced with:

> For non-text-generation pipelines, each variant still dispatches via `config.IS_APPLE_SILICON` and exposes a function named per the page template (`transcribe`, `synthesize`, `transform_audio`, `caption`, `classify`, etc.). Backend call shapes:
>
> | Pipeline | MLX branch | Transformers branch |
> |---|---|---|
> | `image-to-text`, `image-text-to-text` | `from mlx_vlm import load, generate` → `generate(model, processor, formatted_prompt, image)` | `pipeline("image-to-text", model=config.MODEL_ID)` |
> | `automatic-speech-recognition` | `from mlx_audio.stt.utils import load` → `load(id).generate(audio).text` | `pipeline("automatic-speech-recognition", model=config.MODEL_ID)` |
> | `text-to-speech` | `from mlx_audio.tts.utils import load_model`; iterate `load_model(id).generate(text=..., voice=...)` and concatenate each result's `.audio` | `pipeline("text-to-speech", model=config.MODEL_ID)` |
> | `audio-to-audio` | `mlx_audio.sts.<ModelClass>.from_pretrained(id)` + model-specific method (e.g. `.enhance(audio)`, `.separate_long(...)`). **Apple-only.** | — (`RuntimeError` on non-Apple hosts) |
>
> For `audio-to-audio`, the exact `mlx_audio.sts` class and method depend on the model (SAM-Audio → `separate_long`, MossFormer2 → `enhance`, DeepFilterNet → `enhance`). Step 2 maps the HF card's tags/name to a known `mlx_audio.sts` class; if no mapping exists, the skill reports "no supported STS backend" and emits a General Script page with a manual-wiring TODO instead of scaffolding broken inference code.

### 3. Step 6 deps table — ASR row update and two new rows

The current "Pattern-specific additional deps" table in Step 6 (currently around lines 547-556) is updated as follows. Existing unrelated rows (text generation, image/vision/diffusion, vision-language, embeddings, data processing, visualization) are kept as-is.

Replace the ASR row:

> | Automatic speech recognition | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"`, `"transformers[audio];platform_machine!='arm64' or sys_platform!='darwin'"` |

Insert two new rows:

> | Text to speech | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"`, `"transformers[audio];platform_machine!='arm64' or sys_platform!='darwin'"` |
> | Audio to audio (STS) | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"` — **no fallback** (Apple-only) |

### 4. Step 8 Report — new conditional bullet

Insert a new bullet in Step 8 immediately after the "Chosen model variant" bullet:

> **Apple-Silicon-only warning (when applicable)** — if the classified pipeline is `audio-to-audio`, state: "This scaffold requires Apple Silicon at runtime. On non-Apple hosts, `uv sync` will not install `mlx-audio` and the app will error at model load."

### 5. `references/pipeline-tag-patterns.md` — new `audio-to-audio` entry

The existing ASR and TTS entries are unchanged (their page bodies are library-agnostic — they call `<app_name>.inference.transcribe` and `synthesize` respectively).

Insert a new entry between `text-to-speech` and `image-classification / object-detection`:

> ## Audio transform (speech-to-speech, separation, enhancement)
>
> `pipeline_tag`: `audio-to-audio`
>
> **Apple Silicon only.** The generated `inference.py` raises a clear `RuntimeError` on non-Apple hosts.
>
> ```python
> import streamlit as st
> from <app_name>.inference import transform_audio
>
> st.title("Transform Audio")
> audio = st.audio_input("Record") or st.file_uploader(
>     "Upload", type=["wav", "mp3", "m4a", "flac"]
> )
> if audio and st.button("Transform"):
>     result = transform_audio(audio)
>     # Single-output models (enhancement / denoising) return bytes.
>     # Multi-output models (source separation) return dict[str, bytes].
>     if isinstance(result, dict):
>         for label, audio_bytes in result.items():
>             st.subheader(label)
>             st.audio(audio_bytes, format="audio/wav")
>     else:
>         st.audio(result, format="audio/wav")
> ```
>
> The shape of `transform_audio`'s return is decided at scaffold time based on the HF model card: `separate_long`-style models return a dict of labeled outputs; `enhance`-style models return a single `bytes`. If the card doesn't map to a known `mlx_audio.sts` class, the skill emits a General Script page with a manual-wiring TODO instead of this template.

## Open questions

None. All scope questions resolved in brainstorming:

- Scope of audio support → Option C (expand catalog with `audio-to-audio`).
- STS fallback on non-Apple hosts → Option A (hard Apple-Silicon requirement, `RuntimeError` on load).
- Restructure strategy → Approach 2 (inline swap plus MLX backend index table in `SKILL.md`; no new reference file).

## Files affected

- `streamlit-app-builder/SKILL.md` — cross-cutting principle 2 rewrite (Design §1); `inference.py` post-template paragraph replaced with dispatch table (§2); Step 6 deps table row changes (§3); Step 8 new conditional bullet (§4).
- `streamlit-app-builder/references/pipeline-tag-patterns.md` — new `audio-to-audio` entry (§5).

No changes to `streamlit-app-builder/references/streamlit-docs-index.md` or `streamlit-app-builder/references/license-flags.md`.
