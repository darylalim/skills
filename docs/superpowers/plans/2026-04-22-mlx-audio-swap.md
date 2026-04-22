# MLX Audio Backend Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget `streamlit-app-builder` to generate MLX apps via `mlx-lm` / `mlx-vlm` / `mlx-audio` (replacing `mlx-whisper`), and extend coverage to TTS and audio-to-audio pipelines.

**Architecture:** Edit the skill's two Markdown source files in place. Inline the new MLX backend policy as a compact table in the cross-cutting principle. Split `inference.py` dispatch guidance into a per-pipeline table. Update the deps table and report step. Add an `audio-to-audio` entry to the pipeline catalog. No code, no tests — this is a skill-definition change; verification is grep-based.

**Tech Stack:** Markdown, `grep`, `git`. No runtime test harness (skill files are LLM-interpreted prompts, not executable code).

---

## File Structure

- `streamlit-app-builder/SKILL.md` — four edits (Tasks 1-4): cross-cutting principle rewrite, `inference.py` dispatch-table swap, Step 6 deps-table rows, Step 8 report bullet insertion.
- `streamlit-app-builder/references/pipeline-tag-patterns.md` — one edit (Task 5): insert new `audio-to-audio` entry between TTS and image-classification.
- `docs/superpowers/specs/2026-04-22-mlx-audio-swap-design.md` — read-only reference.

No files are created or deleted. No new directories.

---

## Task 1: Rewrite MLX cross-cutting principle with backend index table

Replaces the prose description of MLX backends with a table listing each `pipeline_tag` → MLX module → PyPI package → Apple-only flag → transformers fallback. Captures `mlx-audio`'s three submodules (STT, TTS, STS) and the Apple-Silicon-only STS rule.

**Files:**
- Modify: `streamlit-app-builder/SKILL.md:41-52`

- [ ] **Step 1: Apply the edit**

Use the Edit tool on `streamlit-app-builder/SKILL.md` with:

`old_string`:
````
### 2. Prefer MLX on Apple Silicon

When the source artifact references a model with an MLX-converted equivalent at `huggingface.co/mlx-community/...`, the generated app uses the MLX backend (`mlx-lm` for text, `mlx-vlm` for vision-language, `mlx-whisper` for ASR) when running on `arm64-darwin`, and falls back to `transformers` / `diffusers` / `huggingface_hub` elsewhere.

The MLX lookup is independent of where the skill itself runs. A Linux developer scaffolding from an HF model card URL still produces an app with MLX support wired in — runtime dispatch activates MLX only when a user later runs the app on a Mac.

MLX support is encoded in the generated app as follows:
- `pyproject.toml` declares MLX and transformers with environment markers so `uv sync` installs the right backend per host.
- `src/<app_name>/inference.py` reads `config.IS_APPLE_SILICON` and dispatches to the appropriate backend at runtime.

**MLX model resolution:** query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` and pick the highest-download-count variant. The chosen variant is noted in `inference.py` with override instructions.
````

`new_string`:
````
### 2. Prefer MLX on Apple Silicon

When the source artifact references a model with an MLX-converted equivalent on HuggingFace, the generated app uses an MLX backend on `arm64-darwin` and falls back to `transformers` / `diffusers` / `huggingface_hub` elsewhere.

**MLX backend index:**

| `pipeline_tag` | MLX module | PyPI | Apple-only? | Transformers fallback |
|---|---|---|---|---|
| `text-generation`, `conversational` | `mlx_lm` | `mlx-lm` | no | `transformers` |
| `image-to-text`, `image-text-to-text` | `mlx_vlm` | `mlx-vlm` | no | `transformers` |
| `automatic-speech-recognition` | `mlx_audio.stt` | `mlx-audio` | no | `transformers[audio]` (via `pipeline("automatic-speech-recognition")`) |
| `text-to-speech` | `mlx_audio.tts` | `mlx-audio` | no | `transformers[audio]` (SpeechT5 / Bark / Parler-TTS) |
| `audio-to-audio` | `mlx_audio.sts` | `mlx-audio` | **yes** | — (`RuntimeError` at model load off Apple Silicon) |

Apple-only rows install no transformers fallback; `inference.py` raises a clear `RuntimeError` at model load on non-Apple hosts, and the generated `README.md` notes the platform requirement.

The MLX lookup is independent of where the skill itself runs. A Linux developer scaffolding from an HF model card still produces an app with MLX support wired in — runtime dispatch activates MLX only when a user later runs the app on a Mac. (Exception: `audio-to-audio` apps run on Apple Silicon only, by design.)

MLX support is encoded in the generated app as follows:
- `pyproject.toml` declares MLX and transformers with environment markers so `uv sync` installs the right backend per host. Audio-to-audio apps declare `mlx-audio` with an Apple-only marker and omit the fallback dep.
- `src/<app_name>/inference.py` reads `config.IS_APPLE_SILICON` and dispatches at runtime.

**MLX model resolution:** query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` and pick the highest-download-count variant. If the base name has no `mlx-community` match, note "no MLX equivalent found" in the final report and generate the app with `transformers` only. Audio-to-audio inputs without a match fail at scaffold time with a clear error — there is no fallback to generate toward.
````

- [ ] **Step 2: Verify**

Run:
```bash
grep -q "MLX backend index" streamlit-app-builder/SKILL.md && echo OK || echo MISSING
grep -c "mlx_audio.sts" streamlit-app-builder/SKILL.md
grep -c "mlx-whisper" streamlit-app-builder/SKILL.md
```

Expected:
- Line 1: `OK`
- Line 2: `1` (the new backend index row)
- Line 3: `2` (the other two `mlx-whisper` references still remain in Tasks 2 & 3)

- [ ] **Step 3: Commit**

```bash
git -C /Users/daryl-lim/Documents/GitHub/skills add streamlit-app-builder/SKILL.md
git -C /Users/daryl-lim/Documents/GitHub/skills commit -m "$(cat <<'EOF'
Rewrite MLX cross-cutting principle with backend index table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Replace `inference.py` non-text-gen paragraph with dispatch table

The single paragraph after the text-generation `inference.py` code block is replaced with a per-pipeline dispatch table showing both MLX and transformers branches. Drops `mlx_whisper.transcribe` in favor of `mlx_audio.stt.utils.load`, adds TTS (`mlx_audio.tts.utils.load_model`) and STS (`mlx_audio.sts.<ModelClass>`) rows.

**Files:**
- Modify: `streamlit-app-builder/SKILL.md:352`

- [ ] **Step 1: Apply the edit**

Use the Edit tool on `streamlit-app-builder/SKILL.md` with:

`old_string`:
````
For non-text-generation pipelines, substitute the library calls: `mlx_vlm.load`/`generate` for vision-language, `mlx_whisper.transcribe` for ASR, `transformers.pipeline(<task>, ...)` for the transformers branch. Each variant still dispatches via `config.IS_APPLE_SILICON` and exposes a function named per the pipeline (`transcribe`, `caption`, `classify`, etc.) matching the page template.
````

`new_string`:
````
For non-text-generation pipelines, each variant still dispatches via `config.IS_APPLE_SILICON` and exposes a function named per the page template (`transcribe`, `synthesize`, `transform_audio`, `caption`, `classify`, etc.). Backend call shapes:

| Pipeline | MLX branch | Transformers branch |
|---|---|---|
| `image-to-text`, `image-text-to-text` | `from mlx_vlm import load, generate` → `generate(model, processor, formatted_prompt, image)` | `pipeline("image-to-text", model=config.MODEL_ID)` |
| `automatic-speech-recognition` | `from mlx_audio.stt.utils import load` → `load(id).generate(audio).text` | `pipeline("automatic-speech-recognition", model=config.MODEL_ID)` |
| `text-to-speech` | `from mlx_audio.tts.utils import load_model`; iterate `load_model(id).generate(text=..., voice=...)` and concatenate each result's `.audio` | `pipeline("text-to-speech", model=config.MODEL_ID)` |
| `audio-to-audio` | `mlx_audio.sts.<ModelClass>.from_pretrained(id)` + model-specific method (e.g. `.enhance(audio)`, `.separate_long(...)`). **Apple-only.** | — (`RuntimeError` on non-Apple hosts) |

For `audio-to-audio`, the exact `mlx_audio.sts` class and method depend on the model (SAM-Audio → `separate_long`, MossFormer2 → `enhance`, DeepFilterNet → `enhance`). Step 2 maps the HF card's tags/name to a known `mlx_audio.sts` class; if no mapping exists, the skill reports "no supported STS backend" and emits a General Script page with a manual-wiring TODO instead of scaffolding broken inference code.
````

- [ ] **Step 2: Verify**

Run:
```bash
grep -q "mlx_audio.stt.utils" streamlit-app-builder/SKILL.md && echo OK || echo MISSING
grep -q "mlx_audio.tts.utils" streamlit-app-builder/SKILL.md && echo OK || echo MISSING
grep -q "mlx_audio.sts" streamlit-app-builder/SKILL.md && echo OK || echo MISSING
grep -c "mlx-whisper" streamlit-app-builder/SKILL.md
grep -c "mlx_whisper.transcribe" streamlit-app-builder/SKILL.md
```

Expected:
- Lines 1-3: three `OK` lines
- Line 4: `1` (only the Step 6 deps-table reference remains, to be fixed in Task 3)
- Line 5: `0` (the `.transcribe` call reference is gone)

- [ ] **Step 3: Commit**

```bash
git -C /Users/daryl-lim/Documents/GitHub/skills add streamlit-app-builder/SKILL.md
git -C /Users/daryl-lim/Documents/GitHub/skills commit -m "$(cat <<'EOF'
Replace inference.py non-text-gen paragraph with dispatch table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update Step 6 deps table — swap ASR row, insert TTS and STS rows

The ASR row swaps `mlx-whisper` for `mlx-audio` and consolidates the fallback on `transformers[audio]`. Two new rows follow: TTS (same Apple/non-Apple split) and audio-to-audio (Apple-only, no fallback).

**Files:**
- Modify: `streamlit-app-builder/SKILL.md:551`

- [ ] **Step 1: Apply the edit**

Use the Edit tool on `streamlit-app-builder/SKILL.md` with:

`old_string`:
````
| ASR                                  | `"mlx-whisper;platform_machine=='arm64' and sys_platform=='darwin'"`, `"openai-whisper;platform_machine!='arm64' or sys_platform!='darwin'"` — or substitute `transformers[audio]` |
| Vision-language                      | `"mlx-vlm;platform_machine=='arm64' and sys_platform=='darwin'"` |
````

`new_string`:
````
| Automatic speech recognition         | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"`, `"transformers[audio];platform_machine!='arm64' or sys_platform!='darwin'"` |
| Text to speech                       | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"`, `"transformers[audio];platform_machine!='arm64' or sys_platform!='darwin'"` |
| Audio to audio (STS)                 | `"mlx-audio;platform_machine=='arm64' and sys_platform=='darwin'"` — **no fallback** (Apple-only) |
| Vision-language                      | `"mlx-vlm;platform_machine=='arm64' and sys_platform=='darwin'"` |
````

- [ ] **Step 2: Verify**

Run:
```bash
grep -c "mlx-whisper" streamlit-app-builder/SKILL.md
grep -c '"mlx-audio;platform_machine' streamlit-app-builder/SKILL.md
grep -c 'transformers\[audio\];platform_machine' streamlit-app-builder/SKILL.md
grep -q "Audio to audio (STS)" streamlit-app-builder/SKILL.md && echo OK || echo MISSING
```

Expected:
- Line 1: `0` (all `mlx-whisper` references gone from `SKILL.md`)
- Line 2: `3` (ASR, TTS, STS rows all cite `mlx-audio` with the Apple marker)
- Line 3: `2` (ASR and TTS fallback rows; STS has no fallback)
- Line 4: `OK`

- [ ] **Step 3: Commit**

```bash
git -C /Users/daryl-lim/Documents/GitHub/skills add streamlit-app-builder/SKILL.md
git -C /Users/daryl-lim/Documents/GitHub/skills commit -m "$(cat <<'EOF'
Update deps table for mlx-audio with audio-to-audio row

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Insert Apple-Silicon-only warning bullet in Step 8 report

Adds a new numbered bullet after "Chosen model variant" in Step 8's Surface list. Because Step 8 uses explicit numbering, bullets 3-6 renumber to 4-7.

**Files:**
- Modify: `streamlit-app-builder/SKILL.md:577-588`

- [ ] **Step 1: Apply the edit**

Use the Edit tool on `streamlit-app-builder/SKILL.md` with:

`old_string`:
````
2. **Chosen model variant** — if MLX resolution returned a match, show `mlx-community/<variant>` alongside the original `<org>/<model>`; otherwise note "no MLX equivalent found, app uses transformers on all platforms."
3. **License + commercial-use flag** — from `references/license-flags.md`, if the model's license matches a flagged entry. Quote the flag text inline.
4. **Gated-model setup** — when the source card had `gated: true`, show the `huggingface-cli login` command and the alternative `HF_TOKEN` path.
5. **Exact local-run command:**

   ```bash
   uv sync
   cp .env.example .env
   streamlit run streamlit_app.py
   ```

6. **Non-goals reminder** — a short list of things the scaffold does NOT include (auth, Docker, CI, DB, observability), explicitly marked as the team's responsibility.
````

`new_string`:
````
2. **Chosen model variant** — if MLX resolution returned a match, show `mlx-community/<variant>` alongside the original `<org>/<model>`; otherwise note "no MLX equivalent found, app uses transformers on all platforms."
3. **Apple-Silicon-only warning (when applicable)** — if the classified pipeline is `audio-to-audio`, state: "This scaffold requires Apple Silicon at runtime. On non-Apple hosts, `uv sync` will not install `mlx-audio` and the app will error at model load."
4. **License + commercial-use flag** — from `references/license-flags.md`, if the model's license matches a flagged entry. Quote the flag text inline.
5. **Gated-model setup** — when the source card had `gated: true`, show the `huggingface-cli login` command and the alternative `HF_TOKEN` path.
6. **Exact local-run command:**

   ```bash
   uv sync
   cp .env.example .env
   streamlit run streamlit_app.py
   ```

7. **Non-goals reminder** — a short list of things the scaffold does NOT include (auth, Docker, CI, DB, observability), explicitly marked as the team's responsibility.
````

- [ ] **Step 2: Verify**

Run:
```bash
grep -q "Apple-Silicon-only warning (when applicable)" streamlit-app-builder/SKILL.md && echo OK || echo MISSING
grep -q "7. \*\*Non-goals reminder\*\*" streamlit-app-builder/SKILL.md && echo OK || echo MISSING
grep -c "^[0-9]\. \*\*" streamlit-app-builder/SKILL.md
```

Expected:
- Line 1: `OK`
- Line 2: `OK` (bullet 7 now ends the list)
- Line 3: `7` (seven numbered bullets in Step 8)

- [ ] **Step 3: Commit**

```bash
git -C /Users/daryl-lim/Documents/GitHub/skills add streamlit-app-builder/SKILL.md
git -C /Users/daryl-lim/Documents/GitHub/skills commit -m "$(cat <<'EOF'
Add Apple-Silicon-only warning to Step 8 report

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add audio-to-audio pipeline entry to catalog

Inserts a new entry between the existing TTS section and the Image classification / object detection section. The page body handles both single-output (enhancement) and multi-output (separation) STS models via an `isinstance(result, dict)` split.

**Files:**
- Modify: `streamlit-app-builder/references/pipeline-tag-patterns.md` (insert after the TTS code block, before `## Image classification / object detection`)

- [ ] **Step 1: Apply the edit**

Use the Edit tool on `streamlit-app-builder/references/pipeline-tag-patterns.md` with:

`old_string`:
````
if st.button("Speak") and text:
    audio_bytes = synthesize(text)
    st.audio(audio_bytes, format="audio/wav")
```

## Image classification / object detection
````

`new_string`:
````
if st.button("Speak") and text:
    audio_bytes = synthesize(text)
    st.audio(audio_bytes, format="audio/wav")
```

## Audio transform (speech-to-speech, separation, enhancement)

`pipeline_tag`: `audio-to-audio`

**Apple Silicon only.** The generated `inference.py` raises a clear `RuntimeError` on non-Apple hosts.

```python
import streamlit as st
from <app_name>.inference import transform_audio

st.title("Transform Audio")
audio = st.audio_input("Record") or st.file_uploader(
    "Upload", type=["wav", "mp3", "m4a", "flac"]
)
if audio and st.button("Transform"):
    result = transform_audio(audio)
    # Single-output models (enhancement / denoising) return bytes.
    # Multi-output models (source separation) return dict[str, bytes].
    if isinstance(result, dict):
        for label, audio_bytes in result.items():
            st.subheader(label)
            st.audio(audio_bytes, format="audio/wav")
    else:
        st.audio(result, format="audio/wav")
```

The shape of `transform_audio`'s return is decided at scaffold time based on the HF model card: `separate_long`-style models return a dict of labeled outputs; `enhance`-style models return a single `bytes`. If the card doesn't map to a known `mlx_audio.sts` class, the skill emits a General Script page with a manual-wiring TODO instead of this template.

## Image classification / object detection
````

- [ ] **Step 2: Verify**

Run:
```bash
grep -q "pipeline_tag\`: \`audio-to-audio\`" streamlit-app-builder/references/pipeline-tag-patterns.md && echo OK || echo MISSING
grep -q "from <app_name>.inference import transform_audio" streamlit-app-builder/references/pipeline-tag-patterns.md && echo OK || echo MISSING
grep -q "Apple Silicon only" streamlit-app-builder/references/pipeline-tag-patterns.md && echo OK || echo MISSING
```

Expected: three `OK` lines.

- [ ] **Step 3: Commit**

```bash
git -C /Users/daryl-lim/Documents/GitHub/skills add streamlit-app-builder/references/pipeline-tag-patterns.md
git -C /Users/daryl-lim/Documents/GitHub/skills commit -m "$(cat <<'EOF'
Add audio-to-audio pattern entry to pipeline-tag catalog

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Final repository-wide verification

No file edits. Runs grep across the whole skill directory to confirm the swap is clean and the new entries are present.

**Files:**
- Read-only: `streamlit-app-builder/**`

- [ ] **Step 1: Scan for residual `mlx-whisper` / `mlx_whisper`**

Run:
```bash
grep -rn "mlx-whisper\|mlx_whisper" streamlit-app-builder/
```

Expected: no output (no lines match). If anything remains, open the matching file and delete it before continuing.

- [ ] **Step 2: Confirm `mlx-audio` coverage**

Run:
```bash
grep -rn "mlx-audio\|mlx_audio" streamlit-app-builder/ | wc -l
```

Expected: at least `8` matches — roughly three in the SKILL.md backend index (STT, TTS, STS rows), three in the `inference.py` dispatch table, three in the Step 6 deps table, plus the new patterns entry. Exact count may vary ±2 based on how grep counts multi-hit lines.

- [ ] **Step 3: Confirm audio-to-audio pattern entry is present**

Run:
```bash
grep -q "Audio transform (speech-to-speech" streamlit-app-builder/references/pipeline-tag-patterns.md && echo OK
grep -q "audio-to-audio" streamlit-app-builder/SKILL.md && echo OK
```

Expected: two `OK` lines.

- [ ] **Step 4: Confirm no unintended changes elsewhere**

Run:
```bash
git -C /Users/daryl-lim/Documents/GitHub/skills log --oneline -10
git -C /Users/daryl-lim/Documents/GitHub/skills status --short
```

Expected:
- Five new commits at the top of the log (Tasks 1-5), each with the corresponding message.
- `status --short` output is empty (working tree clean).

- [ ] **Step 5: No commit needed**

This task only verifies; nothing to commit.

---

## Summary

Five edits across two Markdown files, verified via grep. No runtime tests are in scope — skill files are LLM-interpreted prompts, not executable code. A real end-to-end sanity check requires invoking the skill against a sample HF audio model card (e.g. `mlx-community/Kokoro-82M-bf16`) and confirming the generated app uses the new dispatch paths; that is out of scope for this plan and should be covered in a follow-up QA pass.
