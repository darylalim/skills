---
name: gradio-app-builder
description: >
  Use when the user wants to scaffold a Hugging Face Space (Gradio SDK) for a
  HuggingFace model. Triggers: "build a Gradio Space for an HF model",
  "generate a Gradio demo for huggingface.co/<org>/<model>", "scaffold a
  Hugging Face Space for an HF inference model", "deploy this HF model as a
  Space", "turn an HF model card into a Gradio app", any URL of the shape
  `huggingface.co/<org>/<model>`. Skip when the input is a Python script, a
  Jupyter notebook, a GitHub URL, or any non-HF artifact.
---

# Gradio App Builder (HF model card → Hugging Face Space)

Generate a Hugging Face Space (Gradio SDK) from a HuggingFace model card URL. Output is a flat five-file project (`app.py`, `requirements.txt`, `README.md`, `.env.example`, `test_app.py`) ready to push to a Space repo.

## When to use

**Use:** the user passes a `huggingface.co/<org>/<model>` URL and wants a Space-deployable Gradio demo.

**Don't use** (reject explicitly):
- Python script (`.py` file path or `github.com/.../*.py` blob URL) — not supported
- Jupyter notebook (`.ipynb`) — not supported
- GitHub repo root URL — not supported
- `huggingface.co/datasets/...`, `huggingface.co/spaces/...` — not supported

When rejecting, emit a one-line message telling the user the input shape isn't supported.

## Non-goals (silent — no need to surface in the report)

- Production-grade structure (multipage routing, fail-fast config, src layout) — this is a prototype generator
- Authentication / authorization beyond the gated-model `HF_TOKEN` check
- Custom CSS / theming beyond Gradio defaults
- Auto-pushing to Spaces (the report tells the user how to push manually)
- Building Spaces with Docker SDK or static SDK — only the standard Gradio SDK is in scope
- License or commercial-use surfacing
- Sibling-model suggestions

## Step 1: Identify and load the input

The input MUST be an HF model card URL of the shape `https://huggingface.co/<org>/<model>` (with or without trailing slash; query strings and fragments are stripped before classification).

**Reject early** with a clear message if the input is any other URL shape:
- `github.com/...` (any path) → *"Pass an HF model card URL (`huggingface.co/<org>/<model>`). For Python scripts or notebooks on GitHub, use a general-purpose Gradio prompt without this skill."*
- `*.ipynb` URL → same redirect message as above.
- `huggingface.co/datasets/...` or `huggingface.co/spaces/...` → *"This skill takes model card URLs only. Datasets and existing Spaces aren't supported as input."*
- File path or any other shape → *"Pass an HF model card URL (`huggingface.co/<org>/<model>`)."*

**Load the model card:**

1. Strip query string and fragment from the URL. Resolve `<org>/<model>` from the path.
2. Fetch metadata: `https://huggingface.co/api/models/<org>/<model>` → JSON. Capture `pipeline_tag`, `library_name`, `gated`, plus `sha` (latest commit — needed for the README fallback below).
3. Fetch the README: `https://huggingface.co/<org>/<model>/raw/main/README.md`. On HTTP 404 (repos whose default branch is not `main`), retry with `https://huggingface.co/<org>/<model>/raw/<sha>/README.md` using the `sha` from step 2. Use the YAML frontmatter as a fallback for any metadata fields the API didn't return.
4. On HTTP 404 from the metadata API: fail with *"Model `<org>/<model>` not found on HuggingFace, or it's a private repo without access. Verify the URL."*
5. On HTTP 401/403: *"Access denied to `<org>/<model>`. If gated, run `huggingface-cli login` first; if private, ensure your `HF_TOKEN` has access."*

## Step 2: Build the IR

<!-- skip-validate -->
```python
{
    "model_id": "<org>/<model>",
    "pipeline_tag": "<tag>" or None,                      # from API metadata; None when missing
    "library_name": "transformers" | "diffusers" | "sentence-transformers" | None,
    "is_gated": bool,                                     # api.gated == True
    "deps": ["gradio", "python-dotenv", ...],             # see Step 4
}
```

`library_name` resolution: prefer the API's `library_name` field. If the field is missing, default to `transformers`. If the API's `library_name == "sentence-transformers"` OR the README explicitly recommends `sentence-transformers` (look for `from sentence_transformers import` in the README's code snippets), set `library_name = "sentence-transformers"` regardless of the API value.

`pipeline_tag` normalization:
- When `library_name` resolves to `sentence-transformers`, set `pipeline_tag = "feature-extraction"` regardless of the API value. HF embedding models commonly report `"sentence-similarity"` or `"feature-extraction"` interchangeably; both route to T5 (`embed`) in Step 4.
- When the API returns `pipeline_tag: null` (older models or community uploads without a tag) and the README's YAML frontmatter has no `pipeline_tag` field either, leave `pipeline_tag = None` in the IR. Step 3 treats `None` and any unrecognized value the same way: route to T1 + the **Fallback: General Script** UI body.

## Step 3: Classify the UI pattern

Look up `pipeline_tag` in `references/pipeline-tag-patterns.md`. The matching section's UI body is what gets pasted into `app.py` after the inference function.

**Rejected pipeline tag:**
- `pipeline_tag == "audio-to-audio"` → fail at Step 3 with: *"audio-to-audio has no clean transformers pipeline. This skill can't scaffold a working prototype for audio-to-audio models. For source separation or speech enhancement, use the model's reference implementation directly."*

**Unrecognized or `None` pipeline tag** → fall through to the **Fallback: General Script** section in `references/pipeline-tag-patterns.md`.

## Step 4: Scaffold the five files

### Routing table: `library_name` × `pipeline_tag` → scaffolding template

| `library_name` | `pipeline_tag` | Template (in `references/scaffolding-templates.md`) |
|---|---|---|
| `transformers` | `text-generation`, `conversational` | T2 (`chat`) |
| `transformers` | `text-classification`, `zero-shot-classification`, `token-classification`, `question-answering`, `summarization`, `translation`, `automatic-speech-recognition`, `text-to-speech`, `image-classification`, `object-detection`, `image-to-text`, `image-text-to-text` | T1 (`run_inference`) |
| `transformers`, `sentence-transformers` | `feature-extraction` | T5 (`embed`) |
| `diffusers` | `text-to-image` | T3 (`generate_image`) |
| `diffusers` | `image-to-image` | T4 (`edit_image`) |
| any | unrecognized or None | T1 with the **Fallback: General Script** UI body |

### File 1: `app.py`

Assemble in this order:

1. **Module docstring** — one line describing the model and task.
2. **Imports** — `os`, `gradio as gr`, `dotenv.load_dotenv`, `functools.lru_cache`, plus library-specific imports from the chosen scaffolding template.
3. **`load_dotenv()`** — call once at module level so `.env` values populate `os.environ`.
4. **`MODEL_ID` constant** — hard-coded from Step 1's input URL: `MODEL_ID = "<org>/<model>"`.
5. **Gated-model gate** — see the snippet below; emit only when `is_gated: true`.
6. **`load_model()`** — copy from the chosen scaffolding template, decorated with `@lru_cache(maxsize=1)`.
7. **Inference function** — copy from the chosen scaffolding template (`run_inference` / `chat` / `generate_image` / `edit_image` / `embed`). When using T1, also replace its `PIPELINE_TASK = "<pipeline_tag>"` placeholder with the resolved tag (e.g., `PIPELINE_TASK = "automatic-speech-recognition"`).
8. **UI body** — paste at module scope after the inference function.
9. **`if __name__ == "__main__": demo.launch()`** — at the bottom of the file.

#### Gated-model gate snippet

Insert this after `load_dotenv()` (between assembly step 4 and step 6) when the model card has `gated: true`:

<!-- skip-validate -->
```python
if not os.getenv("HF_TOKEN"):
    with gr.Blocks() as demo:
        gr.Markdown(
            f"## Configuration required\n\n"
            f"This Space wraps the gated model `{MODEL_ID}`. "
            f"Add `HF_TOKEN` under **Settings → Variables and secrets**, "
            f"or set it in `.env` locally."
        )
    if __name__ == "__main__":
        demo.launch()
    raise SystemExit(0)
```

### File 2: `requirements.txt`

Flat dependency list. Pin `gradio` to the version that the Space will use (resolved after `pip install gradio`):

```
gradio==<pinned-version>
python-dotenv
```

Plus library-specific deps per the routing table:

- `transformers` (T1, T2): add `transformers`, `torch`.
- ASR (T1, `automatic-speech-recognition`): add `transformers[audio]` instead of plain `transformers`.
- TTS (T1, `text-to-speech`): add `transformers[audio]` and `scipy`.
- `diffusers` (T3, T4): add `diffusers`, `transformers`, `torch`, `accelerate`, `pillow`.
- `sentence-transformers` (T5): add `sentence-transformers` (replaces `transformers`).

Emit only the rows that apply.

### File 3: `README.md` (Spaces config + brief description)

YAML frontmatter at the top — `sdk_version` matches the version pinned in `requirements.txt`:

```
---
title: <org>/<model>
emoji: 🤗
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: <pinned-version>
pinned: false
---
```

Body (below the frontmatter): one-paragraph description of the model + task, plus a "Run locally" snippet:

````markdown
# <org>/<model>

A Hugging Face Space wrapping `<org>/<model>` for `<pipeline_tag>` inference.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env       # add HF_TOKEN if the model is gated
python app.py
```
````

`app_file` is omitted from the frontmatter — Spaces defaults to `app.py`.

### File 4: `.env.example`

Always emit this file. Template:

```
# Required for gated models (e.g. meta-llama/*, mistralai/*).
# Get a token at https://huggingface.co/settings/tokens.
# On Spaces, set this under Settings → Variables and secrets instead.
HF_TOKEN=
```

### File 5: `test_app.py`

Copy template T6 from `references/scaffolding-templates.md` and adapt the body of `test_inference_function_returns_expected_type` per the in-use scaffolding template — the comment hints inside T6 spell out each adaptation explicitly.

## Step 5: Code-quality gate

Run, in order. All four must pass before reporting the scaffold as complete.

```bash
pip install -r requirements.txt
pip install ruff ty pytest
ruff check --fix app.py test_app.py
ruff format app.py test_app.py
ty check app.py
pytest test_app.py -v
```

Fix failures by adjusting the generated code. Do not weaken the test to make it pass.

`ty check` may flag Gradio calls whose type stubs are incomplete. When the call is correct as written, add a narrowly-scoped `# type: ignore[<code>]` on that line using the exact code ty reports — don't remove the call.

## Step 6: Report to the user

Emit exactly two items, plus a third when the model is gated.

**Always:**

1. **Files created.** List the five paths.
2. **Run command.**

   ```bash
   pip install -r requirements.txt
   cp .env.example .env       # then add your HF_TOKEN if needed
   python app.py              # local

   # to deploy to Spaces:
   git init && git remote add space https://huggingface.co/spaces/<user>/<name>
   # .env.example and test_app.py stay local — they're not part of the Space
   git add app.py requirements.txt README.md && git commit -m "init"
   git push space main
   ```

**Conditional (when `is_gated: true`):**

3. **Gated-model setup.** *"This model is gated. On Spaces, add `HF_TOKEN` under Settings → Variables and secrets. Locally, set `HF_TOKEN=` in `.env` or run `huggingface-cli login`. Without a token, the app shows a configuration-required screen at startup."*

## Output checklist

- [ ] Five files created: `app.py`, `requirements.txt`, `README.md`, `.env.example`, `test_app.py`
- [ ] `app.py` has `@lru_cache(maxsize=1)` on `load_model`
- [ ] `MODEL_ID` is hard-coded as a module-level constant (not env-driven)
- [ ] If `is_gated`, the gated-model gate runs before any model load
- [ ] `requirements.txt` pins `gradio==<version>` and lists library-specific runtime deps
- [ ] `README.md` frontmatter declares `sdk: gradio` and `sdk_version` matching `requirements.txt`
- [ ] `.env.example` always present, contains `HF_TOKEN=` line and Spaces-Secrets pointer
- [ ] `ruff check --fix`, `ruff format`, `ty check`, and `pytest` all pass clean
- [ ] Step 6 report has 2 always-present items (or 3 when gated)

## References

- `references/pipeline-tag-patterns.md` — UI body templates indexed by `pipeline_tag`
- `references/scaffolding-templates.md` — `load_model` + inference-function templates indexed by `library_name` × `pipeline_tag`
