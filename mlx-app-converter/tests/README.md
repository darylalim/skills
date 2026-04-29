# MLX App Converter Tests

Static validator and structural-consistency tests for the skill's Markdown files. Runs locally; not invoked at skill-runtime.

## What it checks

For every fenced ` ```python ` block in `SKILL.md` and `references/rewrite-templates.md`:

1. **Syntax** — `ast.parse` after substituting documented placeholders.
2. **Lint** — `ruff check --select E,F,I`.

Plus the following structural-consistency and spec-alignment checks:

- **Template-name references** — every `T<n>` mentioned in `SKILL.md` has a `## Template T<n>:` heading in `rewrite-templates.md`.
- **Rejection-message presence** — every rejection message in the spec appears verbatim in `SKILL.md`.
- **Skip-validate marker count** — zero `<!-- skip-validate -->` markers in `rewrite-templates.md`.
- **File-list ↔ outputs parity** — concrete files mentioned in `SKILL.md`'s Outputs section are also referenced in its Workflow section.
- **SKILL.md Step 3 model-ID dedup** — Step 3 documents that detected model IDs are deduplicated by string value before matrix construction.
- **SKILL.md Step 5 sampler routing** — Step 5's T2 description names `make_sampler` / `make_logits_processors` (regression guard against direct-rename phrasing for sampling kwargs).
- **T1 cache decorator preservation** — T1 mentions both `@st.cache_resource` and `@lru_cache(maxsize=1)`.
- **T2 token kwarg mapping** — T2 documents `max_new_tokens` → `max_tokens` and shows the literal `mlx_lm.generate(model, tokenizer, prompt, max_tokens=` call shape.
- **T2 sampler-helper construction** — T2 names `make_sampler`, `make_logits_processors`, and `from mlx_lm.sample_utils import` for sampling-kwarg routing.
- **T3 platform guard literal** — T3 contains `platform.machine() == "arm64"` and `platform.system() == "Darwin"`.
- **T3 arm64-only test side effect** — T3 documents that the runtime guard makes rewritten test files arm64-only at import time.
- **T4 mock target** — T4 mocks `mlx_lm.load` rather than `from_pretrained`.
- **T5 dep commands** — T5 emits `uv add mlx-lm` for Streamlit and a `requirements.txt` append for Gradio.
- **T5 removal hint** — T5 includes the `transformers and torch may now be unused` hint phrase.
- **Variant default precedence** — `variant-resolution.md` contains the literal `bf16 > fp16 > 8bit > 6bit > 4bit`.

## How to run

```bash
cd mlx-app-converter/tests
uv run pytest -v
```

This directory has its own `pyproject.toml` and `.venv`, independent of any user-supplied app.

## Skipping a block

Add `<!-- skip-validate -->` on the line immediately above the ` ```python ` fence. Use sparingly. None should be needed for `rewrite-templates.md`; the structural test enforces this.
