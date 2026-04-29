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
- **File-list ↔ outputs parity** — the file paths in `SKILL.md`'s Pipeline section match the Outputs section.
- **T1 cache decorator preservation** — T1 mentions both `@st.cache_resource` and `@lru_cache(maxsize=1)`.
- **T2 token kwarg mapping** — T2 documents `max_new_tokens` → `max_tokens`.
- **T3 platform guard literal** — T3 contains `platform.machine() == "arm64"` and `platform.system() == "Darwin"`.
- **T4 mock target** — T4 mocks `mlx_lm.load` rather than `from_pretrained`.
- **T5 dep commands** — T5 emits `uv add mlx-lm` for Streamlit and a `requirements.txt` append for Gradio.
- **Variant default precedence** — `variant-resolution.md` contains the literal `bf16 > fp16 > 8bit > 6bit > 4bit`.

## How to run

```bash
cd mlx-app-converter/tests
uv run pytest -v
```

This directory has its own `pyproject.toml` and `.venv`, independent of any user-supplied app.

## Skipping a block

Add `<!-- skip-validate -->` on the line immediately above the ` ```python ` fence. Use sparingly. None should be needed for `rewrite-templates.md`; the structural test enforces this.
