# Streamlit App Builder Tests

Static validator and structural-consistency tests for the skill's Markdown files. Runs locally; not invoked at skill-runtime.

## What it checks

For every fenced ` ```python ` block in `SKILL.md`, `references/pipeline-tag-patterns.md`, and `references/scaffolding-templates.md`:

1. **Syntax** — `ast.parse` after substituting documented placeholders (`<org>/<model>`).
2. **Lint** — `ruff check --select E,F,I` (matches the ruleset that scaffolded apps use).

Plus the following structural-consistency checks across the same files:

- **Routing-table coverage** — every `pipeline_tag` in `pipeline-tag-patterns.md` is handled by `SKILL.md`'s routing table (or rejected at Step 3) and vice versa.
- **Template-name references** — every `T<n>` mentioned in `SKILL.md` and `pipeline-tag-patterns.md` has a `## Template T<n>:` heading in `scaffolding-templates.md`.
- **Inference-function name resolution** — every function name in `Use template T<n> (\`fn\`)` lines has a matching `def fn(` in `scaffolding-templates.md`.
- **Rejection-message sync** — the `audio-to-audio` rejection message in `SKILL.md` Step 3 matches the one in `pipeline-tag-patterns.md`'s rejected-tags table verbatim.
- **Skip-validate marker count** — every ` ```python ` fragment in `pipeline-tag-patterns.md` has a `<!-- skip-validate -->` marker on the line immediately before it.
- **File-list ↔ checklist parity** — the four `### File N:` sub-headings in `SKILL.md` Step 4 match the file list in the Output checklist.

## How to run

```bash
cd streamlit-app-builder/tests
uv run pytest -v
```

This directory has its own `pyproject.toml` and `.venv`, independent of any scaffolded app.

## Skipping a block

Add `<!-- skip-validate -->` on the line immediately above the ` ```python ` fence. Use sparingly — only for blocks that can't be validated standalone (e.g., UI fragments without surrounding imports). Prefer fixing the template over adding a skip marker.
