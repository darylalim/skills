# streamlit-app-builder tests

Static validator for the Python code blocks embedded in this skill's Markdown files. Runs locally; not invoked at skill-runtime.

## What it checks

For every fenced ` ```python ` block in `SKILL.md`, `references/pipeline-tag-patterns.md`, and `references/scaffolding-templates.md`:

1. **Syntax** — `ast.parse` after substituting documented placeholders (`<org>/<model>`, `<id>`).
2. **Lint** — `ruff check --select E,F,I` (matches the ruleset that scaffolded apps use).

## How to run

```bash
cd streamlit-app-builder/tests
uv run pytest -v
```

## Skipping a block

Add `<!-- skip-validate -->` on the line immediately above the ` ```python ` fence. Use sparingly — only for blocks that can't be validated standalone (e.g., illustrative fragments without surrounding imports). Prefer fixing the template over adding a skip marker.
