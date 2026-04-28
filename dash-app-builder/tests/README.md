# Dash App Builder Tests

Static validator and structural-consistency tests for the skill's Markdown files. Runs locally; not invoked at skill-runtime.

## What it checks

For every fenced ` ```python ` block in `SKILL.md` and `references/scaffolding-templates.md`:

1. **Syntax** — `ast.parse` after substituting documented placeholders (`<org>/<dataset>`).
2. **Lint** — `ruff check --select E,F,I` (matches the ruleset that scaffolded apps use).

Plus the following structural-consistency checks:

- **Template-name references** — every `T<n>` mentioned in `SKILL.md` has a `## Template T<n>:` heading in `scaffolding-templates.md`.
- **Rejection-message presence** — both rejection messages (wrong-skill and wrong-modality) appear verbatim in `SKILL.md`.
- **File-list ↔ checklist parity** — Step 4's file list and the Output Checklist agree on the four output files.
- **Helper-function name resolution** — `load_dataframe`, `build_filter_for_column`, `pick_chart` each appear exactly once as a `def` in `scaffolding-templates.md`.
- **Skip-validate marker count** — total `<!-- skip-validate -->` markers across both Markdown files is asserted against an expected value.

Plus Dash-specific spec-alignment checks:

- **T1 contains required env-var references** (`DATASET_ID`, `MAX_ROWS`, `HF_TOKEN`).
- **T1 caches `load_dataframe`** with some form of `lru_cache` / `cache` decorator.
- **T1 caps rows** — `MAX_ROWS` is referenced inside the `load_dataset(...)` call's argument expressions.
- **T2's `build_filter_for_column` returns a Dash widget or `None`.**
- **T3's `pick_chart` returns a `plotly.graph_objects.Figure`.**
- **T4 contains `dash_table.DataTable(` and `dcc.Graph(`.**
- **T5 covers the empty-DataFrame edge case for `pick_chart`.**

## How to run

```bash
cd dash-app-builder/tests
uv run pytest -v
```

This directory has its own `pyproject.toml` and `.venv`, independent of any scaffolded app.

## Skipping a block

Add `<!-- skip-validate -->` on the line immediately above the ` ```python ` fence. Use sparingly — only for blocks that can't be validated standalone (e.g., UI fragments without surrounding imports). Prefer fixing the template over adding a skip marker.
