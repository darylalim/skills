# Variant Resolution

This document describes how `mlx-app-converter` queries Hugging Face for MLX variants of a detected model and presents them to the user as a selectable matrix. The actual algorithm lives in `lib/variant_resolution.py`; this file is design rationale and helper invocation reference.

## Implementation

The logic described in this document is implemented in `lib/variant_resolution.py`. The module exposes these public functions:

| Function | Purpose |
|---|---|
| `parse_param_count(name)` | Extract `"8B"`, `"0.5B"`, etc. from a model name |
| `parse_quantization(name)` | Extract `"bf16"`, `"4bit"`, etc. from a model name |
| `query_mlx_variants(base_name, *, list_models=None)` | Query HF Hub and return sorted `Variant` objects |
| `pick_default(variants, original_param_count)` | Apply default-pick rule; return best `Variant` |
| `render_matrix(variants, original_param_count, default, *, model_id)` | Render the user-facing matrix string |
| `find_closest_siblings(base_name, *, list_models=None, k=3)` | Return top-k closest model IDs by edit distance |
| `parse_reply(reply, variants, default, *, allow_skip=False)` | Validate a user reply; return `Variant`, `"default"`, `"skip"`, or `None` |

`QUANTIZATION_PRECEDENCE = ("bf16", "fp16", "8bit", "6bit", "4bit")` is the module-level constant that drives both column ordering and default-pick.

### Invocation

Run from the skill root (`mlx-app-converter/`):

```bash
# Render the selection matrix for a model
python -m lib.variant_resolution query \
    --base-name "Llama-3.1-8B-Instruct" \
    --orig-param-count "8B" \
    --model-id "meta-llama/Llama-3.1-8B-Instruct"

# Find closest siblings when no variants match
python -m lib.variant_resolution siblings \
    --base-name "Llama-3.1-8B-Instruct"
```

## Matrix layout

The presentation matrix has:
- **Rows:** parameter counts in ascending order (e.g., `1B`, `3B`, `8B`, `70B`).
- **Columns:** quantization in fixed precedence order from highest to lowest precision: `bf16, fp16, 8bit, 6bit, 4bit`. Empty columns (no variants in any row) are omitted to keep the matrix narrow.

Example layout (for a model with no `6bit` variants in any row):

```
              bf16   fp16   8bit   4bit
  1B           ✓      –      ✓      ✓
  3B           ✓      –      ✓      ✓
* 8B (orig)    ✓ ★    ✓      ✓      ✓
  70B          –      –      –      ✓
```

- Row prefix `*` marks the row whose parameter count matches the user's original model.
- Cell `★` marks the default pick.
- Cell `✓` indicates the variant exists; `–` indicates it does not.

## Default-pick precedence

The default cell is the intersection of:
- The row matching the user's original model's parameter count (if such a row has any variants); otherwise the closest smaller available row.
- The column with the highest available precision in that row, where the precedence is `bf16 > fp16 > 8bit > 6bit > 4bit`.

Rationale: preserve the user's original parameter-count choice (they sized the model deliberately) while picking the highest-quality MLX variant available. The user can always override via the matrix.

## Multi-model apps

Each detected model is resolved independently: one matrix per model, presented in source-order (line number ascending). The user may answer all matrices in a single reply or one at a time.

## Reply validation

A user reply to a matrix prompt is valid if it is one of:

- The literal string `default` (case-sensitive) — accepts the highlighted default cell.
- A cell address of the form `<size>@<quantization>` where `<size>` matches one of the row labels (e.g., `1B`, `3B`, `8B`, `0.5B`) and `<quantization>` matches one of the column labels (e.g., `bf16`, `4bit`).
- The literal string `skip` (case-sensitive, only when the no-match fallback is in play) — leaves this model unchanged in the rewritten code.

Any other reply is rejected with a re-prompt:

- **Missing `@`** (e.g., user replies `8B`): `Reply must include the quantization (e.g., 8B@bf16) or be the literal "default". Try again.`
- **Unrecognized cell** (e.g., user replies `8B@5bit` when 5bit is not in the matrix): `8B@5bit is not in the matrix. Available cells: <comma-separated list of populated cells>. Try again.`
- **Typos / unknown words** (e.g., user replies `defualt`): same as the unrecognized-cell branch — list the valid options and re-prompt.

The skill never proceeds with a malformed reply; it loops until the user provides a valid cell, `default`, or `skip`. After three consecutive invalid replies, the skill exits with: `Too many invalid replies for <model>. Skipping this model.` This bounds the prompt loop in case of accidental copy-paste loops or terminal corruption.
