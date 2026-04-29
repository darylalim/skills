# Variant Resolution

This document describes how `mlx-app-converter` queries Hugging Face for MLX variants of a detected model and presents them to the user as a selectable matrix.

## Query strategy

For each detected model ID `<base_org>/<base_name>` (e.g., `meta-llama/Llama-3.1-8B-Instruct`):

1. Query `huggingface_hub.list_models(author="mlx-community", search="<base_name>")`. The `search` parameter does substring matching; `Llama-3.1-8B-Instruct` will match `mlx-community/Llama-3.1-8B-Instruct-bf16`, `mlx-community/Llama-3.1-8B-Instruct-4bit`, and others.
2. Filter the results to those whose name ends in one of the MLX quantization suffixes: `-bf16`, `-fp16`, `-8bit`, `-6bit`, `-4bit`. Names without a recognized suffix are dropped (heuristic: assume they are not standard mlx-lm variants).
3. From the surviving list, parse `(parameter_count, quantization)` from each name.

## Parameter-count parsing

The base name typically encodes parameter count as `<digits>B` or `<digits>.<digits>B`. Examples handled:

| Input fragment | Parsed param count |
|---|---|
| `Llama-3.2-1B-Instruct-bf16` | `1B` |
| `Llama-3.1-8B-Instruct-4bit` | `8B` |
| `Qwen2.5-0.5B-Instruct-bf16` | `0.5B` |
| `Qwen2.5-1.5B-Instruct-bf16` | `1.5B` |
| `Llama-3.1-70B-Instruct-4bit` | `70B` |

Regex: `\b(\d+(?:\.\d+)?)B\b` — matches the first occurrence in the name. Models whose name doesn't contain a parameter token are bucketed under "unknown" and shown in a separate row.

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

## No-match fallback

If the search yields zero MLX variants for a detected model:

1. Re-query with the broader pattern: `huggingface_hub.list_models(author="mlx-community", search="<first-word-of-base-name>")`. For example, `Llama-3.1-8B-Instruct` falls back to a `Llama` search.
2. Compute Levenshtein edit distance between each candidate's name (without quantization suffix) and the original `<base_name>`.
3. Return the top 3 candidates by smallest edit distance.

The skill presents these as "closest siblings" and asks the user whether to pick one, skip the model unchanged, or abort.

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
