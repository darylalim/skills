"""Variant resolution helper for mlx-app-converter.

Implements the logic described in references/variant-resolution.md:
- Query HuggingFace Hub for mlx-community variants of a given model.
- Filter by quantization suffix and parse (param_count, quantization) pairs.
- Build a (param-count x quantization) matrix and pick a default cell.
- Fall back to closest siblings via edit distance when no variants match.
- Validate user replies to the matrix prompt.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# THE source of truth for quantization precedence (highest → lowest precision).
# This string must match the literal "bf16 > fp16 > 8bit > 6bit > 4bit" in
# references/variant-resolution.md.
QUANTIZATION_PRECEDENCE: tuple[str, ...] = ("bf16", "fp16", "8bit", "6bit", "4bit")

# Regex to extract parameter count like "8B", "0.5B", "70B" from a model name.
# Case-insensitive: mlx-community has historically published variants with
# mixed casing (e.g., `paligemma-3b-mix-224` vs `Qwen2-VL-7B-Instruct`).
# parse_param_count normalizes the output to uppercase regardless of input.
_PARAM_COUNT_RE = re.compile(r"\b(\d+(?:\.\d+)?)B\b", re.IGNORECASE)

# Suffix patterns that mark a standard mlx-lm quantized variant.
_QUANT_SUFFIXES = tuple(f"-{q}" for q in QUANTIZATION_PRECEDENCE)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_param_count(name: str) -> str | None:
    """Return the first 'NB' or 'N.NB' token found in *name*, or None."""
    m = _PARAM_COUNT_RE.search(name)
    if m is None:
        return None
    return f"{m.group(1)}B"


def parse_quantization(name: str) -> str | None:
    """Return the quantization suffix of *name* (e.g. 'bf16'), or None."""
    for suffix in _QUANT_SUFFIXES:
        if name.endswith(suffix):
            # strip "mlx-community/" prefix if present
            stem = name.split("/")[-1] if "/" in name else name
            if stem.endswith(suffix):
                return suffix.lstrip("-")
    return None


# ---------------------------------------------------------------------------
# Variant dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    full_name: str    # "mlx-community/Llama-3.1-8B-Instruct-4bit"
    param_count: str  # "8B"
    quantization: str  # "4bit"


# ---------------------------------------------------------------------------
# Sorting helpers
# ---------------------------------------------------------------------------


def _param_count_numeric(param_count: str) -> float:
    """Convert '8B' -> 8.0, '0.5B' -> 0.5, etc."""
    return float(param_count[:-1])


def _quant_index(quantization: str) -> int:
    """Lower index = higher precision (bf16=0, 4bit=4)."""
    try:
        return QUANTIZATION_PRECEDENCE.index(quantization)
    except ValueError:
        return len(QUANTIZATION_PRECEDENCE)


def _sort_key(v: Variant) -> tuple[float, int]:
    return (_param_count_numeric(v.param_count), _quant_index(v.quantization))


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _get_model_id(model_obj: object) -> str:
    """Return the model ID from a HF model info object.

    Handles both .id (current API) and .modelId (older versions).
    """
    if hasattr(model_obj, "id"):
        return str(model_obj.id)  # type: ignore[attr-defined]
    if hasattr(model_obj, "modelId"):
        return str(model_obj.modelId)  # type: ignore[attr-defined]
    raise AttributeError(f"Cannot find model ID on object: {model_obj!r}")


def query_mlx_variants(
    base_name: str,
    *,
    list_models: Callable[..., Iterable[object]] | None = None,
) -> list[Variant]:
    """Query HF Hub for mlx-community variants matching *base_name*.

    Args:
        base_name: The base model name to search for (e.g. "Llama-3.1-8B-Instruct").
        list_models: Optional callable replacing huggingface_hub.list_models
            for testing.

    Returns:
        Sorted list of Variant objects (smaller param count first, then
        higher precision first).
    """
    if list_models is None:
        from huggingface_hub import (  # type: ignore[import-untyped]
            list_models as _list_models,
        )

        list_models = _list_models

    raw = list_models(author="mlx-community", search=base_name)
    variants: list[Variant] = []
    for model in raw:
        model_id = _get_model_id(model)
        param_count = parse_param_count(model_id)
        quantization = parse_quantization(model_id)
        if param_count is None or quantization is None:
            continue
        variants.append(
            Variant(
                full_name=model_id,
                param_count=param_count,
                quantization=quantization,
            )
        )

    return sorted(variants, key=_sort_key)


# ---------------------------------------------------------------------------
# Default pick
# ---------------------------------------------------------------------------


def pick_default(
    variants: list[Variant],
    original_param_count: str | None,
) -> Variant | None:
    """Pick the default variant according to the variant-resolution spec.

    Rule:
    1. Filter to variants matching original_param_count.
    2. Among those, pick the highest precision (lowest QUANTIZATION_PRECEDENCE
       index).
    3. If none match, fall back to the largest param count smaller than
       original.
    4. If nothing smaller, fall back to the smallest param count larger than
       original.
    5. Return None if variants is empty.
    """
    if not variants:
        return None

    # Helper: best variant in a group (highest precision)
    def best_in_group(group: list[Variant]) -> Variant:
        return min(group, key=lambda v: _quant_index(v.quantization))

    # Step 1+2: exact param-count match
    if original_param_count is not None:
        matched = [v for v in variants if v.param_count == original_param_count]
        if matched:
            return best_in_group(matched)

    # Step 3: closest smaller
    if original_param_count is not None:
        orig_val = _param_count_numeric(original_param_count)
        smaller = [
            v for v in variants
            if _param_count_numeric(v.param_count) < orig_val
        ]
        if smaller:
            # largest among smaller
            largest_val = max(_param_count_numeric(v.param_count) for v in smaller)
            group = [
                v for v in smaller
                if _param_count_numeric(v.param_count) == largest_val
            ]
            return best_in_group(group)

        # Step 4: smallest larger
        larger = [
            v for v in variants
            if _param_count_numeric(v.param_count) > orig_val
        ]
        if larger:
            smallest_val = min(_param_count_numeric(v.param_count) for v in larger)
            group = [
                v for v in larger
                if _param_count_numeric(v.param_count) == smallest_val
            ]
            return best_in_group(group)

    # original_param_count is None or nothing matches: pick overall best
    return best_in_group(variants)


# ---------------------------------------------------------------------------
# Matrix renderer
# ---------------------------------------------------------------------------


def render_matrix(
    variants: list[Variant],
    original_param_count: str | None,
    default: Variant | None,
    *,
    model_id: str,
) -> str:
    """Render the variant selection matrix as a human-readable string.

    Layout: rows = param counts ascending, columns = quantization in
    QUANTIZATION_PRECEDENCE order (empty columns omitted).
    """
    if not variants:
        return f"No MLX variants found for {model_id}."

    # Collect unique param counts (ascending) and quantizations present
    all_params_sorted = sorted(
        {v.param_count for v in variants},
        key=_param_count_numeric,
    )
    # Only include quantization columns that have at least one variant
    present_quants = {v.quantization for v in variants}
    col_order = [q for q in QUANTIZATION_PRECEDENCE if q in present_quants]

    # Build lookup: (param_count, quantization) -> Variant | None
    lookup: dict[tuple[str, str], Variant] = {}
    for v in variants:
        lookup[(v.param_count, v.quantization)] = v

    # Count distinct param counts and quantizations for the header
    n_sizes = len(all_params_sorted)
    n_quants = len(col_order)

    lines: list[str] = []
    lines.append(
        f"Found {n_sizes} size × {n_quants} quantization variants for {model_id}:"
    )
    lines.append("")

    # Column header line
    col_width = 7  # width per column cell
    header_indent = " " * 14
    col_headers = "".join(f"{q:<{col_width}}" for q in col_order)
    lines.append(f"{header_indent}{col_headers}")

    for param in all_params_sorted:
        is_orig = original_param_count is not None and param == original_param_count
        prefix = "* " if is_orig else "  "
        label = f"{param} (orig)" if is_orig else param
        row_label = f"{prefix}{label}"
        # pad row label to 14 chars
        row_label_padded = f"{row_label:<14}"

        cells: list[str] = []
        for quant in col_order:
            variant = lookup.get((param, quant))
            if variant is None:
                cell = "–"
            elif default is not None and variant == default:
                cell = "✓ ★"
            else:
                cell = "✓"
            cells.append(f"{cell:<{col_width}}")

        lines.append(f"{row_label_padded}{''.join(cells)}")

    lines.append("")

    # Default description line
    if default is not None:
        if (
            original_param_count is not None
            and default.param_count == original_param_count
        ):
            reason = (
                "same parameters as your original, "
                "highest available precision."
            )
        elif original_param_count is not None:
            orig_val = _param_count_numeric(original_param_count)
            default_val = _param_count_numeric(default.param_count)
            if default_val < orig_val:
                reason = (
                    f"closest smaller param count to your original "
                    f"{original_param_count}, highest available precision."
                )
            else:
                reason = (
                    f"closest larger param count to your original "
                    f"{original_param_count}, highest available precision."
                )
        else:
            reason = "best available precision."
        lines.append(
            f"Default: {default.param_count} @ {default.quantization} "
            f"(★) — {reason}"
        )
    lines.append('Reply with a variant (e.g., "8B@bf16") or "default" to accept.')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Levenshtein edit distance (standalone, no third-party dep required)
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # DP with two rows
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[len(b)]


def _stem(model_id: str) -> str:
    """Strip mlx-community/ prefix and quantization suffix from a model ID."""
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    for suffix in _QUANT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


# ---------------------------------------------------------------------------
# Closest siblings (no-match fallback)
# ---------------------------------------------------------------------------


def find_closest_siblings(
    base_name: str,
    *,
    list_models: Callable[..., Iterable[object]] | None = None,
    k: int = 3,
) -> list[str]:
    """Return the top-k closest mlx-community model IDs to *base_name*.

    Queries a broader pattern (first word of base_name), then ranks by
    Levenshtein distance of the stem to base_name.

    Args:
        base_name: The original base name to find siblings for.
        list_models: Optional callable replacing huggingface_hub.list_models.
        k: Maximum number of results to return.

    Returns:
        List of model IDs sorted by (edit_distance, name).
    """
    if list_models is None:
        from huggingface_hub import (  # type: ignore[import-untyped]
            list_models as _list_models,
        )

        list_models = _list_models

    # Broader search: first word of base_name
    first_word = base_name.split("-")[0]
    raw = list_models(author="mlx-community", search=first_word)

    candidates: list[tuple[int, str, str]] = []  # (distance, name, full_id)
    for model in raw:
        model_id = _get_model_id(model)
        stem = _stem(model_id)
        dist = _levenshtein(stem, base_name)
        candidates.append((dist, model_id, model_id))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return [full_id for _, _, full_id in candidates[:k]]


# ---------------------------------------------------------------------------
# Reply parser
# ---------------------------------------------------------------------------


def parse_reply(
    reply: str,
    variants: list[Variant],
    default: Variant | None,
    *,
    allow_skip: bool = False,
) -> Variant | str | None:
    """Parse a user reply to a matrix prompt.

    Returns:
        - A Variant if reply is a valid "NB@quant" cell address.
        - The string "default" if reply is exactly "default".
        - The string "skip" if allow_skip=True and reply is exactly "skip".
        - None if the reply is invalid.
    """
    stripped = reply.strip()

    if stripped == "default":
        return "default"

    if allow_skip and stripped == "skip":
        return "skip"

    # Must contain "@"
    if "@" not in stripped:
        return None

    parts = stripped.split("@", 1)
    if len(parts) != 2:
        return None

    size_str, quant_str = parts[0].strip(), parts[1].strip()

    # Check against known variants
    for v in variants:
        if v.param_count == size_str and v.quantization == quant_str:
            return v

    return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli_query(args: argparse.Namespace) -> None:
    variants = query_mlx_variants(args.base_name)
    default = pick_default(variants, args.orig_param_count)
    display_id = args.model_id or args.base_name
    print(render_matrix(variants, args.orig_param_count, default, model_id=display_id))


def _cli_siblings(args: argparse.Namespace) -> None:
    siblings = find_closest_siblings(args.base_name, k=args.k)
    if siblings:
        print("Closest siblings:")
        for s in siblings:
            print(f"  {s}")
    else:
        print("No siblings found.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m lib.variant_resolution",
        description=(
            "Query mlx-community for model variants and render a selection matrix."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Query variants and print the selection matrix.")
    q.add_argument(
        "--base-name",
        required=True,
        help="Model base name (e.g. Llama-3.1-8B-Instruct).",
    )
    q.add_argument(
        "--orig-param-count",
        default=None,
        help="Original param count (e.g. 8B).",
    )
    q.add_argument(
        "--model-id",
        default=None,
        help=(
            "Full original model ID for display "
            "(e.g. meta-llama/Llama-3.1-8B-Instruct)."
        ),
    )
    q.set_defaults(func=_cli_query)

    s = sub.add_parser(
        "siblings",
        help="Find closest sibling models when no variants match.",
    )
    s.add_argument("--base-name", required=True, help="Model base name.")
    s.add_argument("--k", type=int, default=3, help="Number of siblings to return.")
    s.set_defaults(func=_cli_siblings)

    parsed = parser.parse_args(argv)
    parsed.func(parsed)


if __name__ == "__main__":
    main(sys.argv[1:])
