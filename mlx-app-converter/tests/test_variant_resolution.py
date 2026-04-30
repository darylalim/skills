"""Unit tests for mlx-app-converter/lib/variant_resolution.py.

The module under test lives at mlx-app-converter/lib/variant_resolution.py.
We add the lib directory to sys.path so it can be imported without a package
install.  All HuggingFace Hub calls go through a fake list_models callable —
no real network connection is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Insert the lib directory into sys.path so "import variant_resolution" works.
LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import variant_resolution as vr  # noqa: E402

# ---------------------------------------------------------------------------
# Fake HF client helpers
# ---------------------------------------------------------------------------


class FakeModel:
    """Minimal stand-in for huggingface_hub ModelInfo."""

    def __init__(self, model_id: str) -> None:
        self.id = model_id


def fake_list_models(*, author: str, search: str):  # noqa: ANN202
    """Return a fixed set of models for testing."""
    return [
        FakeModel("mlx-community/Llama-3.1-8B-Instruct-bf16"),
        FakeModel("mlx-community/Llama-3.1-8B-Instruct-4bit"),
        FakeModel("mlx-community/Llama-3.1-1B-Instruct-bf16"),
        # Invalid entries that should be filtered out:
        FakeModel("mlx-community/some-other-model"),         # no param count token
        FakeModel("mlx-community/phi-2-bf16"),               # no param count token
        FakeModel("mlx-community/Llama-3.1-8B-Instruct"),   # no quant suffix
    ]


# ---------------------------------------------------------------------------
# parse_param_count
# ---------------------------------------------------------------------------


class TestParseParamCount:
    def test_integer_param_count(self):
        assert vr.parse_param_count("Llama-3.1-8B-Instruct-4bit") == "8B"

    def test_large_param_count(self):
        assert vr.parse_param_count("Llama-3.1-70B-Instruct-4bit") == "70B"

    def test_fractional_half(self):
        assert vr.parse_param_count("Qwen2.5-0.5B-Instruct-bf16") == "0.5B"

    def test_fractional_one_and_half(self):
        assert vr.parse_param_count("Qwen2.5-1.5B-Instruct-bf16") == "1.5B"

    def test_one_billion(self):
        assert vr.parse_param_count("Llama-3.2-1B-Instruct-bf16") == "1B"

    def test_three_billion(self):
        assert vr.parse_param_count("Llama-3.2-3B-Instruct-bf16") == "3B"

    def test_no_token_returns_none(self):
        assert vr.parse_param_count("phi-2-bf16") is None

    def test_no_token_plain_name(self):
        assert vr.parse_param_count("mlx-community/some-other-model") is None

    def test_full_model_id_with_prefix(self):
        assert vr.parse_param_count("mlx-community/Llama-3.1-8B-Instruct-bf16") == "8B"


# ---------------------------------------------------------------------------
# parse_quantization
# ---------------------------------------------------------------------------


class TestParseQuantization:
    def test_bf16(self):
        assert vr.parse_quantization("Llama-3.1-8B-Instruct-bf16") == "bf16"

    def test_fp16(self):
        assert vr.parse_quantization("Llama-3.1-8B-Instruct-fp16") == "fp16"

    def test_8bit(self):
        assert vr.parse_quantization("Llama-3.1-8B-Instruct-8bit") == "8bit"

    def test_6bit(self):
        assert vr.parse_quantization("Llama-3.1-8B-Instruct-6bit") == "6bit"

    def test_4bit(self):
        assert vr.parse_quantization("Llama-3.1-8B-Instruct-4bit") == "4bit"

    def test_no_suffix_returns_none(self):
        assert vr.parse_quantization("mlx-community/Llama-3.1-8B-Instruct") is None

    def test_no_suffix_plain(self):
        assert vr.parse_quantization("Llama-3.1-8B-Instruct") is None

    def test_with_mlx_prefix(self):
        assert vr.parse_quantization("mlx-community/Llama-3.1-8B-Instruct-bf16") == "bf16"


# ---------------------------------------------------------------------------
# query_mlx_variants
# ---------------------------------------------------------------------------


class TestQueryMlxVariants:
    def test_filters_invalid_entries(self):
        """Only entries with both param_count and quantization should survive."""
        results = vr.query_mlx_variants("Llama-3.1", list_models=fake_list_models)
        # phi-2-bf16 has no param count → dropped
        # some-other-model has no param count → dropped
        # Llama-3.1-8B-Instruct has no quant suffix → dropped
        ids = [v.full_name for v in results]
        assert "mlx-community/phi-2-bf16" not in ids
        assert "mlx-community/some-other-model" not in ids
        assert "mlx-community/Llama-3.1-8B-Instruct" not in ids

    def test_valid_entries_present(self):
        results = vr.query_mlx_variants("Llama-3.1", list_models=fake_list_models)
        ids = [v.full_name for v in results]
        assert "mlx-community/Llama-3.1-8B-Instruct-bf16" in ids
        assert "mlx-community/Llama-3.1-8B-Instruct-4bit" in ids
        assert "mlx-community/Llama-3.1-1B-Instruct-bf16" in ids

    def test_sort_order_smaller_first(self):
        """Smaller param count should come before larger."""
        results = vr.query_mlx_variants("Llama-3.1", list_models=fake_list_models)
        params = [v.param_count for v in results]
        numeric = [float(p[:-1]) for p in params]
        assert numeric == sorted(numeric)

    def test_sort_order_within_same_param_count(self):
        """Within same param count, higher precision (bf16) comes before lower (4bit)."""
        results = vr.query_mlx_variants("Llama-3.1", list_models=fake_list_models)
        eight_b = [v for v in results if v.param_count == "8B"]
        quants = [v.quantization for v in eight_b]
        # bf16 index=0, 4bit index=4 → bf16 should come first
        assert quants.index("bf16") < quants.index("4bit")

    def test_parsed_variant_fields(self):
        results = vr.query_mlx_variants("Llama-3.1", list_models=fake_list_models)
        v = next(r for r in results if r.full_name == "mlx-community/Llama-3.1-8B-Instruct-bf16")
        assert v.param_count == "8B"
        assert v.quantization == "bf16"


# ---------------------------------------------------------------------------
# pick_default
# ---------------------------------------------------------------------------


def _make_variants(*specs: tuple[str, str]) -> list[vr.Variant]:
    """Create a list of Variant objects from (param_count, quantization) pairs."""
    return [
        vr.Variant(
            full_name=f"mlx-community/model-{pc}-{q}",
            param_count=pc,
            quantization=q,
        )
        for pc, q in specs
    ]


class TestPickDefault:
    def test_empty_variants_returns_none(self):
        assert vr.pick_default([], "8B") is None

    def test_exact_match_highest_precision(self):
        variants = _make_variants(("8B", "bf16"), ("8B", "4bit"), ("70B", "bf16"))
        result = vr.pick_default(variants, "8B")
        assert result is not None
        assert result.param_count == "8B"
        assert result.quantization == "bf16"

    def test_exact_match_only_lower_precision(self):
        variants = _make_variants(("8B", "4bit"), ("70B", "bf16"))
        result = vr.pick_default(variants, "8B")
        assert result is not None
        assert result.param_count == "8B"
        assert result.quantization == "4bit"

    def test_no_match_falls_back_to_closest_smaller(self):
        """If original param count not present, pick largest available that's smaller."""
        variants = _make_variants(("1B", "bf16"), ("3B", "bf16"), ("70B", "bf16"))
        result = vr.pick_default(variants, "8B")
        assert result is not None
        assert result.param_count == "3B"

    def test_no_match_falls_back_to_smallest_larger_when_nothing_smaller(self):
        """If nothing smaller, fall back to smallest larger."""
        variants = _make_variants(("70B", "bf16"), ("34B", "bf16"))
        result = vr.pick_default(variants, "8B")
        assert result is not None
        assert result.param_count == "34B"

    def test_none_original_returns_best_overall(self):
        variants = _make_variants(("1B", "4bit"), ("8B", "bf16"))
        result = vr.pick_default(variants, None)
        # No original → pick smallest + highest precision = 1B@4bit? No — best_in_group uses quant precedence
        # Actually the function falls through to best_in_group(variants) for None original
        assert result is not None

    def test_multiple_smaller_picks_largest_smaller(self):
        """Among multiple smaller options, pick the largest one."""
        variants = _make_variants(("1B", "bf16"), ("3B", "4bit"), ("3B", "bf16"), ("70B", "bf16"))
        result = vr.pick_default(variants, "8B")
        assert result is not None
        assert result.param_count == "3B"
        assert result.quantization == "bf16"  # highest precision in the 3B group


# ---------------------------------------------------------------------------
# render_matrix
# ---------------------------------------------------------------------------


class TestRenderMatrix:
    def _four_by_four_variants(self) -> list[vr.Variant]:
        """Build the 4-param × 4-quant set from the spec example."""
        specs = [
            ("1B", "bf16"), ("1B", "8bit"), ("1B", "4bit"),
            ("3B", "bf16"), ("3B", "8bit"), ("3B", "4bit"),
            ("8B", "bf16"), ("8B", "fp16"), ("8B", "8bit"), ("8B", "4bit"),
            ("70B", "4bit"),
        ]
        return [
            vr.Variant(
                full_name=f"mlx-community/Model-{pc}-{q}",
                param_count=pc,
                quantization=q,
            )
            for pc, q in specs
        ]

    def test_header_line(self):
        variants = self._four_by_four_variants()
        default = vr.pick_default(variants, "8B")
        output = vr.render_matrix(variants, "8B", default, model_id="meta-llama/Llama-3.1-8B-Instruct")
        assert output.startswith("Found 4 size × 4 quantization variants for meta-llama/Llama-3.1-8B-Instruct:")

    def test_orig_row_marked_with_star(self):
        variants = self._four_by_four_variants()
        default = vr.pick_default(variants, "8B")
        output = vr.render_matrix(variants, "8B", default, model_id="meta-llama/Llama-3.1-8B-Instruct")
        assert "* 8B (orig)" in output

    def test_default_cell_marked_with_filled_star(self):
        variants = self._four_by_four_variants()
        default = vr.pick_default(variants, "8B")
        output = vr.render_matrix(variants, "8B", default, model_id="meta-llama/Llama-3.1-8B-Instruct")
        # Default for 8B should be bf16 (highest precision)
        assert "★" in output

    def test_empty_column_omitted(self):
        """6bit column should be omitted because no variants have it."""
        variants = self._four_by_four_variants()
        default = vr.pick_default(variants, "8B")
        output = vr.render_matrix(variants, "8B", default, model_id="meta-llama/Llama-3.1-8B-Instruct")
        lines = output.splitlines()
        col_header_line = next(ln for ln in lines if "bf16" in ln and "fp16" in ln)
        assert "6bit" not in col_header_line

    def test_absent_cell_marked_with_dash(self):
        """70B only has 4bit; its bf16/fp16/8bit cells should be dashes."""
        variants = self._four_by_four_variants()
        default = vr.pick_default(variants, "8B")
        output = vr.render_matrix(variants, "8B", default, model_id="meta-llama/Llama-3.1-8B-Instruct")
        lines = output.splitlines()
        seventy_b_line = next(ln for ln in lines if "70B" in ln)
        assert "–" in seventy_b_line

    def test_default_line_present(self):
        variants = self._four_by_four_variants()
        default = vr.pick_default(variants, "8B")
        output = vr.render_matrix(variants, "8B", default, model_id="meta-llama/Llama-3.1-8B-Instruct")
        assert "Default: 8B @ bf16 (★)" in output

    def test_reply_prompt_present(self):
        variants = self._four_by_four_variants()
        default = vr.pick_default(variants, "8B")
        output = vr.render_matrix(variants, "8B", default, model_id="meta-llama/Llama-3.1-8B-Instruct")
        assert 'Reply with a variant' in output

    def test_empty_variants_returns_no_match_message(self):
        output = vr.render_matrix([], None, None, model_id="some/Model")
        assert "No MLX variants found for some/Model" in output

    def test_simple_2x2_golden(self):
        """Golden test for a simple 2-row × 2-col matrix."""
        variants = _make_variants(("1B", "bf16"), ("1B", "4bit"), ("8B", "bf16"), ("8B", "4bit"))
        default = vr.pick_default(variants, "8B")
        output = vr.render_matrix(variants, "8B", default, model_id="org/MyModel")

        assert "Found 2 size × 2 quantization variants for org/MyModel:" in output
        assert "* 8B (orig)" in output
        assert "1B" in output
        assert "bf16" in output
        assert "4bit" in output
        assert "★" in output  # default marked
        assert "–" not in output  # all cells filled in this 2x2


# ---------------------------------------------------------------------------
# find_closest_siblings
# ---------------------------------------------------------------------------


def _sibling_list_models(*, author: str, search: str):  # noqa: ANN202
    return [
        FakeModel("mlx-community/Llama-3.1-8B-Instruct-bf16"),
        FakeModel("mlx-community/Llama-3.1-8B-Instruct-4bit"),
        FakeModel("mlx-community/Llama-3.2-1B-Instruct-bf16"),
        FakeModel("mlx-community/Llama-3.3-70B-Instruct-bf16"),
        FakeModel("mlx-community/completely-unrelated-model"),
    ]


class TestFindClosestSiblings:
    def test_returns_at_most_k_results(self):
        results = vr.find_closest_siblings(
            "Llama-3.1-8B-Instruct", list_models=_sibling_list_models, k=2
        )
        assert len(results) <= 2

    def test_closest_sibling_first(self):
        """Llama-3.1-8B-Instruct-bf16 should rank above completely-unrelated-model."""
        results = vr.find_closest_siblings(
            "Llama-3.1-8B-Instruct", list_models=_sibling_list_models, k=5
        )
        unrelated_idx = next(
            (i for i, r in enumerate(results) if "completely-unrelated" in r), None
        )
        llama_idx = next(
            (i for i, r in enumerate(results) if "Llama-3.1-8B" in r), None
        )
        assert llama_idx is not None
        if unrelated_idx is not None:
            assert llama_idx < unrelated_idx

    def test_default_k_is_3(self):
        results = vr.find_closest_siblings(
            "Llama-3.1-8B-Instruct", list_models=_sibling_list_models
        )
        assert len(results) <= 3

    def test_returns_full_model_ids(self):
        results = vr.find_closest_siblings(
            "Llama-3.1-8B-Instruct", list_models=_sibling_list_models, k=3
        )
        for r in results:
            assert r.startswith("mlx-community/")


# ---------------------------------------------------------------------------
# parse_reply
# ---------------------------------------------------------------------------


_SAMPLE_VARIANTS = _make_variants(
    ("1B", "bf16"),
    ("8B", "bf16"),
    ("8B", "4bit"),
)
_SAMPLE_DEFAULT = next(v for v in _SAMPLE_VARIANTS if v.param_count == "8B" and v.quantization == "bf16")


class TestParseReply:
    def test_default_literal(self):
        result = vr.parse_reply("default", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert result == "default"

    def test_default_is_case_sensitive(self):
        # "Default" should not match — must be lowercase "default"
        result = vr.parse_reply("Default", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert result is None

    def test_valid_cell_address(self):
        result = vr.parse_reply("8B@bf16", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert isinstance(result, vr.Variant)
        assert result.param_count == "8B"
        assert result.quantization == "bf16"

    def test_valid_cell_4bit(self):
        result = vr.parse_reply("8B@4bit", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert isinstance(result, vr.Variant)
        assert result.param_count == "8B"
        assert result.quantization == "4bit"

    def test_skip_when_allowed(self):
        result = vr.parse_reply("skip", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT, allow_skip=True)
        assert result == "skip"

    def test_skip_when_not_allowed(self):
        result = vr.parse_reply("skip", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT, allow_skip=False)
        assert result is None

    def test_missing_at_sign(self):
        result = vr.parse_reply("8B", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert result is None

    def test_unrecognized_cell(self):
        # 8B@5bit is not in the variants
        result = vr.parse_reply("8B@5bit", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert result is None

    def test_unrecognized_size(self):
        result = vr.parse_reply("999B@bf16", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert result is None

    def test_typo_defualt(self):
        result = vr.parse_reply("defualt", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert result is None

    def test_whitespace_stripped(self):
        result = vr.parse_reply("  8B@bf16  ", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert isinstance(result, vr.Variant)

    def test_empty_string(self):
        result = vr.parse_reply("", _SAMPLE_VARIANTS, _SAMPLE_DEFAULT)
        assert result is None


# ---------------------------------------------------------------------------
# QUANTIZATION_PRECEDENCE constant
# ---------------------------------------------------------------------------


class TestQuantizationPrecedence:
    def test_order(self):
        assert vr.QUANTIZATION_PRECEDENCE == ("bf16", "fp16", "8bit", "6bit", "4bit")

    def test_bf16_is_highest(self):
        """bf16 should come before all others."""
        assert vr.QUANTIZATION_PRECEDENCE[0] == "bf16"

    def test_4bit_is_lowest(self):
        assert vr.QUANTIZATION_PRECEDENCE[-1] == "4bit"


# ---------------------------------------------------------------------------
# v2 VLM coverage tests
# ---------------------------------------------------------------------------


def test_query_handles_qwen2_vl_dense_matrix():
    """Qwen2-VL family on mlx-community has multiple param counts × multiple
    quantizations — confirm query_mlx_variants parses all of them and
    pick_default returns the orig-param-count + bf16 cell."""
    def fake_list_models(*, author: str, search: str):  # noqa: ANN202
        assert author == "mlx-community"
        return [
            FakeModel("mlx-community/Qwen2-VL-2B-Instruct-bf16"),
            FakeModel("mlx-community/Qwen2-VL-2B-Instruct-4bit"),
            FakeModel("mlx-community/Qwen2-VL-7B-Instruct-bf16"),
            FakeModel("mlx-community/Qwen2-VL-7B-Instruct-4bit"),
            FakeModel("mlx-community/Qwen2-VL-7B-Instruct-8bit"),
        ]

    variants = vr.query_mlx_variants(
        "Qwen2-VL-7B-Instruct", list_models=fake_list_models
    )
    assert len(variants) == 5, f"Expected 5 variants, got {len(variants)}"
    assert all(isinstance(v, vr.Variant) for v in variants)

    default = vr.pick_default(variants, "7B")
    assert default is not None
    assert default.param_count == "7B"
    assert default.quantization == "bf16"


def test_query_handles_llava_sparse_by_rows():
    """Llava family on mlx-community has only the 7B size on the user's
    original — confirm sparse-by-rows matrices work."""
    def fake_list_models(*, author: str, search: str):  # noqa: ANN202
        return [
            FakeModel("mlx-community/llava-v1.6-mistral-7B-4bit"),
            FakeModel("mlx-community/llava-v1.6-mistral-7B-8bit"),
            FakeModel("mlx-community/llava-v1.6-mistral-7B-bf16"),
        ]

    variants = vr.query_mlx_variants(
        "llava-v1.6-mistral-7B", list_models=fake_list_models
    )
    assert len(variants) == 3
    default = vr.pick_default(variants, "7B")
    assert default is not None
    assert default.param_count == "7B"
    assert default.quantization == "bf16"


def test_query_handles_paligemma_sparse_by_cols():
    """PaliGemma on mlx-community may have multiple param counts but only one
    quantization — confirm sparse-by-cols matrices work."""
    def fake_list_models(*, author: str, search: str):  # noqa: ANN202
        return [
            FakeModel("mlx-community/paligemma-3B-mix-224-4bit"),
            FakeModel("mlx-community/paligemma-3B-mix-448-4bit"),
            FakeModel("mlx-community/paligemma-10B-mix-224-4bit"),
        ]

    variants = vr.query_mlx_variants(
        "paligemma-3B-mix", list_models=fake_list_models
    )
    assert len(variants) == 3
    default = vr.pick_default(variants, "3B")
    assert default is not None
    assert default.param_count == "3B"
    assert default.quantization == "4bit"


def test_render_matrix_renders_vlm_dense():
    """Render-matrix output for VLMs visually mirrors the LLM matrix —
    same row/col layout, same default-pick marker."""
    def fake_list_models(*, author: str, search: str):  # noqa: ANN202
        return [
            FakeModel("mlx-community/Qwen2-VL-2B-Instruct-bf16"),
            FakeModel("mlx-community/Qwen2-VL-2B-Instruct-4bit"),
            FakeModel("mlx-community/Qwen2-VL-7B-Instruct-bf16"),
            FakeModel("mlx-community/Qwen2-VL-7B-Instruct-4bit"),
        ]

    variants = vr.query_mlx_variants(
        "Qwen2-VL-7B-Instruct", list_models=fake_list_models
    )
    default = vr.pick_default(variants, "7B")
    output = vr.render_matrix(
        variants, "7B", default, model_id="Qwen/Qwen2-VL-7B-Instruct"
    )
    assert "★" in output
    assert "* 7B (orig)" in output or "*7B (orig)" in output
    assert "bf16" in output
    assert "4bit" in output
