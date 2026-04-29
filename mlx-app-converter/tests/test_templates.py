"""Static validator and structural-consistency tests for mlx-app-converter."""
from __future__ import annotations

import re
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]  # mlx-app-converter/


# === Spec alignment tests ===

VARIANT_RESOLUTION_MD = SKILL_ROOT / "references" / "variant-resolution.md"


def test_variant_default_precedence_documented():
    """variant-resolution.md must document the precedence string verbatim."""
    text = VARIANT_RESOLUTION_MD.read_text()
    assert "bf16 > fp16 > 8bit > 6bit > 4bit" in text, (
        "Default-pick precedence string missing from variant-resolution.md"
    )


REWRITE_TEMPLATES_MD = SKILL_ROOT / "references" / "rewrite-templates.md"


def _template_section(name: str) -> str:
    """Extract the body of `## Template T<name>: ...` up to the next `## ` header."""
    text = REWRITE_TEMPLATES_MD.read_text()
    pattern = re.compile(
        rf"^## Template {re.escape(name)}:.*?(?=^## |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(text)
    assert m, f"Section '## Template {name}:' not found in rewrite-templates.md"
    return m.group(0)


def test_t1_preserves_cache_decorators():
    """T1 must show both Streamlit (@st.cache_resource) and Gradio (@lru_cache) variants."""
    section = _template_section("T1")
    assert "@st.cache_resource" in section, "T1 missing @st.cache_resource"
    assert "@lru_cache(maxsize=1)" in section, "T1 missing @lru_cache(maxsize=1)"


def test_t2_maps_max_new_tokens_to_max_tokens():
    """T2 must document the max_new_tokens → max_tokens kwarg rename."""
    section = _template_section("T2")
    assert "max_new_tokens" in section, "T2 missing reference to max_new_tokens"
    assert "max_tokens" in section, "T2 missing reference to max_tokens"
    # both must appear so the rename is documented
