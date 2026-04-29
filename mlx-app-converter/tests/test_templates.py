"""Static validator and structural-consistency tests for mlx-app-converter."""
from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]  # mlx-app-converter/

CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
SKIP_MARKER = "<!-- skip-validate -->"


@dataclass(frozen=True)
class CodeBlock:
    source_file: Path
    line_no: int
    content: str


def find_python_blocks(path: Path) -> list[CodeBlock]:
    text = path.read_text()
    blocks: list[CodeBlock] = []
    for m in CODE_BLOCK_RE.finditer(text):
        before = text[: m.start()]
        last_line_before = before.rstrip("\n").rsplit("\n", 1)[-1] if before else ""
        if SKIP_MARKER in last_line_before:
            continue
        line_no = before.count("\n") + 2
        blocks.append(CodeBlock(source_file=path, line_no=line_no, content=m.group(1)))
    return blocks


PLACEHOLDER_SUBSTITUTIONS = {
    "<MODEL_ID>": "mlx-community/Llama-3.1-8B-Instruct-bf16",
    "<orig_model_id>": "meta-llama/Llama-3.1-8B-Instruct",
}


def substitute_placeholders(content: str) -> str:
    for placeholder, replacement in PLACEHOLDER_SUBSTITUTIONS.items():
        content = content.replace(placeholder, replacement)
    return content


# === Static validation tests ===

MARKDOWN_FILES = [
    SKILL_ROOT / "SKILL.md",
    SKILL_ROOT / "references" / "rewrite-templates.md",
]


def _all_blocks() -> list[CodeBlock]:
    out: list[CodeBlock] = []
    for f in MARKDOWN_FILES:
        if f.exists():
            out.extend(find_python_blocks(f))
    return out


@pytest.mark.parametrize(
    "block", _all_blocks(), ids=lambda b: f"{b.source_file.name}:{b.line_no}"
)
def test_block_parses_as_python(block):
    substituted = substitute_placeholders(block.content)
    try:
        ast.parse(substituted)
    except SyntaxError as e:
        pytest.fail(
            f"{block.source_file.name}:{block.line_no} SyntaxError: {e.msg} "
            f"(line {e.lineno}, col {e.offset})"
        )


def _lint_block(block: CodeBlock) -> tuple[int, str]:
    substituted = substitute_placeholders(block.content)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(substituted)
        tmp_path = tf.name
    try:
        result = subprocess.run(
            ["ruff", "check", "--select", "E,F,I", "--no-fix", tmp_path],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    "block", _all_blocks(), ids=lambda b: f"{b.source_file.name}:{b.line_no}"
)
def test_block_passes_ruff_check(block):
    rc, output = _lint_block(block)
    if rc != 0:
        pytest.fail(
            f"{block.source_file.name}:{block.line_no} ruff failures:\n{output}"
        )


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
    """T2 must document the max_new_tokens → max_tokens kwarg rename and show
    the literal mlx_lm.generate call shape with max_tokens=."""
    section = _template_section("T2")
    assert "max_new_tokens" in section, "T2 missing reference to max_new_tokens"
    assert "max_tokens" in section, "T2 missing reference to max_tokens"
    assert "mlx_lm.generate(model, tokenizer, prompt, max_tokens=" in section, (
        "T2 missing the literal mlx_lm.generate call shape"
    )


def test_t2_documents_sampler_helper_construction():
    """T2 must show that sampling parameters require make_sampler /
    make_logits_processors helpers (not direct kwargs to mlx_lm.generate)."""
    section = _template_section("T2")
    assert "make_sampler" in section, (
        "T2 missing make_sampler helper (sampling kwargs need wrapper)"
    )
    assert "make_logits_processors" in section, (
        "T2 missing make_logits_processors helper (repetition_penalty needs wrapper)"
    )
    assert "from mlx_lm.sample_utils import" in section, (
        "T2 missing the sample_utils import showing helper origin"
    )


def test_t3_contains_apple_silicon_check():
    """T3 must contain the literal platform.machine and platform.system checks."""
    section = _template_section("T3")
    assert 'platform.machine() == "arm64"' in section, (
        "T3 missing platform.machine() == \"arm64\" check"
    )
    assert 'platform.system() == "Darwin"' in section, (
        "T3 missing platform.system() == \"Darwin\" check"
    )


def test_t4_mocks_mlx_lm_load():
    """T4 must show the mock target as mlx_lm.load (not from_pretrained)."""
    section = _template_section("T4")
    assert "mlx_lm.load" in section, "T4 missing mlx_lm.load as mock target"


def test_t5_emits_uv_add_for_streamlit():
    """T5 must instruct `uv add mlx-lm` for Streamlit (uv-managed)."""
    section = _template_section("T5")
    assert "uv add mlx-lm" in section, "T5 missing 'uv add mlx-lm' command for Streamlit"


def test_t5_appends_to_requirements_for_gradio():
    """T5 must instruct appending mlx-lm to requirements.txt for Gradio."""
    section = _template_section("T5")
    assert "requirements.txt" in section, "T5 missing requirements.txt reference"
    assert "mlx-lm" in section, "T5 missing mlx-lm reference"


def test_t5_prints_removal_hint():
    """T5 must include the transformers/torch removal hint string."""
    section = _template_section("T5")
    assert "transformers and torch may now be unused" in section, (
        "T5 missing the standard removal-hint phrase"
    )
