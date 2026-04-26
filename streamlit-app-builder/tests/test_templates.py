"""Static validator for code templates in skill markdown files."""
from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]  # streamlit-app-builder/

CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
SKIP_MARKER = "<!-- skip-validate -->"


@dataclass(frozen=True)
class CodeBlock:
    source_file: Path
    line_no: int  # 1-indexed start line of the code body
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


def test_extract_finds_python_blocks(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("intro\n\n```python\nx = 1\n```\n\ntail\n")
    blocks = find_python_blocks(f)
    assert len(blocks) == 1
    assert blocks[0].content == "x = 1\n"
    assert blocks[0].line_no == 4


def test_extract_skips_non_python_fences(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("```bash\necho hi\n```\n")
    assert find_python_blocks(f) == []


def test_extract_honors_skip_marker(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("<!-- skip-validate -->\n```python\nbroken syntax !!!\n```\n")
    assert find_python_blocks(f) == []


PLACEHOLDER_SUBSTITUTIONS = {
    "<org>/<model>": "test-org/test-model",
    "<id>": "test-org/test-model",
}


def substitute_placeholders(content: str) -> str:
    for placeholder, replacement in PLACEHOLDER_SUBSTITUTIONS.items():
        content = content.replace(placeholder, replacement)
    return content


def test_substitute_replaces_org_model():
    assert (
        substitute_placeholders('load("<org>/<model>")')
        == 'load("test-org/test-model")'
    )


def test_substitute_passes_unmatched_text_through():
    assert substitute_placeholders("plain text") == "plain text"


MARKDOWN_FILES = [
    REPO_ROOT / "SKILL.md",
    REPO_ROOT / "references" / "pipeline-tag-patterns.md",
    REPO_ROOT / "references" / "scaffolding-templates.md",
]


def _all_blocks() -> list[CodeBlock]:
    out: list[CodeBlock] = []
    for f in MARKDOWN_FILES:
        if f.exists():
            out.extend(find_python_blocks(f))
    return out


@pytest.mark.parametrize("block", _all_blocks(), ids=lambda b: f"{b.source_file.name}:{b.line_no}")
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
    """Run ruff check on a substituted block. Returns (returncode, combined_output)."""
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


@pytest.mark.parametrize("block", _all_blocks(), ids=lambda b: f"{b.source_file.name}:{b.line_no}")
def test_block_passes_ruff_check(block):
    rc, output = _lint_block(block)
    if rc != 0:
        pytest.fail(
            f"{block.source_file.name}:{block.line_no} ruff failures:\n{output}"
        )
