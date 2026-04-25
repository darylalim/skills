"""Static validator for code templates in skill markdown files."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest  # noqa: F401  # used in Task 4 parametrize

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
