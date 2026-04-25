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
    "<app_name>": "app",
    "<App Name>": "App",
    "<app-name>": "app",
    "<default_height>": "1024",
    "<default_steps>": "20",
    "<default_width>": "1024",
    "<org>/<model>": "test-org/test-model",
    "<mlx-community/...>": "mlx-community/test",
    "<family>": "flux2",
    "<feature>": "feature",
    "<ModelClass>": "Model",
    "<id>": "test-org/test-model",
    "<name>": "name",
}


def substitute_placeholders(content: str) -> str:
    for placeholder, replacement in PLACEHOLDER_SUBSTITUTIONS.items():
        content = content.replace(placeholder, replacement)
    return content


def test_substitute_replaces_app_name():
    assert substitute_placeholders("from <app_name> import x") == "from app import x"


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
    REPO_ROOT / "references" / "mflux-families.md",
]


def _all_blocks() -> list[CodeBlock]:
    out: list[CodeBlock] = []
    for f in MARKDOWN_FILES:
        if f.exists():
            out.extend(find_python_blocks(f))
    # Append scaffolding-templates.md once it exists (added in Task 8).
    extra = REPO_ROOT / "references" / "scaffolding-templates.md"
    if extra.exists():
        out.extend(find_python_blocks(extra))
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


# Canonical model IDs from mflux-families.md prose. Each entry: (id, expected_family).
CANONICAL_MFLUX_IDS: list[tuple[str, str]] = [
    ("black-forest-labs/FLUX.1-schnell", "flux"),
    ("black-forest-labs/FLUX.1-dev", "flux"),
    ("black-forest-labs/FLUX.1-Kontext-dev", "flux"),
    ("black-forest-labs/FLUX.2-Klein", "flux2"),
    ("black-forest-labs/FLUX.2-Klein-Edit", "flux2"),
    ("Qwen/Qwen-Image", "qwen_image"),
    ("Qwen/Qwen-Image-Edit", "qwen_image"),
    ("Qwen/Qwen-Image-Edit-2509", "qwen_image"),
    ("briaai/FIBO", "fibo"),
    ("briaai/FIBO-lite", "fibo"),
    ("briaai/Fibo-Edit", "fibo"),
    ("briaai/Fibo-Edit-RMBG", "fibo"),
    ("Tongyi-MAI/Z-Image", "z_image"),
    ("filipstrand/Z-Image-Turbo-mflux-4bit", "z_image"),
]


MFLUX_TABLE_ROW_RE = re.compile(
    r"^\|\s*`(?P<key>\w+)`\s*\|\s*`(?P<regex>[^`]+)`\s*\|", re.MULTILINE
)


def _load_mflux_routing_table() -> list[tuple[str, re.Pattern[str]]]:
    text = (REPO_ROOT / "references" / "mflux-families.md").read_text()
    rows: list[tuple[str, re.Pattern[str]]] = []
    for m in MFLUX_TABLE_ROW_RE.finditer(text):
        key = m.group("key")
        # Skip the header divider row (regex column would not be a real regex).
        if key in {"Family key", "key"}:
            continue
        # Markdown tables require | to be escaped as \| to avoid breaking columns.
        # Unescape before compiling so alternation works correctly.
        raw_regex = m.group("regex").replace(r"\|", "|")
        try:
            pattern = re.compile(raw_regex)
        except re.error:
            continue  # not a routing-table row
        rows.append((key, pattern))
    return rows


@pytest.mark.parametrize("model_id,expected_family", CANONICAL_MFLUX_IDS)
def test_canonical_id_matches_expected_family(model_id, expected_family):
    rows = _load_mflux_routing_table()
    matched: list[str] = [key for key, pat in rows if pat.match(model_id)]
    assert matched, f"{model_id} matched no row in mflux-families.md Part A"
    assert matched[0] == expected_family, (
        f"{model_id} expected to match {expected_family} but first match was "
        f"{matched[0]} (full match list: {matched})"
    )
