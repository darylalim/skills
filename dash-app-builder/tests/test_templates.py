"""Static validator for code templates in skill markdown files."""
from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]  # dash-app-builder/

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
    "<org>/<dataset>": "test-org/test-dataset",
}


def substitute_placeholders(content: str) -> str:
    for placeholder, replacement in PLACEHOLDER_SUBSTITUTIONS.items():
        content = content.replace(placeholder, replacement)
    return content


def test_substitute_replaces_org_dataset():
    assert (
        substitute_placeholders('load("<org>/<dataset>")')
        == 'load("test-org/test-dataset")'
    )


def test_substitute_passes_unmatched_text_through():
    assert substitute_placeholders("plain text") == "plain text"


MARKDOWN_FILES = [
    SKILL_ROOT / "SKILL.md",
    SKILL_ROOT / "references" / "scaffolding-templates.md",
]


def _all_blocks() -> list[CodeBlock]:
    out: list[CodeBlock] = []
    for f in MARKDOWN_FILES:
        if f.exists():
            out.extend(find_python_blocks(f))
    return out


@pytest.mark.parametrize(
    "block", _all_blocks(),
    ids=lambda b: f"{b.source_file.name}:{b.line_no}",
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
            capture_output=True, text=True,
        )
        return result.returncode, result.stdout + result.stderr
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    "block", _all_blocks(),
    ids=lambda b: f"{b.source_file.name}:{b.line_no}",
)
def test_block_passes_ruff_check(block):
    rc, output = _lint_block(block)
    if rc != 0:
        pytest.fail(
            f"{block.source_file.name}:{block.line_no} ruff failures:\n{output}"
        )


# === Structural consistency tests ===

SCAFFOLDING_TEMPLATES_MD = SKILL_ROOT / "references" / "scaffolding-templates.md"
SKILL_MD = SKILL_ROOT / "SKILL.md"

TEMPLATE_HEADING_RE = re.compile(r"^## Template (T\d+):", re.MULTILINE)
TEMPLATE_REF_RE = re.compile(r"\bT\d+\b")
DEF_RE = re.compile(r"^def (\w+)\(", re.MULTILINE)


def test_template_references_resolve():
    """Every T<n> referenced in SKILL.md has a `## Template T<n>:` heading
    in scaffolding-templates.md."""
    defined = set(TEMPLATE_HEADING_RE.findall(SCAFFOLDING_TEMPLATES_MD.read_text()))
    referenced = set(TEMPLATE_REF_RE.findall(SKILL_MD.read_text()))
    unresolved = referenced - defined
    assert not unresolved, (
        f"Template names referenced in SKILL.md but not defined in "
        f"scaffolding-templates.md: {sorted(unresolved)}"
    )


WRONG_SKILL_MSG = (
    "This skill builds Dash apps from Hugging Face dataset cards "
    "(`huggingface.co/datasets/<org>/<dataset>`). For scripts, notebooks, or "
    "other inputs, use a general-purpose Dash app prompt without this skill."
)
WRONG_MODALITY_MSG_PREFIX = (
    "This skill builds Dash apps from **tabular** Hugging Face datasets."
)


def test_wrong_skill_rejection_message_present():
    """The wrong-skill rejection message appears verbatim in SKILL.md."""
    text = SKILL_MD.read_text()
    assert WRONG_SKILL_MSG in text, (
        "Wrong-skill rejection message missing or changed in SKILL.md"
    )


def test_wrong_modality_rejection_message_present():
    """The wrong-modality rejection message prefix appears in SKILL.md."""
    text = SKILL_MD.read_text()
    assert WRONG_MODALITY_MSG_PREFIX in text, (
        "Wrong-modality rejection message missing or changed in SKILL.md"
    )


STEP3_FILE_HEADING_RE = re.compile(r"^### File \d+: `([^`]+)`", re.MULTILINE)
CHECKLIST_FILES_RE = re.compile(r"^- \[ \] Four files created: (.+)$", re.MULTILINE)


def test_step3_files_match_output_checklist():
    """The four file paths under SKILL.md Step 3's `### File N:` sub-headings
    match the list in the Output checklist's `Four files created:` line."""
    text = SKILL_MD.read_text()
    step3 = set(STEP3_FILE_HEADING_RE.findall(text))
    checklist_match = CHECKLIST_FILES_RE.search(text)
    assert checklist_match, "Output checklist's 'Four files created:' line not found"
    checklist = set(re.findall(r"`([^`]+)`", checklist_match.group(1)))
    assert step3 == checklist, (
        f"Step 3 sub-headings list {sorted(step3)} but Output checklist lists "
        f"{sorted(checklist)}"
    )


HELPER_FUNCTIONS = ("load_dataframe", "build_filter_for_column", "pick_chart")


def test_helper_function_names_resolve_uniquely():
    """Each helper function name appears exactly once as a `def` across
    scaffolding-templates.md."""
    templates = SCAFFOLDING_TEMPLATES_MD.read_text()
    for fname in HELPER_FUNCTIONS:
        count = len(re.findall(rf"^def {fname}\(", templates, re.MULTILINE))
        assert count == 1, (
            f"Helper `{fname}` defined {count} times in scaffolding-templates.md "
            f"(expected exactly 1)"
        )


EXPECTED_SKIP_VALIDATE_COUNT = 1


def test_skip_validate_marker_count():
    """Total skip-validate markers across both Markdown files is asserted against
    an expected value to catch accidental additions."""
    total = 0
    for f in (SKILL_MD, SCAFFOLDING_TEMPLATES_MD):
        if f.exists():
            total += f.read_text().count(SKIP_MARKER)
    assert total == EXPECTED_SKIP_VALIDATE_COUNT, (
        f"Expected {EXPECTED_SKIP_VALIDATE_COUNT} `<!-- skip-validate -->` markers; "
        f"found {total}. If you added a new skip, update EXPECTED_SKIP_VALIDATE_COUNT."
    )


# === Dash-specific spec-alignment tests ===

TEMPLATE_BLOCK_RE = re.compile(
    r"^## Template (T\d+):.*?\n```python\n(.*?)```",
    re.MULTILINE | re.DOTALL,
)


def _template_blocks() -> dict[str, str]:
    """Map T<n> -> template body source."""
    text = SCAFFOLDING_TEMPLATES_MD.read_text()
    return {m.group(1): m.group(2) for m in TEMPLATE_BLOCK_RE.finditer(text)}


def test_t1_contains_required_env_vars():
    """T1 references DATASET_ID, MAX_ROWS, and HF_TOKEN."""
    t1 = _template_blocks().get("T1", "")
    for name in ("DATASET_ID", "MAX_ROWS", "HF_TOKEN"):
        assert name in t1, f"T1 must reference `{name}`"


def _is_cache_decorator(node: ast.expr) -> bool:
    """True if `node` is a decorator expression naming lru_cache or cache."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id in {"lru_cache", "cache"}
    if isinstance(target, ast.Attribute):
        return target.attr in {"lru_cache", "cache"}
    return False


def test_t1_caches_load_dataframe():
    """The `load_dataframe` definition in T1 carries an lru_cache or cache
    decorator (Name, Call, or Attribute access)."""
    t1 = _template_blocks().get("T1", "")
    tree = ast.parse(substitute_placeholders(t1))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_dataframe":
            assert any(_is_cache_decorator(d) for d in node.decorator_list), (
                "load_dataframe must be decorated with lru_cache or cache"
            )
            return
    pytest.fail("T1 must define `load_dataframe`")


def test_t1_caps_rows():
    """`MAX_ROWS` is referenced inside the `load_dataset(...)` call's argument
    expressions (covers slice syntax, streaming + take, or any other in-call cap)."""
    t1 = _template_blocks().get("T1", "")
    tree = ast.parse(substitute_placeholders(t1))

    def call_uses_max_rows(call: ast.Call) -> bool:
        for child in ast.walk(call):
            if isinstance(child, ast.Name) and child.id == "MAX_ROWS":
                return True
        return False

    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_dataset"
        ):
            assert call_uses_max_rows(node), (
                "load_dataset(...) call must reference MAX_ROWS in its arguments"
            )
            found = True
    assert found, "T1 must call load_dataset(...)"


def test_t2_returns_dash_widgets():
    """`build_filter_for_column` returns dbc.Col(...) or None — every Return."""
    t2 = _template_blocks().get("T2", "")
    tree = ast.parse(substitute_placeholders(t2))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_filter_for_column":
            for ret in (n for n in ast.walk(node) if isinstance(n, ast.Return)):
                v = ret.value
                if v is None or (isinstance(v, ast.Constant) and v.value is None):
                    continue  # bare `return` or `return None`
                if (
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Attribute)
                    and v.func.attr == "Col"
                ):
                    continue  # `dbc.Col(...)`
                pytest.fail(
                    f"build_filter_for_column has unexpected return: "
                    f"{ast.dump(v)[:100]}"
                )
            return
    pytest.fail("T2 must define `build_filter_for_column`")


def test_t3_returns_figure():
    """`pick_chart` is annotated `-> go.Figure` and every return yields a
    `go.Figure(...)` constructor or a `px.<chart>(...)` call."""
    t3 = _template_blocks().get("T3", "")
    tree = ast.parse(substitute_placeholders(t3))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "pick_chart":
            ann = node.returns
            assert ann is not None, "pick_chart must have a return annotation"
            ann_text = ast.unparse(ann)
            assert "Figure" in ann_text, (
                f"pick_chart return annotation must mention Figure, got `{ann_text}`"
            )
            for ret in (n for n in ast.walk(node) if isinstance(n, ast.Return)):
                v = ret.value
                if not isinstance(v, ast.Call):
                    pytest.fail(
                        f"pick_chart returns non-call expression: "
                        f"{ast.unparse(v) if v else '<None>'}"
                    )
                fn_text = ast.unparse(v.func)
                ok = (
                    "Figure" in fn_text  # go.Figure() constructor
                    or fn_text.startswith("px.")  # plotly.express call
                )
                assert ok, (
                    f"pick_chart return must be go.Figure(...) or px.<chart>(...), "
                    f"got `{ast.unparse(v)[:80]}`"
                )
            return
    pytest.fail("T3 must define `pick_chart`")


def test_t4_contains_layout_primitives():
    """T4 contains `dash_table.DataTable(` and `dcc.Graph(` (substring checks
    are sufficient — both are required for the canonical UI)."""
    t4 = _template_blocks().get("T4", "")
    assert "dash_table.DataTable(" in t4, "T4 must contain dash_table.DataTable(...)"
    assert "dcc.Graph(" in t4, "T4 must contain dcc.Graph(...)"


def test_t4_parses_cleanly():
    """T4 has cross-template references that fail F821 in standalone lint
    (and thus carries a skip-validate marker), but it must still parse as
    valid Python — `ast.parse` doesn't care about undefined names."""
    t4 = _template_blocks().get("T4", "")
    ast.parse(substitute_placeholders(t4))


def test_t5_covers_empty_dataframe_edge_case():
    """T5 contains a test function whose name matches `test_pick_chart_empty*`."""
    t5 = _template_blocks().get("T5", "")
    pattern = re.compile(r"^def (test_pick_chart_empty\w*)\(", re.MULTILINE)
    assert pattern.search(t5), (
        "T5 must define a test function named `test_pick_chart_empty*`"
    )
