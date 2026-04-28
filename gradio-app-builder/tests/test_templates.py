"""Static validator for code templates in skill markdown files."""
from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]  # gradio-app-builder/

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
    "<pipeline_tag>": "text-classification",
    "<pinned-version>": "5.0.0",
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
    SKILL_ROOT / "SKILL.md",
    SKILL_ROOT / "references" / "pipeline-tag-patterns.md",
    SKILL_ROOT / "references" / "scaffolding-templates.md",
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


# === Structural consistency tests ===

PIPELINE_TAG_PATTERNS_MD = SKILL_ROOT / "references" / "pipeline-tag-patterns.md"
SCAFFOLDING_TEMPLATES_MD = SKILL_ROOT / "references" / "scaffolding-templates.md"
SKILL_MD = SKILL_ROOT / "SKILL.md"

PIPELINE_TAG_LINE_RE = re.compile(r"^`pipeline_tag`:\s*([^.]+)\.", re.MULTILINE)
BACKTICK_IDENT_RE = re.compile(r"`([a-z][a-z0-9-]*)`")
TEMPLATE_HEADING_RE = re.compile(r"^## Template (T\d+):", re.MULTILINE)
TEMPLATE_REF_RE = re.compile(r"\bT\d+\b")
REJECTED_TAGS_ROW_RE = re.compile(r"^\|\s*`([a-z][a-z0-9-]*)`\s*\|", re.MULTILINE)


def _pipeline_tags_in_catalog() -> set[str]:
    """Tags in pipeline-tag-patterns.md, including the rejected-tags table."""
    text = PIPELINE_TAG_PATTERNS_MD.read_text()
    tags: set[str] = set()
    for m in PIPELINE_TAG_LINE_RE.finditer(text):
        body = m.group(1)
        if "missing or unrecognized" in body:
            continue  # Fallback: General Script — no specific tag
        tags.update(BACKTICK_IDENT_RE.findall(body))
    rejected_section = re.search(
        r"^## Rejected pipeline tags\s*\n(.*?)(?=^## |\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    if rejected_section:
        for row in REJECTED_TAGS_ROW_RE.finditer(rejected_section.group(1)):
            tag = row.group(1)
            if tag != "pipeline_tag":  # skip the table header
                tags.add(tag)
    return tags


def _pipeline_tags_in_skill_routing() -> set[str]:
    """Tags in SKILL.md's Step 4 routing table."""
    text = SKILL_MD.read_text()
    section = re.search(
        r"^### Routing table.*?\n(.*?)(?=^###|\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    if not section:
        return set()
    tags: set[str] = set()
    for line in section.group(1).split("\n"):
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) != 3 or "library_name" in cols[0]:
            continue  # divider, header, or malformed
        tags.update(BACKTICK_IDENT_RE.findall(cols[1]))
    return tags


def test_routing_table_matches_catalog():
    """Every pipeline_tag in pipeline-tag-patterns.md is handled by SKILL.md's routing table
    (or rejected at Step 3), and vice versa."""
    catalog = _pipeline_tags_in_catalog()
    routing = _pipeline_tags_in_skill_routing()
    rejected = {"audio-to-audio"}
    missing_from_routing = (catalog - rejected) - routing
    missing_from_catalog = routing - catalog
    assert not missing_from_routing, (
        f"Tags in catalog but not in routing table: {sorted(missing_from_routing)}"
    )
    assert not missing_from_catalog, (
        f"Tags in routing table but missing from catalog: {sorted(missing_from_catalog)}"
    )


def test_template_references_resolve():
    """Every T<n> referenced in SKILL.md and pipeline-tag-patterns.md has a matching
    `## Template T<n>:` heading in scaffolding-templates.md."""
    defined = set(TEMPLATE_HEADING_RE.findall(SCAFFOLDING_TEMPLATES_MD.read_text()))
    referenced: set[str] = set()
    for f in (SKILL_MD, PIPELINE_TAG_PATTERNS_MD):
        referenced |= set(TEMPLATE_REF_RE.findall(f.read_text()))
    unresolved = referenced - defined
    assert not unresolved, (
        f"Template names referenced but not defined in scaffolding-templates.md: "
        f"{sorted(unresolved)}"
    )


def test_audio_to_audio_rejection_messages_match():
    """The audio-to-audio rejection message in SKILL.md Step 3 matches the one in
    pipeline-tag-patterns.md's rejected-tags table verbatim."""
    msg = (
        "audio-to-audio has no clean transformers pipeline. "
        "This skill can't scaffold a working prototype for audio-to-audio models. "
        "For source separation or speech enhancement, use the model's reference "
        "implementation directly."
    )
    skill_text = SKILL_MD.read_text()
    catalog_text = PIPELINE_TAG_PATTERNS_MD.read_text()
    assert msg in skill_text, (
        "audio-to-audio rejection message missing or changed in SKILL.md Step 3"
    )
    assert msg in catalog_text, (
        "audio-to-audio rejection message missing or changed in "
        "pipeline-tag-patterns.md's rejected-tags table"
    )


def test_skip_validate_markers_cover_all_python_blocks_in_catalog():
    """Every ```python block in pipeline-tag-patterns.md has a `<!-- skip-validate -->`
    marker on the line immediately before — these UI fragments depend on symbols defined
    in the assembled file and would fail F821 if validated standalone."""
    text = PIPELINE_TAG_PATTERNS_MD.read_text()
    total = 0
    skipped = 0
    for m in CODE_BLOCK_RE.finditer(text):
        total += 1
        before = text[: m.start()]
        last_line = before.rstrip("\n").rsplit("\n", 1)[-1] if before else ""
        if SKIP_MARKER in last_line:
            skipped += 1
    assert total == skipped, (
        f"pipeline-tag-patterns.md has {total} ```python blocks but only {skipped} "
        f"have <!-- skip-validate --> markers immediately before them"
    )


def test_no_skip_validate_markers_in_scaffolding_templates():
    """scaffolding-templates.md must have ZERO skip-validate markers preceding
    Python blocks — every T1-T6 block is validated by ast.parse + ruff. A
    marker here would silently drop a template from validation."""
    text = SCAFFOLDING_TEMPLATES_MD.read_text()
    skipped_lines: list[int] = []
    for m in CODE_BLOCK_RE.finditer(text):
        before = text[: m.start()]
        last_line = before.rstrip("\n").rsplit("\n", 1)[-1] if before else ""
        if SKIP_MARKER in last_line:
            skipped_lines.append(before.count("\n") + 1)
    assert not skipped_lines, (
        f"scaffolding-templates.md has unexpected skip-validate markers before "
        f"Python blocks at lines {skipped_lines} — these templates must be validated"
    )


STEP4_FILE_HEADING_RE = re.compile(r"^### File \d+: `([^`]+)`", re.MULTILINE)
CHECKLIST_FILES_RE = re.compile(r"^- \[ \] Five files created: (.+)$", re.MULTILINE)
INFERENCE_FN_REF_RE = re.compile(r"Use (?:scaffolding )?template T\d+ \(`(\w+)`\)")
DEF_RE = re.compile(r"^def (\w+)\(", re.MULTILINE)


def test_step4_files_match_output_checklist():
    """The five file paths under SKILL.md Step 4's `### File N:` sub-headings match
    the list in the Output checklist's `Five files created:` line."""
    text = SKILL_MD.read_text()
    step4 = set(STEP4_FILE_HEADING_RE.findall(text))
    checklist_match = CHECKLIST_FILES_RE.search(text)
    assert checklist_match, "Output checklist's 'Five files created:' line not found"
    checklist = set(re.findall(r"`([^`]+)`", checklist_match.group(1)))
    assert step4 == checklist, (
        f"Step 4 sub-headings list {sorted(step4)} but Output checklist lists "
        f"{sorted(checklist)}"
    )


def test_inference_function_names_resolve():
    """Every inference-function name mentioned in pipeline-tag-patterns.md's
    `Use template Tn (\\`name\\`)` lines has a matching `def name(` in
    scaffolding-templates.md."""
    catalog = PIPELINE_TAG_PATTERNS_MD.read_text()
    templates = SCAFFOLDING_TEMPLATES_MD.read_text()
    referenced = set(INFERENCE_FN_REF_RE.findall(catalog))
    defined = set(DEF_RE.findall(templates))
    unresolved = referenced - defined
    assert not unresolved, (
        f"Inference function names referenced in pipeline-tag-patterns.md but not "
        f"defined in scaffolding-templates.md: {sorted(unresolved)}"
    )


# === Gradio-specific structural checks ===

SEMVER_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")
README_FRONTMATTER_RE = re.compile(
    r"^```\n---\n(.*?)\n---\n```",
    re.MULTILINE | re.DOTALL,
)


def _extract_readme_frontmatter_text() -> str:
    """Return the raw YAML lines inside SKILL.md's File 3 frontmatter block."""
    text = SKILL_MD.read_text()
    file3_section = re.search(
        r"^### File 3: `README\.md`.*?\n(.*?)(?=^### |\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    assert file3_section, (
        "SKILL.md is missing the `### File 3: \\`README.md\\`` section"
    )
    match = README_FRONTMATTER_RE.search(file3_section.group(1))
    assert match, "README.md frontmatter block not found in File 3 section"
    return match.group(1)


def _frontmatter_field(field: str) -> str | None:
    """Return the trimmed value of a top-level YAML key, or None if absent."""
    for line in _extract_readme_frontmatter_text().split("\n"):
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return None


def test_readme_frontmatter_declares_gradio_sdk():
    sdk = _frontmatter_field("sdk")
    assert sdk == "gradio", f"Expected sdk: gradio, got {sdk!r}"


def test_readme_frontmatter_has_sdk_version_placeholder():
    sdk_version = _frontmatter_field("sdk_version")
    assert sdk_version is not None, "sdk_version missing from README.md frontmatter"
    # Either the literal placeholder, or a concrete semver-like version.
    assert sdk_version == "<pinned-version>" or SEMVER_RE.match(sdk_version), (
        f"sdk_version {sdk_version!r} is neither the documented placeholder "
        f"nor a semver string"
    )


def test_readme_frontmatter_app_file_default():
    # app_file should be absent (defaults to app.py) or explicitly "app.py".
    app_file = _frontmatter_field("app_file")
    if app_file is not None:
        assert app_file in ("app.py", '"app.py"', "'app.py'"), (
            f"app_file must be omitted or 'app.py', got {app_file!r}"
        )


def test_t2_uses_chat_interface():
    """T2 must use gr.ChatInterface, not gr.Blocks. Catches drift toward streamlit-style
    chat patterns in pipeline-tag-patterns.md's text-generation section."""
    catalog = PIPELINE_TAG_PATTERNS_MD.read_text()
    section = re.search(
        r"^## Text generation / chat\s*\n.*?\n(.*?)(?=^## |\Z)",
        catalog, re.MULTILINE | re.DOTALL,
    )
    assert section, (
        "Text generation / chat section missing from pipeline-tag-patterns.md"
    )
    body = section.group(1)
    assert "gr.ChatInterface(" in body, (
        "Text generation / chat section must use gr.ChatInterface — "
        "found no `gr.ChatInterface(` call"
    )
    assert "gr.Blocks(" not in body, (
        "Text generation / chat section must NOT use gr.Blocks — "
        "use gr.ChatInterface instead"
    )


def _template_section(name: str) -> str:
    """Return the body of a `## Template Tn:` section from scaffolding-templates.md."""
    templates = SCAFFOLDING_TEMPLATES_MD.read_text()
    section = re.search(
        rf"^## Template {name}:.*?\n(.*?)(?=^## |\Z)",
        templates, re.MULTILINE | re.DOTALL,
    )
    assert section, f"Template {name} section missing from scaffolding-templates.md"
    return section.group(1)


def test_t3_uses_guidance_scale():
    """T3 (text-to-image) must expose `guidance_scale`, not `seed`. guidance_scale
    is the primary quality/speed knob for text-to-image diffusers pipelines."""
    body = _template_section("T3")
    assert "guidance_scale" in body, "T3 must expose guidance_scale"
    assert not re.search(r"\bseed\b", body), (
        "T3 must not reference seed — guidance_scale is the spec-aligned parameter"
    )


def test_t4_uses_strength():
    """T4 (image-to-image) must expose `strength`, not `seed`. strength is the
    primary control axis for image-to-image diffusers pipelines."""
    body = _template_section("T4")
    assert "strength" in body, "T4 must expose strength"
    assert not re.search(r"\bseed\b", body), (
        "T4 must not reference seed — strength is the spec-aligned parameter"
    )


def test_t5_returns_cosine_similarity():
    """T5 (sentence-transformers) must compute cosine similarity from two text
    inputs and return a float. The function takes two text arguments, imports
    `cos_sim`, and returns a float."""
    body = _template_section("T5")
    assert "from sentence_transformers.util import cos_sim" in body, (
        "T5 must import cos_sim from sentence_transformers.util"
    )
    # Check name + arg names + return type separately so future line wraps
    # don't break the test.
    assert "def embed(" in body, "T5 must define a function named `embed`"
    assert "text_a: str" in body, "T5's embed must accept `text_a: str`"
    assert "text_b: str" in body, "T5's embed must accept `text_b: str`"
    assert "-> float:" in body, "T5's embed must return float"


WRONG_SKILL_MSG = (
    "Pass an HF model card URL (`huggingface.co/<org>/<model>`). "
    "For Python scripts or notebooks on GitHub, use a general-purpose "
    "Gradio prompt without this skill."
)


def test_wrong_skill_rejection_message_present():
    """The wrong-skill rejection message appears verbatim in SKILL.md.
    Locks in the post-narrowing wording so future drift is caught."""
    text = SKILL_MD.read_text()
    assert WRONG_SKILL_MSG in text, (
        "Wrong-skill rejection message missing or changed in SKILL.md"
    )
