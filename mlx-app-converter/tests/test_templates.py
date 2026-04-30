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


def test_variant_resolution_documents_reply_validation():
    """variant-resolution.md must have a 'Reply validation' section covering
    malformed replies (missing @, unrecognized cell, typos)."""
    text = VARIANT_RESOLUTION_MD.read_text()
    assert "## Reply validation" in text, (
        "variant-resolution.md missing '## Reply validation' section"
    )
    assert "Missing `@`" in text, (
        "Reply validation must cover the missing-@ case"
    )
    assert "Unrecognized cell" in text, (
        "Reply validation must cover the unrecognized-cell case"
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


def _vlm_subsection(template_name: str, *, scope: str = "loose") -> str:
    """Return the VLM subsection of template T<N>.

    scope='loose'  → from '### VLM' onwards (covers VLM form + any subsequent
                     shared subsections that follow it).
    scope='strict' → exactly the '### VLM form (...)' subsection up to the next
                     '### ' heading. Use when negative assertions must not
                     accidentally match shared-subsection text.
    """
    section = _template_section(template_name)
    parts = section.split("### VLM", 1)
    assert len(parts) == 2, (
        f"Template {template_name} missing '### VLM' subsection"
    )
    body = parts[1]
    if scope == "strict":
        body = body.split("### ", 1)[0]
    return body


def test_t1_preserves_cache_decorators():
    """T1 must show both Streamlit (@st.cache_resource) and Gradio (@lru_cache) variants."""
    section = _template_section("T1")
    assert "@st.cache_resource" in section, "T1 missing @st.cache_resource"
    assert "@lru_cache(maxsize=1)" in section, "T1 missing @lru_cache(maxsize=1)"


def test_t1_has_vlm_subsection():
    """T1 must contain a VLM subsection covering both Streamlit and Gradio."""
    section = _template_section("T1")
    assert "### VLM" in section, "T1 missing '### VLM' subsection"


def test_t1_vlm_uses_mlx_vlm_load():
    """T1 VLM form must use mlx_vlm.load (not from_pretrained)."""
    vlm_section = _vlm_subsection("T1")
    assert "mlx_vlm.load" in vlm_section, (
        "T1 VLM form missing mlx_vlm.load loader call"
    )
    assert "AutoProcessor" in vlm_section, (
        "T1 VLM form missing AutoProcessor source-pattern reference"
    )


def test_t1_vlm_preserves_cache_decorators():
    """T1 VLM form must preserve both Streamlit and Gradio cache decorators."""
    vlm_section = _vlm_subsection("T1")
    assert "@st.cache_resource" in vlm_section, (
        "T1 VLM form missing @st.cache_resource (Streamlit cache decorator)"
    )
    assert "@lru_cache(maxsize=1)" in vlm_section, (
        "T1 VLM form missing @lru_cache(maxsize=1) (Gradio cache decorator)"
    )


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


def test_t2_has_vlm_subsection():
    """T2 must contain a VLM subsection covering mlx_vlm.generate."""
    section = _template_section("T2")
    assert "### VLM" in section, "T2 missing '### VLM' subsection"


def test_t2_vlm_uses_mlx_vlm_generate():
    """T2 VLM form must show the mlx_vlm.generate call shape with image arg."""
    vlm_section = _vlm_subsection("T2")
    assert "mlx_vlm.generate(" in vlm_section, (
        "T2 VLM form missing mlx_vlm.generate call"
    )
    assert "image" in vlm_section, (
        "T2 VLM form missing image arg in inference signature"
    )


def test_t2_vlm_extracts_text_from_generation_result():
    """T2 VLM form must extract .text from mlx_vlm.generate's GenerationResult.
    mlx_vlm.generate returns a dataclass with .text attribute, not a bare str —
    preserving the source's str return type requires .text extraction."""
    vlm_section = _vlm_subsection("T2")
    assert ".text" in vlm_section, (
        "T2 VLM form missing .text extraction — mlx_vlm.generate returns "
        "GenerationResult, not a str. Use result.text to preserve source contract."
    )


def test_t2_vlm_uses_direct_sampling_kwargs():
    """T2 VLM form must document that sampling kwargs (temperature, top_p,
    top_k, repetition_penalty) are passed DIRECTLY to mlx_vlm.generate —
    NOT via make_sampler / make_logits_processors helpers (those are mlx-lm).
    """
    vlm_form_only = _vlm_subsection("T2", scope="strict")
    assert "temperature" in vlm_form_only, (
        "T2 VLM missing 'temperature' kwarg (must be direct kwarg, not "
        "helper-constructed)"
    )
    assert "make_sampler" not in vlm_form_only, (
        "T2 VLM form must NOT use make_sampler — that's an mlx-lm helper. "
        "mlx-vlm accepts sampling kwargs directly."
    )
    assert "make_logits_processors" not in vlm_form_only, (
        "T2 VLM form must NOT use make_logits_processors — mlx-vlm accepts "
        "repetition_penalty as a direct kwarg."
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


def test_t3_documents_arm64_test_side_effect():
    """T3 must note that the runtime guard makes the rewritten test file
    effectively arm64-only at import time (pytest collection raises
    RuntimeError on x86 CI)."""
    section = _template_section("T3")
    assert "arm64-only" in section, (
        "T3 missing 'arm64-only' note about test-collection side effect"
    )


def test_t4_mocks_mlx_lm_load():
    """T4 must show the mock target as mlx_lm.load (not from_pretrained)."""
    section = _template_section("T4")
    assert "mlx_lm.load" in section, "T4 missing mlx_lm.load as mock target"


def test_t4_has_vlm_subsection():
    """T4 must contain a VLM subsection covering mlx_vlm.load mock target."""
    section = _template_section("T4")
    assert "### VLM" in section, "T4 missing '### VLM' subsection"


def test_t4_vlm_mocks_mlx_vlm_load():
    """T4 VLM form must show patch target as <module>.mlx_vlm.load."""
    vlm_section = _vlm_subsection("T4")
    assert "mlx_vlm.load" in vlm_section, (
        "T4 VLM form missing mlx_vlm.load as mock target"
    )
    assert "mock_processor" in vlm_section, (
        "T4 VLM form missing mock_processor in mock return tuple"
    )


def test_t4_vlm_mock_returns_text_attribute():
    """T4 VLM form's mlx_vlm.generate mock must return an object with .text
    attribute, matching mlx_vlm's actual GenerationResult dataclass. A bare
    string return will not match the real API and the test will mislead."""
    vlm_section = _vlm_subsection("T4")
    assert ".text" in vlm_section, (
        "T4 VLM form missing .text attribute on mlx_vlm.generate's mock — "
        "the real GenerationResult has .text, not a bare str return."
    )


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
    """T5 must include the transformers/torch removal hint phrase."""
    section = _template_section("T5")
    assert "transformers and torch are no longer needed for inference" in section, (
        "T5 missing the standard removal-hint phrase"
    )


def test_t5_uses_framework_driven_dep_detection():
    """T5 must pick the dep manifest by framework (not by file presence) so
    that Streamlit-on-Spaces (which has both pyproject.toml and requirements.txt)
    routes correctly."""
    section = _template_section("T5")
    assert "Framework = Streamlit" in section, (
        "T5 missing 'Framework = Streamlit' branch — should pick by framework"
    )
    assert "Framework = Gradio" in section, (
        "T5 missing 'Framework = Gradio' branch — should pick by framework"
    )


def test_t5_warns_about_multi_file_imports():
    """T5 removal hint must hedge: other files in the project may still import
    transformers/torch (e.g., tokenizer-only uses), so the user must audit
    before removing the deps."""
    section = _template_section("T5")
    assert "other files in your project may still import them" in section, (
        "T5 removal hint missing the multi-file caveat"
    )
    # Line-broken in the rendered hint, so check fragments separately.
    assert "do NOT run a removal" in section, (
        "T5 removal hint missing the explicit don't-run-removal warning"
    )


def test_t5_documents_modality_set_parameterization():
    """T5 must parameterize the dep delta on the set of target packages
    (LLM only, VLM only, multi-modal). Each modality combination must show
    in the section."""
    section = _template_section("T5")
    # LLM-only case (existing)
    assert "uv add mlx-lm" in section, (
        "T5 missing 'uv add mlx-lm' for LLM-only case"
    )
    # VLM-only case (new)
    assert "uv add mlx-vlm" in section, (
        "T5 missing 'uv add mlx-vlm' for VLM-only case"
    )
    # Multi-modal case (new)
    assert "uv add mlx-lm mlx-vlm" in section, (
        "T5 missing 'uv add mlx-lm mlx-vlm' for multi-modal case"
    )


def test_t5_requirements_txt_handles_both_packages():
    """T5 must show that requirements.txt receives both mlx-lm and mlx-vlm
    lines for multi-modal Gradio apps."""
    section = _template_section("T5")
    assert "mlx-vlm" in section, (
        "T5 missing mlx-vlm reference for Gradio multi-modal case"
    )


def test_t5_modality_set_table_present():
    """T5 must contain a parameterization table mapping detected modalities
    to target package set."""
    section = _template_section("T5")
    assert "{mlx-lm}" in section or "{mlx-vlm}" in section, (
        "T5 missing modality-set notation in parameterization table"
    )


# === Structural consistency tests ===

SKILL_MD = SKILL_ROOT / "SKILL.md"

TEMPLATE_HEADING_RE = re.compile(r"^## Template (T\d+):", re.MULTILINE)
TEMPLATE_REF_RE = re.compile(r"\bT\d+\b")

REJECTION_MESSAGES = [
    # gate 1 — wrong skill
    "mlx-app-converter operates on an existing Streamlit or Gradio app file in the current working directory. For other inputs, use the appropriate skill (streamlit-app-builder, gradio-app-builder) or a general-purpose prompt.",
    # gate 2 — hardware
    "MLX requires Apple Silicon (arm64 macOS). Detected: <machine>/<system>. Run this skill on an Apple Silicon Mac.",
    # gate 3 — app file not found
    "No app file found. Expected one of app.py, streamlit_app.py, gradio_app.py in the current directory.",
    # gate 3' — multiple app files
    "Found multiple app files: <list>. Tell me which one to convert.",
    # gate 4 — framework
    "does not import streamlit or gradio at the top level. mlx-app-converter only supports Streamlit and Gradio apps.",
    # gate 5 — git-clean
    "Commit or stash before running mlx-app-converter so the rewrite is reviewable via git diff.",
    # gate 6 — no detectable models
    "No HF model IDs found in",
    "mlx-app-converter requires statically-known model IDs (string literal or simple constant). Dynamic IDs (env var, UI input) are not supported in v1.",
    # gate 7 — no MLX variants (soft)
    "No MLX variants found for",
    "Pick one or reply \"skip\" to leave this model unchanged.",
    # gate 8 — dynamic model arg (soft)
    "model ID is dynamic (env var or runtime input). v1 supports only statically-known model IDs.",
]


def test_template_references_resolve():
    """Every T<n> in SKILL.md has a matching `## Template T<n>:` heading in
    rewrite-templates.md."""
    defined = set(TEMPLATE_HEADING_RE.findall(REWRITE_TEMPLATES_MD.read_text()))
    referenced = set(TEMPLATE_REF_RE.findall(SKILL_MD.read_text()))
    unresolved = referenced - defined
    assert not unresolved, (
        f"Template names referenced in SKILL.md but not defined in "
        f"rewrite-templates.md: {sorted(unresolved)}"
    )


def test_all_rejection_messages_present_in_skill_md():
    """Every rejection message snippet from the spec appears in SKILL.md."""
    skill_text = SKILL_MD.read_text()
    missing = [msg for msg in REJECTION_MESSAGES if msg not in skill_text]
    assert not missing, (
        "Rejection messages missing from SKILL.md:\n  - "
        + "\n  - ".join(missing)
    )


def test_skill_md_step3_documents_model_id_dedup():
    """SKILL.md Step 3 must document that detected model IDs are deduplicated
    by string value before matrix construction (so two from_pretrained calls
    referencing the same MODEL_ID produce one matrix prompt, not two)."""
    text = SKILL_MD.read_text()
    assert "Deduplicate by model ID" in text, (
        "SKILL.md missing model-ID deduplication note in Step 3"
    )


def test_skill_md_step5_t2_describes_sampler_routing():
    """SKILL.md Step 5's T2 description must name the sampler-helper route
    (regression guard against the original misleading 'temperature → temp'
    direct-rename phrasing — sampling kwargs are not direct kwargs to
    mlx_lm.generate)."""
    text = SKILL_MD.read_text()
    assert "make_sampler" in text, (
        "SKILL.md missing 'make_sampler' — Step 5 T2 description must "
        "name the helper route for sampling kwargs"
    )
    assert "make_logits_processors" in text, (
        "SKILL.md missing 'make_logits_processors' — Step 5 T2 description "
        "must name the helper route for repetition_penalty"
    )


def test_skill_md_step3_documents_all_soft_rejections_exit():
    """SKILL.md Step 3 must document the exit behavior when every detected
    model hits a soft rejection (all dynamic args, or all skipped via no-match
    fallback) — otherwise the skill flow is undefined for that case."""
    text = SKILL_MD.read_text()
    assert "Nothing to convert" in text, (
        "SKILL.md missing the 'Nothing to convert' exit message for "
        "all-soft-rejections case"
    )


def test_skip_validate_marker_count_is_zero():
    """rewrite-templates.md must have zero `<!-- skip-validate -->` markers.
    Every code block must be valid standalone."""
    text = REWRITE_TEMPLATES_MD.read_text()
    count = text.count(SKIP_MARKER)
    assert count == 0, (
        f"rewrite-templates.md has {count} skip-validate markers; "
        f"expected 0. Fix the underlying template instead."
    )


_FILE_TOKEN_RE = re.compile(r"`([^`]+\.(?:py|toml|txt|example))`")


def _files_in_outputs_section() -> set[str]:
    """Extract file references from the SKILL.md '## Outputs (in-place edits)' section.

    Captures all backtick-wrapped tokens on each bullet line that look like
    filenames (have a recognised extension), so multi-token bullets like
    '`requirements.txt` or `pyproject.toml`' are fully covered while
    non-file inline-code tokens (package names, import paths) are excluded.
    """
    text = SKILL_MD.read_text()
    section = re.search(
        r"^## Outputs \(in-place edits\).*?\n(.*?)(?=^## |\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    assert section, "SKILL.md missing '## Outputs (in-place edits)' section"
    files: set[str] = set()
    for line in section.group(1).splitlines():
        if line.startswith("- "):
            files.update(_FILE_TOKEN_RE.findall(line))
    return files


def _files_in_workflow_section() -> set[str]:
    """Extract file references from the SKILL.md '## Workflow' section."""
    text = SKILL_MD.read_text()
    section = re.search(
        r"^## Workflow.*?\n(.*?)(?=^## |\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    assert section, "SKILL.md missing '## Workflow' section"
    return set(re.findall(r"`([^`]+\.(?:py|toml|txt|example))`", section.group(1)))


def test_outputs_section_files_referenced_in_workflow():
    """Every concrete file mentioned in '## Outputs' is also referenced in '## Workflow'.

    The check is one-way: outputs ⊆ workflow. The workflow may mention
    additional files (intermediate paths) that aren't outputs; that's OK.
    """
    outputs_files = _files_in_outputs_section()
    workflow_files = _files_in_workflow_section()

    # Normalize: outputs may use generic patterns ("test_*.py", "<app file>");
    # for parity, only check files that look like concrete paths.
    concrete_outputs = {
        f for f in outputs_files
        if not f.startswith("<") and "*" not in f
    }
    missing = concrete_outputs - workflow_files
    assert not missing, (
        f"Files in Outputs section but not referenced in Workflow: {sorted(missing)}"
    )
