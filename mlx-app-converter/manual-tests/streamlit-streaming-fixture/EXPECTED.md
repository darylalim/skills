# Streamlit streaming fixture — expected post-conversion-attempt state

Unlike the other v2 fixtures, this one is **expected to be REJECTED** by
mlx-app-converter. The skill's v2 streaming soft-reject gate (SKILL.md
Step 3 — TextIteratorStreamer detection + yield+streamer= detection)
should detect `TextIteratorStreamer` and exit without modifying
any file.

After running `mlx-app-converter` on this fixture, walk through each
invariant below. Mark **✓** if it holds, **✗** if not.

## Streaming gate fires

- [ ] **Skill prints the streaming-detected soft-reject message and exits without modifying the file.** Expected message phrasing: `Streaming inference detected at <call site>. v2 supports non-streaming inference only; streaming is planned for a follow-up version.`
- [ ] Skill mentions the call site in its rejection (line number or function name) so the user can locate the streaming pattern.

## Source files unmodified

- [ ] `streamlit_app.py` is byte-identical to the pre-conversion baseline.
- [ ] `pyproject.toml` is byte-identical (no `mlx-lm` or `mlx-vlm` dependency added).
- [ ] `test_streamlit_app.py` is byte-identical (mocks still target `transformers`, not `mlx_*`).
- [ ] No `import mlx_lm` or `import mlx_vlm` introduced anywhere.
- [ ] No Apple Silicon runtime guard inserted at top of file.

## All-soft-rejections exit

This fixture has only one detected model (`meta-llama/Llama-3.1-8B-Instruct`)
and that model hits the streaming gate. So the skill should also print the
file-level all-soft-rejections exit message:

- [ ] Skill prints: `Nothing to convert — every detected model was skipped. The app file was not modified.`

## Verification

- [ ] `git status` shows zero changes after the conversion attempt.
- [ ] `git diff` is empty.

## What this fixture catches

If the streaming gate ever stops firing (e.g., the detection logic in
SKILL.md Step 3 is rewritten and silently regresses), this fixture would
show the skill silently converting the streamer-based source to a
non-streaming `mlx_lm.generate` call — breaking the source's `Iterator[str]`
return contract by replacing it with `str`. The non-modification invariants
above are the canary.
