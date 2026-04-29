# MLX App Converter — Manual Integration Tests

The static tests in `mlx-app-converter/tests/` verify documentation invariants
but do not exercise the actual rewrite end-to-end. This directory contains
two small fixture apps and expected-output checklists so a human can confirm
that the skill produces correct results against a known input.

**These are NOT auto-run by pytest.** Run them manually when changing the
templates or rewrite logic.

---

## Directory layout

```
manual-tests/
├── README.md                           # this file
├── streamlit-llm-fixture/
│   ├── streamlit_app.py                # input: transformers-based Streamlit chat app
│   ├── pyproject.toml                  # uv-managed deps
│   ├── test_streamlit_app.py           # mocked inference test (pre-conversion)
│   └── EXPECTED.md                     # post-rewrite invariants checklist
└── gradio-llm-fixture/
    ├── app.py                          # input: transformers-based Gradio chat app
    ├── requirements.txt                # pip deps
    ├── test_app.py                     # mocked inference test (pre-conversion)
    └── EXPECTED.md                     # post-rewrite invariants checklist
```

---

## How to run a manual test

### Step 1 — Pick a fixture

Choose either `streamlit-llm-fixture` (Streamlit + `pyproject.toml`) or
`gradio-llm-fixture` (Gradio + `requirements.txt`). Run both to get full
coverage across the two framework paths.

### Step 2 — Copy to a scratch location

```bash
cp -r mlx-app-converter/manual-tests/streamlit-llm-fixture/ /tmp/scratch-streamlit/
# or
cp -r mlx-app-converter/manual-tests/gradio-llm-fixture/ /tmp/scratch-gradio/
```

### Step 3 — Initialise a clean git tree

The skill checks for a clean working tree before rewriting. Set that up:

```bash
cd /tmp/scratch-streamlit   # or scratch-gradio
git init
git add .
git commit -m "fixture: pre-conversion baseline"
```

### Step 4 — Invoke Claude Code

Open Claude Code in the scratch directory and run a conversion prompt:

```
Convert this app to MLX.
```

Or more explicitly:

```
This is a Streamlit app that uses transformers for inference. Convert it to
use mlx-lm so it runs on my Apple Silicon Mac.
```

The skill will:
1. Detect the framework and the `run_inference` function.
2. Present a variant-selection matrix for the model
   (`meta-llama/Llama-3.1-8B-Instruct`).
3. Rewrite `streamlit_app.py` (or `app.py`) and the test file.
4. Print the dep-update command (`uv add mlx-lm` for Streamlit; append line
   for Gradio).

### Step 5 — Walk through the EXPECTED.md checklist

Open the fixture's `EXPECTED.md` and go through each item:

- Mark **✓** if the invariant holds in the actual output.
- Mark **✗** if it does not.

Each section maps to one template (T1 loader, T2 inference, T3 platform guard,
T4 test rewrite, T5 dep manifest).

### Step 6 — Triage mismatches

If an invariant is not met, decide whether it is:

- A **real bug** in the skill (e.g., `@st.cache_resource` was stripped, or
  sampling kwargs were not routed through `make_sampler`). File a fix.
- **Stylistic variation** that does not affect correctness (e.g., blank-line
  count, comment wording). No action needed.

---

## What this catches

These are failure modes that static tests cannot surface — they require
actually running the skill against real input:

- **Cache decorator stripped accidentally** — T1 preservation rule: the
  decorator must survive the body rewrite exactly as written.
- **Sampling kwargs passed directly instead of through `make_sampler`** —
  `mlx_lm.generate` does not accept `temperature` or `top_p` directly;
  passing them causes a `TypeError` at runtime.
- **Test file mocks not updated** — if `AutoTokenizer.from_pretrained` is
  still being patched after conversion, the test will fail with an
  `ImportError` because `transformers` is no longer imported.
- **Apple Silicon guard placed wrong** — inserting the guard *after*
  `import mlx_lm` means x86 hosts get an `ImportError` (MLX not installed)
  instead of the cleaner `RuntimeError` that the guard is meant to produce.
  The guard must come before any MLX import.
- **Dep manifest path picked incorrectly** — when a Gradio app has both a
  `pyproject.toml` and a `requirements.txt`, T5 must append to
  `requirements.txt` (Spaces convention), not call `uv add`.

---

## When to run this

Run the manual tests in these situations:

- After modifying any T1–T5 template in `references/rewrite-templates.md`.
- After modifying `SKILL.md` Step 5 (the rewrite logic section).
- After modifying `lib/variant_resolution.py`.
- Before pushing significant changes upstream to verify end-to-end correctness.
