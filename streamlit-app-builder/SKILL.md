---
name: streamlit-app-builder
description: >
  Generate a production-structured Streamlit app from an existing artifact: a
  local Python script, a Jupyter notebook (local or URL), or a HuggingFace
  model card URL. Triggers: build a Streamlit app for production, wrap a
  notebook into a Streamlit app, generate a UI for a HuggingFace model, any
  link to a `.ipynb` or to `huggingface.co/<org>/<model>`, turn a script into
  a multipage Streamlit app, scaffold a Streamlit app intended for paying
  customers.
---

# Streamlit App Builder

Generate a production-structured Streamlit app package from an existing artifact. Output is a `src/<app_name>/` Python package with a multipage `st.navigation` router, env-based config with fail-fast validation, and linted/typed/tested code — ready to plug into the team's own production infrastructure.

## Non-goals

The skill **does not** produce, and the generated `README.md` documents these as the team's responsibility:

- Authentication / authorization
- Deployment artifacts (Dockerfile, Kubernetes manifests, CI workflows)
- Database or storage layers
- Observability (structured logging infra, metrics, tracing)
- Secrets management at deploy time — the skill covers only the local `.env` source; production secrets come from the deployment platform

## Cross-cutting principles

### 1. Always verify against live `docs.streamlit.io`

Streamlit's API evolves. Before generating code, fetch the current Streamlit docs for high-churn topics and confirm API shapes. Canonical URLs live in `references/streamlit-docs-index.md`.

**Mandatory fetches** before code generation:
- Multipage + `st.navigation` + `st.Page`
- Caching (`@st.cache_resource`, `@st.cache_data`)
- App testing (`streamlit.testing.v1.AppTest`)
- Secrets
- File organization
- Plus the widget page(s) for the classified UI pattern (e.g., `st.chat_input` for chat, `st.audio_input` for ASR)

### 2. Prefer MLX on Apple Silicon

When the source artifact references a model with an MLX-converted equivalent at `huggingface.co/mlx-community/...`, the generated app uses the MLX backend (`mlx-lm` for text, `mlx-vlm` for vision-language, `mlx-whisper` for ASR) when running on `arm64-darwin`, and falls back to `transformers` / `diffusers` / `huggingface_hub` elsewhere.

The MLX lookup is independent of where the skill itself runs. A Linux developer scaffolding from an HF model card URL still produces an app with MLX support wired in — runtime dispatch activates MLX only when a user later runs the app on a Mac.

MLX support is encoded in the generated app as follows:
- `pyproject.toml` declares MLX and transformers with environment markers so `uv sync` installs the right backend per host.
- `src/<app_name>/inference.py` reads `config.IS_APPLE_SILICON` and dispatches to the appropriate backend at runtime.

**MLX model resolution:** query `https://huggingface.co/api/models?author=mlx-community&search=<base-name>` and pick the highest-download-count variant. The chosen variant is noted in `inference.py` with override instructions.
