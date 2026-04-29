# Rewrite Templates

This document defines the code transformations `mlx-app-converter` applies to convert a Streamlit or Gradio app from `transformers`-based inference to MLX-based inference. Each template has a before/after pair plus a list of preservation rules.

Templates are referenced from `SKILL.md` by name (T1–T5).

## Template T1: Loader

Replace the `transformers`-based model-loading function with an `mlx_lm.load`-based equivalent. The cache decorator (`@st.cache_resource` for Streamlit, `@lru_cache(maxsize=1)` for Gradio) is preserved verbatim.

**Streamlit form (before):**

```python
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tokenizer
```

**Streamlit form (after):**

```python
import mlx_lm
import streamlit as st

MODEL_ID = "mlx-community/Llama-3.1-8B-Instruct-bf16"


@st.cache_resource
def load_model():
    return mlx_lm.load(MODEL_ID)
```

**Gradio form (before):**

```python
from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tokenizer
```

**Gradio form (after):**

```python
from functools import lru_cache

import mlx_lm

MODEL_ID = "mlx-community/Llama-3.1-8B-Instruct-bf16"


@lru_cache(maxsize=1)
def load_model():
    return mlx_lm.load(MODEL_ID)
```

**Preservation rules:**
- The cache decorator (`@st.cache_resource` or `@lru_cache(maxsize=1)`) is preserved verbatim — including any user-supplied arguments to it.
- The `MODEL_ID` constant name is preserved; only its value is rewritten to the user-selected MLX variant.
- The function name (`load_model` in the canonical case) and its return shape (`(model, tokenizer)` tuple) are preserved — `mlx_lm.load` returns the same tuple shape.

**Import handling:**
- Remove `from transformers import AutoTokenizer, AutoModelForCausalLM` (and any subset).
- Remove unused `from transformers import AutoTokenizer` / `AutoModelForCausalLM` standalone imports.
- Add `import mlx_lm` if not already present.
- Other `transformers` imports (e.g., `pipeline`, `AutoConfig` used elsewhere in the file) are left alone.
