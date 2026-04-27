# Scaffolding templates

Inference-function templates consumed by `SKILL.md` Step 4. Each template is a complete `load_model()` + inference-function pair, paste-ready into the single-file `app.py`. Pick the template by `(library_name, pipeline_tag)` per the routing table in `SKILL.md`.

All templates assume:
- `MODEL_ID: str` is a module-level constant (hard-coded at scaffold time).
- `import gradio as gr` and `import os` are already in the file header.
- For gated models, the `app.py` header already performs the `HF_TOKEN` check via the gated-model gate snippet before any of these functions run.
- `@lru_cache(maxsize=1)` from `functools` decorates `load_model` so the model loads once per process.

## Template T1: transformers `pipeline()` (covers most pipeline tags)

Used for `library_name = transformers` and `pipeline_tag` in:
`text-classification`, `zero-shot-classification`, `token-classification`, `question-answering`, `summarization`, `translation`, `feature-extraction` (when not sentence-transformers), `automatic-speech-recognition`, `text-to-speech`, `image-classification`, `object-detection`, `image-to-text`, `image-text-to-text`.

```python
import os
from functools import lru_cache

from transformers import pipeline

MODEL_ID = "<org>/<model>"
PIPELINE_TASK = "<pipeline_tag>"  # e.g. "text-classification"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the transformers pipeline."""
    return pipeline(PIPELINE_TASK, model=MODEL_ID, token=os.getenv("HF_TOKEN"))


def run_inference(*args, **kwargs):
    """Invoke the pipeline. Pass kwargs matching the pipeline's call signature."""
    return load_model()(*args, **kwargs)
```

Per-tag invocation hint (for the UI body in `pipeline-tag-patterns.md`):
- `text-classification` / `zero-shot-classification` / `token-classification` / `summarization` / `translation` / `feature-extraction`: `run_inference(text)`
- `question-answering`: `run_inference(question=q, context=c)`
- `automatic-speech-recognition`: `run_inference(audio_path)` — returns dict with `text` key
- `text-to-speech`: `run_inference(text)` — returns dict with `audio` (numpy array) and `sampling_rate`
- `image-classification` / `object-detection` / `image-to-text`: `run_inference(image)` — accepts PIL or path
- `image-text-to-text`: `run_inference(text=prompt, images=image)` — argument names vary by model; check the model card

## Template T2: transformers causal-LM with `gr.ChatInterface`

Used for `library_name = transformers` and `pipeline_tag` in `{text-generation, conversational}`.

```python
import os
from functools import lru_cache

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "<org>/<model>"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the causal-LM model and tokenizer."""
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=token)
    return model, tokenizer


def chat(message: str, history: list[dict[str, str]]) -> str:
    """Generate a response from history + new message.

    Designed for `gr.ChatInterface(fn=chat, type="messages")`, where `history`
    is a list of `{"role": ..., "content": ...}` dicts. When the tokenizer
    lacks a `chat_template`, falls back to a generic `role: content` joined
    prompt — best-effort; models without a chat template may produce
    uncalibrated output.
    """
    model, tokenizer = load_model()
    messages = list(history) + [{"role": "user", "content": message}]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        ) + "\nassistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
```

## Template T3: diffusers `AutoPipelineForText2Image`

Used for `library_name = diffusers` and `pipeline_tag = text-to-image`.

```python
import os
from functools import lru_cache

import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "<org>/<model>"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the diffusers pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=dtype, token=os.getenv("HF_TOKEN")
    ).to(device)
    return pipe


def generate_image(
    prompt: str, num_inference_steps: int = 20, guidance_scale: float = 7.5
):
    """Generate one image from a prompt."""
    pipe = load_model()
    return pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
```

## Template T4: diffusers `AutoPipelineForImage2Image`

Used for `library_name = diffusers` and `pipeline_tag = image-to-image`.

```python
import os
from functools import lru_cache

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

MODEL_ID = "<org>/<model>"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the diffusers img2img pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID, torch_dtype=dtype, token=os.getenv("HF_TOKEN")
    ).to(device)
    return pipe


def edit_image(
    prompt: str,
    image: Image.Image,
    num_inference_steps: int = 20,
    strength: float = 0.8,
):
    """Edit a reference image conditioned on a prompt."""
    pipe = load_model()
    return pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        strength=strength,
    ).images[0]
```

## Template T5: sentence-transformers (embeddings)

Used for `library_name = sentence-transformers` (or when `library_name = transformers` and the model card explicitly recommends sentence-transformers in the README — common for `feature-extraction` models).

```python
import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

MODEL_ID = "<org>/<model>"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(MODEL_ID, token=os.getenv("HF_TOKEN"))


def embed(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    model = load_model()
    embeddings = model.encode([text_a, text_b])
    return float(cos_sim(embeddings[0], embeddings[1]).item())
```

## Template T6: `test_app.py` skeleton

Used in every scaffold. Provides one mocked-model test that validates the inference function's signature without hitting the network.

```python
"""Tests for the scaffolded app.py inference function."""
import os

import pytest

# For gated models, set a placeholder HF_TOKEN before importing app so the
# gated-model gate is bypassed during testing. For non-gated models this is a
# no-op. The fixture imports `app` at function scope so this assignment runs
# first.
os.environ.setdefault("HF_TOKEN", "test-token")


@pytest.fixture
def mock_load_model(monkeypatch):
    """Replace load_model with a stub matching the in-use template.

    Five stubs below — keep the one that matches your scaffolding template
    (T1/T2/T3/T4/T5) and delete the other four at scaffold time. Each block
    defines `_stub_factory` so the `monkeypatch.setattr` line at the bottom
    is template-agnostic.
    """
    import app

    # T1 (transformers pipeline). load_model returns a callable.
    class _Stub:
        def __call__(self, *args, **kwargs):
            return [{"label": "POSITIVE", "score": 0.99}]

    def _stub_factory():
        return _Stub()

    # # T2 (transformers causal-LM). load_model returns (model, tokenizer).
    # import torch
    # class _Tokenizer:
    #     chat_template = None  # forces the fallback prompt path
    #     def __call__(self, prompt, **kw):
    #         return type("Inputs", (), {
    #             "input_ids": type("T", (), {"shape": (1, 4)})(),
    #             "to": lambda self, _device: self,
    #         })()
    #     def decode(self, ids, **kw):
    #         return "stub response"
    # class _Model:
    #     device = "cpu"
    #     def generate(self, **kw):
    #         return torch.tensor([[0, 1, 2, 3, 4, 5]])
    # def _stub_factory():
    #     return _Model(), _Tokenizer()

    # # T3 / T4 (diffusers). load_model returns a callable whose result has .images[0].
    # from PIL import Image
    # class _Stub:
    #     def __call__(self, **kw):
    #         out = type("Out", (), {})
    #         out.images = [Image.new("RGB", (8, 8))]
    #         return out
    # def _stub_factory():
    #     return _Stub()

    # # T5 (sentence-transformers). load_model returns object with .encode([a, b]).
    # import numpy as np
    # class _Stub:
    #     def encode(self, texts):
    #         return np.array([[1.0, 0.0], [0.7, 0.7]])
    # def _stub_factory():
    #     return _Stub()

    if hasattr(app.load_model, "cache_clear"):
        app.load_model.cache_clear()
    monkeypatch.setattr(app, "load_model", _stub_factory)
    return _stub_factory


def test_inference_function_returns_expected_type(mock_load_model):
    # Adapt the call signature and assertion to whichever template is in use:
    # T1: result = app.run_inference("hi"); assert isinstance(result, list)
    # T2: result = app.chat("hi", []); assert isinstance(result, str)
    # T3: result = app.generate_image("a cat"); assert hasattr(result, "size")
    # T4: img = PIL.Image.new("RGB", (8, 8))
    #     result = app.edit_image("a cat", img); assert hasattr(result, "size")
    # T5: result = app.embed("hi", "world"); assert isinstance(result, float)
    # Implement per the in-use template; this skeleton is replaced at scaffold time.
    pass
```

The scaffold-time substitution replaces the body of `test_inference_function_returns_expected_type` with the appropriate adaptation — the comment hints inside T6 spell out each adaptation explicitly.
