# HuggingFace `pipeline_tag` → UI Pattern Catalog

When the input is an HF model card, the skill reads `pipeline_tag` from `https://huggingface.co/api/models/<id>` (or the YAML frontmatter of `README.md` as fallback) and selects a UI pattern from this catalog. Each entry specifies the primary Streamlit widgets and a minimal page body.

## Text generation / chat

`pipeline_tag`: `text-generation`, `conversational`

```python
# Chat page body
import streamlit as st
from <app_name>.inference import generate_response

st.title("Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

## Text classification

`pipeline_tag`: `text-classification`, `zero-shot-classification`

```python
import streamlit as st
from <app_name>.inference import classify

st.title("Classify Text")
text = st.text_area("Input", height=200)
if st.button("Classify") and text:
    result = classify(text)
    st.json(result)
```

## Token classification (NER)

`pipeline_tag`: `token-classification`

```python
import streamlit as st
from <app_name>.inference import extract_entities

st.title("Entity Recognition")
text = st.text_area("Input", height=200)
if st.button("Extract") and text:
    entities = extract_entities(text)
    st.write(entities)
    # Render highlighted spans via markdown with inline HTML,
    # or use a third-party component if preferred.
```

## Question answering

`pipeline_tag`: `question-answering`

```python
import streamlit as st
from <app_name>.inference import answer

st.title("Question Answering")
context = st.text_area("Context", height=250)
question = st.text_input("Question")
if st.button("Answer") and context and question:
    st.write(answer(context=context, question=question))
```

## Summarization / translation

`pipeline_tag`: `summarization`, `translation`

```python
import streamlit as st
from <app_name>.inference import transform_text

st.title("Transform Text")
src = st.text_area("Input", height=250)
if st.button("Run") and src:
    st.text_area("Output", transform_text(src), height=250)
```

## Feature extraction (embeddings)

`pipeline_tag`: `feature-extraction`

```python
import streamlit as st
from <app_name>.inference import embed

st.title("Embeddings")
text = st.text_area("Input", height=200)
if st.button("Embed") and text:
    vec = embed(text)
    st.write(f"Dim: {len(vec)}")
    st.line_chart(vec)  # quick visualization
```

## Automatic speech recognition (ASR)

`pipeline_tag`: `automatic-speech-recognition`

```python
import streamlit as st
from <app_name>.inference import transcribe

st.title("Transcribe Audio")
audio = st.audio_input("Record") or st.file_uploader("Upload", type=["wav", "mp3", "m4a", "flac"])
if audio and st.button("Transcribe"):
    text = transcribe(audio)
    st.text_area("Transcript", text, height=200)
```

## Text to speech

`pipeline_tag`: `text-to-speech`

```python
import streamlit as st
from <app_name>.inference import synthesize

st.title("Text to Speech")
text = st.text_area("Input", height=150)
if st.button("Speak") and text:
    audio_bytes = synthesize(text)
    st.audio(audio_bytes, format="audio/wav")
```

## Audio transform (speech-to-speech, separation, enhancement)

`pipeline_tag`: `audio-to-audio`

**Apple-Silicon-only.** The generated `inference.py` raises a clear `RuntimeError` on non-Apple-Silicon hosts.

```python
import streamlit as st
from <app_name>.inference import transform_audio

st.title("Transform Audio")
audio = st.audio_input("Record") or st.file_uploader(
    "Upload", type=["wav", "mp3", "m4a", "flac"]
)
if audio and st.button("Transform"):
    result = transform_audio(audio)
    # Single-output models (enhancement / denoising) return bytes.
    # Multi-output models (source separation) return dict[str, bytes].
    if isinstance(result, dict):
        for label, audio_bytes in result.items():
            st.subheader(label)
            st.audio(audio_bytes, format="audio/wav")
    else:
        st.audio(result, format="audio/wav")
```

The shape of `transform_audio`'s return is decided at scaffold time based on the HF model card: `separate_long`-style models return a dict of labeled outputs; `enhance`-style models return a single `bytes`. If the card doesn't map to a known `mlx_audio.sts` class, the skill emits a General Script page with a manual-wiring TODO instead of this template.

## Image classification / object detection

`pipeline_tag`: `image-classification`, `object-detection`

```python
import streamlit as st
from <app_name>.inference import classify_image

st.title("Classify Image")
img = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"])
if img and st.button("Classify"):
    st.image(img)
    st.json(classify_image(img))
```

## Image to text (captioning, VQA)

`pipeline_tag`: `image-to-text`

```python
import streamlit as st
from <app_name>.inference import caption

st.title("Image Captioning")
img = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"])
if img and st.button("Caption"):
    st.image(img)
    st.write(caption(img))
```

## Text to image

`pipeline_tag`: `text-to-image`

Scaffold-time substitution: the `value=` arguments on the width/height/steps sliders below are replaced with the defaults from the matched family's `references/mflux-families.md` Part B block (e.g., FLUX.2 Klein: `steps=4, width=1024, height=560`). When `mflux_family` is `None` (no Part A match), use generic defaults `(width=1024, height=1024, steps=20, seed=42)`.

```python
"""Text-to-image page."""
import streamlit as st

from <app_name> import inference


def render() -> None:
    st.title("Text-to-Image")
    prompt = st.text_area("Prompt", height=100)
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider("Width", 256, 2048, <default_width>, 64)
        steps = st.slider("Steps", 1, 100, <default_steps>)
    with col2:
        height = st.slider("Height", 256, 2048, <default_height>, 64)
        seed = st.number_input("Seed", value=42, step=1)
    if st.button("Generate", type="primary", disabled=not prompt.strip()):
        with st.spinner("Generating..."):
            image = inference.generate_image(
                prompt=prompt, width=width, height=height,
                num_inference_steps=steps, seed=int(seed),
            )
        st.image(image)
```

## Image to image

`pipeline_tag`: `image-to-image`

Scaffold-time substitution: the `value=` argument on the steps slider is replaced with the matched family's `i2i` default from `references/mflux-families.md` Part B (e.g., Qwen-Image-Edit: `steps=30`). Width and height are not exposed as sliders — mflux edit variants either derive output dimensions from the reference image (Flux2KleinEdit, FIBOEdit) or accept width/height in `inference.edit_image`'s family-specific path (Qwen-Image-Edit, Flux1Kontext). The scaffold consumer decides per family whether to pass `width` / `height` into the inference call.

```python
"""Image-to-image page."""
import tempfile
from pathlib import Path

import streamlit as st

from <app_name> import inference


def render() -> None:
    st.title("Image-to-Image")
    uploaded = st.file_uploader(
        "Reference image(s)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    prompt = st.text_area("Prompt", height=100)
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider("Steps", 1, 100, <default_steps>)
    with col2:
        seed = st.number_input("Seed", value=42, step=1)
    if st.button("Generate", type="primary",
                 disabled=not (prompt.strip() and uploaded)):
        with tempfile.TemporaryDirectory() as td:
            image_paths = []
            for up in uploaded:
                p = Path(td) / up.name
                p.write_bytes(up.read())
                image_paths.append(str(p))
            with st.spinner("Generating..."):
                image = inference.edit_image(
                    prompt=prompt,
                    image_paths=image_paths,
                    num_inference_steps=steps,
                    seed=int(seed),
                )
        st.image(image)
```

## Fallback: General Script

`pipeline_tag`: missing / unrecognized / not applicable (code-based input without a clear pattern)

```python
import streamlit as st
import pandas as pd
from <app_name>.inference import run  # or equivalent entry from source

st.title("Run")
# TODO: replace with widgets that expose run()'s parameters
param = st.text_input("Parameter")
if st.button("Run") and param:
    result = run(param)
    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif isinstance(result, (dict, list)):
        st.json(result)
    else:
        st.write(result)
```
