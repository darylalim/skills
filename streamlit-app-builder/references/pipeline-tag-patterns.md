# HuggingFace `pipeline_tag` → UI Pattern Catalog

When the input is an HF model card, the skill reads `pipeline_tag` from `https://huggingface.co/api/models/<id>` (or the YAML frontmatter of `README.md` as fallback) and selects a UI pattern from this catalog. Each entry specifies the primary Streamlit widgets and an inline UI body. The UI body is pasted directly into `streamlit_app.py` after the `load_model()` and inference-function definitions from `references/scaffolding-templates.md` (the inference function name varies per scaffolding template — `run_inference`, `generate_response`, `generate_image`, `edit_image`, or `embed`).

The `<!-- skip-validate -->` marker before each block tells the static validator to skip parsing — these are fragments that depend on symbols defined in the surrounding assembled file.

## Text generation / chat

`pipeline_tag`: `text-generation`, `conversational`. Use scaffolding template T2 (`generate_response`).

<!-- skip-validate -->
```python
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
        response = generate_response(prompt, max_new_tokens=256)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

## Text classification

`pipeline_tag`: `text-classification`, `zero-shot-classification`. Use template T1 (`run_inference`).

<!-- skip-validate -->
```python
st.title("Classify Text")
text = st.text_area("Input", height=200)
if st.button("Classify") and text:
    result = run_inference(text)
    st.json(result)
```

## Token classification (NER)

`pipeline_tag`: `token-classification`. Use template T1.

<!-- skip-validate -->
```python
st.title("Entity Recognition")
text = st.text_area("Input", height=200)
if st.button("Extract") and text:
    entities = run_inference(text)
    st.write(entities)
```

## Question answering

`pipeline_tag`: `question-answering`. Use template T1; call as `run_inference(question=..., context=...)`.

<!-- skip-validate -->
```python
st.title("Question Answering")
context = st.text_area("Context", height=250)
question = st.text_input("Question")
if st.button("Answer") and context and question:
    st.write(run_inference(question=question, context=context))
```

## Summarization / translation

`pipeline_tag`: `summarization`, `translation`. Use template T1.

<!-- skip-validate -->
```python
st.title("Transform Text")
src = st.text_area("Input", height=250)
if st.button("Run") and src:
    result = run_inference(src)
    # Pipeline returns [{"summary_text": "..."}] for summarization,
    # [{"translation_text": "..."}] for translation.
    output = result[0].get("summary_text") or result[0].get("translation_text") or str(result)
    st.text_area("Output", output, height=250)
```

## Feature extraction (embeddings)

`pipeline_tag`: `feature-extraction`. Use template T5 (`embed`) — `sentence-transformers` is the right backend for most embedding models.

<!-- skip-validate -->
```python
st.title("Embeddings")
text = st.text_area("Input", height=200)
if st.button("Embed") and text:
    vec = embed(text)
    st.write(f"Dim: {vec.shape[-1]}")
    st.line_chart(vec)
```

## Automatic speech recognition (ASR)

`pipeline_tag`: `automatic-speech-recognition`. Use template T1.

<!-- skip-validate -->
```python
st.title("Transcribe Audio")
audio = st.audio_input("Record") or st.file_uploader(
    "Upload", type=["wav", "mp3", "m4a", "flac"]
)
if audio and st.button("Transcribe"):
    audio_bytes = audio.getvalue() if hasattr(audio, "getvalue") else audio.read()
    result = run_inference(audio_bytes)
    st.text_area("Transcript", result.get("text", str(result)), height=200)
```

## Text to speech

`pipeline_tag`: `text-to-speech`. Use template T1.

<!-- skip-validate -->
```python
import io

import scipy.io.wavfile

st.title("Text to Speech")
text = st.text_area("Input", height=150)
if st.button("Speak") and text:
    result = run_inference(text)
    # transformers.pipeline("text-to-speech") returns {"audio": np.ndarray, "sampling_rate": int}
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, result["sampling_rate"], result["audio"].squeeze())
    st.audio(buf.getvalue(), format="audio/wav")
```

(`scipy` is added to dependencies for TTS apps.)

## Image classification / object detection

`pipeline_tag`: `image-classification`, `object-detection`. Use template T1.

<!-- skip-validate -->
```python
st.title("Classify Image")
img = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"])
if img and st.button("Classify"):
    st.image(img)
    st.json(run_inference(img))
```

## Image to text (captioning)

`pipeline_tag`: `image-to-text`. Use template T1.

<!-- skip-validate -->
```python
st.title("Image Captioning")
img = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"])
if img and st.button("Caption"):
    st.image(img)
    result = run_inference(img)
    st.write(result[0].get("generated_text", str(result)) if result else "")
```

## Image-text-to-text (visual question answering, multimodal chat)

`pipeline_tag`: `image-text-to-text`. Use template T1. Argument names vary by model — `text` and `images` are the most common keywords; check the model card.

<!-- skip-validate -->
```python
from PIL import Image

st.title("Image + Text")
img = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"])
question = st.text_area("Question about the image", height=100)
if st.button("Answer", disabled=not (img and question.strip())):
    st.image(img)
    image = Image.open(img)
    result = run_inference(text=question, images=image)
    st.write(result[0].get("generated_text") if isinstance(result, list) else result)
```

## Text to image

`pipeline_tag`: `text-to-image`. Use template T3 (`generate_image`).

<!-- skip-validate -->
```python
st.title("Text-to-Image")
prompt = st.text_area("Prompt", height=100)
col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Steps", 1, 100, 20)
with col2:
    seed = st.number_input("Seed", value=42, step=1)
if st.button("Generate", type="primary", disabled=not prompt.strip()):
    with st.spinner("Generating..."):
        image = generate_image(prompt=prompt, num_inference_steps=steps, seed=int(seed))
    st.image(image)
```

## Image to image

`pipeline_tag`: `image-to-image`. Use template T4 (`edit_image`).

<!-- skip-validate -->
```python
from PIL import Image

st.title("Image-to-Image")
uploaded = st.file_uploader(
    "Reference image", type=["jpg", "jpeg", "png", "webp"]
)
prompt = st.text_area("Prompt", height=100)
col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Steps", 1, 100, 20)
with col2:
    seed = st.number_input("Seed", value=42, step=1)
if st.button("Generate", type="primary",
             disabled=not (prompt.strip() and uploaded)):
    reference = Image.open(uploaded)
    with st.spinner("Generating..."):
        image = edit_image(
            prompt=prompt, image=reference,
            num_inference_steps=steps, seed=int(seed),
        )
    st.image(image)
```

## Fallback: General Script

`pipeline_tag` missing or unrecognized. Use template T1 with `run_inference(input)`.

<!-- skip-validate -->
```python
st.title("Run")
param = st.text_input("Input")
if st.button("Run") and param:
    result = run_inference(param)
    st.json(result) if isinstance(result, (dict, list)) else st.write(result)
```

## Rejected pipeline tags

| `pipeline_tag` | Rejection message |
|---|---|
| `audio-to-audio` | *"audio-to-audio has no clean transformers pipeline. This skill can't scaffold a working prototype for audio-to-audio models. For source separation or speech enhancement, use the model's reference implementation directly."* |
