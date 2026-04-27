# HuggingFace `pipeline_tag` → UI Pattern Catalog

When the input is an HF model card, the skill reads `pipeline_tag` from `https://huggingface.co/api/models/<id>` (or the YAML frontmatter of `README.md` as fallback) and selects a UI pattern from this catalog. Each entry specifies the primary Gradio components and an inline UI body. The UI body is pasted directly into `app.py` after the `load_model()` and inference-function definitions from `references/scaffolding-templates.md` (the inference function name varies per scaffolding template — `run_inference`, `chat`, `generate_image`, `edit_image`, or `embed`).

The `<!-- skip-validate -->` marker before each block tells the static validator to skip parsing — these are fragments that depend on symbols defined in the surrounding assembled file.

## Text generation / chat

`pipeline_tag`: `text-generation`, `conversational`. Use scaffolding template T2 (`chat`).

<!-- skip-validate -->
```python
demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title=f"Chat — {MODEL_ID}",
)
```

## Text classification

`pipeline_tag`: `text-classification`, `zero-shot-classification`. Use template T1 (`run_inference`).

<!-- skip-validate -->
```python
demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Textbox(label="Input", lines=5),
    outputs=gr.JSON(label="Predictions"),
    title=f"Classify — {MODEL_ID}",
    flagging_mode="never",
)
```

## Token classification (NER)

`pipeline_tag`: `token-classification`. Use template T1.

<!-- skip-validate -->
```python
demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Textbox(label="Input", lines=5),
    outputs=gr.JSON(label="Entities"),
    title=f"Entities — {MODEL_ID}",
    flagging_mode="never",
)
```

## Question answering

`pipeline_tag`: `question-answering`. Use template T1; call as `run_inference(question=..., context=...)`.

<!-- skip-validate -->
```python
def _answer(question: str, context: str):
    return run_inference(question=question, context=context)


demo = gr.Interface(
    fn=_answer,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Textbox(label="Context", lines=10),
    ],
    outputs=gr.JSON(label="Answer"),
    title=f"QA — {MODEL_ID}",
    flagging_mode="never",
)
```

## Summarization / translation

`pipeline_tag`: `summarization`, `translation`. Use template T1.

<!-- skip-validate -->
```python
def _transform(text: str) -> str:
    result = run_inference(text)
    # Pipeline returns [{"summary_text": "..."}] for summarization,
    # [{"translation_text": "..."}] for translation.
    return result[0].get("summary_text") or result[0].get("translation_text") or str(result)


demo = gr.Interface(
    fn=_transform,
    inputs=gr.Textbox(label="Input", lines=10),
    outputs=gr.Textbox(label="Output", lines=10),
    title=f"Transform — {MODEL_ID}",
    flagging_mode="never",
)
```

## Feature extraction (embeddings)

`pipeline_tag`: `feature-extraction`. Use template T5 (`embed`) — `sentence-transformers` is the right backend for most embedding models.

<!-- skip-validate -->
```python
demo = gr.Interface(
    fn=embed,
    inputs=[
        gr.Textbox(label="Text A", lines=3),
        gr.Textbox(label="Text B", lines=3),
    ],
    outputs=gr.Number(label="Cosine similarity", interactive=False),
    title=f"Embed — {MODEL_ID}",
    flagging_mode="never",
)
```

## Automatic speech recognition (ASR)

`pipeline_tag`: `automatic-speech-recognition`. Use template T1.

<!-- skip-validate -->
```python
def _transcribe(audio_path: str) -> str:
    result = run_inference(audio_path)
    return result.get("text", str(result))


demo = gr.Interface(
    fn=_transcribe,
    inputs=gr.Audio(type="filepath", label="Audio"),
    outputs=gr.Textbox(label="Transcript", lines=5),
    title=f"Transcribe — {MODEL_ID}",
    flagging_mode="never",
)
```

## Text to speech

`pipeline_tag`: `text-to-speech`. Use template T1.

<!-- skip-validate -->
```python
def _synthesize(text: str):
    result = run_inference(text)
    # transformers.pipeline("text-to-speech") returns {"audio": np.ndarray, "sampling_rate": int}
    return (result["sampling_rate"], result["audio"].squeeze())


demo = gr.Interface(
    fn=_synthesize,
    inputs=gr.Textbox(label="Input", lines=5),
    outputs=gr.Audio(label="Output"),
    title=f"Speak — {MODEL_ID}",
    flagging_mode="never",
)
```

(`scipy` is added to dependencies for TTS apps.)

## Image classification / object detection

`pipeline_tag`: `image-classification`, `object-detection`. Use template T1.

<!-- skip-validate -->
```python
demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Image(type="pil", label="Image"),
    outputs=gr.JSON(label="Predictions"),
    title=f"Classify — {MODEL_ID}",
    flagging_mode="never",
)
```

## Image to text (captioning)

`pipeline_tag`: `image-to-text`. Use template T1.

<!-- skip-validate -->
```python
def _caption(image) -> str:
    result = run_inference(image)
    return result[0].get("generated_text", str(result)) if result else ""


demo = gr.Interface(
    fn=_caption,
    inputs=gr.Image(type="pil", label="Image"),
    outputs=gr.Textbox(label="Caption", lines=2),
    title=f"Caption — {MODEL_ID}",
    flagging_mode="never",
)
```

## Image-text-to-text (visual question answering, multimodal chat)

`pipeline_tag`: `image-text-to-text`. Use template T1. Argument names vary by model — `text` and `images` are the most common keywords; check the model card.

<!-- skip-validate -->
```python
def _vqa(image, question: str) -> str:
    result = run_inference(text=question, images=image)
    if isinstance(result, list) and result:
        return result[0].get("generated_text", str(result))
    return str(result)


demo = gr.Interface(
    fn=_vqa,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Textbox(label="Question", lines=2),
    ],
    outputs=gr.Textbox(label="Answer", lines=3),
    title=f"VQA — {MODEL_ID}",
    flagging_mode="never",
)
```

## Text to image

`pipeline_tag`: `text-to-image`. Use template T3 (`generate_image`).

<!-- skip-validate -->
```python
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", lines=3),
        gr.Slider(1, 100, value=20, step=1, label="Steps"),
        gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance scale"),
    ],
    outputs=gr.Image(type="pil", label="Output"),
    title=f"Generate — {MODEL_ID}",
    flagging_mode="never",
)
```

## Image to image

`pipeline_tag`: `image-to-image`. Use template T4 (`edit_image`).

<!-- skip-validate -->
```python
demo = gr.Interface(
    fn=edit_image,
    inputs=[
        gr.Textbox(label="Prompt", lines=3),
        gr.Image(type="pil", label="Reference"),
        gr.Slider(1, 100, value=20, step=1, label="Steps"),
        gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Strength"),
    ],
    outputs=gr.Image(type="pil", label="Output"),
    title=f"Edit — {MODEL_ID}",
    flagging_mode="never",
)
```

## Fallback: General Script

`pipeline_tag` missing or unrecognized. Use template T1 with `run_inference(input)`.

<!-- skip-validate -->
```python
demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.JSON(label="Output"),
    title=f"Run — {MODEL_ID}",
    flagging_mode="never",
)
```

## Rejected pipeline tags

| `pipeline_tag` | Rejection message |
|---|---|
| `audio-to-audio` | *"audio-to-audio has no clean transformers pipeline. This skill can't scaffold a working prototype for audio-to-audio models. For source separation or speech enhancement, use the model's reference implementation directly."* |
