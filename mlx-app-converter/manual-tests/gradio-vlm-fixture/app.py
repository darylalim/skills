"""Gradio VLM chat fixture — pre-conversion baseline (transformers)."""
from __future__ import annotations

from functools import lru_cache

import gradio as gr
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


@lru_cache(maxsize=1)
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID)
    return model, processor


def run_inference(
    prompt: str,
    image: Image.Image,
    model,
    processor,
    max_new_tokens: int = 200,
) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(outputs[0], skip_special_tokens=True)


def chat(prompt: str, image: Image.Image) -> str:
    model, processor = load_model()
    return run_inference(prompt, image, model, processor)


demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Prompt", value="Describe this image."),
        gr.Image(type="pil", label="Image"),
    ],
    outputs=gr.Textbox(label="Response"),
    title="VLM chat — Qwen2-VL fixture",
)


if __name__ == "__main__":
    demo.launch()
