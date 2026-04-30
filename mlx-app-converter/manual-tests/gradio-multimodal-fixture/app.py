"""Gradio multi-modal fixture — VLM for image understanding + LLM for follow-up.

Pre-conversion baseline. Loads two models in one file so mlx-app-converter v2
must route each independently, union the imports, and combine the dep manifest
(append two lines to requirements.txt — Gradio convention is per-line, NOT
a single combined uv add).
"""
from __future__ import annotations

from functools import lru_cache

import gradio as gr
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

VLM_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
LLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@lru_cache(maxsize=1)
def load_vlm():
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(VLM_MODEL_ID)
    return model, processor


@lru_cache(maxsize=1)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID)
    return model, tokenizer


def describe_image(
    prompt: str,
    image: Image.Image,
    model,
    processor,
    max_new_tokens: int = 200,
) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(outputs[0], skip_special_tokens=True)


def follow_up(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 200,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def chat(image: Image.Image) -> tuple[str, str]:
    vlm_model, processor = load_vlm()
    llm_model, tokenizer = load_llm()
    description = describe_image(
        "Describe this image.", image, vlm_model, processor
    )
    follow_up_prompt = (
        f"The image was described as: {description}\n"
        "What might the person in the image be thinking?"
    )
    answer = follow_up(follow_up_prompt, llm_model, tokenizer)
    return description, answer


demo = gr.Interface(
    fn=chat,
    inputs=gr.Image(type="pil", label="Image"),
    outputs=[
        gr.Textbox(label="Description (VLM)"),
        gr.Textbox(label="Reasoning (LLM)"),
    ],
    title="Multi-modal: image description + follow-up reasoning",
)


if __name__ == "__main__":
    demo.launch()
