"""Streamlit VLM chat fixture — pre-conversion baseline (transformers).

This file is a fixture for mlx-app-converter manual tests. It must remain
runnable as-is (transformers-based) before the converter is applied.
"""
from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


@st.cache_resource
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


def main() -> None:
    st.title("VLM chat — Qwen2-VL fixture")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    prompt = st.text_input("Prompt", value="Describe this image.")
    if uploaded is None or not prompt:
        st.info("Upload an image and enter a prompt to start.")
        return
    image = Image.open(BytesIO(uploaded.read()))
    st.image(image, caption="Uploaded image")
    model, processor = load_model()
    if st.button("Generate"):
        with st.spinner("Generating..."):
            response = run_inference(prompt, image, model, processor)
        st.markdown(response)


if __name__ == "__main__":
    main()
