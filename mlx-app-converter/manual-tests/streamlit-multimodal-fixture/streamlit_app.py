"""Streamlit multi-modal fixture — VLM for image understanding + LLM for follow-up.

Pre-conversion baseline. Loads two models in one file so mlx-app-converter v2
must route each independently, union the imports, and combine the dep manifest.
"""
from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

VLM_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
LLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@st.cache_resource
def load_vlm():
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(VLM_MODEL_ID)
    return model, processor


@st.cache_resource
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


def main() -> None:
    st.title("Multi-modal: image description + follow-up reasoning")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to start.")
        return
    image = Image.open(BytesIO(uploaded.read()))
    st.image(image, caption="Uploaded image")
    if st.button("Describe + reason"):
        vlm_model, processor = load_vlm()
        llm_model, tokenizer = load_llm()
        with st.spinner("VLM: describing..."):
            description = describe_image(
                "Describe this image.", image, vlm_model, processor
            )
        st.markdown("**Description:** " + description)
        with st.spinner("LLM: following up..."):
            follow_up_prompt = (
                f"The image was described as: {description}\n"
                "What might the person in the image be thinking?"
            )
            answer = follow_up(follow_up_prompt, llm_model, tokenizer)
        st.markdown("**Reasoning:** " + answer)


if __name__ == "__main__":
    main()
