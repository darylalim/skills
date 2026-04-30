"""Streamlit streaming chat fixture — pre-conversion baseline (transformers).

Streams tokens to the UI via TextIteratorStreamer + a background thread.
This is the canonical streaming pattern; mlx-app-converter v2 should detect
the streamer usage and soft-reject conversion (streaming is deferred to a
follow-up version of the skill).
"""
from __future__ import annotations

from threading import Thread
from typing import Iterator

import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tokenizer


def run_inference(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 200,
) -> Iterator[str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for chunk in streamer:
        yield chunk


def main() -> None:
    st.title("Streaming chat — Llama-3.1 fixture")
    prompt = st.text_input("Prompt", value="Hello, who are you?")
    if not prompt:
        return
    model, tokenizer = load_model()
    if st.button("Generate"):
        st.write_stream(run_inference(prompt, model, tokenizer))


if __name__ == "__main__":
    main()
