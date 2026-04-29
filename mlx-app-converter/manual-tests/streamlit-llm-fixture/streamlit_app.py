"""Minimal Streamlit chat app — pre-conversion baseline (transformers-based)."""

import streamlit as st
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

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
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- UI ---

st.title("LLM Chat")
st.caption(f"Model: {MODEL_ID}")

model, tokenizer = load_model()

prompt = st.text_input("Your message", placeholder="Ask me anything…")

if st.button("Send") and prompt:
    with st.spinner("Thinking…"):
        response = run_inference(prompt, model, tokenizer)
    st.markdown("**Response:**")
    st.write(response)
