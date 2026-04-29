"""Minimal Gradio chat app — pre-conversion baseline (transformers-based)."""

from functools import lru_cache

import gradio as gr
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@lru_cache(maxsize=1)
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


def run_inference_for_chat(message: str, history: list) -> str:
    model, tokenizer = load_model()
    return run_inference(message, model, tokenizer)


# --- UI ---

demo = gr.ChatInterface(
    fn=run_inference_for_chat,
    title="LLM Chat",
    description=f"Model: {MODEL_ID}",
)

if __name__ == "__main__":
    demo.launch()
