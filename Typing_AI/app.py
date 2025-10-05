import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# ----------------- Load GPT-2 on CPU -----------------
@st.cache_resource
def load_model():
    model_name = "gpt2"  # change to "gpt2-medium" if your PC has enough RAM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None)
    model.to(torch.device("cpu"))  # force CPU
    return tokenizer, model

tokenizer, model = load_model()

# ----------------- Function to Generate Text -----------------
def generate_text(prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}  # ensure CPU
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------- Streamlit UI -----------------
st.title("📝 My AI Writer")

# Sidebar settings
st.sidebar.header("Settings")
max_length = st.sidebar.slider("Max Text Length", 50, 500, 150)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.5, 0.7)
top_k = st.sidebar.slider("Top-k Sampling", 10, 200, 50)
top_p = st.sidebar.slider("Top-p Sampling", 0.1, 1.0, 0.95)

# Prompt input
prompt = st.text_area("Enter your prompt here:")

if st.button("Generate Text"):
    if prompt.strip() != "":
        with st.spinner("Generating..."):
            result = generate_text(prompt, max_length, temperature, top_k, top_p)
        st.subheader("Generated Text:")
        st.write(result)

        # Save option
        if st.button("Save to File"):
            file_name = "generated_text.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(result)
            st.success(f"Text saved to {os.path.abspath(file_name)}")
    else:
        st.warning("Please enter a prompt!")
