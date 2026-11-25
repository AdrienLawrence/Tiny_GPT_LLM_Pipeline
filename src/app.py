import streamlit as st
import torch
import torch.nn.functional as F

from config import BLOCK_SIZE as block_size
from inference import load_checkpoint

device = torch.device("cpu")


@st.cache_resource
def load_model():
    return load_checkpoint(device)


def generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.5, top_k=15):
    input_ids = tokenizer.encode(prompt).ids

    if len(input_ids) == 0:
        return "Prompt produced no tokens."

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -block_size:]

        with torch.no_grad():
            logits = model(context)

        logits = logits[:, -1, :]
        logits = logits / temperature

        top_k_actual = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k_actual)

        probs = F.softmax(values, dim=-1)
        next_token = indices.gather(
            -1,
            torch.multinomial(probs, num_samples=1)
        )

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())


st.set_page_config(
    page_title="TinyDarcy Mini GPT",
    page_icon="🧠",
    layout="centered"
)

st.title("TinyDarcy: From-Scratch Mini GPT")
st.caption("A small decoder-only transformer trained from scratch for next-token prediction.")

tokenizer, model = load_model()

prompt = st.text_input("Prompt", value="The old man")

max_new_tokens = st.slider("Max new tokens", 20, 150, 80)
temperature = st.slider("Temperature", 0.1, 1.5, 0.5, 0.1)
top_k = st.slider("Top-k", 5, 100, 15)

if st.button("Generate"):
    with st.spinner("TinyDarcy is thinking, which is generous phrasing."):
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    st.subheader("Output")
    st.write(output)

st.divider()

st.subheader("Model Details")
st.write({
    "Architecture": "Decoder-only Transformer / GPT-style",
    "Parameters": "2,469,184",
    "Vocabulary size": "8,000",
    "Context length": block_size,
    "Training device": "CPU",
    "Final training loss": "5.3993",
    "Validation loss": "5.5072",
    "Validation perplexity": "246.4540"
})

st.caption("This is a small educational model, not an instruction-tuned chatbot.")
