import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPT

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tokenizer_path = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "model.pt")

# ----------------------------
# Model config
# MUST match training config
# ----------------------------
block_size = 128
embed_size = 128
num_layers = 2
heads = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load tokenizer
# ----------------------------
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

# ----------------------------
# Load model
# ----------------------------
model = GPT(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_layers=num_layers,
    heads=heads,
    max_length=block_size
).to(device)

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print(f"Loaded model on device: {device}")


# ----------------------------
# Text generation
# ----------------------------
def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.6,
    top_k=20
):
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    input_ids = tokenizer.encode(prompt).ids

    if len(input_ids) == 0:
        raise ValueError("Prompt produced no tokens. Use a different prompt.")

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Keep only the most recent block_size tokens
        context = input_ids[:, -block_size:]

        with torch.no_grad():
            logits = model(context)

        # Only use logits for the final position
        logits = logits[:, -1, :]

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            probs = F.softmax(values, dim=-1)

            next_token = indices.gather(
                -1,
                torch.multinomial(probs, num_samples=1)
            )
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# ----------------------------
# Test prompts
# ----------------------------
if __name__ == "__main__":
    prompts = [
        "He went",
        "The girl",
        "In the end,",
        "The old house",
        "It was a strange"
    ]

    for prompt in prompts:
        print("\n" + "=" * 80)
        print("Prompt:", prompt)
        print("=" * 80)
        print(generate(prompt))