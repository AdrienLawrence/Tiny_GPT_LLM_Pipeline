import torch
import torch.nn.functional as F

from config import BLOCK_SIZE as block_size
from inference import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_checkpoint(device)

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
