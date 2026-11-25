import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import TextDataset
from config import BLOCK_SIZE as block_size
from config import TEXT_PATH as text_path
from config import TOKENIZER_PATH as tokenizer_path
from inference import load_checkpoint


batch_size = 16

eval_batches = 200
max_new_tokens = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Load tokenizer
# ----------------------------
tokenizer, model = load_checkpoint(device)
vocab_size = tokenizer.get_vocab_size()


# ----------------------------
# Dataset
# ----------------------------
dataset = TextDataset(tokenizer_path, text_path, block_size=block_size)

val_size = max(1, int(len(dataset) * 0.10))
train_size = len(dataset) - val_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)



criterion = nn.CrossEntropyLoss()


# ----------------------------
# Parameter count
# ----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Evaluation loss/perplexity
# ----------------------------
def evaluate_loss():
    total_loss = 0.0
    batches_seen = 0
    total_tokens = 0

    start_time = time.time()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            logits = logits.view(-1, vocab_size)
            y = y.view(-1)

            loss = criterion(logits, y)

            total_loss += loss.item()
            batches_seen += 1
            total_tokens += y.numel()

            if batches_seen >= eval_batches:
                break

    elapsed = time.time() - start_time
    avg_loss = total_loss / batches_seen
    perplexity = math.exp(avg_loss)
    tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0

    return avg_loss, perplexity, tokens_per_second, batches_seen, total_tokens


# ----------------------------
# Text generation
# ----------------------------
def generate(prompt, temperature=0.9, top_k=40):
    input_ids = tokenizer.encode(prompt).ids

    if len(input_ids) == 0:
        return "[Prompt produced no tokens]"

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -block_size:]

        with torch.no_grad():
            logits = model(context)

        logits = logits[:, -1, :]
        logits = logits / temperature

        top_k_actual = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k_actual)

        probs = torch.softmax(values, dim=-1)
        next_token = indices.gather(
            -1,
            torch.multinomial(probs, num_samples=1)
        )

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# ----------------------------
# Run evaluation
# ----------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    print(f"Device: {device}")
    print(f"Dataset examples: {len(dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Block size: {block_size}")
    print(f"Parameter count: {count_parameters(model):,}")

    avg_loss, perplexity, tokens_per_second, batches_seen, total_tokens = evaluate_loss()

    print("\nQuantitative Metrics")
    print("-" * 80)
    print(f"Validation batches evaluated: {batches_seen}")
    print(f"Validation tokens evaluated: {total_tokens:,}")
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Validation perplexity: {perplexity:.4f}")
    print(f"Evaluation speed: {tokens_per_second:.2f} tokens/sec")

    prompts = [
        "The old man",
        "In the end,",
        "She looked at the",
        "The city was",
        "Once upon a time"
    ]

    print("\nQualitative Samples")
    print("-" * 80)

    for prompt in prompts:
        print("\nPrompt:", prompt)
        print("Generated:")
        print(generate(prompt))
        print("-" * 80)
