import os
import csv
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import GPT
from tokenizers import Tokenizer


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tokenizer_path = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
text_path = os.path.join(BASE_DIR, "data", "input.txt")

checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
loss_log_path = os.path.join(checkpoint_dir, "loss_log.csv")

os.makedirs(checkpoint_dir, exist_ok=True)


# ----------------------------
# Hyperparameters
# ----------------------------
batch_size = 16
block_size = 128
embed_size = 128
num_layers = 2
heads = 4
learning_rate = 1e-4
epochs = 2

max_steps = None

checkpoint_every = 1000
print_every = 200


# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Load tokenizer
# ----------------------------
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()


# ----------------------------
# Dataset + DataLoader
# ----------------------------
dataset = TextDataset(tokenizer_path, text_path, block_size=block_size)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)


# ----------------------------
# Model
# ----------------------------
model = GPT(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_layers=num_layers,
    heads=heads,
    max_length=block_size
).to(device)


# ----------------------------
# Loss + Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# ----------------------------
# Utility functions
# ----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, epoch, step, global_step, loss):
    torch.save(model.state_dict(), path)

    latest_info_path = os.path.join(checkpoint_dir, "latest_checkpoint_info.txt")
    with open(latest_info_path, "w", encoding="utf-8") as f:
        f.write(f"epoch={epoch}\n")
        f.write(f"step={step}\n")
        f.write(f"global_step={global_step}\n")
        f.write(f"loss={loss:.6f}\n")
        f.write(f"path={path}\n")

    print(f"Checkpoint saved: {path}")


def append_loss_log(epoch, step, global_step, loss, elapsed_seconds):
    file_exists = os.path.exists(loss_log_path)

    with open(loss_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "epoch",
                "step",
                "global_step",
                "loss",
                "elapsed_seconds"
            ])

        writer.writerow([
            epoch,
            step,
            global_step,
            loss,
            round(elapsed_seconds, 4)
        ])


# ----------------------------
# Training info
# ----------------------------
batches_per_epoch = len(loader)
total_steps = batches_per_epoch * epochs

if max_steps is not None:
    total_steps = min(total_steps, max_steps)

print("=" * 80)
print("TRAINING CONFIG")
print("=" * 80)
print(f"Device: {device}")
print(f"Vocab size: {vocab_size}")
print(f"Dataset examples: {len(dataset):,}")
print(f"Batch size: {batch_size}")
print(f"Block size: {block_size}")
print(f"Batches per epoch: {batches_per_epoch:,}")
print(f"Epochs: {epochs}")
print(f"Total planned steps: {total_steps:,}")
print(f"Model parameters: {count_parameters(model):,}")
print(f"Checkpoint path: {checkpoint_path}")
print(f"Loss log path: {loss_log_path}")
print("=" * 80)


# ----------------------------
# Training loop
# ----------------------------
model.train()

global_step = 0
training_start = time.time()
last_loss = None

try:
    for epoch in range(epochs):
        epoch_start = time.time()

        for i, (x, y) in enumerate(loader):
            step_start = time.time()

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            logits = logits.view(-1, vocab_size)
            y = y.view(-1)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            global_step += 1
            last_loss = loss.item()

            elapsed_total = time.time() - training_start
            append_loss_log(epoch, i, global_step, last_loss, elapsed_total)

            if global_step % print_every == 0 or global_step == 1:
                seconds_per_step = elapsed_total / global_step
                remaining_steps = total_steps - global_step
                eta_seconds = max(0, remaining_steps * seconds_per_step)
                eta_minutes = eta_seconds / 60

                epoch_elapsed = time.time() - epoch_start

                print(
                    f"Epoch {epoch}, "
                    f"Step {i}/{batches_per_epoch}, "
                    f"Global Step {global_step}/{total_steps}, "
                    f"Loss: {last_loss:.4f}, "
                    f"Perplexity: {math.exp(min(last_loss, 20)):.2f}, "
                    f"Step Time: {time.time() - step_start:.3f}s, "
                    f"Epoch Elapsed: {epoch_elapsed / 60:.1f} min, "
                    f"ETA: {eta_minutes:.1f} min"
                )

            if global_step % checkpoint_every == 0:
                numbered_checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"model_step_{global_step}.pt"
                )

                save_checkpoint(
                    numbered_checkpoint_path,
                    epoch,
                    i,
                    global_step,
                    last_loss
                )

                save_checkpoint(
                    checkpoint_path,
                    epoch,
                    i,
                    global_step,
                    last_loss
                )

            if max_steps is not None and global_step >= max_steps:
                print("Max steps reached. Stopping early.")
                raise StopIteration

        epoch_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"model_epoch_{epoch}.pt"
        )

        save_checkpoint(
            epoch_checkpoint_path,
            epoch,
            batches_per_epoch,
            global_step,
            last_loss
        )

        save_checkpoint(
            checkpoint_path,
            epoch,
            batches_per_epoch,
            global_step,
            last_loss
        )

        print(f"Epoch {epoch} complete.")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    if last_loss is not None:
        interrupted_path = os.path.join(
            checkpoint_dir,
            f"model_interrupted_step_{global_step}.pt"
        )

        save_checkpoint(
            interrupted_path,
            epoch,
            i,
            global_step,
            last_loss
        )

        save_checkpoint(
            checkpoint_path,
            epoch,
            i,
            global_step,
            last_loss
        )

        print("Interrupted model saved. Your suffering was not entirely wasted.")

except StopIteration:
    pass


# ----------------------------
# Final save
# ----------------------------
if last_loss is not None:
    save_checkpoint(
        checkpoint_path,
        epoch,
        i,
        global_step,
        last_loss
    )

total_elapsed = time.time() - training_start

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Final loss: {last_loss:.4f}" if last_loss is not None else "No training steps completed.")
print(f"Total steps completed: {global_step:,}")
print(f"Total time: {total_elapsed / 60:.2f} minutes")
print(f"Final model saved to: {checkpoint_path}")
print(f"Loss log saved to: {loss_log_path}")