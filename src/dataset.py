import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TextDataset(Dataset):
    def __init__(self, tokenizer_path, text_path, block_size=64):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokens = self.tokenizer.encode(text).ids
        self.block_size = block_size

    def __len__(self):
        return max(0, (len(self.tokens) - self.block_size - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.tokens[start:start + self.block_size + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tokenizer_path = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
    text_path = os.path.join(BASE_DIR, "data", "input.txt")

    dataset = TextDataset(tokenizer_path, text_path, block_size=128)

    print("Dataset size:", len(dataset))

    x, y = dataset[0]

    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    print("First x tokens:", x[:10])
    print("First y tokens:", y[:10])