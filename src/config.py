from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = str(ROOT / "tokenizer" / "tokenizer.json")
CHECKPOINT_PATH = ROOT / "checkpoints" / "model.pt"
TEXT_PATH = ROOT / "data" / "input.txt"

BLOCK_SIZE = 128
MODEL_SETTINGS = {
    "embed_size": 128,
    "num_layers": 2,
    "heads": 4,
    "max_length": BLOCK_SIZE,
}
