import torch
from tokenizers import Tokenizer

from config import CHECKPOINT_PATH, MODEL_SETTINGS, TOKENIZER_PATH
from model import GPT


def load_checkpoint(device):
    """Load the tokenizer and trained model on the requested device."""
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model = GPT(vocab_size=tokenizer.get_vocab_size(), **MODEL_SETTINGS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    return tokenizer, model
