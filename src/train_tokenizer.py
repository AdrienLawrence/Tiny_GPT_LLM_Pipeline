from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Paths
INPUT_FILE = "../data/input.txt"
TOKENIZER_PATH = "../tokenizer/tokenizer.json"

# Initialize tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Trainer
trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train
tokenizer.train([INPUT_FILE], trainer)

# Save
tokenizer.save(TOKENIZER_PATH)

print("Tokenizer trained and saved.")