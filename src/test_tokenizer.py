import os
from tokenizers import Tokenizer

# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")

# Load tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Test text
text = "The model learns to generate text."

# Encode
encoded = tokenizer.encode(text)

print("Original:", text)
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)

# Decode back
decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)