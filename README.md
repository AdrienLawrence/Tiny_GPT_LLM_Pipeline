# TinyDarcy: From-Scratch Mini GPT LLM Pipeline

TinyDarcy is a small GPT-style language model built from scratch in PyTorch for the class CS4985 GenAI final project. The goal of this project was to implement an end-to-end LLM pipeline, including data preparation, tokenization, model architecture, training, evaluation, generation, and deployment.

This is not a production-quality chatbot or a mature instruction-following LLM. It is a small educational decoder-only transformer trained for next-token prediction.

## Project Goal

The goal was to build a miniature LLM pipeline from scratch and demonstrate the major stages of modern language model development:

1. Text data preparation
2. BPE tokenizer training
3. Dataset creation and batching
4. GPT-style decoder-only transformer implementation
5. Next-token prediction training
6. Checkpoint saving
7. Evaluation with loss and perplexity
8. Text generation
9. Local deployment with Streamlit

## Model Name

The trained model is named TinyDarcy (Be nice to him).

The name comes from the model's small size and the literary training data, which included public-domain literature such as Pride and Prejudice along with other text sources.

## Architecture

TinyDarcy uses a small decoder-only transformer architecture inspired by GPT.

Main components:

- Token embeddings
- Positional embeddings
- Causal self-attention
- Multi-head attention
- Feed-forward layers
- GELU activation
- Dropout
- Pre-layer normalization
- Final layer normalization
- Linear output head for next-token prediction

The causal mask ensures that the model can only attend to previous tokens when predicting the next token. This prevents future-token leakage during training.

## Model Configuration

| Setting | Value |
|---|---:|
| Vocabulary size | 8,000 |
| Context length | 128 tokens |
| Embedding size | 128 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Trainable parameters | 2,469,184 |
| Training device | CPU |
| Final training loss | 5.3993 |
| Validation loss | 5.5072 |
| Validation perplexity | 246.4540 |
| Total training steps | 5,716 |
| Total training time | 25.76 minutes |

## Dataset

The dataset was built from a local input.txt file containing public-domain text and some mixed reference-style text. The corpus included mostly literary prose, but some Wikipedia/reference-style content remained in the dataset.

This affected generation quality. Some generated samples drifted into reference-like fragments involving government, country names, dates, or abbreviations. This showed the importance of data cleaning and corpus quality in LLM training.

Future versions would use cleaner literature-only data and remove:

- Wikipedia-style entries
- Census/statistical data
- References and citations
- URLs
- Project Gutenberg boilerplate
- Illustration/copyright markers
- Tables and metadata

## Tokenization

The tokenizer was trained using Byte Pair Encoding (BPE) with the Hugging Face tokenizers library.

Tokenizer settings:

- BPE model
- Vocabulary size: 8,000
- Minimum frequency: 2
- Whitespace pre-tokenizer
- Special tokens:
  - [PAD]
  - [UNK]
  - [CLS]
  - [SEP]
  - [MASK]

The tokenizer converts raw text into token IDs. The model then maps these token IDs into learned embedding vectors.

## Training Objective

TinyDarcy was trained using next-token prediction.

For each sequence of tokens, the model receives an input sequence and learns to predict the same sequence shifted one token forward.

Example input:

The girl opened the

Example target:

girl opened the door

The model predicts the next token at every position in the block. Cross-entropy loss measures how much probability the model assigned to the correct next token.

## Training Results

The model trained for 2 epochs on CPU.

Training started near random prediction:

Initial loss: 9.1606  
Initial perplexity: 9515.04

Final training result:

Final loss: 5.3993  
Total steps completed: 5,716  
Total time: 25.76 minutes

The loss decreased substantially during training, showing that the model learned meaningful next-token structure from the dataset.

## Evaluation

Evaluation was performed using a held-out validation split.

Results:

| Metric | Value |
|---|---:|
| Validation examples | 4,573 |
| Validation tokens evaluated | 409,600 |
| Validation batches evaluated | 200 |
| Validation loss | 5.5072 |
| Validation perplexity | 246.4540 |
| Evaluation speed | 39,193.88 tokens/sec |

The validation loss was close to the final training loss, suggesting that the model was not severely overfitting. However, both losses remained high compared to mature language models, meaning the model still produced rough and repetitive text.

## Qualitative Generation Results

Example prompt:

The old man

Example output:

The old man was to the old and had been a word is the first at her his way to , I have it of the long they saw , with the world were she should . “ and a I had no in his t the most my dear for that be a new...

The model learned some common English structure, dialogue punctuation, and phrase patterns, but it did not consistently generate coherent long-form English.

Observed behavior:

- Learned common words and punctuation
- Generated dialogue-like fragments
- Produced repeated phrases
- Sometimes drifted into noisy Wikipedia/reference-style content
- Did not reliably maintain grammar or meaning over long generations

## Why MMLU or TruthfulQA Were Not Used

Benchmarks like MMLU, TruthfulQA, and HellaSwag are designed for mature instruction-following LLMs with broad pretrained knowledge.

TinyDarcy is a small base language model trained from scratch for next-token prediction. It is not instruction-tuned and is not expected to answer questions reliably.

Because of that, evaluation focused on metrics appropriate for a small from-scratch language model:

- Training loss
- Validation loss
- Perplexity
- Parameter count
- Runtime
- Generated text samples

## Deployment

A local Streamlit app was created to deploy the trained model for inference.

The app:

- Loads the trained model checkpoint
- Loads the trained tokenizer
- Accepts a user prompt
- Generates text from the model
- Displays model details and evaluation metrics

To run the app locally:

streamlit run src/app.py

If using the project virtual environment on Windows:

.\.venv\Scripts\python.exe -m streamlit run .\src\app.py

## Project Structure

Tiny_GPT_LLM_Pipeline/
|
├── checkpoints/
|   ├── model.pt
|   └── loss_log.csv
|
├── data/
|   └── input.txt
|
├── src/
|   ├── app.py
|   ├── dataset.py
|   ├── evaluate.py
|   ├── generate.py
|   ├── model.py
|   ├── train.py
|   └── train_tokenizer.py
|
├── tokenizer/
|   └── tokenizer.json
|
├── requirements.txt
└── README.md

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train tokenizer

python src/train_tokenizer.py

### 3. Train model

python src/train.py

### 4. Evaluate model

python src/evaluate.py

### 5. Generate text

python src/generate.py

### 6. Run Streamlit app

streamlit run src/app.py

## Requirements

Main libraries:

- Python
- PyTorch
- Hugging Face tokenizers
- Streamlit

Example requirements.txt:

torch  
tokenizers  
streamlit

## Limitations

TinyDarcy is intentionally small and was trained under limited compute.

Main limitations:

- CPU-only training
- Small model size
- Short training time
- No instruction fine-tuning
- No RLHF or preference tuning
- Limited context length
- No large-scale benchmark performance
- Mixed/noisy dataset
- Repetitive and unstable generation

## Future Improvements

Future improvements would include:

- Clean literature-only training data
- Better dataset filtering
- More training epochs
- GPU training
- Larger model size
- Larger context length
- Learning rate scheduling
- Weight tying
- Train/validation loss plotting
- Instruction fine-tuning
- Better sampling controls
- Hosted Streamlit deployment

## Conclusion

This project successfully implemented a miniature LLM pipeline from scratch. The model is small and limited, but the pipeline includes the major stages of LLM development: tokenization, dataset preparation, transformer model implementation, causal language modeling, training, checkpointing, evaluation, generation, and deployment.

The final model showed measurable learning through decreasing loss and validation perplexity, while also demonstrating the importance of data quality, training scale, and compute resources in modern LLM development.