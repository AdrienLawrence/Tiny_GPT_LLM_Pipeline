import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()

        if embed_size % heads != 0:
            raise ValueError("embed_size must be divisible by heads")

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(batch_size, seq_len, self.heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.heads, self.head_dim)
        queries = queries.view(batch_size, seq_len, self.heads, self.head_dim)

        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        energy = torch.matmul(queries, keys.transpose(-2, -1))
        energy = energy / (self.head_dim ** 0.5)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        ).bool()

        energy = energy.masked_fill(causal_mask == 0, float("-inf"))

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, values)

        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, embed_size)

        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.attention = SelfAttention(embed_size, heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(self.norm1(x))
        x = x + self.dropout(attention)

        forward = self.feed_forward(self.norm2(x))
        x = x + forward

        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=128,
        num_layers=2,
        heads=4,
        max_length=64,
        dropout=0.1
    ):
        super().__init__()

        self.embed_size = embed_size
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm_final = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape

        if seq_length > self.max_length:
            raise ValueError(
                f"Input sequence length {seq_length} exceeds max_length {self.max_length}"
            )

        positions = torch.arange(0, seq_length, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)

        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_final(x)

        logits = self.fc_out(x)

        return logits


if __name__ == "__main__":
    vocab_size = 8000
    model = GPT(
        vocab_size=vocab_size,
        embed_size=128,
        num_layers=2,
        heads=4,
        max_length=128,
        dropout=0.1
    )

    x = torch.randint(0, vocab_size, (2, 128))

    out = model(x)

    print("Output shape:", out.shape)