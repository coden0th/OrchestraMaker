"""A small GPT-style decoder (nanoGPT-like) over note tokens."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head, self.dropout = c.n_head, c.dropout
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd), nn.Dropout(c.dropout))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in self.qkv(self.ln1(x)).split(C, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None, ignore_index=0):
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1], device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=ignore_index)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.95, eos=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])
            probs = F.softmax(logits[:, -1].float() / temperature, dim=-1)
            sorted_p, order = probs.sort(descending=True)
            sorted_p[sorted_p.cumsum(-1) - sorted_p > top_p] = 0  # nucleus sampling
            nxt = order.gather(-1, torch.multinomial(sorted_p, 1))
            idx = torch.cat([idx, nxt], dim=1)
            if eos is not None and (nxt == eos).all():
                break
        return idx
