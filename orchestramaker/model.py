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

    def forward(self, x, past=None):
        """past: this layer's cached (keys, values) when decoding one token at a time."""
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in self.qkv(self.ln1(x)).split(C, dim=2))
        if past is not None:
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + self.mlp(self.ln2(x)), (k, v)


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
        logits, _ = self._run(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=ignore_index)
        return logits, loss

    def _run(self, idx, past=None):
        offset = past[0][0].shape[2] if past else 0
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(offset, offset + idx.shape[1], device=idx.device)))
        cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, past[i] if past else None)
            cache.append(kv)
        return self.head(self.ln_f(x)), cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.95, eos=None, memory=0, min_p=0.0,
                 return_logprob=False):
        """Sample with a KV cache. idx starts with BOS STYLE INST; when the context is full, restart the
        cache from STYLE INST + the most recent tokens (like the mid-piece windows seen in training).
        memory > 0 also keeps the piece's first `memory` tokens (its opening) in every restarted context,
        so the model can still "hear" the opening theme minutes later.
        temperature may be a float or a per-token tensor (see Tokenizer.temperatures).
        min_p drops tokens whose probability is below min_p x the most likely token's.
        return_logprob also returns each row's mean log-probability of its sampled tokens under the
        untempered model: how much the model itself "believes" what it played (a coherence score)."""
        block = self.config.block_size
        prefix = idx[:, 1:3]
        if not isinstance(temperature, (int, float)):
            temperature = torch.as_tensor(temperature, device=idx.device)
        pending, past = idx[:, -block:], None
        logp_sum = torch.zeros(idx.shape[0], device=idx.device)
        count = torch.zeros(idx.shape[0], device=idx.device)
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            if past is not None and past[0][0].shape[2] + pending.shape[1] > block:
                recent = block * 3 // 4 - memory
                pending, past = torch.cat([prefix, idx[:, 3:3 + memory], idx[:, -recent:]], dim=1), None
            logits, past = self._run(pending, past)
            logits = logits[:, -1].float()
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_p, order = probs.sort(descending=True)
            sorted_p[sorted_p.cumsum(-1) - sorted_p > top_p] = 0  # nucleus sampling
            sorted_p[sorted_p < min_p * sorted_p[:, :1]] = 0
            nxt = order.gather(-1, torch.multinomial(sorted_p, 1))
            chosen = F.log_softmax(logits, dim=-1).gather(-1, nxt).squeeze(1)
            logp_sum += torch.where(finished, 0.0, chosen)
            count += (~finished).float()
            if eos is not None:
                finished |= nxt.squeeze(1) == eos
            idx = torch.cat([idx, nxt], dim=1)
            pending = nxt
            if eos is not None and finished.all():
                break
        return (idx, (logp_sum / count.clamp(min=1)).tolist()) if return_logprob else idx
