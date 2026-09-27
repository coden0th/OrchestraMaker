"""A small decoder-only transformer over note tokens.

arch="gpt2" (v0, v1): learned positions, LayerNorm, GELU MLP.
arch="llama" (v1.5+): rotary positions (RoPE), RMSNorm, SwiGLU MLP, no biases.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    arch: str = "gpt2"
    mlp_hidden: int | None = None  # llama: SwiGLU width (default ~8/3 * n_embd, rounded to 64)
    rope_base: float = 10000.0

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

    def forward(self, x, past=None, pos=None):
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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope_tables(pos, head_dim, base):
    """Rotary position embedding angles for these positions, computed once per forward pass."""
    half = head_dim // 2
    freqs = base ** (-torch.arange(half, device=pos.device, dtype=torch.float32) / half)
    angles = pos.float()[:, None] * freqs[None]                     # (T, half)
    return angles.cos(), angles.sin()


def rotate(x, rope):
    """Rotate each (first half, second half) channel pair by a position-dependent angle, so attention
    scores depend on relative distance."""
    cos, sin = (t.to(x.dtype) for t in rope)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class LlamaBlock(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head, self.dropout = c.n_head, c.dropout
        hidden = c.mlp_hidden or 64 * round(8 * c.n_embd / 3 / 64)
        self.ln1, self.ln2 = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.gate = nn.Linear(c.n_embd, hidden, bias=False)
        self.up = nn.Linear(c.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, c.n_embd, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, past=None, pos=None):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in self.qkv(self.ln1(x)).split(C, dim=2))
        q, k = rotate(q, pos), rotate(k, pos)   # pos: (cos, sin) tables; cached keys are already rotated
        if past is not None:
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        h = self.ln2(x)
        return x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h))), (k, v)


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        llama = c.arch == "llama"
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = None if llama else nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList((LlamaBlock if llama else Block)(c) for _ in range(c.n_layer))
        self.ln_f = RMSNorm(c.n_embd) if llama else nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.grad_checkpoint = False  # recompute activations in backward: less memory, ~30% slower
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) - (self.pos_emb.weight.numel() if self.pos_emb else 0)

    def forward(self, idx, targets=None, ignore_index=0):
        logits, _ = self._run(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=ignore_index)
        return logits, loss

    def _run(self, idx, past=None):
        offset = past[0][0].shape[2] if past else 0
        pos = torch.arange(offset, offset + idx.shape[1], device=idx.device)
        x = self.tok_emb(idx) + (self.pos_emb(pos) if self.pos_emb is not None else 0)
        if self.pos_emb is None:  # llama blocks take the rotary tables instead of positions
            pos = rope_tables(pos, self.config.n_embd // self.config.n_head, self.config.rope_base)
        x = self.drop(x)
        cache = []
        for i, block in enumerate(self.blocks):
            if self.grad_checkpoint and self.training:
                x = checkpoint(lambda h, b=block: b(h, None, pos)[0], x, use_reentrant=False)
                continue
            x, kv = block(x, past[i] if past else None, pos)
            cache.append(kv)
        return self.head(self.ln_f(x)), cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.95, eos=None, memory=0, min_p=0.0,
                 return_logprob=False, time_steps=None, min_steps=0):
        """Sample with a KV cache. idx starts with BOS STYLE INST; when the context is full, restart the
        cache from STYLE INST + the most recent tokens (like the mid-piece windows seen in training).
        memory > 0 also keeps the piece's first `memory` tokens (its opening) in every restarted context,
        so the model can still "hear" the opening theme minutes later.
        temperature may be a float or a per-token tensor (see Tokenizer.temperatures).
        time_steps (per token, see Tokenizer.time_steps) + min_steps: EOS is not allowed before each row has
        played that long (pieces in the data are often single movements that end after 1-3 minutes).
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
        clock = torch.zeros(idx.shape[0], device=idx.device)
        if time_steps is not None:
            time_steps = torch.as_tensor(time_steps, device=idx.device)
        for _ in range(max_new_tokens):
            if past is not None and past[0][0].shape[2] + pending.shape[1] > block:
                recent = block * 3 // 4 - memory
                pending, past = torch.cat([prefix, idx[:, 3:3 + memory], idx[:, -recent:]], dim=1), None
            logits, past = self._run(pending, past)
            logits = logits[:, -1].float()
            if eos is not None and time_steps is not None:
                logits[:, eos] = torch.where(clock < min_steps, float("-inf"), logits[:, eos])
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_p, order = probs.sort(descending=True)
            sorted_p[sorted_p.cumsum(-1) - sorted_p > top_p] = 0  # nucleus sampling
            sorted_p[sorted_p < min_p * sorted_p[:, :1]] = 0
            nxt = order.gather(-1, torch.multinomial(sorted_p, 1))
            chosen = F.log_softmax(logits, dim=-1).gather(-1, nxt).squeeze(1)
            logp_sum += torch.where(finished, 0.0, chosen)
            count += (~finished).float()
            if time_steps is not None:
                clock += time_steps[nxt.squeeze(1)]
            if eos is not None:
                finished |= nxt.squeeze(1) == eos
            idx = torch.cat([idx, nxt], dim=1)
            pending = nxt
            if eos is not None and finished.all():
                break
        return (idx, (logp_sum / count.clamp(min=1)).tolist()) if return_logprob else idx
