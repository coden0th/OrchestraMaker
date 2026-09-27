"""Training windows: sample by source (so the small jazz set isn't drowned out), crop, transpose."""

import numpy as np
import torch

from .instruments import INSTRUMENTS
from .tokenizer import Tokenizer


class BatchSampler:
    def __init__(self, pieces: list[dict], tokenizer: Tokenizer, split: str, source_weights: dict[str, float],
                 block_size: int, max_transpose: int = 0, seed: int = 0):
        self.tok, self.block, self.max_transpose = tokenizer, block_size, max_transpose
        self.rng = np.random.default_rng(seed)
        self.sources, self.pieces, self.piece_probs = [], {}, {}
        for source in source_weights:
            chosen = [p for p in pieces if p["source"] == source and p["split"] == split]
            if chosen:
                self.sources.append(source)
                self.pieces[source] = chosen
                lengths = np.array([len(p["tokens"]) for p in chosen], dtype=float)
                self.piece_probs[source] = lengths / lengths.sum()  # longer pieces get more windows
        w = np.array([source_weights[s] for s in self.sources], dtype=float)
        self.source_probs = w / w.sum()
        vocab = tokenizer.vocab
        self.is_group_start = np.array([t.startswith(("TIME_", "PITCH_")) for t in vocab])
        self.is_pitch = np.array([t.startswith("PITCH_") for t in vocab])

    def batch(self, batch_size: int, source: str | None = None, device="cpu"):
        rows = np.stack([self._window(source) for _ in range(batch_size)])
        rows = torch.from_numpy(rows).to(device, non_blocking=True)
        return rows[:, :-1], rows[:, 1:]

    def _window(self, source=None) -> np.ndarray:
        source = source or self.rng.choice(self.sources, p=self.source_probs)
        piece = self.pieces[source][self.rng.choice(len(self.pieces[source]), p=self.piece_probs[source])]
        ids = piece["tokens"].astype(np.int64)
        n = self.block + 1
        start = 0 if len(ids) <= n else int(self.rng.integers(0, len(ids) - n))
        if start == 0:
            seq = ids[:n]
        else:
            # Mid-piece window: re-state style + instrument so the conditioning is always visible.
            while not self.is_group_start[ids[start]]:
                start += 1
            seq = np.concatenate([ids[1:3], ids[start:start + n - 2]])
        if self.max_transpose:
            seq = self._transpose(seq, piece["instrument"])
        return np.pad(seq, (0, n - len(seq)), constant_values=self.tok.pad)

    def _transpose(self, seq, instrument):
        pitches = seq[self.is_pitch[seq]] - self.tok.pitch0
        if not len(pitches):
            return seq
        inst = INSTRUMENTS[instrument]
        lo, hi = pitches.min(), pitches.max()
        # Never push notes further outside the instrument's range than the original player went.
        down = max(-self.max_transpose, min(inst.low, lo) - lo)
        up = min(self.max_transpose, max(inst.high, hi) - hi)
        return self.tok.transpose(seq, int(self.rng.integers(down, up + 1)))
