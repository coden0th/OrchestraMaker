"""Training windows: sample by source (so the small jazz set isn't drowned out), crop, transpose."""

import json
from pathlib import Path

import numpy as np
import torch

from .instruments import INSTRUMENTS
from .tokenizer import Tokenizer


def load_pieces(data_dir: str | Path) -> list[dict]:
    """v0: pieces.pt (token arrays in memory). v1+: index.json + tokens.bin (memory-mapped uint16)."""
    data_dir = Path(data_dir)
    if (data_dir / "pieces.pt").exists():
        return torch.load(data_dir / "pieces.pt", weights_only=False)["pieces"]
    index = json.loads((data_dir / "index.json").read_text())
    tokens = np.memmap(data_dir / "tokens.bin", dtype=np.uint16, mode="r")
    return [{"source": src, "style": style, "split": split, "title": title, "instrument": index["instrument"],
             "tokens": tokens[off:off + n]}
            for src, style, split, title, off, n in zip(index["source"], index["style"], index["split"],
                                                        index["title"], index["offset"], index["length"])]


class BatchSampler:
    def __init__(self, pieces: list[dict], tokenizer: Tokenizer, split: str, source_weights: dict[str, float],
                 block_size: int, max_transpose: int = 0, seed: int = 0):
        self.tok, self.block, self.max_transpose = tokenizer, block_size, max_transpose
        self.rng = np.random.default_rng(seed)
        self.sources, self.pieces, self.piece_cdf = [], {}, {}
        for source in source_weights:
            chosen = [p for p in pieces if p["source"] == source and p["split"] == split]
            if chosen:
                self.sources.append(source)
                self.pieces[source] = chosen
                lengths = np.array([len(p["tokens"]) for p in chosen], dtype=float)
                self.piece_cdf[source] = np.cumsum(lengths) / lengths.sum()  # longer pieces get more windows
        w = np.array([source_weights[s] for s in self.sources], dtype=float)
        self.source_probs = w / w.sum()
        self.is_group_start = tokenizer.is_group_start

    def batch(self, batch_size: int, source: str | None = None, device="cpu"):
        rows = np.stack([self._window(source) for _ in range(batch_size)])
        rows = torch.from_numpy(rows).to(device, non_blocking=True)
        return rows[:, :-1], rows[:, 1:]

    def _window(self, source=None) -> np.ndarray:
        source = source or self.rng.choice(self.sources, p=self.source_probs)
        cdf = self.piece_cdf[source]
        piece = self.pieces[source][min(int(np.searchsorted(cdf, self.rng.random())), len(cdf) - 1)]
        ids = piece["tokens"]  # may be a memory map: only the window is read from disk
        n = self.block + 1
        start = 0 if len(ids) <= n else int(self.rng.integers(0, len(ids) - n))
        if start == 0:
            seq = ids[:n].astype(np.int64)
        else:
            # Mid-piece window: re-state style + instrument so the conditioning is always visible.
            while not self.is_group_start[ids[start]]:
                start += 1
            seq = np.concatenate([ids[1:3], ids[start:start + n - 2]]).astype(np.int64)
        if self.max_transpose:
            seq = self._transpose(seq, piece["instrument"])
        return np.pad(seq, (0, n - len(seq)), constant_values=self.tok.pad)

    def _transpose(self, seq, instrument):
        pitches = self.tok.pitch_of[seq]
        pitches = pitches[pitches >= 0]
        if not len(pitches):
            return seq
        inst = INSTRUMENTS[instrument]
        lo, hi = pitches.min(), pitches.max()
        # Never push notes further outside the instrument's range than the original player went.
        down = max(-self.max_transpose, min(inst.low, lo) - lo)
        up = min(self.max_transpose, max(inst.high, hi) - hi)
        return self.tok.transpose(seq, int(self.rng.integers(down, up + 1)))
