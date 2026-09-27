"""Notes <-> token ids.

A piece is `BOS STYLE_x INST_y` followed by one group per note, then `EOS`:

    [TIME_k ...] PITCH_p VEL_v DUR_d

Time is performance time (seconds as played, not a score grid), so swing and rubato survive.
TIME_k moves the clock forward k * 10 ms; longer gaps chain several TIME tokens; chord notes
share an onset and need none.
"""

import json
from pathlib import Path

import numpy as np

from .instruments import Note

TIME_STEP = 0.01
MAX_SHIFT = 100                               # steps per TIME token (1 s)
DUR_BINS = np.geomspace(0.02, 8.0, 64)        # seconds, log-spaced
VEL_BINS = 16


class Tokenizer:
    def __init__(self, styles: list[str], instruments: list[str]):
        self.styles, self.instruments = list(styles), list(instruments)
        self.vocab = (["PAD", "BOS", "EOS"]
                      + [f"STYLE_{s}" for s in self.styles]
                      + [f"INST_{i}" for i in self.instruments]
                      + [f"TIME_{k}" for k in range(1, MAX_SHIFT + 1)]
                      + [f"PITCH_{p}" for p in range(128)]
                      + [f"VEL_{v}" for v in range(VEL_BINS)]
                      + [f"DUR_{d}" for d in range(len(DUR_BINS))])
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.pad, self.bos, self.eos = 0, 1, 2
        self.pitch0 = self.index["PITCH_0"]

    def __len__(self):
        return len(self.vocab)

    def encode(self, notes: list[Note], style: str, instrument: str) -> list[int]:
        ids = [self.bos, self.index[f"STYLE_{style}"], self.index[f"INST_{instrument}"]]
        clock = 0  # in TIME_STEPs, quantized from absolute onsets so rounding errors don't accumulate
        for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
            shift = max(0, round(n.start / TIME_STEP) - clock)
            clock += shift
            while shift > 0:
                step = min(shift, MAX_SHIFT)
                ids.append(self.index[f"TIME_{step}"])
                shift -= step
            dur = int(np.abs(np.log(DUR_BINS) - np.log(max(n.duration, 1e-3))).argmin())
            vel = min(VEL_BINS - 1, n.velocity * VEL_BINS // 128)
            ids += [self.pitch0 + n.pitch, self.index[f"VEL_{vel}"], self.index[f"DUR_{dur}"]]
        return ids + [self.eos]

    def decode(self, ids) -> tuple[str | None, str | None, list[Note]]:
        """Tolerant of malformed model output: incomplete note groups are dropped."""
        style = instrument = pitch = None
        clock, velocity, notes = 0, 80, []
        for i in ids:
            kind, _, value = self.vocab[int(i)].partition("_")
            if kind == "EOS":
                break
            if kind == "STYLE":
                style = value
            elif kind == "INST":
                instrument = value
            elif kind == "TIME":
                clock += int(value)
            elif kind == "PITCH":
                pitch = int(value)
            elif kind == "VEL":
                velocity = int(value) * 128 // VEL_BINS + 128 // VEL_BINS // 2
            elif kind == "DUR" and pitch is not None:
                notes.append(Note(pitch, clock * TIME_STEP, float(DUR_BINS[int(value)]), velocity))
                pitch = None
        return style, instrument, notes

    def transpose(self, ids: np.ndarray, semitones: int) -> np.ndarray:
        ids = ids.copy()
        pitches = (ids >= self.pitch0) & (ids < self.pitch0 + 128)
        ids[pitches] = np.clip(ids[pitches] + semitones, self.pitch0, self.pitch0 + 127)
        return ids

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"styles": self.styles, "instruments": self.instruments}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls(**json.loads(Path(path).read_text()))
