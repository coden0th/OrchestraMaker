"""Notes <-> token ids.

A piece is `BOS STYLE_x INST_y` followed by one group per note, then `EOS`:

    [TIME_k ...] PITCH_p VEL_v DUR_d        (v0, v1)
    [TIME_k ...] NOTE_p_v DUR_d             (compact: pitch and velocity in one token, piano range only)

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
PIANO_LOW, PIANO_HIGH = 21, 108               # compact tokens cover the 88 piano keys


class Tokenizer:
    def __init__(self, styles: list[str], instruments: list[str], compact: bool = False):
        self.styles, self.instruments, self.compact = list(styles), list(instruments), compact
        notes = ([f"NOTE_{p}_{v}" for p in range(PIANO_LOW, PIANO_HIGH + 1) for v in range(VEL_BINS)] if compact
                 else [f"PITCH_{p}" for p in range(128)] + [f"VEL_{v}" for v in range(VEL_BINS)])
        self.vocab = (["PAD", "BOS", "EOS"]
                      + [f"STYLE_{s}" for s in self.styles]
                      + [f"INST_{i}" for i in self.instruments]
                      + [f"TIME_{k}" for k in range(1, MAX_SHIFT + 1)]
                      + notes
                      + [f"DUR_{d}" for d in range(len(DUR_BINS))])
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.pad, self.bos, self.eos = 0, 1, 2
        # Per-token lookups shared by the sampler, the memorization check and the model's sampling:
        self.pitch_of = np.array([int(t.split("_")[1]) if t.startswith(("PITCH_", "NOTE_")) else -1
                                  for t in self.vocab])                       # -1: not a pitch token
        self.is_group_start = np.array([t.startswith(("TIME_", "PITCH_", "NOTE_")) for t in self.vocab])
        self.is_note_end = np.array([t.startswith("DUR_") for t in self.vocab])  # exactly one per note

    def __len__(self):
        return len(self.vocab)

    def config(self) -> dict:
        return {"styles": self.styles, "instruments": self.instruments, "compact": self.compact}

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
            if self.compact:
                pitch = n.pitch + 12 * ((PIANO_LOW - n.pitch + 11) // 12) if n.pitch < PIANO_LOW else n.pitch
                pitch = pitch - 12 * ((pitch - PIANO_HIGH + 11) // 12) if pitch > PIANO_HIGH else pitch
                ids += [self.index[f"NOTE_{pitch}_{vel}"], self.index[f"DUR_{dur}"]]
            else:
                ids += [self.index[f"PITCH_{n.pitch}"], self.index[f"VEL_{vel}"], self.index[f"DUR_{dur}"]]
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
            elif kind == "NOTE":
                p, v = value.split("_")
                pitch, velocity = int(p), int(v) * 128 // VEL_BINS + 128 // VEL_BINS // 2
            elif kind == "VEL":
                velocity = int(value) * 128 // VEL_BINS + 128 // VEL_BINS // 2
            elif kind == "DUR" and pitch is not None:
                notes.append(Note(pitch, clock * TIME_STEP, float(DUR_BINS[int(value)]), velocity))
                pitch = None
        return style, instrument, notes

    def time_steps(self) -> np.ndarray:
        """How far each token moves the clock, in TIME_STEPs (TIME_k -> k, everything else 0)."""
        return np.array([int(t[5:]) if t.startswith("TIME_") else 0 for t in self.vocab], dtype=np.float32)

    def dur_steps(self) -> np.ndarray:
        """How long each DUR token lasts, in TIME_STEPs (-1 for other tokens): lets sampling track sounding notes."""
        return np.array([DUR_BINS[int(t[4:])] / TIME_STEP if t.startswith("DUR_") else -1 for t in self.vocab],
                        dtype=np.float32)

    def temperatures(self, default: float, pitch: float | None = None) -> np.ndarray:
        """Per-token sampling temperature: `pitch` for pitch tokens (which notes), `default` for the rest
        (when, how long, how loud)."""
        t = np.full(len(self.vocab), default, dtype=np.float32)
        if pitch is not None:
            t[self.pitch_of >= 0] = pitch
        return t

    def transpose(self, ids: np.ndarray, semitones: int) -> np.ndarray:
        """Move every pitch token by `semitones` (callers keep the result inside the instrument's range)."""
        ids = ids.copy()
        pitched = self.pitch_of[ids] >= 0
        step = VEL_BINS if self.compact else 1  # compact NOTE tokens are laid out pitch-major
        low, high = (PIANO_LOW, PIANO_HIGH) if self.compact else (0, 127)
        shift = np.clip(self.pitch_of[ids[pitched]] + semitones, low, high) - self.pitch_of[ids[pitched]]
        ids[pitched] += shift * step
        return ids

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.config(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls(**json.loads(Path(path).read_text()))
