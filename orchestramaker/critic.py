"""The critic: predicts how the listener would rate a take (1-5), by rendering and hearing it.

Trained by scripts/train_critic.py from blind ratings; ridge regression on one MERT layer (ear.py).
"""

import tempfile
from pathlib import Path

import numpy as np

from .ear import Ear
from .instruments import Instrument, Note
from .render import render

CRITIC = Path(__file__).resolve().parent.parent / "checkpoints/critic.npz"


class Critic:
    def __init__(self, path: Path = CRITIC, device="cuda"):
        m = np.load(path)
        self.layer, self.mu, self.sd, self.w, self.b = int(m["layer"]), m["mu"], m["sd"], m["w"], float(m["b"])
        self.aggregate = str(m["aggregate"]) if "aggregate" in m else "take"
        self.info = {"cv_spearman": float(m["cv_spearman"]), "ratings": int(m["n"]), "aggregate": self.aggregate}
        self.ear = Ear(device)

    def score(self, instrument: Instrument, notes: list[Note]) -> float:
        with tempfile.TemporaryDirectory() as tmp:
            wav = render([(instrument, notes)], Path(tmp) / "take.wav")
            whole, windows = self.ear.hear(wav)
        if self.aggregate == "take":
            features = whole[self.layer][None]
        else:  # windows: the take is as good as its weakest (min) or average (mean) 10-second stretch
            features = windows[:, self.layer]
        scores = ((features - self.mu) / self.sd) @ self.w + self.b
        return float(np.clip(scores.min() if self.aggregate == "min" else scores.mean(), 1, 5))
