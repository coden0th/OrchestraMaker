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
        self.info = {"cv_spearman": float(m["cv_spearman"]), "ratings": int(m["n"])}
        self.ear = Ear(device)

    def score(self, instrument: Instrument, notes: list[Note]) -> float:
        with tempfile.TemporaryDirectory() as tmp:
            wav = render([(instrument, notes)], Path(tmp) / "take.wav")
            features = self.ear.listen(wav)[self.layer]
        return float(np.clip(((features - self.mu) / self.sd) @ self.w + self.b, 1, 5))
