"""The "ear": listen to a rendered take with a pretrained music-audio model and describe what it hears.

MERT-v1-95M (m-a-p, CC BY-NC 4.0) is a self-supervised model trained on music audio; its hidden states
capture pitch, harmony, rhythm and timbre. We keep, per layer, the mean and standard deviation over time,
so a take becomes a fixed-size vector a small critic can learn the listener's taste from.
"""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL = "m-a-p/MERT-v1-95M"
RATE = 24000          # MERT's sample rate
WINDOW = 10           # seconds per forward pass (MERT was trained on short clips)
CACHE = Path(__file__).resolve().parent.parent / "outputs/ear_cache"


def resample(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    """FFT resampling: plenty for a 30-second clip (and no extra dependency)."""
    if rate == target:
        return x
    n = int(round(len(x) * target / rate))
    spectrum = np.fft.rfft(x)
    keep = n // 2 + 1
    spectrum = spectrum[:keep] if len(spectrum) >= keep else np.pad(spectrum, (0, keep - len(spectrum)))
    return np.fft.irfft(spectrum, n) * (n / len(x))


class Ear:
    def __init__(self, device="cuda"):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.device = device
        self.model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(device).eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, trust_remote_code=True)
        # MERT's remote code ignores output_hidden_states under transformers 5, so collect the 13 states
        # (encoder input after its layer norm, then each of the 12 layers) with forward hooks.
        self.states = []
        keep = lambda module, inputs, output: self.states.append(output[0] if isinstance(output, tuple) else output)
        self.model.encoder.layer_norm.register_forward_hook(keep)
        for layer in self.model.encoder.layers:
            layer.register_forward_hook(keep)

    @torch.no_grad()
    def listen(self, wav: str | Path) -> np.ndarray:
        """(layers, 2 * hidden): per-layer mean and std over time of the whole clip. Cached by file content."""
        wav = Path(wav)
        key = hashlib.sha1(wav.read_bytes()).hexdigest()
        cached = CACHE / f"{key}.npy"
        if cached.exists():
            return np.load(cached)
        audio, rate = sf.read(wav, dtype="float32")
        audio = resample(audio.mean(axis=1) if audio.ndim == 2 else audio, rate, RATE)
        states = []
        for start in range(0, max(len(audio) - RATE, 1), WINDOW * RATE):
            chunk = audio[start:start + WINDOW * RATE]
            inputs = self.processor(chunk, sampling_rate=RATE, return_tensors="pt").to(self.device)
            self.states.clear()
            self.model(**inputs)
            states.append(torch.stack(self.states)[:, 0].float().cpu())                # (layers, T, H)
        frames = torch.cat(states, dim=1)
        vector = torch.cat([frames.mean(1), frames.std(1)], dim=1).numpy()
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cached, vector)
        return vector
