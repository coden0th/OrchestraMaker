"""Simple, interpretable statistics of a performance, to compare generated music with real music."""

import numpy as np

from .instruments import Note

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])  # Krumhansl-Kessler
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def describe(notes: list[Note], n: int = 8) -> dict[str, float]:
    if len(notes) < n + 2:
        return {}
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))
    pitch = np.array([x.pitch for x in notes])
    start = np.array([x.start for x in notes])
    dur = np.array([x.duration for x in notes])
    length = max(start[-1] - start[0], 1e-3)

    gaps = np.diff(start)
    gaps = gaps[gaps > 0.02]  # ignore notes of the same chord
    hist = np.histogram(np.log(gaps), bins=12, range=(np.log(0.02), np.log(4)))[0] if len(gaps) else np.ones(12)
    p = hist[hist > 0] / hist.sum()

    grams = [tuple(pitch[i:i + n]) for i in range(len(pitch) - n + 1)]
    weights = np.bincount(pitch % 12, weights=dur, minlength=12)
    key_fit = max(np.corrcoef(np.roll(profile, k), weights)[0, 1] for profile in (MAJOR, MINOR) for k in range(12))
    return {
        "notes_per_s": len(notes) / length,
        "distinct_pitches": len(set(pitch.tolist())),
        "rhythm_bits": float(-(p * np.log2(p)).sum()),        # variety of time gaps between notes
        "repetition": 1 - len(set(grams)) / len(grams),        # share of n-note sequences that are repeats
        "key_clarity": float(key_fit),                          # how well it fits one major/minor key
    }


KEYS = ["notes_per_s", "distinct_pitches", "rhythm_bits", "repetition", "key_clarity"]


def classify(stats: dict, ref: list[dict]) -> str:
    """repetitive: repeats itself more than 90% of real windows; sparse: fewer notes or pitches than 90%."""
    q = {k: np.percentile([r[k] for r in ref], [10, 90]) for k in KEYS}
    if stats["repetition"] > q["repetition"][1]:
        return "repetitive"
    if stats["notes_per_s"] < q["notes_per_s"][0] or stats["distinct_pitches"] < q["distinct_pitches"][0]:
        return "sparse"
    return "typical"


def most_typical(takes: list[list[Note]], ref: list[dict]) -> tuple[int, str]:
    """Index of the take closest to real music (distance to the real medians in units of the 10-90% spread),
    preferring takes classified as typical."""
    med = {k: np.median([r[k] for r in ref]) for k in KEYS}
    spread = {k: np.subtract(*np.percentile([r[k] for r in ref], [90, 10])) + 1e-6 for k in KEYS}
    scored = []
    for i, notes in enumerate(takes):
        stats = describe(notes)
        if stats:
            cls = classify(stats, ref)
            scored.append((cls != "typical", sum(abs(stats[k] - med[k]) / spread[k] for k in KEYS), i, cls))
    if not scored:
        return 0, "empty"
    _, _, i, cls = min(scored)
    return i, f"{cls}, best of {len(takes)} ({sum(c == 'typical' for *_, c in scored)} typical)"


def reference_windows(pieces: list[dict], tokenizer, style: str, seconds: float, count: int, seed: int = 0,
                      from_start: bool = False, instrument: str | None = None):
    """describe() of `seconds`-long windows of real validation performances of a style: random windows,
    or the openings (from_start), which is what a take generated from scratch should be compared with."""
    rng = np.random.default_rng(seed)
    chosen = [p for p in pieces if p["style"] == style and p["split"] == "validation"
              and (instrument is None or p["instrument"] == instrument)]
    out = []
    for i in rng.permutation(len(chosen)):
        _, _, notes = tokenizer.decode(np.asarray(chosen[i]["tokens"]))
        end = max(n.end for n in notes)
        if end < seconds * 1.5:
            continue
        t0 = min(n.start for n in notes) if from_start else rng.uniform(0, end - seconds)
        stats = describe([Note(n.pitch, n.start - t0, n.duration, n.velocity) for n in notes
                          if t0 <= n.start < t0 + seconds])
        if stats:
            out.append(stats)
        if len(out) == count:
            break
    return out
