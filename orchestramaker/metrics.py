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


QUOTE_SHARE = 0.2  # full-data fingerprint: original takes 0-8%, quotes 41-55%, real excerpts ~57%


def popular_share(notes: list[Note], fingerprint: np.ndarray, n: int = 8) -> float:
    """Share of a take's n-note pitch sequences found in the popular-repertoire fingerprint
    (scripts/build_fingerprint.py). Original takes: 0-8%; takes replaying K.545 or a meme-famous piece: 41-55%."""
    p = np.array([x.pitch for x in sorted(notes, key=lambda x: (x.start, x.pitch))], dtype=np.int64)
    if len(p) < n:
        return 0.0
    h = np.zeros(len(p) - n + 1, dtype=np.int64)
    for i in range(n):
        h = h * 131 + p[i:len(p) - n + 1 + i]
    h = np.unique(h)
    at = np.minimum(np.searchsorted(fingerprint, h), len(fingerprint) - 1)  # fingerprint is sorted (np.unique)
    return float((fingerprint[at] == h).mean())


def pick_take(takes: list[list[Note]], ref: list[dict], scores, fingerprint: np.ndarray | None = None,
              label: str = "log-prob") -> tuple[int, str]:
    """Among takes whose statistics look like real music of the style (not sparse, not looping) and that don't
    quote well-known pieces, the one with the highest score. scores: the model's coherence (mean
    log-probability), or a callable giving the critic's predicted rating for take i (only called for takes
    that pass the checks, since listening costs time).
    Coherence alone favours memorized passages - the model is most sure of what it has heard most often."""
    checked = []
    for i, notes in enumerate(takes):
        stats = describe(notes)
        if stats:
            cls = classify(stats, ref)
            if cls == "typical" and fingerprint is not None and popular_share(notes, fingerprint) > QUOTE_SHARE:
                cls = "quotes"
            checked.append((i, cls))
    if not checked:
        return 0, "empty"
    pool = [(i, c) for i, c in checked if c == "typical"] or checked
    score = scores if callable(scores) else scores.__getitem__
    best_score, i, cls = max((score(i), i, c) for i, c in pool)
    typical = sum(c == "typical" for _, c in checked)
    quotes = sum(c == "quotes" for _, c in checked)
    return i, (f"{cls}, best of {len(takes)} by {label} ({typical} typical"
               f"{f', {quotes} quoting known pieces' if quotes else ''}), {label} {best_score:.2f}")


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


def top_line(notes: list[Note], chord_window: float = 0.03) -> list[Note]:
    """The highest note of each onset (notes starting within chord_window count as one onset): roughly the melody."""
    line = []
    for n in sorted(notes, key=lambda x: x.start):
        if line and n.start - line[-1].start < chord_window:
            if n.pitch > line[-1].pitch:
                line[-1] = n
        else:
            line.append(n)
    return line


def theme_return(notes: list[Note], opening: float = 20, n: int = 5) -> float:
    """Share of the opening's melodic shapes (n successive intervals of the top line, so a theme counts in any
    key) that come back in the last third of the piece."""
    line = top_line(notes)
    if len(line) < 3 * n:
        return 0.0
    t0, end = line[0].start, line[-1].start  # recordings can begin with silence
    shapes = lambda part: {tuple(np.diff([x.pitch for x in part[i:i + n + 1]])) for i in range(len(part) - n)}
    first = shapes([x for x in line if x.start < t0 + opening])
    last = shapes([x for x in line if x.start >= t0 + (end - t0) * 2 / 3])
    return len(first & last) / len(first) if first else 0.0


CLASHES = (1, 6, 11)  # minor 2nd, tritone, major 7th (in any octave)


def dissonance(notes: list[Note], step: float = 0.25) -> float:
    """Share of simultaneously sounding pitch pairs (sampled every `step` s) that clash (CLASHES)."""
    if not notes:
        return 0.0
    bad = total = 0
    for t in np.arange(0, max(n.end for n in notes), step):
        p = sorted({n.pitch for n in notes if n.start <= t < n.end})
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                total += 1
                bad += (p[j] - p[i]) % 12 in CLASHES
    return bad / total if total else 0.0


def key_clarity(notes: list[Note]) -> float:
    w = np.bincount([n.pitch % 12 for n in notes], weights=[n.duration for n in notes], minlength=12)
    if not w.sum():
        return 0.0
    return float(max(np.corrcoef(np.roll(p, k), w)[0, 1] for p in (MAJOR, MINOR) for k in range(12)))


def harmony_profile(notes: list[Note], window: float = 10) -> dict[str, float]:
    """Harmonic "chaos" of a take, including its worst stretch - blind ratings follow the worst part:
    worst-window dissonance correlates -0.43 with them, peak note density only -0.17."""
    end = max((n.start for n in notes), default=0)
    parts = [[n for n in notes if w <= n.start < w + window] for w in np.arange(0, max(end - 5, 0) + 1, window)]
    parts = [p for p in parts if len(p) >= 8] or [notes]
    ds, kc = [dissonance(p) for p in parts], [key_clarity(p) for p in parts]
    return {"worst_dissonance": max(ds), "mean_dissonance": float(np.mean(ds)),
            "worst_key_clarity": min(kc), "mean_key_clarity": float(np.mean(kc))}
