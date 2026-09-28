"""Teach a critic the listener's taste from their blind 1-5 ratings (outputs/ratings.json).

Each rated take is heard by the ear (orchestramaker/ear.py, MERT). A ridge regression on one layer's
features predicts the rating. Which layer and how much regularization are chosen inside each training
fold (nested cross-validation), so the reported accuracy is not flattered by that choice. The same critic
built from note statistics alone (metrics.describe) shows whether listening adds anything.
Takes that quote well-known pieces are left out: rating a famous melody highly must not teach the critic
to reward memorization.

Run: .venv/bin/python scripts/train_critic.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestramaker.instruments import Note  # noqa: E402
from orchestramaker.metrics import KEYS, QUOTE_SHARE, describe, harmony_profile, popular_share  # noqa: E402

RATINGS = ROOT / "outputs/ratings.json"
CRITIC = ROOT / "checkpoints/critic.npz"


def ridge_fit(X, y, alpha):
    """Ridge regression in dual form: a (samples x samples) solve, fast with ~200 ratings and ~1500 features."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    a = np.linalg.solve(Z @ Z.T + alpha * np.eye(len(Z)), y - y.mean())
    return {"mu": mu, "sd": sd, "w": Z.T @ a, "b": y.mean()}


def ridge_predict(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


ALPHAS = [1, 10, 100, 1000, 10000]


class TakeView:
    """One feature vector per take."""
    def __init__(self, X):
        self.X = X

    def fit(self, idx, y, alpha):
        return ridge_fit(self.X[idx], y[idx], alpha)

    def predict(self, model, idx):
        return ridge_predict(model, self.X[idx])


class WindowView:
    """One feature vector per ~10 s window: trained with every window labelled by its take's rating,
    a take scored by its weakest window (agg=min) or its average window - "a good start does not
    save a chaotic ending", as the listener rates."""
    def __init__(self, windows, agg):
        self.windows, self.agg = windows, agg

    def fit(self, idx, y, alpha):
        X = np.concatenate([self.windows[i] for i in idx])
        return ridge_fit(X, np.concatenate([np.full(len(self.windows[i]), y[i]) for i in idx]), alpha)

    def predict(self, model, idx):
        return np.array([self.agg(ridge_predict(model, self.windows[i])) for i in idx])


def select(views, y, folds=3, seed=0):
    """(view index, alpha) with the best inner-CV Spearman."""
    order = np.random.default_rng(seed).permutation(len(y))
    best, best_rho = (0, ALPHAS[0]), -2.0
    for v, view in enumerate(views):
        for alpha in ALPHAS:
            pred = np.zeros(len(y))
            for f in range(folds):
                test = order[f::folds]
                train = np.setdiff1d(order, test)
                pred[test] = view.predict(view.fit(train, y, alpha), test)
            rho = spearman(pred, y)
            if rho > best_rho:
                best, best_rho = (v, alpha), rho
    return best


def nested_cv(views, y, folds=5, repeats=5):
    """Out-of-fold Spearman and mean error, with the view and alpha chosen inside each training fold,
    so the choice among ~65 settings does not flatter the result."""
    rhos, maes = [], []
    for r in range(repeats):
        order = np.random.default_rng(100 + r).permutation(len(y))
        pred = np.zeros(len(y))
        for f in range(folds):
            test = order[f::folds]
            train = np.setdiff1d(order, test)
            v, alpha = select([Subset(view, train) for view in views], y[train], seed=r)
            pred[test] = views[v].predict(views[v].fit(train, y, alpha), test)
        rhos.append(spearman(pred, y))
        maes.append(np.abs(np.clip(pred, 1, 5) - y).mean())
    return float(np.mean(rhos)), float(np.mean(maes))


class Subset:
    """A view restricted to some takes (re-indexed from 0), for selection inside a training fold."""
    def __init__(self, view, idx):
        self.view, self.idx = view, np.asarray(idx)

    def fit(self, idx, y, alpha):
        full_y = np.zeros(self.idx.max() + 1)
        full_y[self.idx] = y
        return self.view.fit(self.idx[idx], full_y, alpha)

    def predict(self, model, idx):
        return self.view.predict(model, self.idx[idx])


def main():
    import hashlib
    from orchestramaker.ear import Ear
    ratings = json.loads(RATINGS.read_text())
    fingerprint = np.load(ROOT / "data/fingerprint_popular_8.npy")
    ear = Ear()
    ear_X, win_X, stat_X, y, skipped = [], [], [], [], {"changed": 0, "quote": 0}
    for url, r in ratings.items():
        path = ROOT / url.lstrip("/")
        if not path.exists() or hashlib.sha1(path.read_bytes()).hexdigest() != r.get("sha1"):
            skipped["changed"] += 1  # the take was re-rendered after it was rated
            continue
        take = json.loads(path.read_text())
        notes = [Note(*n) for n in take["versions"][0]["tracks"][0]["notes"]]
        if take["meta"].get("source") != "real" and popular_share(notes, fingerprint) > QUOTE_SHARE:
            skipped["quote"] += 1
            continue
        stats = describe(notes) or {k: 0.0 for k in KEYS}
        whole, windows = ear.hear(path.parent / take["versions"][0]["audio"])
        ear_X.append(whole)
        win_X.append(windows)
        stat_X.append([stats[k] for k in KEYS] + list(harmony_profile(notes).values()))
        y.append(r["rating"])
    ear_X, stat_X, y = np.stack(ear_X), np.array(stat_X), np.array(y, dtype=float)
    print(f"{len(y)} ratings used (skipped: {skipped}); distribution: "
          + " ".join(f"{k}:{int((y == k).sum())}" for k in range(1, 6)))

    n_layers = ear_X.shape[1]
    candidates = {
        "note statistics only": [TakeView(stat_X)],
        "ear + note statistics": [TakeView(np.concatenate([ear_X[:, L], stat_X], 1)) for L in range(n_layers)],
        "ear, whole take": [TakeView(ear_X[:, L]) for L in range(n_layers)],
        "ear, weakest 10 s window": [WindowView([w[:, L] for w in win_X], np.min) for L in range(n_layers)],
        "ear, average 10 s window": [WindowView([w[:, L] for w in win_X], np.mean) for L in range(n_layers)],
    }
    results = {name: nested_cv(views, y) for name, views in candidates.items()}
    for name, (rho, mae) in results.items():
        print(f"critic from {name:26s}: Spearman {rho:.2f}, mean error {mae:.2f} stars")
    print(f"(always guessing the average rating: mean error {np.abs(y - y.mean()).mean():.2f} stars)")

    name = max((n for n in results if n.startswith("ear")), key=lambda n: results[n][0])
    views = candidates[name]
    v, alpha = select(views, y)  # the final critic: chosen on all ratings
    model = views[v].fit(np.arange(len(y)), y, alpha)
    rho, mae = results[name]
    print(f"final critic: {name}, MERT layer {v}, alpha {alpha}")
    CRITIC.parent.mkdir(exist_ok=True)
    np.savez(CRITIC, layer=v, alpha=alpha, cv_spearman=rho, cv_mae=mae, n=len(y), with_stats="statistics" in name,
             aggregate="min" if "weakest" in name else "mean" if "average" in name else "take", **model)
    print(f"saved {CRITIC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
