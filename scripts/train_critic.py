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
from orchestramaker.metrics import KEYS, QUOTE_SHARE, describe, popular_share  # noqa: E402

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


def select(views, y, folds=3, seed=0):
    """(view index, alpha) with the best inner-CV Spearman; views = list of feature matrices."""
    order = np.random.default_rng(seed).permutation(len(y))
    best, best_rho = (0, ALPHAS[0]), -2.0
    for v, X in enumerate(views):
        for alpha in ALPHAS:
            pred = np.zeros(len(y))
            for f in range(folds):
                test = order[f::folds]
                train = np.setdiff1d(order, test)
                pred[test] = ridge_predict(ridge_fit(X[train], y[train], alpha), X[test])
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
            v, alpha = select([X[train] for X in views], y[train], seed=r)
            pred[test] = ridge_predict(ridge_fit(views[v][train], y[train], alpha), views[v][test])
        rhos.append(spearman(pred, y))
        maes.append(np.abs(np.clip(pred, 1, 5) - y).mean())
    return float(np.mean(rhos)), float(np.mean(maes))


def main():
    import hashlib
    from orchestramaker.ear import Ear
    ratings = json.loads(RATINGS.read_text())
    fingerprint = np.load(ROOT / "data/fingerprint_popular_8.npy")
    ear = Ear()
    ear_X, stat_X, y, skipped = [], [], [], {"changed": 0, "quote": 0}
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
        ear_X.append(ear.listen(path.parent / take["versions"][0]["audio"]))
        stat_X.append([stats[k] for k in KEYS])
        y.append(r["rating"])
    ear_X, stat_X, y = np.stack(ear_X), np.array(stat_X), np.array(y, dtype=float)
    print(f"{len(y)} ratings used (skipped: {skipped}); distribution: "
          + " ".join(f"{k}:{int((y == k).sum())}" for k in range(1, 6)))

    layers = [ear_X[:, layer] for layer in range(ear_X.shape[1])]
    stats_rho, stats_mae = nested_cv([stat_X], y)
    rho, mae = nested_cv(layers, y)
    both_rho, both_mae = nested_cv([np.concatenate([L, stat_X], axis=1) for L in layers], y)
    print(f"critic from note statistics only : Spearman {stats_rho:.2f}, mean error {stats_mae:.2f} stars")
    print(f"critic from the ear (MERT)       : Spearman {rho:.2f}, mean error {mae:.2f} stars")
    print(f"ear + statistics                 : Spearman {both_rho:.2f}, mean error {both_mae:.2f} stars")
    print(f"(always guessing the average rating: mean error {np.abs(y - y.mean()).mean():.2f} stars)")

    layer, alpha = select(layers, y)  # the final critic: chosen on all ratings
    print(f"final critic: MERT layer {layer}, alpha {alpha}")
    model = ridge_fit(ear_X[:, layer], y, alpha)
    CRITIC.parent.mkdir(exist_ok=True)
    np.savez(CRITIC, layer=layer, alpha=alpha, cv_spearman=rho, cv_mae=mae, n=len(y), **model)
    print(f"saved {CRITIC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
