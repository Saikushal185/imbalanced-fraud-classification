"""SMOTE — Synthetic Minority Over-sampling, implemented from scratch.

For each minority sample we interpolate toward one of its k minority nearest
neighbours, creating new synthetic positives along the line segments between
them. Kept dependency-free (numpy + sklearn's NearestNeighbors) so the
mechanics are explicit rather than hidden behind `imbalanced-learn`.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(X_min, n_synthetic, k=5, seed=0):
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_min))).fit(X_min)
    neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]
    out = np.empty((n_synthetic, X_min.shape[1]))
    for i in range(n_synthetic):
        j = rng.integers(len(X_min))
        nbr = X_min[rng.choice(neigh[j])]
        gap = rng.random()
        out[i] = X_min[j] + gap * (nbr - X_min[j])
    return out


def smote_resample(X, y, seed=0):
    """Oversample the minority class up to parity with the majority."""
    X_min = X[y == 1]
    n_needed = int((y == 0).sum() - (y == 1).sum())
    synth = smote(X_min, n_needed, seed=seed)
    X_res = np.vstack([X, synth])
    y_res = np.r_[y, np.ones(len(synth), dtype=int)]
    return X_res, y_res
