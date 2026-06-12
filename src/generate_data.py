"""Synthetic card-transaction data with ~1% fraud.

Legitimate and fraudulent transactions are drawn from overlapping
distributions over a few engineered features (amount, hour, distance from
home, recent velocity, merchant risk). Fraud is rare and only partially
separable — exactly the regime where accuracy is a useless metric.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
SEED = 17
N = 40000
FRAUD_RATE = 0.012
RNG = np.random.default_rng(SEED)


def main():
    n_fraud = int(N * FRAUD_RATE)
    n_legit = N - n_fraud

    def block(n, amount, hour, dist, velocity, mrisk):
        return np.column_stack([
            np.clip(RNG.lognormal(*amount, n), 1, None),
            RNG.normal(*hour, n) % 24,
            np.clip(RNG.normal(*dist, n), 0, None),
            np.clip(RNG.normal(*velocity, n), 0, None),
            np.clip(RNG.normal(*mrisk, n), 0, 1),
        ])

    legit = block(n_legit, (3.4, 0.7), (14, 4), (8, 6), (2, 1.5), (0.3, 0.15))
    fraud = block(n_fraud, (4.2, 1.0), (2, 5), (60, 40), (6, 3), (0.6, 0.2))

    X = np.vstack([legit, fraud])
    y = np.r_[np.zeros(n_legit), np.ones(n_fraud)].astype(int)
    order = RNG.permutation(N)
    X, y = X[order], y[order]

    df = pd.DataFrame(X, columns=[
        "amount", "hour", "dist_from_home", "velocity_1h", "merchant_risk"])
    df = df.round(3)
    df["is_fraud"] = y
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} transactions to {OUT}  "
          f"({y.sum()} fraud = {y.mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
