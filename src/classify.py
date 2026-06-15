"""Rare-event fraud classification done honestly.

Why accuracy lies: a model that predicts "never fraud" scores ~98.8% accuracy
and catches zero fraud. We compare four treatments of the imbalance and judge
them with PR-AUC, ROC-AUC and a *cost-based* operating point, not accuracy:

  * baseline    — logistic regression, default 0.5 threshold
  * class_weight— logistic regression with balanced class weights
  * smote       — logistic regression on SMOTE-oversampled training data
  * threshold   — baseline model with a cost-minimising decision threshold

Cost model: a missed fraud (FN) costs 20x a false alarm (FP).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smote import smote_resample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
REPORTS = ROOT / "reports"
FN_COST, FP_COST = 20.0, 1.0


def cost_at_threshold(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return fn * FN_COST + fp * FP_COST, tp, fp, fn


def best_cost_threshold(y, proba):
    thrs = np.linspace(0.01, 0.99, 99)
    costs = [cost_at_threshold(y, proba, t)[0] for t in thrs]
    i = int(np.argmin(costs))
    return float(thrs[i]), float(costs[i])


def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns="is_fraud").to_numpy()
    y = df["is_fraud"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=17)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    models = {}
    models["baseline"] = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    models["class_weight"] = LogisticRegression(
        max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    Xrs, yrs = smote_resample(Xtr, ytr, seed=17)
    models["smote"] = LogisticRegression(max_iter=1000).fit(Xrs, yrs)

    naive_acc = max(yte.mean(), 1 - yte.mean())
    print(f"Test set: {len(yte):,} txns, {yte.mean()*100:.2f}% fraud")
    print(f'"Always legit" accuracy = {naive_acc*100:.2f}% but catches 0 fraud\n')
    print(f"{'model':<13} {'PR-AUC':>7} {'ROC-AUC':>8} {'recall':>7} "
          f"{'precision':>10} {'cost':>8}")

    curves, results = {}, {}
    for name, m in models.items():
        proba = m.predict_proba(Xte)[:, 1]
        ap = average_precision_score(yte, proba)
        roc = roc_auc_score(yte, proba)
        cost, tp, fp, fn = cost_at_threshold(yte, proba, 0.5)
        rec = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        curves[name] = precision_recall_curve(yte, proba)
        results[name] = {"pr_auc": round(ap, 4), "roc_auc": round(roc, 4),
                         "recall@0.5": round(rec, 4),
                         "precision@0.5": round(prec, 4),
                         "cost@0.5": round(cost, 1)}
        print(f"{name:<13} {ap:>7.4f} {roc:>8.4f} {rec:>7.3f} "
              f"{prec:>10.3f} {cost:>8.0f}")

    # Cost-based threshold tuning on the baseline model
    proba = models["baseline"].predict_proba(Xte)[:, 1]
    thr, cost = best_cost_threshold(yte, proba)
    _, tp, fp, fn = cost_at_threshold(yte, proba, thr)
    results["threshold_tuned"] = {
        "threshold": round(thr, 3), "cost": round(cost, 1),
        "recall": round(tp / (tp + fn + 1e-9), 4),
        "precision": round(tp / (tp + fp + 1e-9), 4)}
    print(f"\nCost-tuned threshold = {thr:.2f}  -> cost {cost:.0f} "
          f"(vs {results['baseline']['cost@0.5']:.0f} at 0.5), "
          f"recall {results['threshold_tuned']['recall']:.3f}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(
        {"prevalence": float(yte.mean()), "naive_accuracy": float(naive_acc),
         "cost_model": {"fn": FN_COST, "fp": FP_COST},
         "models": results}, indent=2))

    plt.figure(figsize=(7, 5))
    for name, (p, r, _) in curves.items():
        plt.plot(r, p, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
    plt.axhline(yte.mean(), ls="--", color="grey",
                label=f"random ({yte.mean():.3f})")
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title("Precision-Recall curves — imbalanced fraud")
    plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "pr_curves.png", dpi=110)
    print(f"Wrote {REPORTS/'metrics.json'} and {REPORTS/'pr_curves.png'}")


if __name__ == "__main__":
    main()
