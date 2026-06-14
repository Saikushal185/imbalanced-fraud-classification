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
