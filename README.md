# 💳 Imbalanced Fraud Classification

Rare-event classification done the right way. With only **~1.2% fraud**, a
model that predicts "never fraud" scores **98.8% accuracy** and catches *zero*
fraud — so this project ignores accuracy entirely and compares four imbalance
treatments on **PR-AUC, recall, and a business cost model**.

## What this project demonstrates
- **Why accuracy lies** — the trivial "always legit" classifier and its 98.8%
  accuracy is the opening exhibit.
- **Three imbalance treatments** — balanced **class weights**, **SMOTE**
  oversampling (implemented from scratch in `src/smote.py`), and **decision-
  threshold tuning** — against a plain baseline.
- **Cost-based operating points** — a missed fraud costs 20× a false alarm, so
  the right threshold is far below 0.5; we pick it by minimising expected cost.
- **PR curves over ROC** — precision-recall is the honest picture under heavy
  imbalance.

