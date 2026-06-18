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

## Demo

```text
$ python3 src/classify.py
Test set: 12,000 txns, 1.20% fraud
"Always legit" accuracy = 98.80% but catches 0 fraud

model          PR-AUC  ROC-AUC  recall  precision     cost
baseline       0.9362   0.9930   0.875      0.969      364
class_weight   0.9384   0.9930   0.944      0.402      362
smote          0.9410   0.9936   0.951      0.449      308

Cost-tuned threshold = 0.07  -> cost 255 (vs 364 at 0.5), recall 0.924
```

Class weights and SMOTE trade precision for the recall that actually matters
when misses are expensive; **tuning the threshold to the cost model** beats all
of them on total cost. `reports/pr_curves.png` plots precision-recall against
the random-classifier floor.

## Treatments

| Treatment | Where | Idea |
|---|---|---|
| Baseline | `src/classify.py` | logistic regression, threshold 0.5 |
| Class weights | `src/classify.py` | reweight the loss toward the minority |
| SMOTE | `src/smote.py` | synthesize minority samples to parity |
| Threshold tuning | `src/classify.py` | minimise FN×20 + FP×1 expected cost |

## Project structure
```
imbalanced-fraud-classification/
├── data/transactions.csv   # generated ~1.2%-fraud transactions
├── src/
│   ├── generate_data.py    # overlapping legit/fraud distributions
│   ├── smote.py            # from-scratch SMOTE oversampling
│   └── classify.py         # 4 treatments, PR-AUC, cost-based threshold
├── reports/                # metrics.json, pr_curves.png
├── requirements.txt
├── torun.txt
└── license.md
```

## Run it
```bash
./run.sh        # or see torun.txt
```
