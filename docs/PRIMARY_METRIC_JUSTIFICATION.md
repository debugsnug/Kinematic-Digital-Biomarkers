# Primary Evaluation Metric — Justification

## Metric Selected: Balanced Accuracy (Primary) + F1-Score (Co-primary)

### Why NOT plain Accuracy?

The dataset is **class-imbalanced**: 30 DCP vs 21 TD (≈ 59%:41%).
A trivial model that always predicts DCP would reach 59% accuracy — but zero
clinical value. Plain accuracy is therefore a misleading single metric here.

### Primary metric: Balanced Accuracy

```
Balanced Accuracy = (Sensitivity + Specificity) / 2
                  = (TPR + TNR) / 2
```

- Treats both classes equally regardless of their size ratio.
- Bounded in [0, 1]; 0.5 = random guessing.
- Recommended by Brodersen et al. (2010) and the scikit-learn documentation
  for imbalanced binary classification.

Reported value (nested CV): **0.814** [95% CI 0.699–0.918]

### Co-primary metric: F1-Score (class 1 = DCP)

```
F1 = 2 × (Precision × Sensitivity) / (Precision + Sensitivity)
```

- Emphasises correct identification of the minority/clinically-relevant class.
- Penalises both false negatives (missed DCP) and false positives (false DCP alarm).

Reported value (nested CV): **0.852** [95% CI 0.745–0.935]

### Supporting metrics also reported

| Metric           | Value (nested CV) | Purpose |
|--|--|--|
| Accuracy         | 0.824             | Baseline comparison |
| Sensitivity      | 0.867             | Miss-rate for DCP (clinical safety) |
| Specificity      | 0.762             | False-alarm rate |
| MCC              | 0.634             | Accounts for all 4 cells of confusion matrix |
| ROC-AUC          | 0.968 (mean fold) | Threshold-free ranking quality |
| PR-AUC           | reported per fold | Better than ROC for imbalanced sets |

### Metric used for inner-CV model selection

Inner hyperparameter search uses **accuracy** because:
1. It is what the course slide specifies ("highest-accuracy selection").
2. At the inner fold level, classes are approximately balanced by
   StratifiedGroupKFold, reducing the bias from using accuracy.
3. All class-weight-aware models use `class_weight="balanced"` anyway.

The inner criterion is documented in `src/model_catalog.py` (line 13).
