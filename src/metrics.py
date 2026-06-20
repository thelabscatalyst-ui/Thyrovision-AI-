"""The metric block — computed one place, reused by training and evaluation.

Rule 4 (CLAUDE.md): never report accuracy alone. Every report gives accuracy,
sensitivity, specificity, precision, F1, AUC and the confusion matrix together.
Positive class = malignant (label 1), so:
  * sensitivity = of the real cancers, how many we caught (the metric that matters
    most clinically — a missed cancer is the worst error),
  * specificity = of the benign nodules, how many we correctly cleared.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """All metrics from ground-truth labels and malignant probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp

    def safe(num, den):
        return float(num) / float(den) if den else float("nan")

    sensitivity = safe(tp, tp + fn)          # recall of malignant
    specificity = safe(tn, tn + fp)          # recall of benign
    precision = safe(tp, tp + fp)
    f1 = safe(2 * precision * sensitivity, precision + sensitivity)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "accuracy": safe(tp + tn, total),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def format_metrics(m: dict) -> str:
    """Human-readable block including the confusion matrix."""
    cm = m["confusion_matrix"]
    return (
        f"  accuracy    : {m['accuracy']:.4f}\n"
        f"  sensitivity : {m['sensitivity']:.4f}  (malignant recall)\n"
        f"  specificity : {m['specificity']:.4f}  (benign recall)\n"
        f"  precision   : {m['precision']:.4f}\n"
        f"  f1          : {m['f1']:.4f}\n"
        f"  AUC         : {m['auc']:.4f}\n"
        f"  confusion   : TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}\n"
        f"                (rows=true [benign,malignant], positive=malignant)"
    )
