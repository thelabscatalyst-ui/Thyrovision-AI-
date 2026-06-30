"""Final evaluation — the ONLY place the sealed test set is read (rule 3).

Loads the best checkpoint, runs the 1,000-image test split once, prints the full
metric block, saves a confusion matrix + ROC curve, and walks the honesty
checklist (incl. the >=97% leakage stop-gate and the 80-87% sanity band).

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m src.evaluate
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from . import dataset, model as model_mod, utils
from .metrics import compute_metrics, format_metrics

CKPT_PATH = utils.CHECKPOINTS_DIR / "resnet50_baseline.pt"
CM_PNG = utils.FIG_LEGACY / "baseline_test_confusion_matrix.png"
ROC_PNG = utils.FIG_LEGACY / "baseline_test_roc.png"
METRICS_JSON = utils.JSON_LEGACY / "baseline_test_metrics.json"


@torch.no_grad()
def _infer(model, loader, device):
    probs, ys = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def _plot_confusion(m: dict) -> None:
    cm = m["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred benign", "pred malignant"])
    ax.set_yticks([0, 1], labels=["true benign", "true malignant"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=14)
    ax.set_title("TN5000 test — confusion matrix")
    fig.tight_layout(); fig.savefig(CM_PNG, dpi=120); plt.close(fig)


def _plot_roc(y_true, y_prob, auc) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, label=f"ResNet-50 (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("1 - specificity (FPR)"); ax.set_ylabel("sensitivity (TPR)")
    ax.set_title("TN5000 test — ROC"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(ROC_PNG, dpi=120); plt.close(fig)


def honesty_checklist(m: dict) -> None:
    acc = m["accuracy"]
    print("\n=== Honesty checklist ===")
    print("[note] Split is IMAGE-level: TN5000 has no patient ID, so patient overlap")
    print("       cannot be asserted = 0. Documented limitation (rule 2).")
    print(f"[{'STOP' if acc >= 0.97 else ' ok '}] test accuracy {acc:.4f} "
          f"{'>= 0.97 — re-verify split for leakage before trusting!' if acc >= 0.97 else '< 0.97'}")
    band = "within" if 0.80 <= acc <= 0.87 else "OUTSIDE"
    print(f"[{'ok' if band=='within' else 'flag'}] test accuracy {acc:.4f} is {band} the 0.80-0.87 sanity band")
    print("[ ok ] full metric block reported together (acc/sens/spec/prec/F1/AUC/confusion)")
    print("[note] caliper-leakage audit: chance-level (AUC 0.46); Phase-3 Grad-CAM check pending")


def main():
    utils.set_seed()
    utils.ensure_output_dirs()
    device = utils.get_device()
    print(f"Device: {device}")

    model, meta = model_mod.load_checkpoint(CKPT_PATH, device)
    print(f"Loaded checkpoint from best epoch {meta.get('epoch')} "
          f"(val AUC {meta.get('val_metrics', {}).get('auc', float('nan')):.4f})")

    test_ds = dataset.make_split_dataset("test", image_size=meta.get("image_size", 224))
    test_ld = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"Test set: {len(test_ds)} images (sealed until now).")

    y_true, y_prob = _infer(model, test_ld, device)
    m = compute_metrics(y_true, y_prob)

    print("\n=== TN5000 TEST metrics (single official split) ===")
    print(format_metrics(m))
    _plot_confusion(m)
    _plot_roc(y_true, y_prob, m["auc"])
    METRICS_JSON.write_text(json.dumps(m, indent=2))
    honesty_checklist(m)
    print(f"\nFigures: {CM_PNG.name}, {ROC_PNG.name} | metrics: {METRICS_JSON.name}")


if __name__ == "__main__":
    main()
