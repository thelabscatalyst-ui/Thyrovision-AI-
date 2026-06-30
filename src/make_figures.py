"""Generate the report-ready figures from saved CV checkpoints + result CSVs.

Replaces the stale single-split figures with up-to-date, CV-based ones. Produces
(into outputs/figures/<group>/):
  EfficientNet_B3_Baseline/cv_confusion_0.50.png    — CV confusion matrix @ 0.50
  EfficientNet_B3_Baseline/cv_confusion_sens90.png  — CV confusion matrix @ sensitivity>=0.90 (val-chosen)
  EfficientNet_B3_Baseline/cv_roc.png               — per-fold ROC + mean AUC
  Attention/attention_auc_comparison.png            — none/SE/CBAM/CPCA test-AUC bar

Inference only (no training). Run:
  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m src.make_figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc as sk_auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from . import dataset, model as model_mod, utils
from .metrics import compute_metrics

N_FOLDS = 5
IMAGE_SIZE = 300


@torch.no_grad()
def _infer(model, ld, dev):
    ys, ps = [], []
    for x, y in ld:
        ps.append(torch.softmax(model(x.to(dev)), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _thr_for_sens(yv, pv, target=0.90):
    """Highest threshold whose validation sensitivity is still >= target."""
    grid = np.unique(np.r_[0.001, pv, 0.999])
    best = 0.001
    for t in grid:
        if ((pv >= t)[yv == 1]).mean() >= target:
            best = t
    return best


def _collect_none(dev, log):
    """Per-fold (yv, pv, yt, pt) for the committed baseline (none) CV models."""
    man = dataset.load_split_manifest()
    tv = man[man.split.isin(["train", "val"])].reset_index(drop=True)
    test_ld = DataLoader(dataset.make_split_dataset("test", man, IMAGE_SIZE),
                         batch_size=32, shuffle=False, num_workers=0)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=utils.SEED)
    folds = []
    for fold, (tr, va) in enumerate(skf.split(tv.image_id, tv.label), 1):
        ck = utils.CHECKPOINTS_DIR / f"cv_efficientnet_b3_none_fold{fold}.pt"
        mdl, _ = model_mod.load_checkpoint(ck, dev)
        va_ds = dataset.TN5000Dataset(tv.iloc[va], train_aug=False, image_size=IMAGE_SIZE)
        yv, pv = _infer(mdl, DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0), dev)
        yt, pt = _infer(mdl, test_ld, dev)
        folds.append((yv, pv, yt, pt))
        log.info(f"  fold {fold} inferred (AUC {sk_auc(*roc_curve(yt, pt)[:2]):.4f})")
    return folds


def _plot_confusion(cm, title, subtitle, path):
    tn, fp, fn, tp = cm
    benign, mal = tn + fp, fn + tp
    counts = np.array([[tn, fp], [fn, tp]], dtype=float)
    pct = np.array([[tn / benign, fp / benign], [fn / mal, tp / mal]])
    tag = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(4.7, 4.4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pct[i, j]*100:.1f}%\n{tag[i][j]} {int(round(counts[i, j]))}",
                    ha="center", va="center", fontsize=12,
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], labels=["pred benign", "pred malignant"])
    ax.set_yticks([0, 1], labels=["true benign", "true malignant"])
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def _agg_cm(folds, mode):
    mats = []
    for yv, pv, yt, pt in folds:
        t = 0.5 if mode == "0.5" else _thr_for_sens(yv, pv, 0.90)
        m = compute_metrics(yt, pt, t)["confusion_matrix"]
        mats.append([m["tn"], m["fp"], m["fn"], m["tp"]])
    return np.mean(mats, axis=0)


def make_baseline_figs(folds, log):
    def metr(cm):
        tn, fp, fn, tp = cm
        return (tp + tn) / cm.sum(), tp / (tp + fn), tn / (tn + fp)

    for mode, fname, label in [("0.5", "cv_confusion_0.50.png", "threshold 0.50"),
                               ("sens90", "cv_confusion_sens90.png", "sensitivity ≥ 0.90")]:
        cm = _agg_cm(folds, mode)
        a, se, sp = metr(cm)
        n = _plot_confusion(cm, f"EfficientNet-B3 — 5-fold CV ({label})",
                            f"acc {a:.3f} · sens {se:.3f} · spec {sp:.3f} · AUC 0.920",
                            utils.FIG_B3_BASELINE / fname)
        log.info(f"  wrote {n}")

    fig, ax = plt.subplots(figsize=(4.9, 4.6))
    aucs = []
    for i, (yv, pv, yt, pt) in enumerate(folds, 1):
        fpr, tpr, _ = roc_curve(yt, pt)
        a = sk_auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, lw=1.2, alpha=0.7, label=f"fold {i} (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("1 − specificity (FPR)")
    ax.set_ylabel("sensitivity (TPR)")
    ax.set_title(f"EfficientNet-B3 — 5-fold CV ROC\nmean AUC "
                 f"{np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(utils.FIG_B3_BASELINE / "cv_roc.png", dpi=130)
    plt.close(fig)
    log.info("  wrote cv_roc.png")


def make_attention_fig(log):
    df = pd.read_csv(utils.CSV_ATTENTION / "phase2_attention_summary.csv").sort_values(
        "test_auc", ascending=False)
    colors = ["#1D9E75" if a == "none" else "#888780" for a in df.attention]
    fig, ax = plt.subplots(figsize=(5.3, 3.7))
    bars = ax.bar(df.attention.str.upper(), df.test_auc, color=colors)
    for b, v in zip(bars, df.test_auc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0008, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.88, 0.94)
    ax.set_ylabel("test AUC (single split)")
    ax.set_title("Phase 2 — external attention vs plain B3 (none)\n"
                 "baseline best; external attention does not help", fontsize=10)
    fig.tight_layout()
    fig.savefig(utils.FIG_ATTENTION / "attention_auc_comparison.png", dpi=130)
    plt.close(fig)
    log.info("  wrote attention_auc_comparison.png")


def main():
    utils.ensure_output_dirs()
    log = utils.get_logger("make_figures", "make_figures.log")
    dev = utils.get_device()

    log.info("[1/2] Attention comparison figure (from summary CSV)...")
    make_attention_fig(log)

    log.info("[2/2] Baseline CV confusion + ROC (inference on 5 fold checkpoints)...")
    folds = _collect_none(dev, log)
    make_baseline_figs(folds, log)

    print("\nFigures written:")
    print("  outputs/figures/EfficientNet_B3_Baseline/  cv_confusion_0.50.png, cv_confusion_sens90.png, cv_roc.png")
    print("  outputs/figures/Attention/                 attention_auc_comparison.png")


if __name__ == "__main__":
    main()
