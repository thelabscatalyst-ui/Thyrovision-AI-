"""5-fold cross-validation — the error-barred final number for a B3 attention arm.

Protocol
--------
* Stratified 5-fold split over the 4,000 trainval images (train+val combined),
  balanced by label. The official 1,000-image TEST set stays sealed (rule 3) and is
  evaluated only after each fold finishes training. This mirrors the official-split
  protocol that produced the committed single-split baseline, so the CV result is the
  honest *error bar* (mean ± SD) on that exact number.
* Each fold trains a fresh model with the IDENTICAL committed EfficientNet-B3
  two-phase recipe (`fit_two_phase`: freeze-head -> unfreeze-backbone, 300px, batch 16,
  dropout 0.4, label-smoothing 0.1, class-weighted loss, light aug). Only the data
  differs between folds — a fairness guarantee.
* `--attention` selects the external attention arm ('none'/'se'/'cbam'/'cpca'); 'none'
  is the plain-B3 baseline. The SAME script CVs the baseline now and any attention
  finalist later — filenames are keyed by backbone+attention so arms never collide.

We report two error-barred numbers (mean ± SD over the 5 folds):
  1. held-out validation metrics (the cross-validated estimate),
  2. each fold model's metrics on the sealed test set.

Resumable: one result JSON per fold, so a Mac sleep costs at most the current fold.

Run (lid OPEN, plugged in):
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dimsu .venv/bin/python -m src.cv_train --attention none
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from . import dataset, model as model_mod, utils
from .evaluate import _infer
from .metrics import compute_metrics
from .train import fit_two_phase

N_FOLDS = 5
METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]

# Committed EfficientNet-B3 two-phase recipe (identical to the single-split baseline
# and to phase2_attention.py — only the data folds differ here).
BACKBONE = "efficientnet_b3"
IMAGE_SIZE = 300
BATCH = 16
WORKERS = 2
TEST_BATCH = 32        # inference is grad-free, so a larger batch fits memory
AUG = "default"
DROP = 0.4
LS = 0.1
FT_LR = 1e-4


def _summarise(per_fold: list[dict], stage: str) -> pd.DataFrame:
    """mean ± SD across folds for each metric."""
    rows = []
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in per_fold], dtype=float)
        rows.append({"stage": stage, "metric": k,
                     "mean": round(vals.mean(), 4), "sd": round(vals.std(ddof=1), 4)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="5-fold CV of a B3 attention arm")
    ap.add_argument("--attention", default="none", choices=["none", "se", "cbam", "cpca"],
                    help="external attention arm to cross-validate (default: none = plain B3)")
    ap.add_argument("--backbone", default=BACKBONE,
                    help=f"timm backbone (default: {BACKBONE})")
    args = ap.parse_args()
    attention, backbone = args.attention, args.backbone
    arm = f"{backbone}_{attention}"          # e.g. efficientnet_b3_cbam — keys all filenames

    utils.set_seed()
    utils.ensure_output_dirs()
    log = utils.get_logger("cv", f"cv_{arm}.log")
    device = utils.get_device()
    log.info(f"Device: {device} | {N_FOLDS}-fold CV | backbone={backbone} attention={attention} "
             f"| recipe: two-phase {IMAGE_SIZE}px batch {BATCH} drop {DROP} ls {LS} ft_lr {FT_LR}")

    # baseline (none) -> EfficientNet_B3_Baseline/, any attention arm -> attention/
    results_dir = utils.CSV_B3_BASELINE if attention == "none" else utils.CSV_ATTENTION
    results_dir.mkdir(parents=True, exist_ok=True)
    cv_csv = results_dir / f"cv_{arm}_results.csv"
    cv_summary_csv = results_dir / f"cv_{arm}_summary.csv"

    manifest = dataset.load_split_manifest()
    trainval = manifest[manifest.split.isin(["train", "val"])].reset_index(drop=True)
    log.info(f"trainval pool: {len(trainval)} images "
             f"(benign {int((trainval.label==0).sum())}, malignant {int((trainval.label==1).sum())})")

    # Sealed test set, built once, evaluated only after each fold finishes training.
    test_ds = dataset.make_split_dataset("test", manifest, IMAGE_SIZE)
    test_ld = DataLoader(test_ds, batch_size=TEST_BATCH, shuffle=False, num_workers=WORKERS)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=utils.SEED)
    val_per_fold, test_per_fold, records = [], [], []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(trainval.image_id, trainval.label), 1):
        json_dir = utils.JSON_B3_BASELINE if attention == "none" else utils.JSON_ATTENTION
        json_dir.mkdir(parents=True, exist_ok=True)
        result_path = json_dir / f"cv_{arm}_fold{fold}_result.json"
        ckpt = utils.CHECKPOINTS_DIR / f"cv_{arm}_fold{fold}.pt"

        if result_path.exists():
            # Resume: this fold already finished in a previous run — reuse it.
            r = json.loads(result_path.read_text())
            val_m, test_m, best_epoch = r["val_metrics"], r["test_metrics"], r["best_epoch"]
            log.info(f"[fold{fold}] RESUMED from cache | val AUC {val_m['auc']:.4f} "
                     f"| TEST AUC {test_m['auc']:.4f}")
        else:
            utils.set_seed(utils.SEED + fold)          # per-fold reproducibility
            tr_ds = dataset.TN5000Dataset(trainval.iloc[tr_idx], train_aug=True,
                                          image_size=IMAGE_SIZE, aug_strength=AUG)
            va_ds = dataset.TN5000Dataset(trainval.iloc[va_idx], train_aug=False,
                                          image_size=IMAGE_SIZE)
            log.info(f"--- fold {fold}/{N_FOLDS}: train {len(tr_ds)}, val {len(va_ds)} ---")

            best_meta, best_epoch, elapsed, _ = fit_two_phase(
                tr_ds, va_ds, device, ckpt, log, tag=f"{arm}-fold{fold}",
                backbone=backbone, attention=attention, image_size=IMAGE_SIZE,
                drop_rate=DROP, label_smoothing=LS, ft_lr=FT_LR,
                batch_size=BATCH, num_workers=WORKERS,
                history_csv=utils.CSV_HISTORY / f"cv_{arm}_fold{fold}_history.csv",
                meta_extra={"fold": fold})
            val_m = best_meta["val_metrics"]

            # Evaluate this fold's best model on the sealed test set.
            fold_model, _ = model_mod.load_checkpoint(ckpt, device)
            y_true, y_prob = _infer(fold_model, test_ld, device)
            test_m = compute_metrics(y_true, y_prob)

            # Mark the fold finished so an interruption never costs this fold twice.
            result_path.write_text(json.dumps(
                {"fold": fold, "best_epoch": best_epoch,
                 "val_metrics": val_m, "test_metrics": test_m}, indent=2))
            log.info(f"[fold{fold}] best epoch {best_epoch} ({elapsed/60:.1f} min) | "
                     f"val AUC {val_m['auc']:.4f} acc {val_m['accuracy']:.4f} | "
                     f"TEST AUC {test_m['auc']:.4f} acc {test_m['accuracy']:.4f}")

        val_per_fold.append(val_m)
        test_per_fold.append(test_m)
        records.append({"fold": fold, "best_epoch": best_epoch,
                        **{f"val_{k}": val_m[k] for k in METRIC_KEYS},
                        **{f"test_{k}": test_m[k] for k in METRIC_KEYS}})

    pd.DataFrame(records).to_csv(cv_csv, index=False)
    summary = pd.concat([_summarise(val_per_fold, "val"),
                         _summarise(test_per_fold, "test")], ignore_index=True)
    summary.to_csv(cv_summary_csv, index=False)

    log.info("=" * 60)
    log.info(f"5-fold CV done ({arm}). Total wall-clock: {(time.time()-t0)/60:.1f} min on {device}.")
    print(f"\n=== 5-fold CV summary — {arm} (mean ± SD) ===")
    for stage in ("val", "test"):
        print(f"\n[{stage}]")
        for _, r in summary[summary.stage == stage].iterrows():
            print(f"  {r['metric']:12s}: {r['mean']:.4f} ± {r['sd']:.4f}")
    print(f"\nPer-fold: {cv_csv}\nSummary:  {cv_summary_csv}")


if __name__ == "__main__":
    main()
