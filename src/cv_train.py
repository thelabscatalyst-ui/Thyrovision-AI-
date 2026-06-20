"""5-fold cross-validation baseline — the error-barred final number.

Protocol
--------
* Stratified 5-fold split over the 4,000 trainval images (train+val combined),
  balanced by label. The official 1,000-image TEST set stays sealed.
* Each fold trains a fresh ResNet-50 with the IDENTICAL `fit()` routine as the
  single-split baseline (only the data differs), early-stopping on that fold's
  held-out validation slice.
* We report two error-barred numbers (mean ± SD over the 5 folds):
    1. held-out validation metrics (the cross-validated estimate),
    2. each fold model's metrics on the sealed test set (test touched only at the
       very end, after all training decisions are locked — rule 3).

Phase 2's CBAM model must be run through this same script for a fair comparison.

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m src.cv_train
"""
from __future__ import annotations

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
from .train import IMAGE_SIZE, fit

N_FOLDS = 5
METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
CV_CSV = utils.LOGS_DIR / "cv_results.csv"
CV_SUMMARY_CSV = utils.LOGS_DIR / "cv_summary.csv"


def _summarise(per_fold: list[dict], stage: str) -> pd.DataFrame:
    """mean ± SD across folds for each metric."""
    rows = []
    for k in METRIC_KEYS:
        vals = np.array([m[k] for m in per_fold], dtype=float)
        rows.append({"stage": stage, "metric": k,
                     "mean": round(vals.mean(), 4), "sd": round(vals.std(ddof=1), 4)})
    return pd.DataFrame(rows)


def main():
    utils.set_seed()
    utils.ensure_output_dirs()
    log = utils.get_logger("cv", "baseline_cv.log")
    device = utils.get_device()
    log.info(f"Device: {device} | {N_FOLDS}-fold CV over trainval, test sealed")

    manifest = dataset.load_split_manifest()
    trainval = manifest[manifest.split.isin(["train", "val"])].reset_index(drop=True)
    log.info(f"trainval pool: {len(trainval)} images "
             f"(benign {int((trainval.label==0).sum())}, malignant {int((trainval.label==1).sum())})")

    # Sealed test set, built once, evaluated only after each fold finishes training.
    test_ds = dataset.make_split_dataset("test", manifest, IMAGE_SIZE)
    test_ld = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=utils.SEED)
    val_per_fold, test_per_fold, records = [], [], []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(trainval.image_id, trainval.label), 1):
        result_path = utils.LOGS_DIR / f"cv_fold{fold}_result.json"
        ckpt = utils.CHECKPOINTS_DIR / f"resnet50_cv_fold{fold}.pt"

        if result_path.exists():
            # Resume: this fold already finished in a previous run — reuse it.
            r = json.loads(result_path.read_text())
            val_m, test_m, best_epoch = r["val_metrics"], r["test_metrics"], r["best_epoch"]
            log.info(f"[fold{fold}] RESUMED from cache | val AUC {val_m['auc']:.4f} "
                     f"| TEST AUC {test_m['auc']:.4f}")
        else:
            utils.set_seed(utils.SEED + fold)          # per-fold reproducibility
            tr_ds = dataset.TN5000Dataset(trainval.iloc[tr_idx], train_aug=True, image_size=IMAGE_SIZE)
            va_ds = dataset.TN5000Dataset(trainval.iloc[va_idx], train_aug=False, image_size=IMAGE_SIZE)
            log.info(f"--- fold {fold}/{N_FOLDS}: train {len(tr_ds)}, val {len(va_ds)} ---")

            best_meta, best_epoch, elapsed, _ = fit(
                tr_ds, va_ds, device, ckpt, log, tag=f"fold{fold}",
                history_csv=utils.LOGS_DIR / f"cv_fold{fold}_history.csv",
                meta_extra={"fold": fold})
            val_m = best_meta["val_metrics"]

            # Evaluate this fold's best model on the sealed test set.
            model, _ = model_mod.load_checkpoint(ckpt, device)
            y_true, y_prob = _infer(model, test_ld, device)
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

    pd.DataFrame(records).to_csv(CV_CSV, index=False)
    summary = pd.concat([_summarise(val_per_fold, "val"),
                         _summarise(test_per_fold, "test")], ignore_index=True)
    summary.to_csv(CV_SUMMARY_CSV, index=False)

    log.info("=" * 60)
    log.info(f"5-fold CV done. Total wall-clock: {(time.time()-t0)/60:.1f} min on {device}.")
    print("\n=== 5-fold CV summary (mean ± SD) ===")
    for stage in ("val", "test"):
        print(f"\n[{stage}]")
        for _, r in summary[summary.stage == stage].iterrows():
            print(f"  {r['metric']:12s}: {r['mean']:.4f} ± {r['sd']:.4f}")
    print(f"\nPer-fold: {CV_CSV}\nSummary:  {CV_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
