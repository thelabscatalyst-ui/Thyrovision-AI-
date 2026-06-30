"""Re-run the baseline on the DE-CONTAMINATED data and compare to the original.

Fix applied: 44 train/val images that near-duplicate a test image were dropped
(`keep=False` in the clean manifest). The official 1,000-image test set is
untouched, so the model can no longer have seen a test image's twin in training.

Produces, with identical settings to the original baseline (same `fit()`):
  1. clean single-split test metrics  (vs original 0.853 / AUC 0.922)
  2. clean 5-fold group-aware CV       (vs original 0.852 / AUC 0.918)
     — StratifiedGroupKFold on duplicate-cluster groups so no cluster spans folds.

Resumable (single-split + per-fold result JSONs). Run under caffeinate:
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is .venv/bin/python -m src.rerun_clean
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from . import dataset, model as model_mod, utils
from .cv_train import METRIC_KEYS, N_FOLDS, _summarise
from .evaluate import _infer
from .metrics import compute_metrics, format_metrics
from .train import IMAGE_SIZE, fit

SINGLE_CKPT = utils.CHECKPOINTS_DIR / "resnet50_baseline_clean.pt"
SINGLE_RESULT = utils.JSON_LEGACY / "baseline_clean_single_result.json"
CV_CSV = utils.CSV_LEGACY / "cv_clean_results.csv"
CV_SUMMARY_CSV = utils.CSV_LEGACY / "cv_clean_summary.csv"


def _test_loader(kept):
    return DataLoader(dataset.make_split_dataset("test", kept, IMAGE_SIZE),
                      batch_size=64, shuffle=False, num_workers=4)


def run_single(kept, device, log):
    if SINGLE_RESULT.exists():
        r = json.loads(SINGLE_RESULT.read_text())
        log.info(f"[single] RESUMED | TEST AUC {r['test_metrics']['auc']:.4f}")
        return r["val_metrics"], r["test_metrics"]
    utils.set_seed()
    tr = dataset.make_split_dataset("train", kept, IMAGE_SIZE)
    va = dataset.make_split_dataset("val", kept, IMAGE_SIZE)
    log.info(f"[single] clean train {len(tr)}, val {len(va)}")
    best_meta, best_epoch, elapsed, _ = fit(
        tr, va, device, SINGLE_CKPT, log, tag="clean-single",
        history_csv=utils.CSV_HISTORY / "baseline_clean_history.csv",
        curve_png=utils.FIG_ARCHIVE / "baseline_clean_training_curve.png")
    model, _ = model_mod.load_checkpoint(SINGLE_CKPT, device)
    y, p = _infer(model, _test_loader(kept), device)
    test_m = compute_metrics(y, p)
    SINGLE_RESULT.write_text(json.dumps(
        {"best_epoch": best_epoch, "val_metrics": best_meta["val_metrics"],
         "test_metrics": test_m}, indent=2))
    log.info(f"[single] done ({elapsed/60:.1f} min) | TEST AUC {test_m['auc']:.4f} "
             f"acc {test_m['accuracy']:.4f}")
    return best_meta["val_metrics"], test_m


def run_cv(kept, device, log):
    tv = kept[kept.split.isin(["train", "val"])].reset_index(drop=True)
    test_ld = _test_loader(kept)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=utils.SEED)
    val_pf, test_pf, records = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(
            sgkf.split(tv.image_id, tv.label, groups=tv.group), 1):
        result_path = utils.JSON_LEGACY / f"cv_clean_fold{fold}_result.json"
        ckpt = utils.CHECKPOINTS_DIR / f"resnet50_cv_clean_fold{fold}.pt"
        if result_path.exists():
            r = json.loads(result_path.read_text())
            val_m, test_m, best_epoch = r["val_metrics"], r["test_metrics"], r["best_epoch"]
            log.info(f"[cv-fold{fold}] RESUMED | TEST AUC {test_m['auc']:.4f}")
        else:
            # sanity: groups must not leak across this fold
            assert set(tv.iloc[tr_idx].group) & set(tv.iloc[va_idx].group) == set(), \
                f"group leak in fold {fold}"
            utils.set_seed(utils.SEED + fold)
            tr_ds = dataset.TN5000Dataset(tv.iloc[tr_idx], train_aug=True, image_size=IMAGE_SIZE)
            va_ds = dataset.TN5000Dataset(tv.iloc[va_idx], train_aug=False, image_size=IMAGE_SIZE)
            log.info(f"--- clean CV fold {fold}/{N_FOLDS}: train {len(tr_ds)}, val {len(va_ds)} ---")
            best_meta, best_epoch, elapsed, _ = fit(
                tr_ds, va_ds, device, ckpt, log, tag=f"cv-fold{fold}",
                history_csv=utils.CSV_HISTORY / f"cv_clean_fold{fold}_history.csv",
                meta_extra={"fold": fold})
            val_m = best_meta["val_metrics"]
            m, _ = model_mod.load_checkpoint(ckpt, device)
            y, p = _infer(m, test_ld, device)
            test_m = compute_metrics(y, p)
            result_path.write_text(json.dumps(
                {"fold": fold, "best_epoch": best_epoch,
                 "val_metrics": val_m, "test_metrics": test_m}, indent=2))
            log.info(f"[cv-fold{fold}] done ({elapsed/60:.1f} min) | "
                     f"TEST AUC {test_m['auc']:.4f} acc {test_m['accuracy']:.4f}")
        val_pf.append(val_m); test_pf.append(test_m)
        records.append({"fold": fold, "best_epoch": best_epoch,
                        **{f"val_{k}": val_m[k] for k in METRIC_KEYS},
                        **{f"test_{k}": test_m[k] for k in METRIC_KEYS}})

    pd.DataFrame(records).to_csv(CV_CSV, index=False)
    summary = pd.concat([_summarise(val_pf, "val"), _summarise(test_pf, "test")],
                        ignore_index=True)
    summary.to_csv(CV_SUMMARY_CSV, index=False)
    return summary


def _compare(log):
    """Print original-vs-clean side by side."""
    old_single = json.loads((utils.JSON_LEGACY / "baseline_test_metrics.json").read_text())
    new_single = json.loads(SINGLE_RESULT.read_text())["test_metrics"]
    print("\n=== Single-split TEST: original vs de-contaminated ===")
    for k in METRIC_KEYS:
        print(f"  {k:12s}: {old_single[k]:.4f}  ->  {new_single[k]:.4f}  "
              f"({new_single[k]-old_single[k]:+.4f})")

    old_cv = pd.read_csv(utils.CSV_LEGACY / "cv_summary.csv")
    new_cv = pd.read_csv(CV_SUMMARY_CSV)
    print("\n=== 5-fold CV TEST (mean ± SD): original vs de-contaminated ===")
    for k in METRIC_KEYS:
        o = old_cv[(old_cv.stage == "test") & (old_cv.metric == k)].iloc[0]
        n = new_cv[(new_cv.stage == "test") & (new_cv.metric == k)].iloc[0]
        print(f"  {k:12s}: {o['mean']:.4f}±{o['sd']:.4f}  ->  "
              f"{n['mean']:.4f}±{n['sd']:.4f}  ({n['mean']-o['mean']:+.4f})")


def main():
    utils.ensure_output_dirs()
    log = utils.get_logger("rerun_clean", "rerun_clean.log")
    device = utils.get_device()
    kept = dataset.load_clean_manifest()
    kept = kept[kept.keep].reset_index(drop=True)
    log.info(f"Device: {device} | clean kept={len(kept)} "
             f"(train {int((kept.split=='train').sum())}, "
             f"val {int((kept.split=='val').sum())}, test {int((kept.split=='test').sum())})")

    t0 = time.time()
    run_single(kept, device, log)
    run_cv(kept, device, log)
    log.info(f"All done in {(time.time()-t0)/60:.1f} min.")
    _compare(log)


if __name__ == "__main__":
    main()
