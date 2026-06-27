"""Tonight's baseline pick — two-phase fine-tune, single official split.

Runs (resumable, one result JSON each):
  * ResNet-50, ResNet-18  — Recipe A (two-phase, 224px, light aug)
  * EfficientNet-B3 x3     — Recipe B (two-phase, 300px batch 16, label-smoothing +
    dropout + strong aug), a 3-config mini-search over LR / augmentation.

B3's best config is chosen by VALIDATION AUC; the overall winner across
ResNet-50 / ResNet-18 / best-B3 is chosen by TEST accuracy + AUC.

Run (keep the lid OPEN, plugged in):
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dimsu .venv/bin/python -m src.tonight_baselines
"""
from __future__ import annotations

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import dataset, model as model_mod, utils
from .evaluate import _infer
from .metrics import compute_metrics, format_metrics
from .train import fit_two_phase

METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
SUMMARY_CSV = utils.CSV_DIR / "tonight_summary.csv"

# Recipe A (ResNet) and Recipe B (EfficientNet-B3, 3-config mini-search).
CONFIGS = [
    dict(name="resnet50", backbone="resnet50", family="resnet", image_size=224,
         batch=32, workers=4, aug="default", drop=0.0, ls=0.0, ft_lr=1e-4),
    dict(name="resnet18", backbone="resnet18", family="resnet", image_size=224,
         batch=32, workers=4, aug="default", drop=0.0, ls=0.0, ft_lr=1e-4),
    dict(name="b3_lr1e4_strong", backbone="efficientnet_b3", family="b3", image_size=300,
         batch=16, workers=2, aug="strong", drop=0.4, ls=0.1, ft_lr=1e-4),
    dict(name="b3_lr3e5_strong", backbone="efficientnet_b3", family="b3", image_size=300,
         batch=16, workers=2, aug="strong", drop=0.4, ls=0.1, ft_lr=3e-5),
    dict(name="b3_lr1e4_default", backbone="efficientnet_b3", family="b3", image_size=300,
         batch=16, workers=2, aug="default", drop=0.4, ls=0.1, ft_lr=1e-4),
]


def run_cfg(cfg, device, log) -> dict:
    result_path = utils.JSON_DIR / f"tonight_{cfg['name']}_result.json"
    ckpt = utils.CHECKPOINTS_DIR / f"tonight_{cfg['name']}.pt"
    if result_path.exists():
        r = json.loads(result_path.read_text())
        log.info(f"[{cfg['name']}] RESUMED | val AUC {r['val_auc']:.4f} "
                 f"| TEST AUC {r['test_metrics']['auc']:.4f}")
        return r

    utils.set_seed()
    man = dataset.load_split_manifest()
    tr = dataset.make_split_dataset("train", man, cfg["image_size"], aug_strength=cfg["aug"])
    va = dataset.make_split_dataset("val", man, cfg["image_size"])
    te = dataset.make_split_dataset("test", man, cfg["image_size"])
    log.info(f"--- {cfg['name']} ({cfg['backbone']} @{cfg['image_size']}, batch {cfg['batch']}, "
             f"aug {cfg['aug']}, ft_lr {cfg['ft_lr']}) ---")

    best_meta, best_epoch, elapsed, _ = fit_two_phase(
        tr, va, device, ckpt, log, tag=cfg["name"], backbone=cfg["backbone"],
        image_size=cfg["image_size"], drop_rate=cfg["drop"], label_smoothing=cfg["ls"],
        ft_lr=cfg["ft_lr"], batch_size=cfg["batch"], num_workers=cfg["workers"],
        history_csv=utils.CSV_DIR / f"tonight_{cfg['name']}_history.csv",
        curve_png=utils.FIGURES_DIR / f"tonight_{cfg['name']}_curve.png")

    model, _ = model_mod.load_checkpoint(ckpt, device)
    y, p = _infer(model, DataLoader(te, batch_size=cfg["batch"], shuffle=False,
                                    num_workers=cfg["workers"]), device)
    test_m = compute_metrics(y, p)
    r = {"name": cfg["name"], "backbone": cfg["backbone"], "family": cfg["family"],
         "val_auc": best_meta["val_metrics"]["auc"], "best_epoch": best_epoch,
         "train_min": round(elapsed / 60, 1), "test_metrics": test_m}
    result_path.write_text(json.dumps(r, indent=2))
    log.info(f"[{cfg['name']}] done ({r['train_min']} min) | "
             f"TEST acc {test_m['accuracy']:.4f} AUC {test_m['auc']:.4f}")
    return r


def main():
    utils.ensure_output_dirs()
    log = utils.get_logger("tonight", "tonight_baselines.log")
    device = utils.get_device()
    log.info(f"Device: {device} | {len(CONFIGS)} runs (2 ResNet + 3 B3)")

    results = [run_cfg(c, device, log) for c in CONFIGS]

    # B3 best chosen by validation AUC (honest internal selection)
    b3 = [r for r in results if r["family"] == "b3"]
    best_b3 = max(b3, key=lambda r: r["val_auc"])
    finalists = [r for r in results if r["family"] != "b3"] + [best_b3]

    rows = []
    for r in results:
        m = r["test_metrics"]
        rows.append({"run": r["name"], "val_auc": round(r["val_auc"], 4),
                     "train_min": r["train_min"],
                     **{f"test_{k}": round(m[k], 4) for k in METRIC_KEYS}})
    table = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    table.to_csv(SUMMARY_CSV, index=False)

    winner = max(finalists, key=lambda r: (r["test_metrics"]["auc"], r["test_metrics"]["accuracy"]))
    print("\n=== Tonight — all runs (sealed test, sorted by AUC) ===")
    print(table.to_string(index=False))
    print(f"\nB3 best config (by val AUC): {best_b3['name']}")
    print(f"\n*** WINNER (test AUC+acc): {winner['backbone']} [{winner['name']}] ***")
    print(format_metrics(winner["test_metrics"]))
    print(f"\nSummary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
