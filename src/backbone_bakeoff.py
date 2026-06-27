"""Backbone bake-off — pick the committed baseline fairly, in OUR pipeline.

Trains each candidate backbone on the SAME official train split, with identical
settings (same `fit()`), evaluates the sealed test once, and emits one comparison
table. This answers the mentor's questions with our own evidence:
  * does ResNet-18 actually beat ResNet-50 here (the overfitting hypothesis)?
  * does EfficientNet win (compact = better regulariser on small data)?

Single fixed split for a fast, fair RELATIVE ranking; full CV is run later on the
winner only. Resumable (per-backbone result JSON) and safe to run under caffeinate.

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -is .venv/bin/python -m src.backbone_bakeoff
"""
from __future__ import annotations

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import dataset, model as model_mod, utils
from .evaluate import _infer
from .metrics import compute_metrics, format_metrics
from .train import fit

METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
BAKEOFF_CSV = utils.CSV_DIR / "backbone_bakeoff.csv"


def run_one(name: str, image_size: int, device, log) -> dict:
    result_path = utils.JSON_DIR / f"bakeoff_{name}_result.json"
    ckpt = utils.CHECKPOINTS_DIR / f"bakeoff_{name}.pt"
    if result_path.exists():
        r = json.loads(result_path.read_text())
        log.info(f"[{name}] RESUMED | TEST AUC {r['test_metrics']['auc']:.4f} "
                 f"acc {r['test_metrics']['accuracy']:.4f}")
        return r

    utils.set_seed()  # same seed for every backbone -> fair comparison
    manifest = dataset.load_split_manifest()
    tr = dataset.make_split_dataset("train", manifest, image_size)
    va = dataset.make_split_dataset("val", manifest, image_size)
    te = dataset.make_split_dataset("test", manifest, image_size)
    # High-res backbones (300px) need a smaller batch + fewer workers to fit 16GB.
    batch, workers = (16, 2) if image_size >= 300 else (32, 4)
    log.info(f"--- {name} @ {image_size}px | train {len(tr)} val {len(va)} "
             f"| batch {batch} workers {workers} ---")

    best_meta, best_epoch, elapsed, _ = fit(
        tr, va, device, ckpt, log, tag=name, backbone=name, image_size=image_size,
        batch_size=batch, num_workers=workers,
        history_csv=utils.CSV_DIR / f"bakeoff_{name}_history.csv",
        curve_png=utils.FIGURES_DIR / f"bakeoff_{name}_curve.png")

    model, _ = model_mod.load_checkpoint(ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    test_batch = 32 if image_size >= 300 else 64
    y, p = _infer(model, DataLoader(te, batch_size=test_batch, shuffle=False, num_workers=workers), device)
    test_m = compute_metrics(y, p)

    r = {"backbone": name, "image_size": image_size, "params_millions": round(n_params/1e6, 1),
         "best_epoch": best_epoch, "train_min": round(elapsed/60, 1),
         "val_metrics": best_meta["val_metrics"], "test_metrics": test_m}
    result_path.write_text(json.dumps(r, indent=2))
    log.info(f"[{name}] done ({r['train_min']} min, {r['params_millions']}M params) | "
             f"TEST AUC {test_m['auc']:.4f} acc {test_m['accuracy']:.4f}")
    log.info("TEST metrics:\n" + format_metrics(test_m))
    return r


def main():
    utils.ensure_output_dirs()
    log = utils.get_logger("bakeoff", "backbone_bakeoff.log")
    device = utils.get_device()
    log.info(f"Device: {device} | backbones: {list(model_mod.BACKBONES)}")

    results = [run_one(name, size, device, log)
               for name, size in model_mod.BACKBONES.items()]

    rows = []
    for r in results:
        row = {"backbone": r["backbone"], "img": r["image_size"],
               "params_M": r["params_millions"], "train_min": r["train_min"],
               "best_epoch": r["best_epoch"]}
        row.update({f"test_{k}": round(r["test_metrics"][k], 4) for k in METRIC_KEYS})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    table.to_csv(BAKEOFF_CSV, index=False)

    print("\n=== Backbone bake-off — sealed TEST (sorted by AUC) ===")
    print(table.to_string(index=False))
    print(f"\nSaved -> {BAKEOFF_CSV}")
    best = table.iloc[0]
    print(f"\nBest by AUC: {best['backbone']} (AUC {best['test_auc']}, acc {best['test_accuracy']})")


if __name__ == "__main__":
    main()
