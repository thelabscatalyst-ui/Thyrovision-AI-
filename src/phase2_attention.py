"""Phase 2 — attention-mechanism comparison on the committed EfficientNet-B3.

The controlled experiment (the core Phase-2 contribution): the SAME B3 backbone +
the SAME two-phase recipe, differing ONLY in the external attention module wrapped
after the backbone:
  * none — plain B3 (baseline arm)
  * se   — Squeeze-Excitation (channel-only; the paper's choice)
  * cbam — channel + spatial attention
  * cpca — Channel-Prior Convolutional Attention

Any metric difference between arms is therefore attributable to attention. Recipe is
the committed B3 recipe (two-phase, 300px, batch 16, dropout 0.4, label-smoothing 0.1,
class-weighted loss, light/default aug). Each arm's config is chosen for it identically.

Honesty rules kept: the sealed TEST set is touched once per arm only after training is
locked; we always report sens/spec/AUC, not accuracy alone. Resumable (one result JSON
per arm) so a Mac sleep never costs more than the current arm.

Run (lid OPEN, plugged in):
  PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dimsu .venv/bin/python -m src.phase2_attention
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
SUMMARY_CSV = utils.CSV_ATTENTION / "phase2_attention_summary.csv"

# Identical backbone + recipe; only `attention` differs (the controlled variable).
BACKBONE = "efficientnet_b3"
IMAGE_SIZE = 300
BATCH = 16
WORKERS = 2
AUG = "default"          # matches the committed B3 (b3_lr1e4_default)
DROP = 0.4
LS = 0.1
FT_LR = 1e-4

ARMS = ["none", "se", "cbam", "cpca"]


def run_arm(attention: str, device, log) -> dict:
    result_path = utils.JSON_ATTENTION / f"phase2_{attention}_result.json"
    ckpt = utils.CHECKPOINTS_DIR / f"phase2_b3_{attention}.pt"
    if result_path.exists():
        r = json.loads(result_path.read_text())
        log.info(f"[{attention}] RESUMED | val AUC {r['val_auc']:.4f} "
                 f"| TEST AUC {r['test_metrics']['auc']:.4f}")
        return r

    utils.set_seed()  # same seed for every arm -> fair comparison
    man = dataset.load_split_manifest()
    tr = dataset.make_split_dataset("train", man, IMAGE_SIZE, aug_strength=AUG)
    va = dataset.make_split_dataset("val", man, IMAGE_SIZE)
    te = dataset.make_split_dataset("test", man, IMAGE_SIZE)
    log.info(f"--- arm '{attention}' ({BACKBONE} @{IMAGE_SIZE}, batch {BATCH}, "
             f"drop {DROP}, ls {LS}, ft_lr {FT_LR}) ---")

    best_meta, best_epoch, elapsed, _ = fit_two_phase(
        tr, va, device, ckpt, log, tag=f"attn-{attention}", backbone=BACKBONE,
        attention=attention, image_size=IMAGE_SIZE, drop_rate=DROP, label_smoothing=LS,
        ft_lr=FT_LR, batch_size=BATCH, num_workers=WORKERS,
        history_csv=utils.CSV_HISTORY / f"phase2_{attention}_history.csv",
        curve_png=utils.FIG_ARCHIVE / f"phase2_{attention}_curve.png")

    model, _ = model_mod.load_checkpoint(ckpt, device)
    y, p = _infer(model, DataLoader(te, batch_size=BATCH, shuffle=False,
                                    num_workers=WORKERS), device)
    test_m = compute_metrics(y, p)
    r = {"attention": attention, "backbone": BACKBONE,
         "val_auc": best_meta["val_metrics"]["auc"], "best_epoch": best_epoch,
         "train_min": round(elapsed / 60, 1), "test_metrics": test_m}
    result_path.write_text(json.dumps(r, indent=2))
    log.info(f"[{attention}] done ({r['train_min']} min) | "
             f"TEST acc {test_m['accuracy']:.4f} AUC {test_m['auc']:.4f}")
    return r


def main():
    utils.ensure_output_dirs()
    log = utils.get_logger("phase2", "phase2_attention.log")
    device = utils.get_device()
    log.info(f"Device: {device} | Phase 2 attention arms: {ARMS}")

    results = [run_arm(a, device, log) for a in ARMS]

    rows = []
    for r in results:
        m = r["test_metrics"]
        rows.append({"attention": r["attention"], "val_auc": round(r["val_auc"], 4),
                     "train_min": r["train_min"],
                     **{f"test_{k}": round(m[k], 4) for k in METRIC_KEYS}})
    table = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    table.to_csv(SUMMARY_CSV, index=False)

    winner = max(results, key=lambda r: (r["test_metrics"]["auc"], r["test_metrics"]["accuracy"]))
    print("\n=== Phase 2 — attention comparison (sealed test, sorted by AUC) ===")
    print(table.to_string(index=False))
    print(f"\n*** Best arm (test AUC+acc): {winner['attention']} ***")
    print(format_metrics(winner["test_metrics"]))
    print(f"\nSummary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
