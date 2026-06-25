"""Fine-tune the ResNet-50 baseline on the TN5000 train split.

What happens here (plain English)
---------------------------------
* We load the train and val splits (the test split is NOT touched — rule 3).
* Each epoch = one full pass over the training images. After each epoch we check
  performance on the val split and remember the weights that gave the best val
  AUC. "Early stopping" halts once val AUC stops improving — saves time and
  prevents over-fitting (memorising the training images).
* Loss is class-weighted so the 71/29 malignant/benign imbalance doesn't let the
  model lazily favour the majority class.

The core training routine lives in `fit()` so the 5-fold CV script reuses the
EXACT same logic (a fairness guarantee — baseline and CV differ only in data).

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m src.train
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import dataset, model as model_mod, utils
from .metrics import compute_metrics, format_metrics

# ── Hyperparameters ────────────────────────────────────────────────────────
EPOCHS = 30
PATIENCE = 5            # early-stop after this many epochs without val-AUC gain
BATCH_SIZE = 32
LR = 1e-4              # low: we're nudging pretrained weights, not learning fresh
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
IMAGE_SIZE = 224
CKPT_PATH = utils.CHECKPOINTS_DIR / "resnet50_baseline.pt"
HISTORY_CSV = utils.LOGS_DIR / "baseline_history.csv"
CURVE_PNG = utils.FIGURES_DIR / "baseline_training_curve.png"


def _run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass. If optimizer is given -> train; else -> evaluate (no grad)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    probs, ys = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        ys.append(y.cpu().numpy())
    metrics = compute_metrics(np.concatenate(ys), np.concatenate(probs))
    return total_loss / n, metrics


def _class_weights(rows) -> torch.Tensor:
    """Inverse-frequency weights from this training subset's labels."""
    counts = rows.label.value_counts().sort_index()
    n, k = len(rows), len(counts)
    return torch.tensor([n / (k * counts[c]) for c in (0, 1)], dtype=torch.float32)


def fit(train_ds, val_ds, device, ckpt_path, log, *, tag="", backbone="resnet50",
        image_size=IMAGE_SIZE, epochs=EPOCHS, patience=PATIENCE,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        history_csv=None, curve_png=None, meta_extra=None):
    """Train one model with early stopping. Returns (best_meta, best_epoch,
    elapsed_sec, history). The best-val-AUC weights are saved to ckpt_path.

    batch_size / num_workers are tunable so large-resolution backbones (e.g.
    EfficientNet-B3 @300px) can use a smaller batch to fit in unified memory."""
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, persistent_workers=num_workers > 0)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0)

    class_weights = _class_weights(train_ds.rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = model_mod.build_model(backbone, num_classes=2, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc, best_epoch, best_meta, no_improve = -1.0, -1, None, 0
    history, t0 = [], time.time()
    pfx = f"[{tag}] " if tag else ""

    for epoch in range(1, epochs + 1):
        tr_loss, tr_m = _run_epoch(model, train_ld, device, criterion, optimizer)
        va_loss, va_m = _run_epoch(model, val_ld, device, criterion)
        scheduler.step()
        if device.type == "mps":
            torch.mps.empty_cache()   # release cached MPS memory between epochs
        log.info(f"{pfx}epoch {epoch:02d} | train loss {tr_loss:.4f} AUC {tr_m['auc']:.4f} "
                 f"| val loss {va_loss:.4f} AUC {va_m['auc']:.4f} "
                 f"sens {va_m['sensitivity']:.3f} spec {va_m['specificity']:.3f}")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_auc": tr_m["auc"],
                        "val_loss": va_loss, "val_auc": va_m["auc"],
                        "val_accuracy": va_m["accuracy"],
                        "val_sensitivity": va_m["sensitivity"],
                        "val_specificity": va_m["specificity"]})

        if va_m["auc"] > best_auc:
            best_auc, best_epoch, no_improve = va_m["auc"], epoch, 0
            best_meta = {"epoch": epoch, "val_metrics": va_m, "backbone": backbone,
                         "class_names": utils.CLASS_NAMES, "image_size": image_size,
                         "split_csv": str(utils.SPLIT_CSV), "device": str(device),
                         "seed": utils.SEED, **(meta_extra or {})}
            model_mod.save_checkpoint(model, ckpt_path, meta=best_meta)
            log.info(f"{pfx}  ↳ new best val AUC {best_auc:.4f} (checkpoint saved)")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"{pfx}early stopping at epoch {epoch} (no val-AUC gain for {patience})")
                break

    elapsed = time.time() - t0
    if history_csv is not None:
        pd.DataFrame(history).to_csv(history_csv, index=False)
    if curve_png is not None:
        _save_curve(history, curve_png)
    return best_meta, best_epoch, elapsed, history


def _save_curve(history: list[dict], path) -> None:
    h = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(h.epoch, h.train_loss, label="train")
    ax1.plot(h.epoch, h.val_loss, label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(h.epoch, h.train_auc, label="train AUC")
    ax2.plot(h.epoch, h.val_auc, label="val AUC")
    ax2.set_title("AUC"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fit_two_phase(train_ds, val_ds, device, ckpt_path, log, *, tag="",
                  backbone="resnet50", image_size=IMAGE_SIZE, drop_rate=0.0,
                  label_smoothing=0.0, head_epochs=6, head_lr=1e-3,
                  ft_epochs=25, ft_lr=1e-4, patience=8, unfreeze="all",
                  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                  history_csv=None, curve_png=None, meta_extra=None):
    """Two-phase fine-tuning (best practice): Phase 1 trains the head with the
    backbone frozen; Phase 2 unfreezes the backbone and fine-tunes at a low LR with
    cosine decay + early stopping. Tracks the best-val-AUC checkpoint across both
    phases. Same return signature as `fit`."""
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, persistent_workers=num_workers > 0)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    cw = _class_weights(train_ds.rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    model = model_mod.build_model(backbone, num_classes=2, pretrained=True,
                                  drop_rate=drop_rate).to(device)

    history, t0 = [], time.time()
    best_auc, best_epoch, best_meta, no_improve, gepoch = -1.0, -1, None, 0, 0
    pfx = f"[{tag}] " if tag else ""

    def _log_save(phase, tr_loss, tr_m, va_loss, va_m):
        nonlocal best_auc, best_epoch, best_meta, no_improve, gepoch
        gepoch += 1
        if device.type == "mps":
            torch.mps.empty_cache()
        log.info(f"{pfx}P{phase} epoch {gepoch:02d} | train loss {tr_loss:.4f} "
                 f"AUC {tr_m['auc']:.4f} | val loss {va_loss:.4f} AUC {va_m['auc']:.4f} "
                 f"sens {va_m['sensitivity']:.3f} spec {va_m['specificity']:.3f}")
        history.append({"epoch": gepoch, "phase": phase, "train_loss": tr_loss,
                        "train_auc": tr_m["auc"], "val_loss": va_loss, "val_auc": va_m["auc"],
                        "val_accuracy": va_m["accuracy"], "val_sensitivity": va_m["sensitivity"],
                        "val_specificity": va_m["specificity"]})
        improved = va_m["auc"] > best_auc
        if improved:
            best_auc, best_epoch, no_improve = va_m["auc"], gepoch, 0
            best_meta = {"epoch": gepoch, "phase": phase, "val_metrics": va_m,
                         "backbone": backbone, "class_names": utils.CLASS_NAMES,
                         "image_size": image_size, "split_csv": str(utils.SPLIT_CSV),
                         "device": str(device), "seed": utils.SEED, **(meta_extra or {})}
            model_mod.save_checkpoint(model, ckpt_path, meta=best_meta)
        return improved

    # ── Phase 1: head only (backbone frozen) ──
    model_mod.set_backbone_trainable(model, False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=head_lr, weight_decay=WEIGHT_DECAY)
    for _ in range(head_epochs):
        tr_loss, tr_m = _run_epoch(model, train_ld, device, criterion, opt)
        va_loss, va_m = _run_epoch(model, val_ld, device, criterion)
        _log_save(1, tr_loss, tr_m, va_loss, va_m)

    # ── Phase 2: unfreeze backbone, fine-tune at low LR with cosine + early stop ──
    model_mod.set_backbone_trainable(model, True)  # ("all" — top-blocks-only not needed; low LR guards)
    opt = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ft_epochs)
    for _ in range(ft_epochs):
        tr_loss, tr_m = _run_epoch(model, train_ld, device, criterion, opt)
        va_loss, va_m = _run_epoch(model, val_ld, device, criterion)
        sched.step()
        if not _log_save(2, tr_loss, tr_m, va_loss, va_m):
            no_improve += 1
            if no_improve >= patience:
                log.info(f"{pfx}early stopping (no val-AUC gain for {patience})")
                break

    elapsed = time.time() - t0
    if history_csv is not None:
        pd.DataFrame(history).to_csv(history_csv, index=False)
    if curve_png is not None:
        _save_curve(history, curve_png)
    return best_meta, best_epoch, elapsed, history


def main():
    utils.set_seed()
    utils.ensure_output_dirs()
    log = utils.get_logger("train", "baseline_train.log")
    device = utils.get_device()
    log.info(f"Device: {device}")

    manifest = dataset.load_split_manifest()
    train_ds = dataset.make_split_dataset("train", manifest, IMAGE_SIZE)
    val_ds = dataset.make_split_dataset("val", manifest, IMAGE_SIZE)
    log.info(f"train={len(train_ds)}  val={len(val_ds)}  (test sealed, not loaded)")

    best_meta, best_epoch, elapsed, _ = fit(
        train_ds, val_ds, device, CKPT_PATH, log,
        history_csv=HISTORY_CSV, curve_png=CURVE_PNG)

    log.info("=" * 60)
    log.info(f"Training done. Wall-clock: {elapsed/60:.1f} min on {device}.")
    log.info(f"Best epoch {best_epoch}, val AUC {best_meta['val_metrics']['auc']:.4f}.")
    print("\nBest-epoch validation metrics:")
    print(format_metrics(best_meta["val_metrics"]))
    print(f"\nWall-clock: {elapsed/60:.1f} min on {device} | best epoch {best_epoch}")


if __name__ == "__main__":
    main()
