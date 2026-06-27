"""Shared helpers: paths, reproducibility, device selection, logging.

Kept deliberately small — everything here is reused across the data, training,
and evaluation code so there's a single source of truth for "where things live"
and "which device we run on".
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
# PROJECT_ROOT is the folder that contains src/, Data/, outputs/, Research/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "Data"

# TN5000 (training set this phase). The dataset proper lives two folders deep.
TN5000_ROOT = DATA_ROOT / "TN5000 Dataset" / "TN5000_forReview" / "TN5000_forReview"
TN5000_IMAGES = TN5000_ROOT / "JPEGImages"
TN5000_ANNOTATIONS = TN5000_ROOT / "Annotations"
TN5000_IMAGESETS = TN5000_ROOT / "ImageSets" / "Main"

# DDTI is QUARANTINED in Phase 1 — defined here only so guardrails can detect it.
DDTI_ROOT = DATA_ROOT / "DDTI Dataset"

# Outputs. Tracked in git: csv/ (frozen split + result tables) and figures/.
# Ignored: checkpoints/ (large weights) and logs/ (run logs + json resume-markers).
OUTPUTS = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS / "checkpoints"
FIGURES_DIR = OUTPUTS / "figures"
LOGS_DIR = OUTPUTS / "logs"
CSV_DIR = OUTPUTS / "csv"             # frozen split + result/history/summary CSVs (tracked)
JSON_DIR = LOGS_DIR / "json"         # per-run result JSONs + metrics (ignored)
LOG_DIR = LOGS_DIR / "logs_files"    # .log / .txt run logs (ignored)
SPLIT_CSV = CSV_DIR / "tn5000_split.csv"

# Class convention: 0 = benign, 1 = malignant (= positive class for sensitivity).
CLASS_NAMES = ("benign", "malignant")

# Standard ImageNet normalisation — required because we fine-tune an
# ImageNet-pretrained backbone, so inputs must be normalised the same way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Make a run reproducible: seed Python, NumPy and torch (CPU + MPS)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS generator
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Return the Apple GPU (MPS) if available, else CPU.

    Sets PYTORCH_ENABLE_MPS_FALLBACK so any op not yet implemented on MPS
    silently falls back to CPU instead of crashing mid-run.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dirs() -> None:
    """Create the outputs/ subfolders if they don't already exist."""
    for d in (CHECKPOINTS_DIR, FIGURES_DIR, CSV_DIR, JSON_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """A small console (+ optional file) logger, no duplicate handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        ensure_output_dirs()
        fh = logging.FileHandler(LOG_DIR / log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
