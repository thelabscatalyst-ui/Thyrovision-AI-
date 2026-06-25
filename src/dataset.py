"""TN5000 data layer: label parsing, frozen split manifest, PyTorch Dataset.

Design notes
------------
* Labels come from each image's Pascal-VOC XML: a single <object><name> of
  "0" (benign) or "1" (malignant).
* We freeze the split to outputs/tn5000_split.csv so every run — training,
  evaluation, future CV — reads the *same* assignment. The test set is therefore
  defined exactly once (rule 3).
* We adopt TN5000's official train/val/test split as published. TN5000 has no
  patient-ID field (verified), so this is an IMAGE-level split; we cannot prove
  zero patient overlap and do not claim it (rule 2 — documented, not silently
  ignored).
* Guardrail: the Dataset refuses any path under DDTI (rule 1, enforced in code).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from . import preprocess, utils

SPLIT_FILES = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}


# ── Label parsing ──────────────────────────────────────────────────────────
def parse_voc_xml(xml_path: str | Path) -> int:
    """Return the binary label (0 benign / 1 malignant) from a VOC annotation."""
    root = ET.parse(xml_path).getroot()
    names = [obj.find("name").text.strip() for obj in root.findall("object")]
    if len(names) != 1:
        raise ValueError(f"{xml_path}: expected exactly 1 object, found {len(names)}")
    label = int(names[0])
    if label not in (0, 1):
        raise ValueError(f"{xml_path}: unexpected class label {label!r}")
    return label


def _read_ids(split: str) -> list[str]:
    txt = utils.TN5000_IMAGESETS / SPLIT_FILES[split]
    return [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]


# ── Frozen split manifest ──────────────────────────────────────────────────
def build_split_manifest(force: bool = False) -> pd.DataFrame:
    """Build (and cache) outputs/tn5000_split.csv: image_id, label, split."""
    if utils.SPLIT_CSV.exists() and not force:
        return pd.read_csv(utils.SPLIT_CSV, dtype={"image_id": str})

    rows = []
    for split in SPLIT_FILES:
        for image_id in _read_ids(split):
            label = parse_voc_xml(utils.TN5000_ANNOTATIONS / f"{image_id}.xml")
            rows.append({"image_id": image_id, "label": label, "split": split})

    df = pd.DataFrame(rows)
    utils.ensure_output_dirs()
    df.to_csv(utils.SPLIT_CSV, index=False)
    return df


def load_split_manifest() -> pd.DataFrame:
    """Load the frozen manifest, building it on first use."""
    return build_split_manifest(force=False)


def load_clean_manifest() -> pd.DataFrame:
    """Load the de-contaminated manifest (image_id, label, split, group, keep).

    Built by near_duplicates.build_clean_manifest(); has near-duplicate leakers
    flagged keep=False and a `group` column for group-aware CV folds.
    """
    clean = utils.OUTPUTS / "tn5000_split_clean.csv"
    if not clean.exists():
        raise FileNotFoundError(
            f"{clean} missing — run near_duplicates.build_clean_manifest() first")
    return pd.read_csv(clean, dtype={"image_id": str})


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split counts of benign/malignant, for the verification report."""
    tab = df.pivot_table(index="split", columns="label", aggfunc="size", fill_value=0)
    tab = tab.rename(columns={0: "benign", 1: "malignant"})
    tab = tab.reindex(["train", "val", "test"])
    tab["total"] = tab.sum(axis=1)
    tab["malignant_%"] = (100 * tab["malignant"] / tab["total"]).round(1)
    return tab


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights from the TRAIN split only.

    weight_c = n_train / (n_classes * count_c) — so the rarer class (benign here)
    gets a larger weight and the model can't just default to the majority class.
    """
    train = df[df.split == "train"]
    counts = train.label.value_counts().sort_index()
    n, k = len(train), len(counts)
    weights = [n / (k * counts[c]) for c in (0, 1)]
    return torch.tensor(weights, dtype=torch.float32)


# ── Transforms ─────────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = 224, aug: str = "default") -> A.Compose:
    """albumentations pipeline. Augmentation is TRAIN-ONLY (rule 5).

    aug="default" = light (flip/rotate/brightness); aug="strong" = heavier set for the
    EfficientNet-B3 recipe (random-resized-crop, bigger rotation, contrast, noise) — the
    paper's biggest lever was augmentation. No vertical flip in either (anatomy)."""
    normalize = A.Normalize(mean=utils.IMAGENET_MEAN, std=utils.IMAGENET_STD)
    tail = [normalize, ToTensorV2()]
    if split != "train":
        return A.Compose([A.Resize(image_size, image_size), *tail])
    if aug == "strong":
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0),
                                ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.6),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(p=0.2),
            *tail,
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),                 # left-right is clinically safe
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.3),
        *tail,
    ])


# ── Dataset ────────────────────────────────────────────────────────────────
class TN5000Dataset(Dataset):
    """Whole-image benign/malignant dataset with on-the-fly artifact cleaning.

    Takes an explicit `rows` DataFrame (needs columns image_id, label) so it
    serves both the official splits and arbitrary CV folds. `train_aug` toggles
    training augmentation (True) vs deterministic eval transforms (False).
    """

    def __init__(self, rows: pd.DataFrame, train_aug: bool,
                 image_size: int = 224, do_calipers: bool = True,
                 aug_strength: str = "default"):
        # ── Rule-1 guardrail: never let DDTI through this class. ──
        root = str(utils.TN5000_IMAGES)
        assert "TN5000" in root, f"images root is not TN5000: {root}"
        assert "DDTI" not in root, "DDTI is quarantined in Phase 1 — refusing to load it"

        self.rows = rows.reset_index(drop=True)
        self.do_calipers = do_calipers
        self.transform = get_transforms("train" if train_aug else "val",
                                        image_size, aug_strength)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        img_path = utils.TN5000_IMAGES / f"{row.image_id}.jpg"
        img = preprocess.load_rgb(img_path)            # raw RGB uint8
        img = preprocess.clean(img, do_calipers=self.do_calipers)  # remove artifacts
        img = self.transform(image=img)["image"]       # -> normalised CHW tensor
        label = torch.tensor(int(row.label), dtype=torch.long)
        return img, label


def make_split_dataset(split: str, manifest: pd.DataFrame | None = None,
                       image_size: int = 224, do_calipers: bool = True,
                       aug_strength: str = "default") -> TN5000Dataset:
    """Build a dataset for an official split ('train' / 'val' / 'test')."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {list(SPLIT_FILES)}, got {split!r}")
    df = manifest if manifest is not None else load_split_manifest()
    return TN5000Dataset(df[df.split == split], train_aug=(split == "train"),
                         image_size=image_size, do_calipers=do_calipers,
                         aug_strength=aug_strength)
