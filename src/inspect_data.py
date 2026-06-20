"""Stage-A verification — run BEFORE any training.

Prints the per-split class balance and saves visual proof to outputs/figures/
that (a) artifact cleaning works and (b) augmentation is sane, so we can eyeball
the data layer before spending compute.

Run:  .venv/bin/python -m src.inspect_data
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # no display needed; we save PNGs
import albumentations as A
import cv2
import matplotlib.pyplot as plt

from . import dataset, preprocess, utils

# Images we know are interesting + a few spread across the ID range.
SAMPLE_IDS = ["000001", "000191", "004755", "000500", "002500", "004000"]
N_AUG = 8


def _viz_aug_pipeline(size: int = 224) -> A.Compose:
    """Same augmentations as training but WITHOUT normalize/tensor, for display."""
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.7),
    ])


def report_split(df) -> None:
    # Sanity: every image appears in exactly one split.
    dup = df.image_id.value_counts()
    assert (dup == 1).all(), "an image_id appears in more than one split!"

    print("\n=== TN5000 split (official, image-level — NO patient ID available) ===")
    print(dataset.class_distribution(df).to_string())
    w = dataset.compute_class_weights(df)
    print(f"\nClass weights (train, inverse-freq): benign={w[0]:.3f}  malignant={w[1]:.3f}")
    print("NOTE: split is IMAGE-level; TN5000 has no patient ID, so patient-level")
    print("      separation cannot be guaranteed. Documented as a limitation.")
    print(f"Frozen manifest: {utils.SPLIT_CSV}")


def save_clean_montage() -> None:
    """Raw vs artifact-cleaned, side by side, for the sample images."""
    n = len(SAMPLE_IDS)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    for i, img_id in enumerate(SAMPLE_IDS):
        path = utils.TN5000_IMAGES / f"{img_id}.jpg"
        raw = preprocess.load_rgb(path)
        _, frac = preprocess.remove_calipers(preprocess.mask_corners(raw))
        cleaned = preprocess.clean(raw)
        axes[i, 0].imshow(raw)
        axes[i, 0].set_title(f"{img_id} — raw")
        axes[i, 1].imshow(cleaned)
        axes[i, 1].set_title(f"{img_id} — cleaned (overlay {frac*100:.2f}%)")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    out = utils.FIGURES_DIR / "stageA_raw_vs_cleaned.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nSaved cleaning montage -> {out}")


def save_aug_montage() -> None:
    """One cleaned training image, augmented N times, to check augmentation."""
    img_id = SAMPLE_IDS[1]
    cleaned = preprocess.clean(preprocess.load_rgb(utils.TN5000_IMAGES / f"{img_id}.jpg"))
    aug = _viz_aug_pipeline()
    cols = 4
    rows = (N_AUG + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for j, ax in enumerate(axes.ravel()):
        if j < N_AUG:
            ax.imshow(aug(image=cleaned)["image"])
            ax.set_title(f"aug #{j+1}")
        ax.axis("off")
    fig.suptitle(f"Training augmentations on {img_id} (cleaned)")
    fig.tight_layout()
    out = utils.FIGURES_DIR / "stageA_augmentations.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved augmentation montage -> {out}")


def main() -> None:
    utils.set_seed()
    utils.ensure_output_dirs()
    df = dataset.build_split_manifest(force=True)  # (re)build the frozen manifest
    report_split(df)
    save_clean_montage()
    save_aug_montage()
    print("\nStage-A verification complete. Review the two figures before training.")


if __name__ == "__main__":
    main()
