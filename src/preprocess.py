"""Artifact removal for TN5000 ultrasound images.

Why this exists
---------------
TN5000 frames have burned-in graphics that a CNN can "cheat" on instead of
learning the nodule:
  * scanner-model text in the top-left dark border (e.g. "LOGIQ E9", "LS7"),
  * a small position marker in the bottom-right corner,
  * caliper "+" marks placed on the lesion the sonographer measured.

If benign/malignant happens to correlate with the scanner used, or with whether
a nodule was measured, the model can read that off the overlay graphics and post
a fake-good score. So we remove these *before* the image ever reaches the model.

Two mechanisms, both operating on pure-white (>=250) overlay pixels — real
tissue is almost never truly 255, whereas machine-drawn graphics are:
  1. Corner masking — blank the fixed border corners where text/markers sit.
  2. Caliper inpainting — fill small pure-white blobs inside the tissue using
     cv2.inpaint, so the nodule texture underneath is reconstructed rather than
     blacked out.

This is *cleaning*, not augmentation: it is applied identically to train, val
AND test. A safety valve skips caliper inpainting if the detected overlay area
is implausibly large (a sign we'd be erasing genuine bright tissue, e.g.
echogenic foci, which are clinically meaningful).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Fraction of (width, height) blanked at each corner.
TL_FRAC = (0.14, 0.12)   # top-left: scanner-model text
BR_FRAC = (0.12, 0.12)   # bottom-right: position marker

OVERLAY_THRESH = 250     # pure-white overlay graphics
CALIPER_MIN_AREA = 2     # px — ignore single-pixel speckle
CALIPER_MAX_AREA = 120   # px — calipers/letters are small; bigger => not a caliper
CALIPER_AREA_SAFETY = 0.02  # if >2% of the frame is "overlay", skip inpainting
INPAINT_RADIUS = 3


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an (H, W, 3) uint8 RGB array."""
    return np.array(Image.open(path).convert("RGB"))


def mask_corners(img: np.ndarray) -> np.ndarray:
    """Black out the top-left and bottom-right corner rectangles."""
    out = img.copy()
    h, w = out.shape[:2]
    tlw, tlh = int(w * TL_FRAC[0]), int(h * TL_FRAC[1])
    brw, brh = int(w * BR_FRAC[0]), int(h * BR_FRAC[1])
    out[0:tlh, 0:tlw] = 0                      # top-left
    out[h - brh:h, w - brw:w] = 0              # bottom-right
    return out


def _overlay_blob_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of small, pure-white, compact blobs (calipers / stray text)."""
    white = (gray >= OVERLAY_THRESH).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    mask = np.zeros_like(white)
    for i in range(1, n):  # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if CALIPER_MIN_AREA <= area <= CALIPER_MAX_AREA:
            mask[labels == i] = 1
    return mask


def remove_calipers(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Inpaint caliper-like overlay blobs. Returns (image, overlay_area_fraction).

    If the overlay area is implausibly large we DON'T inpaint (to avoid erasing
    real bright tissue) and return the image unchanged with the measured
    fraction, so the caller can flag it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = _overlay_blob_mask(gray)
    frac = float(mask.sum()) / float(mask.size)
    if frac > CALIPER_AREA_SAFETY or mask.sum() == 0:
        return img, frac
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    out = cv2.inpaint(img, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return out, frac


def clean(img: np.ndarray, do_calipers: bool = True) -> np.ndarray:
    """Full cleaning pipeline: corner masking, then optional caliper inpainting."""
    out = mask_corners(img)
    if do_calipers:
        out, _ = remove_calipers(out)
    return out


def clean_path(path: str | Path, do_calipers: bool = True) -> np.ndarray:
    """Convenience: load a file and return the cleaned RGB array."""
    return clean(load_rgb(path), do_calipers=do_calipers)
