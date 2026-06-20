"""Caliper-leakage audit (detection only — NOT removal).

Calipers ('+' crosses the sonographer places on a measured nodule) are dim and
overlap echogenic-foci intensity, so we don't inpaint them. But if calipers
appear more often on one class than the other, their mere *presence* leaks the
label — and no amount of inpainting fixes that; it needs a different remedy.

We detect a caliper as an ISOLATED bright cross: a local-maximum pixel whose '+'
arms are bright AND whose diagonal corners are dark (arm_mean - corner_mean >=
MARGIN). The corner-darkness test is what gives specificity — bright *tissue*
sits inside broad bright regions (bright corners too) and is rejected, whereas a
true caliper drawn over tissue has dark diagonals. Validated to fire on exactly
the two real calipers in 000001 while staying silent on clean frames.

The detector is applied identically to both classes, so even if recall is
imperfect, the benign-vs-malignant *comparison* is fair.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from . import dataset, utils

FLOOR = 205         # min brightness for a caliper arm/centre
MARGIN = 45         # required arm-mean minus corner-mean contrast
ARM = 6             # half arm-length in px
NMS_DIST = 10       # merge detections closer than this


def _load_gray(image_id: str) -> np.ndarray:
    return cv2.imread(str(utils.TN5000_IMAGES / f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)


def detect_calipers(gray: np.ndarray) -> list[tuple[int, int, float]]:
    """Return isolated-cross detections [(x, y, contrast), ...] in the tissue area."""
    h, w = gray.shape
    region = np.ones_like(gray, dtype=bool)
    region[: int(h * 0.12), : int(w * 0.14)] = False         # top-left text zone
    region[h - int(h * 0.12):, w - int(w * 0.12):] = False   # bottom-right marker

    g = gray.astype(np.float32)
    locmax = (gray >= FLOOR) & (gray == maximum_filter(gray, size=5)) & region
    ys, xs = np.where(locmax)

    cand: list[tuple[int, int, float]] = []
    a, d = ARM, ARM - 1
    for y, x in zip(ys, xs):
        if x - a < 0 or x + a >= w or y - a < 0 or y + a >= h:
            continue
        arm_h, arm_v = g[y, x - a:x + a + 1], g[y - a:y + a + 1, x]
        arm_mean = (arm_h.sum() + arm_v.sum() - g[y, x]) / (2 * (2 * a + 1) - 1)
        corner_mean = float(np.mean([g[y - d, x - d], g[y - d, x + d],
                                     g[y + d, x - d], g[y + d, x + d]]))
        ends = [g[y, x - a], g[y, x + a], g[y - a, x], g[y + a, x]]
        if arm_mean - corner_mean >= MARGIN and min(ends) >= FLOOR - 30:
            cand.append((int(x), int(y), float(arm_mean - corner_mean)))

    kept: list[tuple[int, int, float]] = []
    for x, y, s in sorted(cand, key=lambda p: -p[2]):
        if all((x - kx) ** 2 + (y - ky) ** 2 >= NMS_DIST ** 2 for kx, ky, _ in kept):
            kept.append((x, y, s))
    return kept


def has_caliper(gray: np.ndarray, min_count: int = 1) -> bool:
    return len(detect_calipers(gray)) >= min_count


# ── Validation: draw detections on a few images ────────────────────────────
def validate(ids=("000001", "000191", "004755", "002500", "000500", "004000")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(ids), figsize=(3.2 * len(ids), 3.4))
    for ax, img_id in zip(axes, ids):
        gray = _load_gray(img_id)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        for x, y, _ in detect_calipers(gray):
            cv2.drawMarker(vis, (x, y), (255, 0, 0), cv2.MARKER_SQUARE, 22, 2)
        ax.imshow(vis)
        ax.set_title(f"{img_id}: {len(detect_calipers(gray))} caliper(s)")
        ax.axis("off")
    fig.tight_layout()
    out = utils.FIGURES_DIR / "stageA_caliper_detection.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved detector validation -> {out}")


# ── The audit: per-class caliper-presence rate ─────────────────────────────
def run_audit(min_count: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = dataset.load_split_manifest()
    df = df.assign(has_caliper=[has_caliper(_load_gray(i), min_count) for i in df.image_id])
    rows = []
    for label, name in ((0, "benign"), (1, "malignant")):
        sub = df[df.label == label]
        rows.append({"class": name, "n": len(sub),
                     "with_caliper": int(sub.has_caliper.sum()),
                     "caliper_%": round(100 * sub.has_caliper.mean(), 1)})
    return pd.DataFrame(rows), df


if __name__ == "__main__":
    utils.ensure_output_dirs()
    validate()
    audit, _ = run_audit()
    print("\n=== Caliper-presence by class (residual detection) ===")
    print(audit.to_string(index=False))
    gap = abs(audit["caliper_%"].iloc[0] - audit["caliper_%"].iloc[1])
    print(f"\nAbsolute gap (|malignant - benign|): {gap:.1f} percentage points")
