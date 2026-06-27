"""Verify TN5000 duplicate images — model-free, reproducible, one command.

Confirms the finding three independent ways:
  1. MD5 of the raw file bytes      -> byte-for-byte identical FILES  (irrefutable)
  2. MD5 of the decoded pixel array -> identical PIXELS (catches re-encodes)
  3. 64-bit dHash (Hamming distance) -> perceptual near-duplicates (no model)

Then cross-tabulates duplicate pairs against TN5000's official train/val/test split
to count the leakage (duplicate pairs that straddle the test boundary), and writes
every duplicate group to outputs/csv/tn5000_duplicates.csv.

The MD5 check uses no model and no threshold, so the result is not a judgement call.

Run:  .venv/bin/python -m src.verify_duplicates
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from . import dataset, utils

DHASH_SIZE = 8
DUPES_CSV = utils.CSV_DIR / "tn5000_duplicates.csv"


def _dhash(img: Image.Image, size: int = DHASH_SIZE) -> np.ndarray:
    """64-bit difference hash (row-wise brightness gradients)."""
    a = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def compute(manifest: pd.DataFrame):
    file_md5, pix_md5, dh = defaultdict(list), defaultdict(list), []
    for image_id in manifest.image_id:
        p = utils.TN5000_IMAGES / f"{image_id}.jpg"
        file_md5[hashlib.md5(p.read_bytes()).hexdigest()].append(image_id)
        img = Image.open(p)
        pix_md5[hashlib.md5(np.asarray(img.convert("RGB")).tobytes()).hexdigest()].append(image_id)
        dh.append(_dhash(img))
    return file_md5, pix_md5, np.array(dh, dtype=np.float32)


def _cross_test_pairs(groups: list[list[str]], split: dict[str, str]) -> int:
    """Count duplicate pairs whose two members fall on opposite sides of the test set."""
    n = 0
    for g in groups:
        tests = [split[i] == "test" for i in g]
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                n += tests[a] != tests[b]
    return n


def main():
    manifest = dataset.load_split_manifest()
    split = dict(zip(manifest.image_id, manifest.split))
    file_md5, pix_md5, dh = compute(manifest)

    file_groups = [v for v in file_md5.values() if len(v) > 1]
    pix_groups = [v for v in pix_md5.values() if len(v) > 1]
    n_imgs = sum(len(g) for g in file_groups)

    print("=== TN5000 duplicate verification (model-free) ===")
    print(f"[MD5 file ] identical-FILE groups : {len(file_groups):3d}  images: {n_imgs}")
    print(f"[MD5 pixel] identical-PIXEL groups: {len(pix_groups):3d}")
    print(f"[MD5 file ] cross-test duplicate pairs (leakage): "
          f"{_cross_test_pairs(file_groups, split)}")

    # perceptual near-duplicates (independent corroboration)
    H = dh @ (1 - dh).T + (1 - dh) @ dh.T
    np.fill_diagonal(H, 99)
    for thr in (0, 3, 5):
        pi, pj = np.where(np.triu(H <= thr, 1))
        ids = manifest.image_id.values
        cross = sum((split[ids[i]] == "test") != (split[ids[j]] == "test")
                    for i, j in zip(pi, pj))
        print(f"[dHash<= {thr}] pairs: {len(pi):4d}  cross-test: {cross}")

    # write every duplicate group + its split membership
    rows = []
    for gi, g in enumerate(file_groups):
        for image_id in g:
            rows.append({"dup_group": gi, "image_id": image_id,
                         "label": int(manifest.set_index("image_id").loc[image_id, "label"]),
                         "split": split[image_id]})
    pd.DataFrame(rows).to_csv(DUPES_CSV, index=False)
    print(f"\nSaved {len(rows)} duplicate-image rows ({len(file_groups)} groups) -> {DUPES_CSV}")


if __name__ == "__main__":
    main()
