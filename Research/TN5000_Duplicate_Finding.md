# Finding: TN5000 Contains Exact-Duplicate Images That Leak Across Its Official Split

**Date:** 2026-06-20 · **Status:** empirically confirmed (model-free, reproducible)

## Summary
TN5000's **official, published image-level split** contains **245 images (4.9%) that are
byte-for-byte identical duplicates** of other images in the dataset — 119 duplicate
groups. **44 of these duplicate pairs straddle the train/val ↔ test boundary**, so a
model trained on the official split is tested on images it effectively already saw
(~4.4% of the 1,000-image test set). The dataset paper does not mention this; neither
does the closest applied paper (Bahmane et al., *Medicina* 2025), which uses the same split.

## Evidence — three independent methods agree
| Method | What it proves | Result |
|--------|----------------|--------|
| **MD5 of raw file bytes** | byte-for-byte identical **files** | **119 groups / 245 images** |
| MD5 of decoded pixel array | identical **pixels** (catches re-encodes) | 119 groups (same) |
| Perceptual dHash (Hamming = 0) | near-identical, **no model** | 136 pairs / 45 cross-test |
| ResNet-50 embeddings (cosine ≥0.98) | "the model sees them as the same" | 133 pairs / 44 cross-test |

The **MD5 result is decisive**: a cryptographic-hash collision cannot occur by chance,
and it uses **no model and no threshold**, so this is not a judgement call — the images
are *literally the same file* stored under different IDs. The four methods agree to
within ~2 pairs.

**Label integrity check:** 100% of duplicate pairs share the same label (as true
duplicates must) — further confirming they are genuine duplicates, not coincidental
look-alikes.

## Root cause
The TN5000 paper states the data was *"randomly split at the image level"* (7:1:2) with
**no de-duplication step** ([Nature Sci Data, s41597-025-05757-4](https://www.nature.com/articles/s41597-025-05757-4)).
A random image-level split with duplicates present inevitably scatters identical images
across train/val/test → leakage. There is no patient-ID field to enable a patient-level
split, so the duplicates cannot be grouped by patient from the metadata.

## Finding
Most important finding which is still making me doubtful that whether this anomaly is correct or not is *Duplicates are found in adjacent photos* so it is still fishy. 

## Impact
- Any same-dataset TN5000 result on the official split is **mildly inflated** (~4.4% of
  the test set is leaked). Effect size is small (estimated <1% accuracy; the de-dup
  re-run quantifies it), but it is real and **shared by every paper using this split**.
- It does **not** affect *relative* comparisons (e.g. baseline vs CBAM) on the same split,
  since the leak hits both arms equally.

## What we do about it (in this project)
- Keep TN5000's official test set intact for benchmark comparability, but **remove the
  44 train/val images that duplicate a test image** (de-contamination), and use
  **group-aware CV** so duplicate clusters never cross folds.
- Report **official vs de-contaminated** numbers side by side as a sensitivity analysis.
- The cross-institution generalisation claim rests on the **DDTI** test, which is
  duplicate- and institution-disjoint by construction.

## Novelty status (honest)
- The **TN5000 paper** and the **Medicina paper** do **not** mention duplicates or
  de-duplication.
- An extensive literature search found **no accessible source that documents TN5000
  exact-duplicates or duplicate-driven split leakage.**
- ⚠️ **One paper to read before any novelty claim:** *"ROI+Context Graph Neural Networks
  for Thyroid Nodule Classification: …Cross-Validation Protocol, and Reproducibility"*
  ([MDPI Electronics 15/1/151](https://www.mdpi.com/2079-9292/15/1/151)) reportedly
  *investigated potential data leakage* in TN5000 during a fold-5 calibration anomaly.
  Its full text was inaccessible (HTTP 403). **Read it to confirm whether it merely
  hypothesised leakage or actually identified the duplicates.** Even if it flagged
  leakage, our exact quantification + fix may remain a distinct contribution.
- **"Nobody has cited it" cannot be proven to 100%** (one cannot prove a negative across
  all literature). Strongest honest claim: *no accessible source documents it, pending
  that one paper.*

## Reproduce (one command, ~10 s)
```bash
.venv/bin/python -m src.verify_duplicates
```
Outputs the table above and writes every duplicate group to `outputs/tn5000_duplicates.csv`
(columns: `dup_group, image_id, label, split`). Method: `src/verify_duplicates.py`.
