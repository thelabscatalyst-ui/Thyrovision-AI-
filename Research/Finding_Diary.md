# ThyroVision — Finding Diary

*Chronological record of what we did and what we found. **⭐ = a finding worth citing
in the paper/report.** Most-recent context lives in the linked detailed docs.*

---

## Phase 1 — Baseline (DONE). Committed backbone: **EfficientNet-B3**.

### Step 1 — Data pipeline (TN5000)
- Built load → artifact-clean (mask burned-in scanner text / calipers) → official
  70/10/20 split.
- ⭐ **TN5000 has NO patient-ID field** (verified across all annotations). Its official
  split is therefore **image-level**, not patient-level — a documented limitation.
  *(detail: `Research/Limitations.md` L1)*

### Step 2 — First baseline + cross-validation (ResNet-50)
- Simple single-phase fine-tune. Single split acc 0.853 / AUC 0.922; 5-fold CV
  **0.852 ± 0.014 / AUC 0.918 ± 0.009**.
- ⭐ **Our plain ResNet-50 (85.2%) reproduces the Medicina paper's ResNet-50 (85.1%)**
  — independent validation that our pipeline is sound.

### Step 3 — ⭐ Data-quality finding: duplicate leakage
- ⭐ **TN5000's official split contains 245 byte-identical duplicate images (119 groups);
  ~44 duplicate pairs straddle the train/test boundary (~4.4% test leakage).** Verified
  3 independent ways (MD5, perceptual hash, embeddings); MD5 is irrefutable.
- ⭐ **Not mentioned in the TN5000 paper nor the Medicina paper** — appears undocumented
  (one reproducibility paper, MDPI Electronics 15/1/151, mentions investigating TN5000
  leakage; must be read in full to fully settle novelty).
  *(detail: `Research/TN5000_Duplicate_Finding.md`; reproduce: `src/verify_duplicates.py`)*

### Step 4 — Backbone bake-off (fair, single recipe)
- ResNet-18/50, EfficientNet-B0/B3 under one identical recipe. ResNet-50 won; both
  EfficientNets underperformed.
- ⭐ **EfficientNet is recipe-sensitive** — under a ResNet-friendly recipe it overfits/
  underperforms; this is a *recipe* artifact, not a true architecture verdict.

### Step 5 — Proper two-phase fine-tuning (the fix)
- Built two-phase recipe (freeze head → unfreeze backbone at low LR) + label smoothing,
  dropout, stronger aug. Re-ran ResNet-50/18 and a 3-config EfficientNet-B3 search.
- ⭐ **Tuned EfficientNet-B3 = 87.1% accuracy / AUC 0.918** (sens 0.895, spec 0.807 at the
  chosen threshold). **Matches the paper's plain EfficientNet-B3 (87.1%) and beats its
  AUC (0.89) — with no GAN.** This is the committed baseline.
  *(detail + full tables: `Research/Baseline_Journey_And_Results.md`)*

### Step 6 — Honest reporting refinements
- ⭐ **Threshold choice matters on imbalanced data:** accuracy at the default 0.5 cutoff
  understated performance; AUC (threshold-free) is the honest headline. Report AUC +
  sensitivity + specificity together; choose the operating threshold by *clinical
  sensitivity*, not raw accuracy.
- Class imbalance handled by **class weighting** (a valid, often-better alternative to
  oversampling/GAN). Ensembling was explored but **dropped** (it blends multiple
  backbones — incompatible with the single-backbone "+CBAM" comparison).

---

## Key findings about the competitor paper (Bahmane 2025, *Medicina*) — all ⭐ citable
- ⭐ Reports **plain EfficientNet-B3 at two different accuracies** (87.1% and 89.7%);
  training time as **both 42 and 12 min**; train/val/test split **stated two ways**
  (70/20/10 vs 70/10/20) — internal inconsistencies.
- ⭐ Their generalisation "proof" is on an **uncited, undescribed "THYROID-DATASET-2022"**
  (NOT DDTI) — not independently verifiable.
- ⭐ Their attention choice (**SE only**) was **never compared** to alternatives.
- ⭐ Their Grad-CAM was **reported, not faithfulness-checked**.
- ⭐ Their biggest accuracy lever was a **GAN (G-RAN), +6.23%** — which we deliberately
  do not use (out of scope; we hit their accuracy without it).

## Findings from the literature (paper-log papers 2/4/6)
- ⭐ **Attention is task-specific** (Paper 2: in a YOLOv8 detector, **CPCA beat CBAM**,
  90.8% vs 91.5% mAP) → we must *test* attention (baseline / +SE / +CBAM), not assume.
- Paper 6 cross-validated classification only internally (DDTI) — our planned
  TN5000→DDTI cross-test is more rigorous.
  *(detail: `Research/Paper_Analysis_2_4_6.md`)*

---

## Where to look
| Topic | File |
|-------|------|
| Full baseline journey + all result tables | `Research/Baseline_Journey_And_Results.md` |
| Duplicate-leakage finding (+ reproduce) | `Research/TN5000_Duplicate_Finding.md` |
| Named limitations (ours + theirs) | `Research/Limitations.md` |
| Gaps → how we tackle each | (in `Limitations.md` + competitor notes) |
| Paper-log deep dive (2/4/6) | `Research/Paper_Analysis_2_4_6.md` |
| Committed B3 confusion matrix | `outputs/figures/FINAL_b3_confusion_matrix.png` |
| Archived (superseded) runs | `outputs/extra/` |
