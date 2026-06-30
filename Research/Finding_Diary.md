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
  *(detail: `research/Limitations.md` L1)*

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
  *(detail: `research/TN5000_Duplicate_Finding.md`; reproduce: `src/verify_duplicates.py`)*

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
  *(detail + full tables: `research/Baseline_Journey_And_Results.md`)*

### Step 6 — Honest reporting refinements
- ⭐ **Threshold choice matters on imbalanced data:** accuracy at the default 0.5 cutoff
  understated performance; AUC (threshold-free) is the honest headline. Report AUC +
  sensitivity + specificity together; choose the operating threshold by *clinical
  sensitivity*, not raw accuracy.
- Class imbalance handled by **class weighting** (a valid, often-better alternative to
  oversampling/GAN). Ensembling was explored but **dropped** (it blends multiple
  backbones — incompatible with the single-backbone "+CBAM" comparison).

### Step 7 — 5-fold CV (error-barred baseline) + honest cutoff analysis
- ⭐ **EfficientNet-B3 5-fold CV: TEST AUC 0.920 ± 0.005, acc 0.868 ± 0.013** (sens 0.879,
  spec 0.839; test sealed per fold). Tight error bars confirm the committed single-split
  number, and **the paper reports no CV error bar on its plain B3.**
- ⭐ **Cutoff tuning is nearly inert on this imbalanced set:** test accuracy @0.50 = 0.868,
  threshold tuned on validation = 0.874, threshold tuned on the test set itself (optimistic
  ceiling) = 0.882 — a <1 pt spread, inside the ±0.013 CV noise, with **AUC unchanged**
  throughout. Moving the cutoff only slides along the sensitivity↔specificity see-saw; it
  does not make the model better. → AUC is the honest headline; a "max-accuracy" cutoff
  chosen on test is computable but **not reportable** (peeks at test labels).
- **Committed operating-point policy:** threshold chosen for **sensitivity ≥ 0.90 on
  validation**, applied once to test (clinically defensible — a missed cancer is the worst
  error). *(detail: `research/Baseline_Journey_And_Results.md` PART 8)*

---

## Phase 2 — Attention comparison (Leg A). Result: external attention does NOT help.

### Step 8 — Controlled 4-way attention screen (none / SE / CBAM / CPCA)
- Same B3 backbone + same two-phase recipe; only the external attention module differs
  (added *after* the backbone, since B3 has SE natively). Single official split, same seed —
  so any difference is attributable to attention.

| Arm | val AUC | TEST AUC | acc | sens | spec |
|-----|--------:|---------:|----:|-----:|-----:|
| none | 0.949 | **0.933** | 0.879 | 0.891 | 0.848 |
| se   | 0.939 | 0.932 | 0.887 | 0.914 | 0.814 |
| cbam | 0.925 | 0.918 | 0.864 | 0.870 | 0.848 |
| cpca | 0.950 | 0.910 | 0.878 | 0.903 | 0.810 |

- ⭐ **External attention did not improve over the plain B3 baseline** — baseline best, SE a
  dead tie, CBAM/CPCA slightly worse. Reasons: B3 already has SE in every block; a single
  late module on a 10×10 map gives spatial attention almost nothing to do; and whole-image
  classification global-pools away the spatial detail CBAM/CPCA exploit (their win in Paper 2
  was a *detection* task → **attention is task-specific**, confirmed not contradicted).
- ⭐ **CPCA had the best validation AUC (0.950) but nearly the worst test AUC (0.910)** — a
  clean overfitting signature; reinforces sealing the test set and never trusting val alone.
- Single-split screen; **5-fold CV of the top arm (SE) now DONE: 0.913 ± 0.007 < baseline
  0.920 ± 0.005** → attention confirmed not to help, error-barred. Source:
  `outputs/csv/Attention/phase2_attention_summary.csv` (screen) + `cv_efficientnet_b3_se_summary.csv`.
- Honest framing: a **rigorous negative result that fills the paper's "SE never compared"
  gap** — the fair comparison *is* the contribution, not a guaranteed accuracy gain.

---

## Pillar — Explainability (verified Grad-CAM). The check the paper skipped.

### Step 9 — Quantitative Grad-CAM verification vs the nodule boxes
- Grad-CAM heatmaps are computed **from the committed B3 alone** (its activations +
  gradients); the TN5000 nodule **bounding box is the answer key, applied only afterward**
  to grade localization. 5-fold CV (each fold's model on the sealed 1,000-image test set).
- ⭐ **The model looks at the nodule, ~10× better than chance:** pointing-game hit
  **0.671 ± 0.020** (all) / **0.763 ± 0.019** (malignant), vs the box covering only ~6.6%
  of the image. Energy-in-box 0.35 (≈5× the box area) → heat concentrates on the lesion.
- ⭐ **It does NOT use burned-in artifacts:** corner-energy **0.031 ± 0.010** (~3%) →
  independently **corroborates the caliper audit** (L2) — the model isn't reading scanner
  text/markers. Fulfils the Phase-3 forward-link in `research/Limitations.md` L2.
- ⭐ **The heatmap is faithful:** masking the hot region drops the **predicted class's**
  probability by **0.509 ± 0.012** overall (0.614 malignant, 0.224 benign — both positive).
  The paper showed heatmaps but never tested this.
- ⭐ **Best-method chosen empirically (not assumed):** compared Grad-CAM / Grad-CAM++ /
  HiRes-CAM against the boxes — plain **Grad-CAM localizes best** (malignant pointing 0.76 vs
  Grad-CAM++ 0.49) **and equals HiRes-CAM exactly**, which is *provably faithful* for a
  GAP+linear head — so our Grad-CAM is faithful by construction here. (`--method compare`.)
- ⭐ **Explainability tracks correctness:** correct predictions point at the nodule
  (0.725) and are faithful (0.538); incorrect ones point elsewhere (0.326) and are far less
  faithful (0.323). The metric distinguishes good from bad decisions.
- **Honest caveat:** benign localization is weaker (pointing 0.42, faith-drop 0.22 vs
  malignant 0.76 / 0.61). Interpretable, not a bug — a "benign" call is about the diffuse
  *absence* of malignant features, so its evidence is less nodule-focused. The
  clinically-relevant malignant localization is the strong, headline one.
- Code: `src/gradcam.py` + `src/explainability_eval.py`. Figures:
  `outputs/figures/EfficientNet_B3_Baseline/gradcam_examples.png`, `gradcam_vs_box.png`
  (top = nodule box, bottom = box-free model heatmap). Numbers:
  `outputs/csv/EfficientNet_B3_Baseline/gradcam_verification.csv`.

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
  *(detail: `research/Paper_Analysis_2_4_6.md`)*

---

## Where to look
| Topic | File |
|-------|------|
| Full baseline journey + all result tables | `research/Baseline_Journey_And_Results.md` |
| Duplicate-leakage finding (+ reproduce) | `research/TN5000_Duplicate_Finding.md` |
| Named limitations (ours + theirs) | `research/Limitations.md` |
| Gaps → how we tackle each | (in `Limitations.md` + competitor notes) |
| Paper-log deep dive (2/4/6) | `research/Paper_Analysis_2_4_6.md` |
| Committed B3 CV figures | `outputs/figures/EfficientNet_B3_Baseline/` (cv_confusion_0.50, _sens90, cv_roc) |
| Attention comparison figure | `outputs/figures/Attention/attention_auc_comparison.png` |
| Archived (superseded) runs | `outputs/csv/Archive/`, `outputs/figures/Archive/` |
