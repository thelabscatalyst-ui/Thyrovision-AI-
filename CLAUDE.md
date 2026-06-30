# ThyroVision AI — Project Memory

## What this is
Binary benign/malignant thyroid ultrasound classifier. **Committed backbone:
EfficientNet-B3** (mentor-confirmed) + attention, benchmarked against a plain
EfficientNet-B3 baseline. Research internship, NIT Delhi (mentor: Dr. Gunjan).
Hard deadline: mid-July 2026.

Full background — the problem statement, why this scope, the gaps vs. the closest
published paper — lives in `research/Extras/Project_Context.md`. Read it in full once now,
and again whenever you need to justify a design decision. It does not need to reload
every message.

## Who you're talking to
2nd-year CS student — comfortable with general programming (functions, loops,
basic Python, git) but no prior ML/DL experience. General coding vocabulary is
fine; it's the ML/DL-specific stuff that's new. Explain ML/DL-specific concepts,
libraries, and conventions (training loops, data loaders, hyperparameters,
GPU/CUDA basics, etc.) in plain English as you introduce them, rather than
assuming familiarity. Plan before you build: for any new stage, lay out the plan
and the folder/file changes, and wait for me to confirm before writing code.

## Current phase: Pillars — Attention ✅ + Explainability ✅ done; TI-RADS next
**Phase 1 baseline = DONE & signed off.** Committed backbone: **EfficientNet-B3**
(two-phase fine-tune, 300px). **5-fold CV: TEST AUC 0.920 ± 0.005, acc 0.868 ± 0.013,
sens 0.879, spec 0.839** — matches the paper's plain B3 on accuracy, beats its AUC (0.89),
no GAN. Tight error bars confirm the single-split number (acc 0.855 / AUC 0.918 @0.50).

**Mentor-confirmed sequencing: (1) attention-mechanism comparison FIRST, (2) detection
module LATER.**
- **Phase 2 (NOW) — attention comparison:** add each mechanism identically to the B3
  backbone and compare: **baseline (none) vs +SE vs +CBAM vs +CPCA**. Same recipe/split;
  this controlled comparison is the core contribution (fixes the paper's "SE never
  compared" gap). **Note:** EfficientNet-B3 has SE *natively*, so each mechanism is added
  as an **external module after the backbone** for a fair like-for-like comparison.
  **Screen result (single split, TEST AUC):** none **0.933** ≈ se 0.932 > cbam 0.918 >
  cpca 0.910 → **external attention did NOT improve over the plain B3 baseline** (SE ties,
  CBAM/CPCA slightly worse). A clean negative result that fills the gap. Next: 5-fold CV the
  SE arm to error-bar it vs the baseline CV. Source: `outputs/csv/Attention/phase2_attention_summary.csv`.
- **Phase 3 (LATER) — detection:** YOLOv8 + **MobileNetV4** backbone (+ attention) to
  localize nodules / cut compute. Scope only after Phase 2 — **do not build YOLO
  machinery yet.**

Plan: `research/Phase2_Plan.md`. Journey + numbers:
`research/Baseline_Journey_And_Results.md`, `research/Finding_Diary.md`.

## Non-negotiable rules

1. **`Data/DDTI Dataset/` is quarantined.** Never let it appear in any training script,
   dataset glob, augmentation pipeline, or tuning loop, in any phase before the
   explicit cross-dataset generalisation test (Phase 3/4). If any code path would
   touch it early, stop and flag it to me first — do not silently work around it.
   This is the one rule that, if broken, invalidates the project's actual
   contribution.
2. **Splits are patient-level, not image-level.** If a dataset has no patient ID
   field, say so explicitly instead of silently falling back to image-level.
3. **The test set is touched exactly once** — at final evaluation, after every
   training and tuning decision is locked.
4. **Never report accuracy alone.** Always give sensitivity, specificity, AUC, and
   a confusion matrix together.
5. **Augmentation applies to the training split only.**
6. **At the end of each session**, write a session brief using
   `research/Extras/Session_Template.md` and save it under `research/Sessions/`.

## Dataset roles

| Dataset | Path                    | Role                                    | Label source                |
|---------|-------------------------|-----------------------------------------|-----------------------------|
| TN5000  | `Data/TN5000 Dataset/`  | Train + same-dataset test               | Biopsy/FNA-confirmed        |
| DDTI    | `Data/DDTI Dataset/`    | Held out — Phase 3/4 generalisation ONLY| TI-RADS category, not biopsy|

## Project structure
`Data/` (never in git — large + restricted research data) · `src/` (code) ·
`outputs/checkpoints,figures,logs/` · `research/` (this context + session history).
Full map in `research/Extras/Project_Context.md` §5.

## Sanity floor
A plain ResNet-50 on TN5000 should land near 85% accuracy (literature baseline).
Far below that means the pipeline is broken before the idea is being tested.

## Compute / infra
Train on the Mac (Apple M5, MPS — no CUDA). **Long jobs must run in the user's own
Terminal under `caffeinate -dimsu` with the lid open** — Claude-launched background
tasks get reaped, and lid-close defeats caffeinate. Keep jobs **resumable**
(per-unit result JSON). EfficientNet-B3 @300 uses **batch 16** (batch 32 OOMs the
16 GB unified memory). Pretrained weights are cached, so training needs no internet.
Python 3.11 venv at `.venv/`. See memory: long-jobs-laptop-sleep.

---

# REFERENCE: Journey & Findings (Phase 1)

## What we did (step by step)
1. Built the TN5000 pipeline: load → clean burned-in artifacts (scanner text top-left,
   calipers, bottom-right marker) → official 70/10/20 split frozen to
   `outputs/csv/Splits/tn5000_split.csv`.
2. First baseline (ResNet-50, simple recipe) + 5-fold CV → validated the pipeline.
3. Found duplicate-image leakage in TN5000 (data-quality finding).
4. Ran a fair backbone bake-off (ResNet-18/50, EfficientNet-B0/B3, one recipe) →
   EfficientNet was under-tuned by that recipe.
5. Built **two-phase fine-tuning** + re-ran ResNet-50/18 and a 3-config B3 search →
   **committed EfficientNet-B3**.
6. Honest reporting: threshold tuning, AUC-vs-accuracy, class weighting for imbalance;
   **dropped ensembling** (incompatible with the single-backbone "+CBAM" comparison).

## Committed baseline — EfficientNet-B3 (TN5000 official split; 5-fold CV)
Recipe: two-phase fine-tune (freeze head 6 ep @1e-3 → unfreeze @1e-4 cosine, early-stop),
**300px, batch 16, dropout 0.4, label-smoothing 0.1, class-weighted loss** (weights
benign 1.70 / malignant 0.71), light aug. Run: `src.cv_train --attention none`.

**5-fold CV — error-barred final number (test sealed per fold, mean ± SD):**
| Stage | Acc | Sens | Spec | AUC |
|-------|----:|-----:|-----:|----:|
| **TEST** | 0.868 ± 0.013 | 0.879 ± 0.022 | 0.839 ± 0.020 | **0.920 ± 0.005** |
| VAL | 0.859 ± 0.008 | 0.885 ± 0.015 | 0.795 ± 0.031 | 0.917 ± 0.007 |

**Operating-point policy (committed):** AUC is the headline (threshold-free). For
accuracy/sens/spec pick the threshold by **sensitivity ≥ 0.90 on validation**, apply once
to test. CV cutoff analysis: acc @0.50 = 0.868, val-tuned = 0.874, test-tuned ceiling =
0.882 → tuning buys <1 pt (within the ±0.013 noise) and never moves AUC. A "max-accuracy"
cutoff chosen on test is **computable but NOT reportable** (it peeks at the test labels).

vs paper's plain EfficientNet-B3 (87.1% / AUC 0.89): **we match accuracy, beat AUC, no GAN.**
The paper's 89.7% headline is its full GAN-augmented hybrid — a different model; Phase-2
attention is our honest lever toward it.

## Key findings (⭐ = citable)
- ⭐ **TN5000 has no patient-ID field** → its official split is image-level (can't prove
  patient separation). Documented as a limitation.
- ⭐ **TN5000's official split contains 245 byte-identical duplicate images (119 groups);
  ~44 duplicate pairs cross train↔test (~4.4% test leakage).** MD5-verified (no model,
  no threshold). Not mentioned in the TN5000 or Medicina papers; one MDPI reproducibility
  paper (Electronics 15/1/151) mentions investigating TN5000 leakage — read it in full
  before any novelty claim. Reproduce: `.venv/bin/python -m src.verify_duplicates`.
- ⭐ **Our plain ResNet-50 (85.2% CV) reproduces the paper's ResNet-50 (85.1%)** →
  independent pipeline validation.
- ⭐ **Tuned EfficientNet-B3 matches the paper's plain B3 on accuracy and beats its AUC,
  without a GAN.** Confirmed by **5-fold CV: TEST AUC 0.920 ± 0.005** (tight error bar) —
  the paper reports no CV error bars on its B3.
- ⭐ **Cutoff-tuning is nearly inert on this imbalanced set:** moving the threshold changed
  accuracy by <1 pt (0.868→0.874 honest, 0.882 even when peeking at test) and never moved
  AUC → confirms AUC is the honest headline and "max-accuracy" numbers are threshold
  artifacts, not better models.
- ⭐ **EfficientNet is recipe-sensitive** — a ResNet-friendly recipe under-tunes it
  (B0 overfit, specificity collapsed); fixed by the two-phase recipe.
- ⭐ **Medicina paper (Bahmane 2025) inconsistencies:** plain B3 reported at 87.1% AND
  89.7%; training time 42 AND 12 min; split stated 70/20/10 vs 70/10/20; "external
  validation" on an uncited **THYROID-DATASET-2022 (NOT DDTI)**; SE-only attention never
  compared; Grad-CAM reported not faithfulness-checked; biggest lever was a GAN (G-RAN,
  +6.23%).
- ⭐ **Attention is task-specific** (Paper 2: in a YOLOv8 detector, CPCA beat CBAM) →
  attention choice must be *tested* (baseline / +SE / +CBAM), not assumed.
- ⭐ **External attention does NOT improve our B3 classifier** (fair single-split screen,
  TEST AUC): none 0.933 / SE 0.932 / CBAM 0.918 / CPCA 0.910 — baseline best, SE ties,
  CBAM/CPCA worse. Consistent with B3's native SE + spatial attention being wasted after
  global pooling on a 10×10 map (CBAM/CPCA's win in Paper 2 was a *detection* task). A clean
  negative result filling the paper's "SE never compared" gap. **CV-confirmed: SE 0.913 ±
  0.007 < baseline 0.920 ± 0.005** — attention does not help, error-barred.
- ⭐ **Verified Grad-CAM (the check the paper skipped):** the committed B3 looks at the nodule
  — pointing-game **0.67 ± 0.02** (0.76 malignant), ~10× the box's 6.6% area; **ignores
  burned-in artifacts** (corner-energy 3.1% ± 1.0%, independently corroborating the caliper
  audit); is **faithful** (masking the hot region drops P(mal) 0.57 ± 0.03 on malignant); and
  the metrics **track correctness** (correct 0.73 pointing / 0.46 drop vs incorrect 0.33 / 0.07).
  Heatmaps computed from the model alone; box used only as the answer key. Benign localization
  weaker (interpretable). Code: `src.explainability_eval`.

## Key decisions & rationale
- **Backbone = EfficientNet-B3:** mentor-confirmed + paper-aligned (their backbone) +
  edges ResNet-18 on AUC/accuracy/sensitivity at sensible thresholds.
- **Imbalance (71% malignant) → class weighting** (committed; train weights benign 1.70 /
  malignant 0.71 — a benign mistake costs 2.39× more). Sets the model's class *bias*, not
  generalization. Oversampling/focal-loss noted as available A/Bs, deliberately **not run**
  so attention stays the single Phase-2 variable.
- **AUC is the honest headline** (threshold-free); always report sens/spec + confusion.
  **Operating point = threshold for sensitivity ≥ 0.90 on validation, applied once to
  test** — never tune the threshold on test to inflate a reported number. See memory
  [[honest-metric-reporting]].
- **No G-RAN GAN** (out of scope; reached paper accuracy without it). **No ensembling.**

## Detailed docs
| Topic | File |
|-------|------|
| Step-by-step journey + ⭐ findings | `research/Finding_Diary.md` |
| Full baseline journey + all tables | `research/Baseline_Journey_And_Results.md` |
| Duplicate-leakage finding (+ reproduce) | `research/TN5000_Duplicate_Finding.md` |
| Named limitations (ours + theirs) | `research/Limitations.md` |
| Paper-log deep dive (papers 2/4/6) | `research/Paper_Analysis_2_4_6.md` |
| Backbone bake-off | `research/Backbone_Bakeoff_Findings.md` |
| Committed B3 CV figures | `outputs/figures/EfficientNet_B3_Baseline/` (cv_confusion_0.50, cv_confusion_sens90, cv_roc) |
| Attention comparison figure | `outputs/figures/Attention/attention_auc_comparison.png` |
| Grad-CAM verification (numbers + figures) | `outputs/csv/EfficientNet_B3_Baseline/gradcam_verification.csv`; `outputs/figures/EfficientNet_B3_Baseline/gradcam_{examples,vs_box}.png` |
| figures layout | grouped like csv/: `Data_Quality/ EfficientNet_B3_Baseline/ Attention/ Bakeoff/ Resnet_50/ Archive/` |
| All tonight runs' numbers / probabilities | `outputs/csv/Tonight/` (tonight_summary.csv, tonight_probs.npz) |
| Baseline 5-fold CV results | `outputs/csv/EfficientNet_B3_Baseline/cv_efficientnet_b3_none_{results,summary}.csv` |
| SE attention 5-fold CV results | `outputs/csv/Attention/cv_efficientnet_b3_se_{results,summary}.csv` |
| csv layout | grouped: `Splits/ Data_Quality/ EfficientNet_B3_Baseline/ Attention/ Bakeoff/ Tonight/ Resnet_50/ Archive/` |

## Open threads
1. ✅ **DONE — Phase 2 attention comparison (Leg A):** screen none 0.933 ≈ se 0.932 > cbam
   0.918 > cpca 0.910; **5-fold CV confirms SE 0.913 ± 0.007 < baseline 0.920 ± 0.005** →
   external attention does not help (error-barred). Results: `outputs/csv/Attention/`.
   ✅ **DONE — Explainability pillar (verified Grad-CAM):** model looks at the nodule (~10×
   chance), ignores artifacts (corner 3%), faithful (mask-drop 0.57 malignant). `src.explainability_eval`.
2. ✅ **DONE — EfficientNet-B3 5-fold CV:** TEST AUC 0.920 ± 0.005, acc 0.868 ± 0.013
   (`src.cv_train --attention none`). Error-barred baseline locked. Next: same CV on the
   winning attention finalist(s) after the single-split screen.
3. **Phase 3 (LATER): detection** — YOLOv8 + MobileNetV4 backbone (+ attention) to localize
   nodules / cut compute. Mentor-confirmed this comes *after* the attention comparison;
   don't build YOLO machinery yet.
4. **Open data question (user-flagged):** the duplicate finding's "adjacent-photo" pattern
   — are the byte-identical duplicates consecutive frames / adjacent image-IDs? Investigate
   before any strong novelty claim. See `research/TN5000_Duplicate_Finding.md` (§Finding).
5. **Future work — accuracy-boost roadmap (POST-completion; user deferred until pillars done):**
   to raise acc+AUC *together* (real model gains, not threshold tricks): Tier 1 = TTA / EMA /
   MixUp (cheap, honest); then ensemble; then a **GAN** (G-RAN-style synthetic benigns) as the
   imbalance lever. Hard rules: synthetic data **TRAIN-ONLY** (never val/test); validate fakes
   (FID + radiologist review + real-data ablation). Context: we already match the externally-
   validated field benchmark (meta-analysis AUC ≈0.92), so this is for a deployable-number push,
   not the research contribution.
