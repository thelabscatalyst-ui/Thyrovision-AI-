# ThyroVision AI — Project Memory

## What this is
Binary benign/malignant thyroid ultrasound classifier. **Committed backbone:
EfficientNet-B3** (mentor-confirmed) + attention, benchmarked against a plain
EfficientNet-B3 baseline. Research internship, NIT Delhi (mentor: Dr. Gunjan).
Hard deadline: mid-July 2026.

Full background — the problem statement, why this scope, the gaps vs. the closest
published paper — lives in `Research/Extras/Project_Context.md`. Read it in full once now,
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

## Current phase: Phase 2 — Attention comparison (Leg A)
**Phase 1 baseline = DONE & signed off.** Committed backbone: **EfficientNet-B3**
(two-phase fine-tune, 300px). Result: acc 0.871 / sens 0.895 / spec 0.807 / AUC 0.918
(single split; matches the paper's plain B3 on accuracy, beats its AUC, no GAN). CV pending.

**Mentor-confirmed sequencing: (1) attention-mechanism comparison FIRST, (2) detection
module LATER.**
- **Phase 2 (NOW) — attention comparison:** add each mechanism identically to the B3
  backbone and compare: **baseline (none) vs +SE vs +CBAM vs +CPCA**. Same recipe/split;
  this controlled comparison is the core contribution (fixes the paper's "SE never
  compared" gap). **Note:** EfficientNet-B3 has SE *natively*, so each mechanism is added
  as an **external module after the backbone** for a fair like-for-like comparison.
- **Phase 3 (LATER) — detection:** YOLOv8 + **MobileNetV4** backbone (+ attention) to
  localize nodules / cut compute. Scope only after Phase 2 — **do not build YOLO
  machinery yet.**

Plan: `Research/Phase2_Plan.md`. Journey + numbers:
`Research/Baseline_Journey_And_Results.md`, `Research/Finding_Diary.md`.

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
   `Research/Extras/Session_Template.md` and save it under `Research/Sessions/`.

## Dataset roles

| Dataset | Path                    | Role                                    | Label source                |
|---------|-------------------------|-----------------------------------------|-----------------------------|
| TN5000  | `Data/TN5000 Dataset/`  | Train + same-dataset test               | Biopsy/FNA-confirmed        |
| DDTI    | `Data/DDTI Dataset/`    | Held out — Phase 3/4 generalisation ONLY| TI-RADS category, not biopsy|

## Project structure
`Data/` (never in git — large + restricted research data) · `src/` (code) ·
`outputs/checkpoints,figures,logs/` · `Research/` (this context + session history).
Full map in `Research/Extras/Project_Context.md` §5.

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
   `outputs/tn5000_split.csv`.
2. First baseline (ResNet-50, simple recipe) + 5-fold CV → validated the pipeline.
3. Found duplicate-image leakage in TN5000 (data-quality finding).
4. Ran a fair backbone bake-off (ResNet-18/50, EfficientNet-B0/B3, one recipe) →
   EfficientNet was under-tuned by that recipe.
5. Built **two-phase fine-tuning** + re-ran ResNet-50/18 and a 3-config B3 search →
   **committed EfficientNet-B3**.
6. Honest reporting: threshold tuning, AUC-vs-accuracy, class weighting for imbalance;
   **dropped ensembling** (incompatible with the single-backbone "+CBAM" comparison).

## Committed baseline — EfficientNet-B3 (TN5000 official split; single split, CV pending)
Recipe: two-phase fine-tune (freeze head 6 ep @1e-3 → unfreeze @1e-4 cosine, early-stop),
**300px, batch 16, dropout 0.4, label-smoothing 0.1, class-weighted loss**, light aug.

| Operating point | Acc | Sens | Spec | Precision | F1 | AUC |
|-----------------|----:|-----:|-----:|----------:|---:|----:|
| max-accuracy (thr 0.43) | 0.871 | 0.895 | 0.807 | 0.926 | 0.910 | 0.918 |
| balanced / Youden (thr 0.50) | 0.855 | 0.867 | 0.822 | 0.930 | 0.897 | 0.918 |

vs paper's plain EfficientNet-B3 (87.1% / AUC 0.89): **we match accuracy, beat AUC, no GAN.**

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
  without a GAN.**
- ⭐ **EfficientNet is recipe-sensitive** — a ResNet-friendly recipe under-tunes it
  (B0 overfit, specificity collapsed); fixed by the two-phase recipe.
- ⭐ **Medicina paper (Bahmane 2025) inconsistencies:** plain B3 reported at 87.1% AND
  89.7%; training time 42 AND 12 min; split stated 70/20/10 vs 70/10/20; "external
  validation" on an uncited **THYROID-DATASET-2022 (NOT DDTI)**; SE-only attention never
  compared; Grad-CAM reported not faithfulness-checked; biggest lever was a GAN (G-RAN,
  +6.23%).
- ⭐ **Attention is task-specific** (Paper 2: in a YOLOv8 detector, CPCA beat CBAM) →
  attention choice must be *tested* (baseline / +SE / +CBAM), not assumed.

## Key decisions & rationale
- **Backbone = EfficientNet-B3:** mentor-confirmed + paper-aligned (their backbone) +
  edges ResNet-18 on AUC/accuracy/sensitivity at sensible thresholds.
- **Imbalance (71% malignant) → class weighting** (valid, often-better alternative to
  oversampling/GAN). Oversampling/focal-loss are easy to A/B if wanted.
- **AUC is the honest headline** (threshold-free); always report sens/spec too; choose
  the operating threshold by **clinical sensitivity**, not raw accuracy.
- **No G-RAN GAN** (out of scope; reached paper accuracy without it). **No ensembling.**

## Detailed docs
| Topic | File |
|-------|------|
| Step-by-step journey + ⭐ findings | `Research/Finding_Diary.md` |
| Full baseline journey + all tables | `Research/Baseline_Journey_And_Results.md` |
| Duplicate-leakage finding (+ reproduce) | `Research/TN5000_Duplicate_Finding.md` |
| Named limitations (ours + theirs) | `Research/Limitations.md` |
| Paper-log deep dive (papers 2/4/6) | `Research/Paper_Analysis_2_4_6.md` |
| Backbone bake-off | `Research/Backbone_Bakeoff_Findings.md` |
| Committed B3 confusion matrix | `outputs/figures/FINAL_b3_confusion_matrix.png` |
| All tonight runs' numbers / probabilities | `outputs/logs/csv/tonight_summary.csv`, `outputs/tonight_probs.npz` |

## Open threads
1. **Phase 2 (NOW): attention comparison** — B3 vs +SE vs +CBAM vs +CPCA, each added as an
   external module after the backbone, identical recipe/split. The core contribution
   (fixes the paper's "SE never compared" gap). Plan in `Research/Phase2_Plan.md`.
2. **Confirm EfficientNet-B3 with 5-fold CV** (final error-barred baseline number) — can
   run alongside / before the attention arms.
3. **Phase 3 (LATER): detection** — YOLOv8 + MobileNetV4 backbone (+ attention) to localize
   nodules / cut compute. Mentor-confirmed this comes *after* the attention comparison;
   don't build YOLO machinery yet.
4. **Open data question (user-flagged):** the duplicate finding's "adjacent-photo" pattern
   — are the byte-identical duplicates consecutive frames / adjacent image-IDs? Investigate
   before any strong novelty claim. See `Research/TN5000_Duplicate_Finding.md` (§Finding).
