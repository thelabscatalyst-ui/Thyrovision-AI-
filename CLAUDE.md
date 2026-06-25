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

## Current phase: Phase 2 — Attention (Leg A)
**Phase 1 baseline = DONE & signed off (mentor-confirmed).** Committed backbone:
**EfficientNet-B3** (two-phase fine-tune, 300px). Result: **acc 0.871 / sensitivity
0.895 / specificity 0.807 / AUC 0.918** (single split; matches the paper's plain B3
on accuracy, beats its AUC, no GAN). Pending: confirm with 5-fold CV.

Phase 2 goal: test attention — **EfficientNet-B3 (baseline) vs +SE vs +CBAM**,
identical otherwise. (Mentor also raised a detection track — YOLOv8 + MobileNetV4 +
CBAM — to be scoped/clarified; see Finding_Diary / latest session.)
Full journey + numbers: `Research/Baseline_Journey_And_Results.md`,
`Research/Finding_Diary.md`.

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