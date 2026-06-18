# ThyroVision AI — Project Memory

## What this is
Binary benign/malignant thyroid ultrasound classifier: ResNet-50 + CBAM attention,
benchmarked against a plain ResNet-50 baseline. Research internship, NIT Delhi
(mentor: Dr. Gunjan). Hard deadline: mid-July 2026.

Full background — the problem statement, why this scope, the gaps vs. the closest
published paper — lives in `Project_Context.md`. Read it in full once now,
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

## Current phase: Phase 1 — Baseline
Goal: a clean, honest fine-tuned ResNet-50 on TN5000, patient-level split, full
metrics reported. Do not start on CBAM or Grad-CAM until this phase is signed off.
(Update this section yourself as we move into Phase 2/3 — don't let it go stale.)

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
   `Research/Session_Template.md` and save it under `Research/Sessions/`.

## Dataset roles

| Dataset | Path                    | Role                                    | Label source                |
|---------|-------------------------|-----------------------------------------|-----------------------------|
| TN5000  | `Data/TN5000 Dataset/`  | Train + same-dataset test               | Biopsy/FNA-confirmed        |
| DDTI    | `Data/DDTI Dataset/`    | Held out — Phase 3/4 generalisation ONLY| TI-RADS category, not biopsy|

## Project structure
`Data/` (never in git — large + restricted research data) · `src/` (code) ·
`outputs/checkpoints,figures,logs/` · `Research/` (this context + session history).
Full map in `Research/Project_Context.md` §5.

## Sanity floor
A plain ResNet-50 on TN5000 should land near 85% accuracy (literature baseline).
Far below that means the pipeline is broken before the idea is being tested.