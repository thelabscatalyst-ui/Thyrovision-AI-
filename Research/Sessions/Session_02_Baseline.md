# ThyroVision — Session Brief

**Session #:** 02  **Date:** 2026-06-19/20  **Week of plan:** Week 2–4 (Phase 1)
**Stage:** Baseline

## 1. Goal for this session
Build the full Phase-1 pipeline and produce an honest, fine-tuned ResNet-50
benign/malignant baseline on TN5000 — data layer → artifact cleaning → patient-
level-aware split → training → sealed-test evaluation → 5-fold CV — with full
metrics, not accuracy alone.

## 2. What actually happened
- **Data layer** (`src/dataset.py`, `preprocess.py`, `utils.py`): parse VOC labels,
  freeze the official split to `outputs/tn5000_split.csv`, clean artifacts, build
  the PyTorch dataset (whole image in; bounding boxes unused, reserved for Phase 3).
- **Split decision:** TN5000 has **no recoverable patient ID** (verified), so used
  the official image-level split (train 3500 / val 500 / test 1000) and documented
  the limitation — see L1 in `research/Limitations.md`.
- **Artifact cleaning:** corner text/marker masking (removes scanner-ID leakage) +
  conservative pure-white caliper inpaint. Dim grey calipers left in (can't be
  removed without erasing echogenic foci).
- **Binding caliper-leakage audit** (you required it before trusting the result):
  benign 50.8% vs malignant 43.3% carry a detected caliper (7.5pp gap), but the
  feature predicts the label at **AUC 0.462 ≈ chance** (negative lift). Per-split
  permutation noise-band check: train significant-but-negligible, **test clean**.
  Decision (governed by effect size, not p-value): proceed; document as inert
  confound; keep Phase-3 Grad-CAM caliper-attention check binding. Full chain in
  `research/Limitations.md` L2 + `outputs/caliper_audit*.csv`.
- **Training** (`src/model.py`, `metrics.py`, `train.py`): ImageNet-pretrained
  ResNet-50, class-weighted CE (71/29 imbalance), AdamW lr 1e-4 cosine, batch 32,
  early stop on val AUC, MPS.
- **Single-split run:** 33 min on the Apple GPU (MPS), best epoch 18.
- **Sealed-test eval** (`src/evaluate.py`): test touched once; full metrics +
  confusion matrix + ROC.
- **5-fold CV** (`src/cv_train.py`): escalated because the single run (33 min) was
  under the 45–60 min line. Made resumable (per-fold result files) + run under
  `caffeinate` after two laptop-sleep interruptions. Completed all 5 folds.

## 3. Results / numbers
**Headline — 5-fold CV on the SEALED test set (mean ± SD):**
Accuracy: 0.852 ± 0.014  Sensitivity: 0.865 ± 0.020  Specificity: 0.816 ± 0.017
AUC: 0.918 ± 0.009  (Precision 0.928 ± 0.006, F1 0.895 ± 0.011)

Cross-validated (out-of-fold validation) AUC: 0.907 ± 0.007, acc 0.843 ± 0.010.

Single-split test (corroborating, best-epoch-18 model): acc 0.853, AUC 0.922,
sens 0.873, spec 0.799 (TN=215 FP=54 FN=93 TP=638).

Dataset = TN5000; split = official image-level; test = the held-out 1,000 images
touched only at final evaluation.

## 4. Sanity checks confirmed
- [~] Split was patient-level — **No: TN5000 has no patient ID; used official
      image-level split, documented explicitly (L1).**
- [x] Test set stayed sealed until final evaluation (read only in evaluate.py / CV
      end, after all training+tuning locked)
- [x] `Data/DDTI Dataset/` was NOT used in training (quarantined; Dataset class
      refuses any DDTI path in code)
- [x] Training took realistic time (single run 33 min; CV 116 min on MPS)
- [x] Reported more than accuracy (sens/spec/precision/F1/AUC/confusion together)
- [x] Test accuracy not ≥97% (max fold 0.863) — no leakage red flag
- [x] Test accuracy in 80–87% sanity band (0.852 ± 0.014); near ~85% literature floor

## 5. What went wrong / surprised us
- **Two laptop-sleep interruptions** killed the CV mid-run. Diagnosed as
  machine-sleep (lid-close defeats `caffeinate -i`), not a code/OOM/time-limit
  issue. Fixed by making CV resumable (each finished fold is cached) and keeping
  the lid open; finished cleanly on the third attempt.
- Python 3.14 (system) too new for DL wheels → used a 3.11 venv (Session 01).
- Caliper auto-removal is unreliable (dim calipers overlap echogenic-foci
  intensity) — handled via the audit + documentation rather than risky inpainting.

## 6. Next session
Phase-1 sign-off, then Phase 2: add **CBAM** to the same ResNet-50 backbone, keep
everything else identical, and run it through the SAME `cv_train.py` for a fair
baseline-vs-attention comparison (Leg A).

## 7. One line for the mentor
"Honest ResNet-50 baseline on TN5000 is done: 5-fold CV test AUC 0.918 ± 0.009,
accuracy 0.852 ± 0.014 (sensitivity 0.865, specificity 0.816) — right at the
literature floor, with patient-ID and caliper limitations documented and a clean
sealed-test protocol — ready to add CBAM."
