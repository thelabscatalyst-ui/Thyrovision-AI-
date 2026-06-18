# ThyroVision — Full Project Context

Read this in full at the start of the project, and dip back in whenever you need to
justify a decision (in a write-up, to the mentor, or to yourself). It is not
auto-loaded every session — CLAUDE.md points here on purpose, so routine sessions
stay lean.

## 1. Problem statement
Thyroid nodules are found in a majority of adults who undergo ultrasound, yet only
5–15% are malignant. The real clinical problem is therefore not detection —
ultrasound already finds nodules reliably — but triage: separating the small
minority that need biopsy from the large majority that don't, without putting
patients through unnecessary fine-needle aspiration (FNA).

This project trains a binary benign-vs-malignant classifier on a ResNet-50
backbone, fine-tuned on thyroid ultrasound data, with CBAM (channel + spatial
attention) added to direct the model toward the nodule itself rather than
surrounding tissue. It is evaluated on three axes: discriminative performance
(sensitivity/specificity/AUC vs. a plain baseline), explainability (Grad-CAM plus a
masking faithfulness check, not just a heatmap on its own), and generalisation
(train on one dataset, test on one the model never saw).

## 2. Why this scope — the gap vs. the closest published work
Closest paper: Bahmane et al. (2025), *Medicina* — hybrid EfficientNet-B3 + SE
blocks on TN5000, 89.7% accuracy. No code or weights released.

Four gaps this project targets:

1. **Attention choice was never tested.** Only SE blocks were tried, with no
   comparison against CBAM's added spatial attention. This project runs that
   comparison directly (Leg A, below).
2. **Generalisation claim isn't verifiable.** Their external validation set,
   "THYROID-DATASET-2022," has no citation anywhere in the paper. This project's
   TN5000→DDTI test (Leg B) is a verifiable version of what they only gestured at.
3. **Grad-CAM was reported, not verified.** They give a localization-accuracy score
   and a clinician-approval percentage, but never check whether the heatmap is
   faithful to the model's actual reasoning. This project adds the masking check.
4. **Their own numbers are internally inconsistent.** E.g. plain EfficientNet-B3 is
   reported at two different accuracies in two different tables, and total
   training time is given as both 42 and 12 minutes for the same model. Treat their
   89.7% / 90.0% / AUC>0.90 as a directional target, not a verified one — there is
   no released code to reproduce it against.

Two honest *shared* limitations — not competitor-only flaws:
- TN5000 is malignant-skewed (3,572 malignant / 1,428 benign) relative to
  real-world prevalence, so "accuracy" here measures discrimination on an enriched
  cohort, not population screening yield.
- Neither their work nor this round of work connects output to TI-RADS categories.

## 3. The two-leg experiment
- **Leg A — Attention.** Plain ResNet-50 vs. ResNet-50+CBAM, identical split and
  settings. Does attention actually help?
- **Leg B — Generalisation.** Train on TN5000, test on DDTI (never seen in
  training). Does the model — and does CBAM specifically — hold up on a different
  institution and a different label standard?

Success is not "the DDTI score is high." It's whether the accuracy **drop** from
TN5000→DDTI is smaller for the CBAM model than for the plain baseline.

## 4. Dataset detail
- **TN5000** — 5,000 images, biopsy/FNA-confirmed labels, single institution
  (Cancer Hospital, Chinese Academy of Medical Sciences, Beijing; GE Logiq
  scanners). Official release split is 70/10/20 at image level; split at patient
  level for this project if patient IDs are available in the metadata, and say so
  explicitly if they aren't.
- **DDTI** — ~480 images / ~390 patients. Labels derived from TI-RADS category
  (1–3 = benign, 4–5 = malignant), not biopsy. Held out entirely. Going
  TN5000→DDTI tests an image-domain shift AND a label-definition shift at once —
  name both in any write-up; don't present DDTI as a clean apples-to-apples
  external set.
- **TN3K** — optional, volunteer-annotated (not biopsy-confirmed) — lower
  confidence labels, use cautiously if at all, and flag it if used.

## 5. Folder structure (target)
```
thyroid-ultrasound/
  Data/              # never committed to git — large + restricted research data
    TN5000 Dataset/
    DDTI Dataset/
  src/               # dataset.py, model.py, train.py, evaluate.py, gradcam.py
  outputs/
    checkpoints/      # saved model weights
    figures/           # confusion matrices, Grad-CAM heatmaps, curves
    logs/
  Research/          # PROJECT_CONTEXT.md, SESSION_BRIEF_TEMPLATE.md, sessions/
  requirements.txt
  README.md
```

## 6. Success criteria
| Outcome                | Target                                                        |
|-------------------------|----------------------------------------------------------------|
| Baseline soundness      | Plain ResNet-50 ≈ 85% accuracy on TN5000                      |
| Attention contribution  | CBAM improves sensitivity/AUC over baseline, same split        |
| Generalisation          | CBAM's TN5000→DDTI drop is smaller than the baseline's drop    |
| Explainability          | Masking the Grad-CAM region measurably lowers the prediction   |