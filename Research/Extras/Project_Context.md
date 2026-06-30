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

This project trains a binary benign-vs-malignant classifier on a committed
**EfficientNet-B3** backbone (mentor-confirmed; the closest paper's backbone),
fine-tuned on thyroid ultrasound, and asks three questions *rigorously*: does adding
attention actually help, does the model explain itself faithfully, and does it
generalise to an unseen dataset. The emphasis is **honesty over leaderboard-chasing**
— every headline number is cross-validated with error bars, the test set is sealed and
touched once, the operating point is chosen on validation (never on test), and the
dataset's flaws are documented rather than hidden. Evaluation axes: discriminative
performance (AUC headline + sensitivity/specificity/confusion vs. a plain baseline),
explainability (Grad-CAM + a masking faithfulness check *and* a check that the heatmap
lands on the annotated nodule, not just a picture), and generalisation (train on one
dataset, test on one the model never saw).

## 2. Why this scope — the gap vs. the closest published work
Closest paper: Bahmane et al. (2025), *Medicina* — hybrid EfficientNet-B3 + SE
blocks on TN5000, 89.7% accuracy. No code or weights released.

Four gaps this project targets:

1. **Attention choice was never tested.** The paper used SE blocks but never isolated
   whether attention helps *at all* (SE was bundled with a residual module and a GAN;
   its own ablation credits the **GAN**, not attention). This project ran the controlled
   comparison — baseline vs +SE vs +CBAM vs +CPCA on an identical B3 — and found that
   **external attention does NOT improve over the plain baseline** (a clean negative
   result that fills the gap). See Leg A.
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

## 3. The experiment — two legs + two pillars
- **Leg A — Attention (DONE).** Plain EfficientNet-B3 vs. +SE / +CBAM / +CPCA, identical
  split and recipe (attention added as an external module after the backbone, since B3 has
  SE natively). **Result: external attention does not help** — 5-fold CV: none AUC
  0.920 ± 0.005 vs SE 0.913 ± 0.007; CBAM/CPCA lower on the single-split screen. The fair
  comparison *is* the contribution; the negative result fills the paper's gap.
- **Leg B — Generalisation (LATER).** Train on TN5000, test on the held-out **DDTI** set
  (never seen). Measure the **committed B3's** honest cross-dataset drop, reported as AUC +
  a re-tuned operating point. Name *both* shifts: a different institution AND a different
  label standard (TI-RADS-derived, not biopsy). DDTI stays **quarantined** until this test.
  (The earlier "does CBAM reduce the drop?" framing is retired — attention didn't win.)

Beyond the two legs, two **pillars** carry the remaining contribution:
- **Explainability (verified).** Grad-CAM + a faithfulness check, *and* — using the TN5000
  nodule bounding boxes — a quantitative test that the heatmap lands on the nodule, not on
  calipers/artifacts. The paper only *claimed* explainability; this verifies it.
- **TI-RADS risk bands.** Map the calibrated probability to clinical-language risk tiers
  (a reporting layer on top of the classifier), aligned to the TI-RADS risk ladder.

## 4. Dataset detail
- **TN5000** — 5,000 images, biopsy/FNA-confirmed labels, single institution
  (Cancer Hospital, Chinese Academy of Medical Sciences, Beijing; GE Logiq
  scanners), with per-nodule **bounding boxes** (used for the explainability check).
  We adopt the official 70/10/20 split. **Two findings of ours:** (1) TN5000 has **no
  patient-ID field** (verified) → the official split is **image-level**, not patient-level,
  and we say so rather than claim patient separation; (2) the official split contains **245
  byte-identical duplicate images (119 groups), ~44 straddling train↔test (~4.4% test
  leakage)** — MD5-verified, undocumented in the TN5000 or Medicina papers. Both numbers sit
  on this (leaky) split for apples-to-apples comparison; the leakage is documented, not hidden.
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
  src/               # dataset, model, attention, train, cv_train, evaluate,
                     #   metrics, phase2_attention, make_figures, (gradcam — to build)
  outputs/
    checkpoints/      # saved model weights (gitignored)
    csv/              # tracked; Splits/ + result tables grouped by experiment
                      #   (EfficientNet_B3_Baseline/ Attention/ Bakeoff/ Tonight/
                      #    Resnet_50/ Data_Quality/ Archive/)
    figures/          # tracked; same grouping (confusion matrices, ROC, curves)
    logs/             # gitignored; per-run json + .log files
  research/          # Project_Context.md, Session_Template.md, Sessions/, finding docs
  requirements.txt
  README.md
```

## 6. Success criteria (with status)
| Outcome | Target | Status |
|---------|--------|--------|
| Baseline soundness | Plain ResNet-50 ≈ 85% on TN5000 | ✅ our ResNet-50 85.2% CV reproduces the paper's 85.1% |
| Committed backbone | Tuned B3 matches/beats the paper without a GAN | ✅ acc 0.868 ± 0.013, **AUC 0.920 ± 0.005 > paper's 0.89** |
| Attention comparison *(the contribution)* | Fairly test whether attention helps | ✅ **negative result — external attention does NOT improve over B3** (CV-confirmed) |
| Explainability | Grad-CAM masking lowers the prediction AND the heatmap lands on the nodule box | ⏳ pillar in progress |
| TI-RADS bands | Calibrated probability → clinical risk tiers | ⏳ later |
| Generalisation | Measure the committed B3's honest TN5000→DDTI drop (AUC + re-tuned point) | ⏳ later (DDTI quarantined) |

The headline framing: a properly-tuned plain B3 already reaches the field's *honest* benchmark
(AUC ≈ 0.92, matching a meta-analysis of externally-validated models, sens 88 / spec 83). The
paper's extra accuracy came from a GAN on a leaky split — not attention. The contribution is
rigour: reproducing their ResNet, exposing the leakage, the fair attention comparison, verified
explainability, and TI-RADS bands. (Accuracy-boost levers — TTA/EMA/ensemble/GAN — are deferred
to *after* the pillars are done; see CLAUDE.md open thread #5.)