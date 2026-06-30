# ThyroVision — Named Limitations (living document)

Honest, named limitations of the data and method. Update as the project moves
through phases. Stating these upfront is rigour, not weakness — reviewers attack
the ones you *don't* name.

---

## L1. No patient ID in TN5000 → image-level split
TN5000's annotations are pure Pascal-VOC with **no patient identifier** in any
field, filename pattern, or other recoverable form (verified: constant annotator
name, contiguous filenames `000001`–`005000`). We therefore adopt TN5000's
**official split as published**, which is **image-level**. We cannot assert that
no patient's images straddle train/val/test, so we do **not** claim patient-level
separation. Mitigating (not proof): the TN5000 paper states curators kept ~one
image per patient-per-view, so image-level ≈ patient-level here. **How we report
it:** state plainly that the split is image-level and patient overlap is
unverifiable.

## L2. Residual calipers + a small (inert) caliper–label association
Calipers ('+' measurement marks) in TN5000 are dim grey (~200–230), overlapping
the intensity of clinically meaningful echogenic foci, so they cannot be safely
inpainted without risking erasure of real tissue. We remove burned-in **corner
text/markers** (the systematic, scanner-correlated leakage path) and inpaint only
**pure-white** overlay; dim calipers remain. We then audited whether their mere
*presence* leaks the label:

**The full chain (binding audit):**
- **Rate gap:** benign **50.8%** vs malignant **43.3%** carry a detected caliper
  (gap 7.5pp; benign has *more*, counter-intuitively).
- **Predictive value:** AUC(caliper-presence → malignant) = **0.462** (≈ chance);
  a caliper-only rule scores *below* the majority baseline (negative lift). The
  feature carries essentially **no usable label information**.
- **Per-split noise-band check (permutation, n-matched):** train AUC 0.452 falls
  **outside** its null band (p<0.001) — statistically significant but driven by
  n=3,500 and a negligible effect size; val (0.491) and test (0.488) are **within**
  the band. The decision is governed by **effect size, not p-value**.
- **Test split is clean:** the split the final number rests on shows no
  significant caliper imbalance.

**Decision:** proceed with training; treat this as a documented, inert confound.
Re-splitting (forfeits official-benchmark comparability) and caliper masking
(risks erasing echogenic foci) were both rejected as worse than the chance-level
risk they'd address. Artifacts: `outputs/caliper_audit.csv`,
`outputs/caliper_audit_summary.csv`, `outputs/figures/stageA_caliper_detection.png`.

**Forward link — ✅ NOW CONFIRMED (Grad-CAM verification):** the explainability pillar
quantified how much heatmap energy lands in the burned-in-artifact corners: **3.1% ± 1.0%
(5-fold CV)** — the model essentially ignores the scanner text/markers and looks at the
nodule (pointing-game 0.76 on malignant, ~10× the box's area). This **independently
corroborates the audit by a different method**: the **audit** showed the *data* offers no
caliper/corner shortcut; the **Grad-CAM check** confirms the *model* didn't construct one
anyway — both point the same way. Code: `src/explainability_eval.py`; see Finding_Diary
Step 9 and [[caliper-audit]].

## L3. TN5000 is malignant-skewed vs real-world prevalence
TN5000 is ~71% malignant (3,574 / 1,426), the opposite of the ~5–15% malignant
real-world prevalence. So "accuracy" here measures discrimination on an *enriched*
cohort, not population screening yield. This is a property of the dataset, shared
with the closest published work (Bahmane et al. 2025).

**How we handle it:** class-weighted loss (train weights benign 1.70 / malignant 0.71 — a
benign mistake costs 2.39× more), not oversampling/GAN. This corrects the *learning* bias so
the minority (benign) isn't ignored. It does **not** make accuracy a meaningful headline (a
trivial "always malignant" model scores ~71%), which is exactly why **AUC is the headline**
and the operating point is chosen by clinical sensitivity (≥0.90 on val), not by accuracy.
Oversampling / focal loss are available A/B alternatives, deliberately not run (keeps
attention the single Phase-2 variable). See [[honest-metric-reporting]].

## L4. Binary output → relative risk tiers (first build), not true ACR TI-RADS
A clinical-language layer now sits on top of the classifier (`src/tirads.py`): calibration
compares **temperature / Platt / isotonic** (fit on validation, scored on test); **Platt wins
— ECE 0.127 → 0.044, now well-calibrated** (<0.05). The model was *under-confident* (the
reliability curve bowed above the diagonal — partly from label smoothing); Platt's shift+scale
corrects it where a single temperature couldn't. The calibrated score maps to **relative
suspicion tiers** (Low / Intermediate / High / Very high) with recommendations + the Grad-CAM
heatmap (report cards). **Named limits:** (1)
calibrated to TN5000's **enriched ~71%-malignant prior**, so tiers are *relative*, not
absolute % risk (absolute needs a prevalence adjustment); (2) it is **risk-tier alignment,
NOT ACR TI-RADS feature scoring** (we don't compute composition/echogenicity/shape/margin/foci);
(3) validating tiers against radiologist TI-RADS needs DDTI (quarantined); and absolute %
risk would need a prevalence/prior adjustment. Those remain future work — the calibration
itself is now good (Platt, ECE 0.044).

## L5. Class weighting sets bias, not generalization — the operating point won't transfer
Class weighting fixes *which class* the model leans toward on TN5000; it is **not** a
generalization mechanism. Consequences for the planned cross-dataset (TN5000→DDTI) test:
- **Class ratio differs across datasets**, so a threshold tuned on TN5000 validation will be
  wrong on DDTI — the operating point must be **re-chosen on the new dataset**. AUC
  (threshold-free) survives this shift; accuracy does not.
- **Label semantics differ** — DDTI labels are TI-RADS categories, not biopsy-confirmed like
  TN5000 — which caps achievable agreement regardless of model quality.
- **Expect a performance drop** on the unseen dataset (domain shift: scanner, population).
  Robustness comes from *features* (backbone / augmentation / attention), not the class
  weights. The honest deliverable is to **measure** the drop (AUC + a re-tuned operating
  point), not to assume strong transfer. DDTI stays quarantined until that Phase-3/4 test.
