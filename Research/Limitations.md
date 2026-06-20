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

**Forward link (Phase 3 — a SECOND, independent check):** the Grad-CAM review
must explicitly check whether high-confidence or misclassified cases show
attention landing on caliper/corner regions rather than the nodule. This covers a
different failure mode than the audit: the **audit** shows the *data* offers no
caliper shortcut; the **Phase-3 check** confirms the *model* didn't construct one
anyway. See [[caliper-audit]].

## L3. TN5000 is malignant-skewed vs real-world prevalence
TN5000 is ~71% malignant (3,574 / 1,426), the opposite of the ~5–15% malignant
real-world prevalence. So "accuracy" here measures discrimination on an *enriched*
cohort, not population screening yield. This is a property of the dataset, shared
with the closest published work (Bahmane et al. 2025).

## L4. Binary output not connected to TI-RADS
This phase produces a benign/malignant probability only; it is not mapped to
TI-RADS categories or calibrated risk bands. Calibration is a planned Phase-4
(stretch) extension.
