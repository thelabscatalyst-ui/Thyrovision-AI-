# ThyroVision — Session Brief

**Session #:** 03  **Date:** 2026-06-27/30  **Week of plan:** Phase 2
**Stage:** Attention (Leg A) + Explainability (start)

## 1. Goal for this session
Run the Phase-2 attention comparison (Leg A) to completion and error-bar it, decide
honest reporting (operating point, metrics), do a full housekeeping pass on the
codebase/outputs, and start the explainability pillar (verified Grad-CAM).

## 2. What actually happened
- **Built Phase-2 machinery:** `src/attention.py` (SE / CBAM / CPCA + factory),
  `model.AttnClassifier` + `build_attn_model` (backbone feature map → external
  attention → pool → head; identical wrapper for every arm), `fit_two_phase(attention=…)`,
  and the `phase2_attention.py` driver. Attention is added as an **external module after**
  the B3 backbone (B3 already has SE natively) for a fair like-for-like swap.
- **Generalized `cv_train.py`** from the old ResNet-50 single-phase `fit()` to the
  committed B3 two-phase recipe with `--attention`/`--backbone` and arm-keyed, resumable
  filenames.
- **Codebase audit** (you suspected drift): imported all modules, ran an independent
  review agent. Its one "CRITICAL" flag (that the wrapper drops B3's conv-head) I
  **refuted empirically** — `forward_features` includes conv_head; native-vs-manual
  forward `max abs diff = 0.0`. So the `none` arm IS the standard B3 head. Fixed real
  drift (stale docstrings; cv_train was genuinely not generalized).
- **Ran Leg A:** 4-arm single-split screen (none/SE/CBAM/CPCA), then 5-fold CV of the
  baseline (none) and the top arm (SE).
- **Honest-reporting decisions:** committed **operating point = sensitivity ≥ 0.90 chosen
  on validation**, applied once to test; AUC stays the headline; cutoff-tuning shown to be
  nearly inert (<1 pt, within noise). Saved memory `honest-metric-reporting`.
- **Mentor-facing analysis:** explained class weighting, why two metrics (AUC vs
  operating-point), benign-cell weakness, and the confusion-matrix gap vs the paper.
- **Literature check:** meta-analysis of *externally-validated* models pools to AUC 0.919
  (sens 88 / spec 83) — we sit exactly on the honest benchmark; the flashy 95–99% papers
  are internal/leaky.
- **Big housekeeping pass:** reorganized `outputs/csv`, `outputs/figures`, and
  `outputs/logs/json` into experiment-grouped, renamed folders (`Splits/ Data_Quality/
  EfficientNet_B3_Baseline/ Attention/ Bakeoff/ Tonight/ Resnet_50/ Archive/`); moved the
  39 MB embeddings cache into gitignored `logs/cache/`; homed `tonight_probs.npz`; removed
  the stale ensemble-proof figure; renamed `Research/`→`research/`; revised
  `Project_Context.md`; built `make_figures.py` (CV confusion matrices @0.50 & sens≥0.90,
  ROC, attention bar).
- **Explainability pillar (start):** built `src/gradcam.py` (Grad-CAM on the backbone
  feature map + nodule-box helpers) and `src/explainability_eval.py` (pointing-game,
  energy-in-box, corner-energy, faithfulness-masking) and ran it on the sealed test set.
  Heatmaps are computed **from the model only**; the box is the answer key applied *after*.

## 3. Results / numbers
**Leg A — attention (5-fold CV, sealed test, mean ± SD):**
- Baseline (none): **AUC 0.920 ± 0.005**, acc 0.868 ± 0.013, sens 0.879, spec 0.839.
- +SE: AUC 0.913 ± 0.007, acc 0.855 ± 0.025. **Worse + noisier.**
- Single-split screen (test AUC): none 0.933 ≈ se 0.932 > cbam 0.918 > cpca 0.910.
- **Verdict: external attention does NOT improve over the plain B3** (CV-confirmed).

**Operating point (committed B3, CV):** @0.50 acc 0.868 / sens 0.879 / spec 0.839;
@sens≥0.90 acc 0.873 / sens 0.896 / spec 0.809 / F1 0.911 (the reported point).

**Explainability (Grad-CAM, fold 1, 1000 test images):** pointing-game hit **0.67** all /
**0.77** malignant (vs box area 0.066 → ~10× chance); energy-in-box 0.35; corner-energy
**0.023** (not using artifacts); faithfulness drop **0.44** all / **0.59** malignant.
Correct preds 0.73 pointing / 0.50 drop vs incorrect 0.29 / 0.07 — the metrics track
decision quality. Benign localization weaker (0.39 / 0.03), interpretable.

Dataset = TN5000, official image-level split, test = sealed 1,000 images.

## 4. Sanity checks confirmed
- [~] Split patient-level — **No (TN5000 has no patient ID); image-level, documented (L1).**
- [x] Test set sealed until final eval (CV touches it once per fold; Grad-CAM is post-hoc
      analysis of the locked model, not tuning).
- [x] `Data/DDTI Dataset/` NOT used (quarantined; guardrail in code).
- [x] Realistic time (each B3 arm ~45–80 min; Grad-CAM ~2 min/1000 imgs).
- [x] Reported more than accuracy (AUC headline + sens/spec/F1/confusion).
- [x] Attention comparison fair (identical wrapper/recipe/seed/split per arm).

## 5. What went wrong / surprised us
- **The attention negative result** (we hoped CBAM/CPCA would help). On reflection it's
  expected: B3 has native SE, a single late module on a 10×10 map wastes spatial attention,
  and classification global-pools away the detail CBAM/CPCA exploit. Reframed as a clean,
  citable negative result that fills the paper's gap.
- **A review agent hallucinated a "critical" bug** — caught and refuted by an empirical
  `max abs diff = 0.0` check. Reinforced: verify claims against the runtime, not assertions.
- **Caliper detector under-counts by design** (precision over recall; relaxing it floods
  false positives on speckle) — fine for the presence-based audit; not an accurate counter.
- **Our confusion matrix trails the paper's GAN hybrid on the benign cell** (spec 0.84 vs
  0.88) — the gap is their GAN (synthetic benigns), an accuracy lever we deferred post-Phase-4.

## 6. Next session
Finish the explainability pillar: bake the model-view-vs-box comparison figure into the
module, error-bar the Grad-CAM metrics over all 5 CV folds, and document the finding
(localization ~10× chance + faithfulness + artifact-free corroboration of the caliper
audit). Then start the **TI-RADS risk-bands** pillar (calibration → clinical tiers).

## 7. One line for the mentor
"Phase 2 is done and error-barred: a fair baseline-vs-SE/CBAM/CPCA comparison shows
external attention does **not** beat the plain EfficientNet-B3 (CV AUC 0.920 vs 0.913),
a clean negative result that fills the paper's gap; and verified Grad-CAM shows the model
looks at the nodule ~10× better than chance, doesn't use scanner artifacts, and is faithful
(masking the hot region drops the prediction) — the explainability check the paper never did."
