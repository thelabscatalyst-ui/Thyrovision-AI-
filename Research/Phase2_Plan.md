# Phase 2 Plan — Attention-Mechanism Comparison (Leg A)

**Status:** active. **Backbone (locked):** EfficientNet-B3. **Mentor-confirmed
sequencing:** attention comparison **first**, detection (YOLO/MobileNetV4) **later**.

## Goal (the core contribution)
The Medicina paper used **SE attention and never compared it** to anything. Our
contribution is the **controlled comparison they skipped** — add several attention
mechanisms to the *same* EfficientNet-B3 backbone, change *nothing else*, and see
which (if any) actually helps. Even "attention didn't help" is a complete, honest result.

## The arms (4) — identical recipe, split, seed
| Arm | What | Why |
|-----|------|-----|
| **baseline** | EfficientNet-B3 → head (no added attention) | reference |
| **+SE** | + Squeeze-Excitation (channel only) | the paper's choice |
| **+CBAM** | + channel **and spatial** attention | "where to look" — the hypothesis |
| **+CPCA** | + Channel-Prior Convolutional Attention | beat CBAM in Paper 2's detector — worth testing |

**Key nuance (document honestly):** EfficientNet-B3 already contains SE *internally*
(every MBConv block). So we add each mechanism as an **external module after the
backbone's final feature map** (before global-pool + head). This makes SE/CBAM/CPCA a
**like-for-like** comparison (all "extra attention on top of the same backbone"), and the
"+SE" arm mirrors what the paper did (SE added on top of B3).

## Recipe (identical to the committed baseline)
Two-phase fine-tune (freeze head → unfreeze @ low LR cosine, early stop), **300px,
batch 16, dropout 0.4, label-smoothing 0.1, class-weighted loss**, light aug, MPS.
Only the attention module differs between arms → any difference is **attributable to
attention** (the controlled experiment).

## Evaluation (honest, same as Phase 1)
- Single official split first → compare arms by **AUC** (threshold-free headline) +
  sensitivity/specificity at a sensible operating point. Then **5-fold CV** on the
  winner (and baseline) for the error-barred final number.
- Same-split leakage hits all arms equally, so the *comparison* is clean regardless.
- Report a 4-row table (baseline/+SE/+CBAM/+CPCA): AUC, acc, sens, spec, params, train time.

## Code plan (built + smoke-tested, then run in user's Terminal under caffeinate)
- `src/attention.py` — SE, CBAM, CPCA modules (small, standard implementations).
- `src/model.py` — `build_model(..., attention="none|se|cbam|cpca")` that wraps the
  timm backbone (features) → attention module → global-pool → 2-class head.
- `src/phase2_attention.py` — driver: runs the 4 arms (reusing `train.fit_two_phase`),
  resumable per-arm, evaluates each on the sealed test, emits the comparison table.
- Reuse everything else (dataset, metrics, evaluate, two-phase trainer) unchanged.

## Housekeeping (do first)
Align the code's output paths to the user's `outputs/logs/{csv,json,logs_files}/` layout
(the approved "Option A") so Phase-2 runs land organised, not at `outputs/` root.

## Explicitly NOT in Phase 2
Detection (YOLOv8 + MobileNetV4) — that's **Phase 3**, scoped only after this comparison.
No GAN. No ensembling. DDTI stays quarantined (Phase 3/4 cross-dataset only).

## Verification
- Each attention module: shape-preserving on a B3 feature map (unit check).
- Each arm trains (two-phase) and beats the ~0.85 sanity floor; no arm ≥0.97 (leakage guard).
- The only difference across arms is the attention block (same data/recipe/seed).
