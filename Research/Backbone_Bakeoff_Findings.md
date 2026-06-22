# Backbone Bake-off — Findings for Mentor Discussion

**Date:** 2026-06-20 · **Purpose:** decide the committed baseline backbone before Phase 2 (CBAM).

## What we did
Ran a fair, **identical-recipe** comparison of four backbones on TN5000, same official
split, same training settings (ImageNet-pretrained, AdamW lr 1e-4, cosine, class-weighted
loss, early stop on val AUC), evaluated once on the sealed 1,000-image test set.

## Result — sealed test (sorted by AUC)
| Backbone | Params | Res | Test AUC | Test Acc | Sens | Spec | Notes |
|----------|-------:|----:|---------:|---------:|-----:|-----:|-------|
| **ResNet-50** | 23.5M | 224 | **0.926** | **0.868** | 0.893 | 0.799 | best overall |
| EfficientNet-B3 | 10.7M | 300 | 0.901 | 0.848 | 0.882 | 0.755 | paper's backbone/res |
| ResNet-18 | 11.2M | 224 | 0.896 | 0.827 | 0.830 | 0.818 | — |
| EfficientNet-B0 | 4.0M | 224 | 0.879 | 0.839 | 0.918 | 0.625 | **overfit** (train AUC 0.998 / val 0.88) |

Figure: `outputs/figures/backbone_bakeoff.png`.

## The honest interpretation (the key point)
- **In our pipeline, ResNet-50 wins and EfficientNets underperform** — this contradicts both the
  mentor's expectation and the general literature ("EfficientNet is better on small data").
- **But the comparison is likely biased.** A single shared recipe was used for fairness, and that
  recipe (AdamW lr 1e-4) is **ResNet-friendly**. EfficientNets are notoriously recipe-sensitive
  (designed with RMSProp, label smoothing, stochastic depth, EMA, AutoAugment). EfficientNet-B0
  **overfit badly** here (specificity collapsed to 0.625) — a symptom of *under-regularisation for
  this recipe*, not of the architecture being bad.
- So we **cannot yet conclude "ResNet-50 > EfficientNet"** — EfficientNet hasn't had a recipe tuned
  for it.

## Questions for the mentor
1. Given EfficientNet underperforms under a *standard* recipe here, do you want us to:
   - (a) **tune an EfficientNet-appropriate recipe** (lower LR, label smoothing, stronger
     augmentation/regularisation) and re-compare fairly — the principled fix; or
   - (b) **commit ResNet-50** (it won the as-is bake-off); or
   - (c) **commit EfficientNet-B3** for direct alignment with the Medicina paper, accepting the
     honest lower number?
2. Does our contribution (CBAM vs baseline + cross-dataset generalisation + verified Grad-CAM)
   require matching the paper's backbone, or is a clearly-justified backbone choice acceptable?

## What does NOT change regardless of the answer
The backbone is a **setup choice**, not the contribution. Leg A (baseline vs CBAM) and Leg B
(cross-dataset) are valid on any backbone — the attention comparison is what matters, and the
leakage/de-dup work, pipeline, and metrics all carry over unchanged.
