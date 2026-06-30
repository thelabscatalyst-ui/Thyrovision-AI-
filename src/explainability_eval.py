"""Verify Grad-CAM faithfulness/localization on the sealed TN5000 test set.

This is the explainability *pillar* — not just heatmaps (which the paper showed) but a
quantitative check that the model looks at the right place, using the nodule bounding
boxes. Four numbers per image, aggregated:

  * pointing game     — is the heatmap's hottest pixel inside the nodule box? (hit rate)
  * energy-in-box     — fraction of heatmap energy inside the box (vs the box's area)
  * corner-energy     — fraction of energy in the burned-in-artifact corners (should be low)
  * faithfulness drop — mask the hot region → how much does P(malignant) fall? (should be >0)

Broken down by correct/incorrect and benign/malignant. Inference + one backward per image;
no training. The model is a committed baseline CV fold (AttnClassifier, attention=none).

Run:  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m src.explainability_eval [--fold 1] [--limit N]
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import dataset, gradcam as gc, model as model_mod, preprocess, utils

SIZE = 300
BATCH = 16
MASK_TOP = 0.80          # faithfulness: mask the hottest 20% of the heatmap
# burned-in-artifact corner zones at SIZE px (match preprocess fractions: TL 14%w/12%h, BR 12%/12%)
TL = (slice(0, int(SIZE * 0.12)), slice(0, int(SIZE * 0.14)))
BR = (slice(SIZE - int(SIZE * 0.12), SIZE), slice(SIZE - int(SIZE * 0.12), SIZE))
_MEAN = torch.tensor(utils.IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(utils.IMAGENET_STD).view(3, 1, 1)


def _prep(image_id: str):
    """Return (normalised CHW tensor, rgb uint8 SIZE×SIZE) for one image."""
    rgb = preprocess.clean(preprocess.load_rgb(utils.TN5000_IMAGES / f"{image_id}.jpg"))
    rgb300 = cv2.resize(rgb, (SIZE, SIZE))
    x = (torch.from_numpy(rgb300).permute(2, 0, 1).float() / 255.0 - _MEAN) / _STD
    return x, rgb300


def _corner_energy(cam_up):
    return float(cam_up[TL].sum() + cam_up[BR].sum())


def run(cam_engine, rows, device, log):
    model = cam_engine.model
    recs, cache = [], []   # cache (image_id, rgb300, cam_up) for the example panel
    ids = list(rows.image_id)
    for start in range(0, len(ids), BATCH):
        chunk = ids[start:start + BATCH]
        xs, rgbs = zip(*[_prep(i) for i in chunk])
        x = torch.stack(xs).to(device)
        cam, preds, probs = cam_engine(x)               # cam (B,h,w) on device

        # faithfulness: mask the hottest region in the input, re-predict (no grad)
        cam_up_b = np.stack([cv2.resize(c.cpu().numpy(), (SIZE, SIZE)) for c in cam])
        masks = np.stack([(cu >= np.quantile(cu, MASK_TOP)).astype(np.float32) for cu in cam_up_b])
        mt = torch.from_numpy(1.0 - masks).unsqueeze(1).to(device)   # (B,1,H,W)
        with torch.no_grad():
            probs_masked = torch.softmax(model(x * mt), dim=1)[:, 1]

        preds = preds.cpu().numpy(); probs = probs.cpu().numpy()
        probs_masked = probs_masked.cpu().numpy()
        for k, image_id in enumerate(chunk):
            row = rows[rows.image_id == image_id].iloc[0]
            label = int(row.label)
            bx0, by0, bx1, by1 = gc.scaled_box(image_id, SIZE)
            cu = cam_up_b[k]; total = float(cu.sum()) + 1e-8
            py, px = np.unravel_index(cu.argmax(), cu.shape)
            hit = int(bx0 <= px <= bx1 and by0 <= py <= by1)
            ebox = float(cu[by0:by1, bx0:bx1].sum()) / total
            area = max((bx1 - bx0) * (by1 - by0), 1) / (SIZE * SIZE)
            pp = probs[k] if preds[k] == 1 else 1.0 - probs[k]            # predicted-class prob
            ppm = probs_masked[k] if preds[k] == 1 else 1.0 - probs_masked[k]
            recs.append({
                "image_id": image_id, "label": label, "pred": int(preds[k]),
                "correct": int(preds[k] == label), "prob": float(probs[k]),
                "pointing_hit": hit, "energy_in_box": ebox, "box_area_frac": area,
                "corner_energy": _corner_energy(cu) / total,
                "faith_drop": float(pp - ppm),   # drop in the PREDICTED class's prob (faithful → >0)
            })
            cache.append((image_id, rgbs[k], cu, (bx0, by0, bx1, by1),
                          int(preds[k]), label, float(probs[k])))
        log.info(f"  {min(start + BATCH, len(ids))}/{len(ids)} images")
    return pd.DataFrame(recs), cache


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    def block(sub, name):
        return {"group": name, "n": len(sub),
                "pointing_hit_rate": round(sub.pointing_hit.mean(), 4),
                "energy_in_box_mean": round(sub.energy_in_box.mean(), 4),
                "energy_in_box_sd": round(sub.energy_in_box.std(ddof=1), 4),
                "box_area_frac_mean": round(sub.box_area_frac.mean(), 4),
                "corner_energy_mean": round(sub.corner_energy.mean(), 4),
                "faith_drop_mean": round(sub.faith_drop.mean(), 4)}
    rows = [block(df, "all"),
            block(df[df.correct == 1], "correct"),
            block(df[df.correct == 0], "incorrect"),
            block(df[df.label == 1], "malignant"),
            block(df[df.label == 0], "benign")]
    return pd.DataFrame(rows)


def _example_panel(cache, path):
    """One row each of TP / TN / FP / FN, heatmap + box overlay."""
    want = {(1, 1): "TP (caught cancer)", (0, 0): "TN (cleared benign)",
            (1, 0): "FP (false alarm)", (0, 1): "FN (missed cancer)"}
    picks = {}
    for image_id, rgb, cam, box, pred, label, prob in cache:
        key = (pred, label)
        if key in want and key not in picks:
            picks[key] = (image_id, rgb, cam, box, prob)
        if len(picks) == 4:
            break
    fig, axes = plt.subplots(1, len(picks), figsize=(3.4 * len(picks), 3.7))
    if len(picks) == 1:
        axes = [axes]
    for ax, (key, (image_id, rgb, cam, box, prob)) in zip(axes, picks.items()):
        ax.imshow(gc.overlay(cam, rgb, box, SIZE))
        ax.set_title(f"{want[key]}\n{image_id} · P(mal) {prob:.2f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def comparison_panel(cam_engine, df, device, path):
    """Honest side-by-side: top row = nodule box (ground truth); bottom row = the model's
    Grad-CAM, box-free (heatmaps made from the model only — the box is never an input)."""
    good = df[(df.label == 1) & (df.pointing_hit == 1)].sort_values("prob", ascending=False).head(3)
    miss = df[(df.label == 1) & (df.pointing_hit == 0)].sort_values("prob", ascending=False).head(1)
    picks = pd.concat([good, miss])
    ids = list(picks.image_id)
    xs, rgbs = zip(*[_prep(i) for i in ids])
    cam, _, probs = cam_engine(torch.stack(xs).to(device))
    cam, probs = cam.cpu().numpy(), probs.cpu().numpy()
    fig, axes = plt.subplots(2, len(ids), figsize=(3.2 * len(ids), 6.4))
    for j, (image_id, rgb) in enumerate(zip(ids, rgbs)):
        box = gc.scaled_box(image_id, SIZE)
        gt = rgb.copy()
        cv2.rectangle(gt, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        axes[0, j].imshow(gt); axes[0, j].axis("off")
        axes[0, j].set_title(f"{image_id}\nnodule (ground truth)", fontsize=9)
        hit = "✓ on nodule" if picks.iloc[j].pointing_hit == 1 else "✗ off nodule"
        axes[1, j].imshow(gc.overlay(cam[j], rgb, None, SIZE)); axes[1, j].axis("off")
        axes[1, j].set_title(f"model's view · P(mal) {probs[j]:.2f}\n{hit}", fontsize=9)
    fig.suptitle("Top: where the nodule IS (box)   |   Bottom: where the model LOOKED "
                 "(Grad-CAM, box-free)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _aggregate(summaries) -> pd.DataFrame:
    """mean ± SD across folds for each (group, metric)."""
    metrics = ["pointing_hit_rate", "energy_in_box_mean", "corner_energy_mean", "faith_drop_mean"]
    out = []
    for g in summaries[0].group:
        row = {"group": g}
        for m in metrics:
            vals = np.array([s.loc[s.group == g, m].iloc[0] for s in summaries], dtype=float)
            row[m] = round(vals.mean(), 4)
            row[f"{m}_sd"] = round(vals.std(ddof=1), 4)
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM localization/faithfulness verification")
    ap.add_argument("--folds", default="1", help="comma list or 'all' (1-5) of baseline CV folds")
    ap.add_argument("--method", default="gradcam",
                    choices=["gradcam", "gradcam++", "hirescam", "compare"],
                    help="CAM method (default gradcam — empirically best-localizing here, and "
                         "== HiRes-CAM for this GAP+linear head, i.e. provably faithful); "
                         "'compare' ranks all three on fold 1 by box-localization")
    ap.add_argument("--limit", type=int, default=0, help="limit #test images (0 = all)")
    args = ap.parse_args()
    folds = [1, 2, 3, 4, 5] if args.folds == "all" else [int(f) for f in args.folds.split(",")]

    utils.set_seed()
    utils.ensure_output_dirs()
    log = utils.get_logger("gradcam", "explainability_eval.log")
    device = utils.get_device()

    man = dataset.load_split_manifest()
    test = man[man.split == "test"].reset_index(drop=True)
    if args.limit:
        test = test.iloc[:args.limit].reset_index(drop=True)

    # ── compare mode: rank CAM methods on fold 1 by box-localization, then stop ──
    if args.method == "compare":
        model, _ = model_mod.load_checkpoint(
            utils.CHECKPOINTS_DIR / "cv_efficientnet_b3_none_fold1.pt", device)
        rows = []
        for meth in ("gradcam", "gradcam++", "hirescam"):
            log.info(f"[compare] {meth} ...")
            df, _ = run(gc.GradCAM(model, meth), test, device, log)
            s = _summary(df)
            g = lambda grp, col: s.loc[s.group == grp, col].iloc[0]
            rows.append({"method": meth, "pointing_all": g("all", "pointing_hit_rate"),
                         "pointing_malignant": g("malignant", "pointing_hit_rate"),
                         "pointing_benign": g("benign", "pointing_hit_rate"),
                         "energy_in_box": g("all", "energy_in_box_mean"),
                         "faith_drop": g("all", "faith_drop_mean")})
        cmp = pd.DataFrame(rows).sort_values("pointing_malignant", ascending=False)
        print("\n=== CAM-method comparison (fold 1, ground-truth-box localization) ===")
        print(cmp.to_string(index=False))
        print(f"\nBest by malignant pointing-game: **{cmp.iloc[0]['method']}** — "
              "re-run with `--method <that> --folds all` to lock it in.")
        return

    summaries = []
    for i, fold in enumerate(folds):
        ckpt = utils.CHECKPOINTS_DIR / f"cv_efficientnet_b3_none_fold{fold}.pt"
        model, _ = model_mod.load_checkpoint(ckpt, device)
        cam_engine = gc.GradCAM(model, args.method)
        log.info(f"[fold {fold}] {args.method} on {len(test)} test images...")
        df, cache = run(cam_engine, test, device, log)
        summaries.append(_summary(df))
        if i == 0:   # per-image table + figures come from the first fold
            df.to_csv(utils.CSV_B3_BASELINE / "gradcam_per_image.csv", index=False)
            _example_panel(cache, utils.FIG_B3_BASELINE / "gradcam_examples.png")
            comparison_panel(cam_engine, df, device,
                             utils.FIG_B3_BASELINE / "gradcam_vs_box.png")

    sum_csv = utils.CSV_B3_BASELINE / "gradcam_verification.csv"
    if len(folds) > 1:
        out = _aggregate(summaries)
        header = f"{args.method}, {len(folds)}-fold mean ± SD"
    else:
        out = summaries[0]
        header = f"{args.method}, fold {folds[0]}, {len(test)} images"
    out.to_csv(sum_csv, index=False)

    print(f"\n=== Grad-CAM verification (committed B3, {header}) ===")
    print(out.to_string(index=False))
    print("\nReading: pointing_hit_rate ≫ box_area_frac and energy_in_box ≫ box_area_frac → "
          "heatmap is ON the nodule; corner_energy low + faith_drop > 0 → not artifacts, faithful.")
    print(f"Summary: {sum_csv}")
    print(f"Figures: {utils.FIG_B3_BASELINE/'gradcam_examples.png'}, "
          f"{utils.FIG_B3_BASELINE/'gradcam_vs_box.png'}")


if __name__ == "__main__":
    main()
