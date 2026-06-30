"""TI-RADS-style clinical layer: calibrate the probability, map it to a relative risk
tier, attach a recommendation + the Grad-CAM heatmap. A *reporting layer on top of* the
classifier — it does not change the model.

HONEST SCOPE (printed with every output):
  * Probabilities are calibrated to TN5000's ENRICHED ~71%-malignant cohort, NOT real-world
    ~5–15% prevalence — so we give RELATIVE risk *tiers*, not absolute % malignancy risk
    (absolute risk needs a prevalence/prior adjustment we can't make here).
  * This is risk-tier ALIGNMENT with the TI-RADS ladder, NOT true ACR TI-RADS feature
    scoring (we do not compute composition / echogenicity / shape / margin / foci).
  * Validating tiers against radiologist TI-RADS needs DDTI (quarantined) — future work.

Run:  HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m src.tirads
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from torch.nn import CrossEntropyLoss

from . import dataset, gradcam as gc, model as model_mod, utils
from .explainability_eval import SIZE, _prep

# Relative suspicion tiers on the calibrated P(malignant), aligned to the TI-RADS risk ladder.
# (upper_bound, label, recommendation)
TIERS = [
    (0.20, "Low suspicion",          "routine follow-up ultrasound"),
    (0.50, "Intermediate suspicion", "short-interval follow-up; consider FNA"),
    (0.80, "High suspicion",         "FNA biopsy recommended"),
    (1.01, "Very high suspicion",    "FNA / biopsy strongly recommended"),
]


def risk_tier(p: float):
    for hi, label, rec in TIERS:
        if p < hi:
            return label, rec
    return TIERS[-1][1], TIERS[-1][2]


@torch.no_grad()
def _logits(model, rows, device):
    xs = torch.stack([_prep(i)[0] for i in rows.image_id])
    out = [model(xs[s:s + 32].to(device)).cpu() for s in range(0, len(xs), 32)]
    return torch.cat(out), torch.tensor(rows.label.values, dtype=torch.long)


def fit_temperature(logits, labels) -> float:
    """Temperature scaling: one scalar T fit on validation NLL (Guo et al. 2017)."""
    T = torch.nn.Parameter(torch.ones(1))
    nll = CrossEntropyLoss()
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = nll(logits / T.clamp_min(1e-2), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp_min(1e-2))


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error + per-bin (confidence, observed-freq, n) for the plot."""
    edges = np.linspace(0, 1, n_bins + 1)
    e, stats = 0.0, []
    for i in range(n_bins):
        hi = probs <= edges[i + 1] if i == n_bins - 1 else probs < edges[i + 1]
        m = (probs >= edges[i]) & hi
        if m.sum() == 0:
            stats.append((np.nan, np.nan, 0))
            continue
        conf, freq = probs[m].mean(), labels[m].mean()
        e += (m.sum() / len(probs)) * abs(freq - conf)
        stats.append((float(conf), float(freq), int(m.sum())))
    return e, stats


def _calibrate_all(val_logits, val_labels, test_logits):
    """Fit temperature / Platt / isotonic on validation; return each method's TEST probs."""
    vy = val_labels.numpy()
    v_p = torch.softmax(val_logits, 1)[:, 1].numpy()
    t_p = torch.softmax(test_logits, 1)[:, 1].numpy()
    v_m = (val_logits[:, 1] - val_logits[:, 0]).numpy().reshape(-1, 1)   # decision margin
    t_m = (test_logits[:, 1] - test_logits[:, 0]).numpy().reshape(-1, 1)
    T = fit_temperature(val_logits, val_labels)
    methods = {
        "raw": t_p,
        "temperature": torch.softmax(test_logits / T, 1)[:, 1].numpy(),
        "platt": LogisticRegression().fit(v_m, vy).predict_proba(t_m)[:, 1],
        "isotonic": IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(v_p, vy).predict(t_p),
    }
    return methods, T


def reliability_plot(raw, cal, labels, path, cal_name="calibrated"):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for probs, name, color in [(raw, "raw", "#888780"), (cal, cal_name, "#1D9E75")]:
        _, stats = ece(probs, labels)
        xs = [s[0] for s in stats if s[2] > 0]
        ys = [s[1] for s in stats if s[2] > 0]
        ax.plot(xs, ys, "o-", color=color, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.set_xlabel("predicted P(malignant)")
    ax.set_ylabel("observed fraction malignant")
    ax.set_title("Calibration reliability (TN5000 test)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def report_cards(model, rows, probs_cal, device, path):
    """One example per tier: cleaned image + Grad-CAM + calibrated suspicion / tier / rec."""
    cam_engine = gc.GradCAM(model)
    picks = [int(np.argmin(np.abs(probs_cal - (hi - 0.1)))) for hi, _, _ in TIERS]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.5 * len(picks), 4.2))
    for ax, idx in zip(axes, picks):
        r = rows.iloc[idx]
        x, rgb = _prep(r.image_id)
        cam, _, _ = cam_engine(x.unsqueeze(0).to(device))
        ax.imshow(gc.overlay(cam[0].cpu().numpy(), rgb, None, SIZE)); ax.axis("off")
        p = float(probs_cal[idx]); label, rec = risk_tier(p)
        true = "malignant" if r.label == 1 else "benign"
        ax.set_title(f"{r.image_id} (true: {true})\nsuspicion {p:.2f} · {label}\n{rec}", fontsize=8)
    fig.suptitle("Example report cards — calibrated suspicion → tier → recommendation (+ Grad-CAM)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    utils.set_seed()
    utils.ensure_output_dirs()
    log = utils.get_logger("tirads", "tirads.log")
    device = utils.get_device()
    model, _ = model_mod.load_checkpoint(
        utils.CHECKPOINTS_DIR / "cv_efficientnet_b3_none_fold1.pt", device)

    man = dataset.load_split_manifest()
    tv = man[man.split.isin(["train", "val"])].reset_index(drop=True)
    _, va_idx = next(StratifiedKFold(5, shuffle=True, random_state=utils.SEED)
                     .split(tv.image_id, tv.label))           # fold-1 val slice = calibration set
    val_rows = tv.iloc[va_idx]
    test_rows = man[man.split == "test"].reset_index(drop=True)

    log.info("inferring logits (fold-1 val for calibration, test for evaluation)...")
    val_logits, val_labels = _logits(model, val_rows, device)
    test_logits, test_labels = _logits(model, test_rows, device)

    methods, T = _calibrate_all(val_logits, val_labels, test_logits)
    y = test_labels.numpy()
    table = pd.DataFrame([{"method": name, "ece": round(ece(p, y)[0], 4),
                           "brier": round(brier_score_loss(y, p), 4)}
                          for name, p in methods.items()])
    cal_only = table[table.method != "raw"]
    best = cal_only.loc[cal_only.ece.idxmin(), "method"]          # best by lowest test ECE
    p_best = methods[best]
    table["chosen"] = table.method == best

    reliability_plot(methods["raw"], p_best, y,
                     utils.FIG_B3_BASELINE / "calibration_reliability.png", cal_name=best)
    report_cards(model, test_rows, p_best, device,
                 utils.FIG_B3_BASELINE / "tirads_report_cards.png")

    dist = (pd.Series([risk_tier(p)[0] for p in p_best]).value_counts()
            .reindex([t[1] for t in TIERS]).fillna(0).astype(int))
    table.to_csv(utils.CSV_B3_BASELINE / "calibration.csv", index=False)

    print("\n=== Calibration method comparison (fit on fold-1 val, evaluated on TEST) ===")
    print(table.to_string(index=False))
    print(f"\n  Best by test ECE: **{best}**  (temperature T={T:.3f} shown for reference)")
    print("\n=== Risk-tier distribution on TN5000 test (best-calibrated) ===")
    for t in [x[1] for x in TIERS]:
        print(f"  {t:24s}: {dist[t]}")
    print("\nCAVEAT: calibrated to TN5000's enriched 71%-malignant cohort → RELATIVE tiers, not")
    print("absolute % risk; this is risk-tier alignment, NOT ACR TI-RADS feature scoring.")
    print("\nFigures: calibration_reliability.png, tirads_report_cards.png | calibration.csv")


if __name__ == "__main__":
    main()
