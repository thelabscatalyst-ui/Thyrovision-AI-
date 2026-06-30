"""Near-duplicate analysis — does the image-level split actually leak?

TN5000 has no patient ID, so we can't split by patient directly. But same-patient
frames of the same nodule look near-identical. We embed every image with a
pretrained ResNet-50 (the SAME feature space the classifier uses — so high cosine
similarity = "the model would see these as the same patient"), find near-duplicate
pairs, group them into pseudo-patients, and — the key number — count how many
near-duplicate pairs **straddle the test boundary** (the actual leakage).

Read-only: this does not retrain or modify the split. It tells us whether the
image-level split is a real problem (regroup needed) or a non-issue (document it).

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m src.near_duplicates
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader

from . import dataset, preprocess, utils

EMB_PATH = utils.CACHE_DIR / "tn5000_embeddings.npy"
THRESHOLDS = (0.95, 0.97, 0.98, 0.99)


@torch.no_grad()
def compute_embeddings(force: bool = False) -> tuple[np.ndarray, "pd.DataFrame"]:
    """2048-d pooled ResNet-50 features for all 5,000 images (cached)."""
    manifest = dataset.load_split_manifest()
    if EMB_PATH.exists() and not force:
        return np.load(EMB_PATH), manifest

    device = utils.get_device()
    model = timm.create_model("resnet50", pretrained=True, num_classes=0).to(device).eval()
    # All images, eval transforms (no aug), cleaned the same way as training.
    ds = dataset.TN5000Dataset(manifest, train_aug=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    embs = []
    for i, (x, _) in enumerate(loader):
        embs.append(model(x.to(device)).cpu().numpy())
        if (i + 1) % 10 == 0:
            print(f"  embedded {(i+1)*64}/{len(ds)}")
    E = np.concatenate(embs).astype(np.float32)
    utils.ensure_output_dirs()
    np.save(EMB_PATH, E)
    print(f"saved embeddings {E.shape} -> {EMB_PATH.name}")
    return E, manifest


def similarity(E: np.ndarray) -> np.ndarray:
    """Cosine-similarity matrix with the diagonal removed."""
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    S = En @ En.T
    np.fill_diagonal(S, -1.0)
    return S


def report(E: np.ndarray, manifest) -> None:
    S = similarity(E)
    splits = manifest.split.values
    nn_sim = S.max(axis=1)

    print("\n=== Nearest-neighbour cosine similarity distribution (per image) ===")
    for p in (50, 90, 95, 99, 99.9):
        print(f"  {p:5.1f}th percentile: {np.percentile(nn_sim, p):.4f}")
    print(f"  max: {nn_sim.max():.4f}")

    print("\n=== Near-duplicate pairs by threshold ===")
    print(f"{'thresh':>7} {'pairs':>7} {'imgs':>6} {'groups':>7} {'maxgrp':>7} "
          f"{'cross-test pairs':>16} {'cross-any pairs':>16}")
    for T in THRESHOLDS:
        iu = np.triu(S >= T, k=1)
        pi, pj = np.where(iu)
        n_pairs = len(pi)
        imgs = np.unique(np.concatenate([pi, pj]))
        # connected components over the pair graph = pseudo-patient groups
        if n_pairs:
            adj = coo_matrix((np.ones(n_pairs), (pi, pj)), shape=(len(E), len(E)))
            n_comp, labels = connected_components(adj, directed=False)
            grp_sizes = np.bincount(labels)
            multi = grp_sizes[grp_sizes >= 2]
            n_groups, max_grp = len(multi), int(multi.max()) if len(multi) else 0
        else:
            n_groups, max_grp = 0, 0
        # pair crosses the test boundary iff exactly one endpoint is in test
        cross_test = int(np.sum((splits[pi] == "test") != (splits[pj] == "test")))
        cross_any = int(np.sum(splits[pi] != splits[pj]))
        print(f"{T:7.2f} {n_pairs:7d} {len(imgs):6d} {n_groups:7d} {max_grp:7d} "
              f"{cross_test:16d} {cross_any:16d}")

    _save_top_pairs(S, manifest, n=8)


def _save_top_pairs(S, manifest, n=8) -> None:
    """Montage of the top-n most-similar pairs, to eyeball whether they're real dups."""
    iu = np.triu_indices(S.shape[0], k=1)
    sims = S[iu]
    order = np.argsort(sims)[::-1][:n]
    ids = manifest.image_id.values
    splits = manifest.split.values
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.6 * n))
    for r, k in enumerate(order):
        i, j = iu[0][k], iu[1][k]
        for c, idx in enumerate((i, j)):
            img = preprocess.load_rgb(utils.TN5000_IMAGES / f"{ids[idx]}.jpg")
            axes[r, c].imshow(img); axes[r, c].axis("off")
            axes[r, c].set_title(f"{ids[idx]} [{splits[idx]}]" +
                                 (f"  sim={sims[k]:.3f}" if c == 0 else ""), fontsize=8)
    fig.suptitle("Most-similar image pairs (verify near-duplicates)")
    fig.tight_layout()
    out = utils.FIG_DATA_QUALITY / "near_duplicate_top_pairs.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\nSaved top-pairs montage -> {out}")


CLEAN_CSV = utils.SPLIT_CLEAN_CSV


def build_clean_manifest(threshold: float = 0.98):
    """Write the de-contaminated manifest: image_id, label, split, group, keep.

    * group = connected-component id over the near-duplicate graph (pseudo-patient;
      singletons get their own id) — used for group-aware CV folds.
    * keep = False for train/val images that near-duplicate a TEST image (the
      leakers we drop). Test images are NEVER dropped — the official 1,000-image
      test set stays whole and benchmark-comparable.
    """
    import pandas as pd  # noqa: F401  (manifest is already a DataFrame)
    E, man = compute_embeddings()
    S = similarity(E)
    n = len(man)
    pi, pj = np.where(np.triu(S >= threshold, k=1))

    adj = coo_matrix((np.ones(len(pi)), (pi, pj)), shape=(n, n))
    _, groups = connected_components(adj, directed=False)

    splits = man.split.values
    is_test = splits == "test"
    cross = is_test[pi] != is_test[pj]
    leakers = {int(a if not is_test[a] else b) for a, b in zip(pi[cross], pj[cross])}
    keep = np.ones(n, dtype=bool)
    keep[list(leakers)] = False

    out = man.copy()
    out["group"] = groups
    out["keep"] = keep
    out.to_csv(CLEAN_CSV, index=False)
    dropped = (~keep)
    print(f"clean manifest -> {CLEAN_CSV.name}")
    print(f"  dropped {int(dropped.sum())} train/val leakers "
          f"({dict(out[dropped].split.value_counts())})")
    print(f"  kept train {int(((out.split=='train') & out.keep).sum())}, "
          f"val {int(((out.split=='val') & out.keep).sum())}, "
          f"test {int((out.split=='test').sum())} (test untouched)")
    return out


def main():
    utils.set_seed()
    E, manifest = compute_embeddings()
    assert len(E) == len(manifest), "embedding/manifest length mismatch"
    report(E, manifest)


if __name__ == "__main__":
    main()
