"""Grad-CAM for the committed EfficientNet-B3 (AttnClassifier) + nodule-box helpers.

Grad-CAM highlights the image regions that most influenced a prediction. We use it to
*verify* explainability (the check the Medicina paper skipped): does the heatmap land on
the annotated nodule (TN5000 bounding box), or on background / burned-in artifacts?

Target layer = the backbone's final spatial feature map (`forward_features`, 10x10x1536).
We expose it directly through the AttnClassifier forward rather than via hooks, so the
gradient w.r.t. that map is clean and the code can't silently grab the wrong layer.

Coordinate note: the dataset resizes with A.Resize (which *squashes* aspect ratio to
square), and preprocess.clean() never crops — so a box in original (W,H) pixels maps to
the size×size input by scaling x and y *independently*.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch

from . import utils


class GradCAM:
    """Grad-CAM on the AttnClassifier's backbone feature map.

    Call returns (cam, preds, probs): cam is per-image normalised to [0,1] at the
    backbone's spatial resolution (10x10); probs is P(malignant)."""

    def __init__(self, model, method: str = "gradcam"):
        self.model = model.eval()
        assert method in ("gradcam", "gradcam++", "hirescam"), f"unknown method {method!r}"
        self.method = method

    def __call__(self, x: torch.Tensor, target=None):
        with torch.enable_grad():
            f = self.model.backbone.forward_features(x)     # (B, C, h, w)
            f.retain_grad()
            fa = self.model.attn(f)
            pooled = self.model.pool(fa).flatten(1)
            logits = self.model.head(self.model.drop(pooled))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            tgt = preds if target is None else torch.as_tensor(target, device=x.device)
            score = logits.gather(1, tgt.view(-1, 1)).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()
            grads = f.grad                                   # (B, C, h, w)

        if self.method == "hirescam":
            # element-wise grad × activation, summed over channels — provably faithful
            cam = torch.relu((grads * f).sum(dim=1))
        elif self.method == "gradcam++":
            # higher-order weighting — sharper, better for small/multiple regions
            g2, g3 = grads ** 2, grads ** 3
            denom = 2 * g2 + (f * g3).sum(dim=(2, 3), keepdim=True)
            alpha = g2 / (denom + 1e-8)
            weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * f).sum(dim=1))
        else:  # plain Grad-CAM (GAP of gradients)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * f).sum(dim=1))       # (B, h, w)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach(), preds.detach(), probs.detach()


# ── Nodule-box helpers ──────────────────────────────────────────────────────
def parse_box(image_id: str) -> tuple[int, int, int, int, int, int]:
    """Return (xmin, ymin, xmax, ymax, W, H) for the nodule (first object) in original px."""
    root = ET.parse(utils.TN5000_ANNOTATIONS / f"{image_id}.xml").getroot()
    size = root.find("size")
    W, H = int(size.find("width").text), int(size.find("height").text)
    bb = root.find("object").find("bndbox")
    box = tuple(int(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
    return (*box, W, H)


def scaled_box(image_id: str, size: int = 300) -> tuple[int, int, int, int]:
    """Box mapped to a size×size input (x and y scaled independently — Resize squashes)."""
    xmin, ymin, xmax, ymax, W, H = parse_box(image_id)
    sx, sy = size / W, size / H
    return (int(xmin * sx), int(ymin * sy), int(xmax * sx), int(ymax * sy))


def overlay(cam: np.ndarray, rgb: np.ndarray, box=None, size: int = 300) -> np.ndarray:
    """Blend a [0,1] cam (any res) over an RGB uint8 size×size image; optional box."""
    cam_up = cv2.resize(cam.astype(np.float32), (size, size))
    heat = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend = (0.55 * rgb + 0.45 * heat).astype(np.uint8)
    if box is not None:
        cv2.rectangle(blend, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return blend
