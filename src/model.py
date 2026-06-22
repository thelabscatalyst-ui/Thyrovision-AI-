"""Model factory: ImageNet-pretrained ResNet-50 for binary classification.

We use `timm` so the backbone + a fresh 2-class head come in one call. Starting
from ImageNet weights (transfer learning) means we only need to *fine-tune* on a
few thousand thyroid images instead of training 25M parameters from scratch.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn


# Backbones we compare, with their native input resolution.
BACKBONES = {
    "resnet18": 224,
    "resnet50": 224,
    "efficientnet_b0": 224,
    "efficientnet_b3": 300,   # native size = the Medicina paper's 300x300
}


def build_model(name: str = "resnet50", num_classes: int = 2,
                pretrained: bool = True) -> nn.Module:
    """Any timm backbone with a fresh `num_classes` head. Output = `num_classes` logits."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def build_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Back-compat alias for the original baseline."""
    return build_model("resnet50", num_classes=num_classes, pretrained=pretrained)


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict | None = None) -> None:
    """Save weights + provenance metadata (epoch, val metrics, class mapping...)."""
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path: str | Path, device: torch.device,
                    num_classes: int = 2) -> tuple[nn.Module, dict]:
    """Rebuild the architecture and load saved weights for evaluation."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    backbone = meta.get("backbone", "resnet50")  # default for older checkpoints
    model = build_model(backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta
