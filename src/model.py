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


def build_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet-50 with a fresh `num_classes` head. Output = 2 logits."""
    return timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict | None = None) -> None:
    """Save weights + provenance metadata (epoch, val metrics, class mapping...)."""
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path: str | Path, device: torch.device,
                    num_classes: int = 2) -> tuple[nn.Module, dict]:
    """Rebuild the architecture and load saved weights for evaluation."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_resnet50(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("meta", {})
