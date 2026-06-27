"""Model factory: ImageNet-pretrained backbones for binary classification.

We use `timm` so any backbone + a fresh 2-class head come in one call. Starting
from ImageNet weights (transfer learning) means we only need to *fine-tune* on a
few thousand thyroid images instead of training from scratch.

Two model shapes live here:
  * `build_model` — a plain timm backbone + its native classifier head (used for
    the ResNet baselines and the committed plain EfficientNet-B3).
  * `build_attn_model` / `AttnClassifier` — Phase-2 wrapper: backbone feature map ->
    external attention module ('none'/'se'/'cbam'/'cpca') -> global-pool -> head, so
    every attention arm shares one identical skeleton and differs only in the module.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

from . import attention as attn_mod


# Backbones we compare, with their native input resolution.
BACKBONES = {
    "resnet18": 224,
    "resnet50": 224,
    "efficientnet_b0": 224,
    "efficientnet_b3": 300,   # native size = the Medicina paper's 300x300
}


def build_model(name: str = "resnet50", num_classes: int = 2,
                pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    """Any timm backbone with a fresh `num_classes` head. Output = `num_classes` logits.

    `drop_rate` sets the classifier dropout (used to regularise EfficientNet)."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                             drop_rate=drop_rate)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze the backbone. The head (and, for AttnClassifier, the attention
    module) stay trainable either way — they're newly-initialised, like the head."""
    if isinstance(model, AttnClassifier):
        for p in model.parameters():          # attn + head always trainable
            p.requires_grad = True
        for p in model.backbone.parameters():  # only the timm backbone is frozen/unfrozen
            p.requires_grad = trainable
    else:
        for p in model.parameters():
            p.requires_grad = trainable
        for p in model.get_classifier().parameters():
            p.requires_grad = True


class AttnClassifier(nn.Module):
    """Backbone feature map -> external attention -> global-pool -> dropout -> head.

    The SAME wrapper for every Phase-2 arm; only the `attention` module differs, so any
    metric difference between arms is attributable to attention (the controlled experiment)."""

    def __init__(self, backbone: str = "efficientnet_b3", attention: str = "none",
                 num_classes: int = 2, drop_rate: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="")
        c = self.backbone.num_features
        self.attention_name = attention
        self.attn = attn_mod.build_attention(attention, c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        f = self.backbone.forward_features(x)   # (B, C, H, W) spatial feature map
        f = self.attn(f)                        # attention-refined (same shape)
        f = self.pool(f).flatten(1)             # (B, C)
        return self.head(self.drop(f))

    def get_classifier(self) -> nn.Module:
        return self.head


def build_attn_model(backbone: str = "efficientnet_b3", attention: str = "none",
                     num_classes: int = 2, drop_rate: float = 0.0,
                     pretrained: bool = True) -> AttnClassifier:
    """Phase-2 model: backbone + external attention ('none'/'se'/'cbam'/'cpca') + head."""
    return AttnClassifier(backbone, attention, num_classes, drop_rate, pretrained)


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
    attention = meta.get("attention")            # set for Phase-2 attention models
    if attention is not None:
        model = build_attn_model(backbone, attention, num_classes=num_classes, pretrained=False)
    else:
        model = build_model(backbone, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta
