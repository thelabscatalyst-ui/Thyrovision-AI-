"""Attention modules for the Phase-2 comparison — added EXTERNALLY after the backbone.

All operate on a feature map (B, C, H, W) and return the same shape (so they slot in
between the backbone and the pooling head). We compare four arms on EfficientNet-B3:
  * none — Identity (baseline)
  * se   — Squeeze-Excitation: channel attention only (the paper's choice)
  * cbam — channel + spatial attention ("what" + "where")
  * cpca — Channel-Prior Convolutional Attention (multi-scale depthwise spatial)

EfficientNet-B3 already has SE *internally*; these are added on top, identically for
every arm, so the only thing that differs between arms is this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: reweight channels by global context."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3), keepdim=True))   # (B,C,1,1)
        return x * w


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1))

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """Channel attention then spatial attention ('what' then 'where')."""
    def __init__(self, channels: int, reduction: int = 16, kernel: int = 7):
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.sa = _SpatialAttention(kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


class CPCA(nn.Module):
    """Channel-Prior Convolutional Attention (Huang et al. 2023), compact form.

    Channel prior (avg+max -> MLP -> sigmoid) reweights channels, then a multi-scale
    depthwise spatial branch (square + strip convs) produces the spatial weights.
    Kernel sizes kept moderate for the ~10x10 B3 feature map.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.dw = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.s7 = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels),
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels)])
        self.s11 = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels),
            nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels)])
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = self.ca(x)                                  # channel prior
        u = self.dw(x)
        att = u + self.s7[0](u) + self.s7[1](u) + self.s11[0](u) + self.s11[1](u)
        return x * torch.sigmoid(self.proj(att))        # spatial reweight


def build_attention(name: str, channels: int) -> nn.Module:
    """Factory: 'none' -> Identity, else the named module."""
    name = name.lower()
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SEBlock(channels)
    if name == "cbam":
        return CBAM(channels)
    if name == "cpca":
        return CPCA(channels)
    raise ValueError(f"unknown attention {name!r}; choose none/se/cbam/cpca")
