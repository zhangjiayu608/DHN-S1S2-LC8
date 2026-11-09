from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PIDBagFusion(nn.Module):
    """PIDNet-style boundary/risk-controlled fusion for decoder features."""

    def __init__(self, channels: int, risk_channels: int = 1) -> None:
        super().__init__()
        self.risk_proj = (
            nn.Identity()
            if risk_channels == channels
            else nn.Sequential(
                nn.Conv2d(risk_channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        )
        self.conv_p = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if d.shape[-2:] != p.shape[-2:]:
            d = F.interpolate(d, size=p.shape[-2:], mode="bilinear", align_corners=False)
        risk = torch.sigmoid(self.risk_proj(d))
        p_add = self.conv_p((1.0 - risk) * i + p)
        i_add = self.conv_i(i + risk * p)
        return p_add + i_add, risk
