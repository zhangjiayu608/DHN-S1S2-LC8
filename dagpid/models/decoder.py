from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .preprocessing import normalize_local


class ConvModule(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DepthwiseSeparableConvModule(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class PPM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale), ConvModule(in_channels, channels, 1)
                )
                for scale in pool_scales
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [
            F.interpolate(
                stage(x), size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            for stage in self.stages
        ]


class TokenPyramidUPerDecoder(nn.Module):
    """Build a multi-scale token pyramid and decode it with a UPerNet-style head."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        channels: int = 192,
        feature_only: bool = False,
    ) -> None:
        super().__init__()
        self.p12 = ConvModule(in_dim, channels, 1)
        self.p24 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvModule(channels, channels, 3, 1),
        )
        self.p48 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvModule(channels, channels, 3, 1),
        )
        self.p96 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvModule(channels, channels, 3, 1),
        )

        self.ppm = PPM(channels, channels)
        self.ppm_bottleneck = ConvModule(channels * 5, channels, 3, 1)
        self.lateral = nn.ModuleList(
            [ConvModule(channels, channels, 1) for _ in range(3)]
        )
        self.fpn = nn.ModuleList(
            [ConvModule(channels, channels, 3, 1) for _ in range(3)]
        )
        self.fpn_bottleneck = ConvModule(channels * 4, channels, 1)
        self.classifier = (
            None
            if feature_only
            else nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(channels, num_classes, 1))
        )

    def forward(
        self, token_map: torch.Tensor, return_feature: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feature = self.forward_feature(token_map)
        if self.classifier is None:
            raise RuntimeError("Feature-only decoder must use forward_feature().")
        logits = self.classifier(feature)
        if return_feature:
            return logits, feature
        return logits

    def forward_feature(self, token_map: torch.Tensor) -> torch.Tensor:
        f12 = self.p12(token_map)
        f24 = self.p24(f12)
        f48 = self.p48(f24)
        f96 = self.p96(f48)
        features = [f96, f48, f24, f12]

        top = torch.cat([features[-1], *self.ppm(features[-1])], dim=1)
        laterals = [lat(feat) for lat, feat in zip(self.lateral, features[:-1])]
        laterals.append(self.ppm_bottleneck(top))
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        outs = [conv(laterals[i]) for i, conv in enumerate(self.fpn)]
        outs.append(laterals[-1])
        outs = [
            F.interpolate(
                out, size=outs[0].shape[-2:], mode="bilinear", align_corners=False
            )
            for out in outs
        ]
        return self.fpn_bottleneck(torch.cat(outs, dim=1))
