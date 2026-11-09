from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


class CromaBackbone(nn.Module):
    """Wrapper around the official pretrained CROMA implementation."""

    pretrained_parameter_prefixes = ("model.",)

    def __init__(
        self,
        croma_dir: str | Path,
        checkpoint: str | Path,
        size: str = "large",
        image_resolution: int = 96,
    ) -> None:
        super().__init__()
        croma_dir = Path(croma_dir).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not (croma_dir / "use_croma.py").exists():
            raise FileNotFoundError(f"Cannot find use_croma.py under {croma_dir}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot find CROMA checkpoint: {checkpoint}")

        sys.path.insert(0, str(croma_dir))
        try:
            from use_croma import PretrainedCROMA
        finally:
            try:
                sys.path.remove(str(croma_dir))
            except ValueError:
                pass

        self.model = PretrainedCROMA(
            pretrained_path=str(checkpoint),
            size=size,
            modality="both",
            image_resolution=image_resolution,
        )
        self.encoder_dim = self.model.encoder_dim
        self.patch_size = self.model.patch_size

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(SAR_images=s1, optical_images=s2)


def build_backbone(config: dict) -> nn.Module:
    name = str(config.get("name", "croma")).lower()
    if name == "croma":
        return CromaBackbone(
            croma_dir=config["croma_dir"],
            checkpoint=config["checkpoint"],
            size=config.get("size", "large"),
            image_resolution=int(config.get("image_resolution", 96)),
        )
    raise ValueError(f"Unknown backbone: {name}")
