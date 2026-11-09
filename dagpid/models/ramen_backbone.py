from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .eo_backbone_utils import batch_robust_norm, load_encoder_weights, map_to_tokens, resolve_path, torch_load


class RAMENBackbone(nn.Module):
    """RAMEN mono-temporal encoder adapted to the common S1/S2 token interface."""

    _S2_10_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

    _WAVELENGTHS = {
        "optical": {
            "B1": 440.0,
            "B2": 490.0,
            "B3": 560.0,
            "B4": 665.0,
            "B5": 705.0,
            "B6": 740.0,
            "B7": 783.0,
            "B8": 832.0,
            "B8A": 865.0,
            "B9": 945.0,
            "B10": 1373.0,
            "B11": 1610.0,
            "B12": 2200.0,
        },
        "sar": {"VV": "asc_vv", "VH": "asc_vh"},
    }

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        image_size: int = 96,
        input_res: float = 10.0,
        output_res: float = 80.0,
        raw_patch_size: int = 8,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        s2_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        repo_dir = resolve_path(repo_dir)
        checkpoint = resolve_path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot find RAMEN checkpoint: {checkpoint}")
        pangaea_dir = repo_dir / "pangaea-bench"
        sys.path.insert(0, str(pangaea_dir))
        try:
            from pangaea.encoders.ramen_encoder import RAMEN_Encoder_MonoTemporal
        finally:
            try:
                sys.path.remove(str(pangaea_dir))
            except ValueError:
                pass

        self.encoder_dim = 768
        self.patch_size = int(raw_patch_size)
        self.image_size = int(image_size)
        self.input_norm = str(input_norm or "batch_robust").lower()
        self.s2_indices = [int(i) for i in (s2_indices or self._S2_10_INDICES)]
        self.output_norm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False) if token_norm else nn.Identity()

        common = dict(
            encoder_weights=checkpoint,
            input_size=self.image_size,
            embed_dim=self.encoder_dim,
            output_dim=self.encoder_dim,
            output_layers=[11],
            download_url="",
            input_res=float(input_res),
            res=float(output_res),
            wavelengths=self._WAVELENGTHS,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
        )
        self.sar_encoder = RAMEN_Encoder_MonoTemporal(input_bands={"sar": ["VV", "VH"]}, **common)
        self.optical_encoder = RAMEN_Encoder_MonoTemporal(
            input_bands={"optical": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]},
            **common,
        )
        self.joint_encoder = RAMEN_Encoder_MonoTemporal(
            input_bands={"sar": ["VV", "VH"], "optical": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]},
            **common,
        )
        pretrained = torch_load(checkpoint)
        for encoder in (self.sar_encoder, self.optical_encoder, self.joint_encoder):
            encoder.modalities = list(encoder.modalities)
            load_encoder_weights(encoder, pretrained, ignored_unexpected_prefixes=("ltae.", "dem_projector.", "radar_projector."))

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(f"RAMEN S1 channel mismatch: got {s1.shape[1]}, expected 2")
        if s2.shape[1] <= max(self.s2_indices):
            raise ValueError(f"RAMEN S2 channel mismatch: got {s2.shape[1]}, need indices {self.s2_indices}")
        s1_in, s2_in = self._normalize_inputs(s1, s2[:, self.s2_indices])
        if s1_in.shape[-1] != self.image_size or s1_in.shape[-2] != self.image_size:
            s1_in = torch.nn.functional.interpolate(s1_in, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            s2_in = torch.nn.functional.interpolate(s2_in, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        sar = self.output_norm(map_to_tokens(self.sar_encoder({"sar": s1_in})[-1]))
        optical = self.output_norm(map_to_tokens(self.optical_encoder({"optical": s2_in})[-1]))
        joint_map = self.joint_encoder({"sar": s1_in, "optical": s2_in})[-1]
        sar_joint, optical_joint = joint_map.split(self.encoder_dim, dim=1)
        joint = self.output_norm(map_to_tokens(0.5 * (sar_joint + optical_joint)))
        return {"SAR_encodings": sar, "optical_encodings": optical, "joint_encodings": joint}

    def _normalize_inputs(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return batch_robust_norm(s1), batch_robust_norm(s2)
        raise ValueError(f"Unknown RAMEN input_norm: {self.input_norm}")


def build_ramen_backbone(config: dict[str, Any]) -> RAMENBackbone:
    return RAMENBackbone(
        repo_dir=config["repo_dir"],
        checkpoint=config["checkpoint"],
        image_size=int(config.get("image_size", 96)),
        input_res=float(config.get("input_res", 10.0)),
        output_res=float(config.get("output_res", 80.0)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        s2_indices=config.get("s2_indices"),
    )
