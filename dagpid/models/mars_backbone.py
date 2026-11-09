from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .eo_backbone_utils import batch_robust_norm, load_encoder_weights, map_to_tokens, resize_image, resolve_path, torch_load


class _MaRSSwinEncoder(nn.Module):
    def __init__(self, model_name: str, checkpoint: Path, in_chans: int, image_size: int) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(model_name, pretrained=False, in_chans=in_chans, img_size=image_size, num_classes=0)
        state = self._convert_state_dict(torch_load(checkpoint))
        if "norm.weight" not in state and "norm.bias" not in state:
            norm = self.encoder.norm
            self.encoder.norm = nn.LayerNorm(norm.normalized_shape, eps=norm.eps, elementwise_affine=False)
        self.load_message = load_encoder_weights(self.encoder, state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder.forward_features(x.float())
        if feat.dim() == 4 and feat.shape[-1] == self.encoder.num_features:
            return feat.permute(0, 3, 1, 2).contiguous()
        if feat.dim() == 4:
            return feat.contiguous()
        raise RuntimeError(f"Unexpected MaRS feature shape: {tuple(feat.shape)}")

    @staticmethod
    def _convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in state.items():
            new_key = key
            if key.startswith("layers_"):
                prefix, rest = key.split(".", 1)
                new_key = prefix.replace("layers_", "layers.") + "." + rest
            out[new_key] = value
        return out


class MaRSBackbone(nn.Module):
    """MaRS RGB/SAR SwinV2 encoders adapted to the token interface."""

    _SIZES = {
        "base": ("swinv2_base_window8_256", 1024),
        "large": ("swinv2_large_window12to16_192to256", 1536),
    }

    def __init__(
        self,
        rgb_checkpoint: str | Path,
        sar_checkpoint: str | Path,
        size: str = "base",
        image_size: int = 256,
        target_side: int = 12,
        raw_patch_size: int = 8,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        rgb_indices: list[int] | None = None,
        sar_reduce: str = "mean",
        joint_mode: str = "average",
    ) -> None:
        super().__init__()
        size = str(size or "base").lower()
        if size not in self._SIZES:
            raise ValueError(f"Unsupported MaRS size: {size}")
        model_name, self.encoder_dim = self._SIZES[size]
        self.patch_size = int(raw_patch_size)
        self.image_size = int(image_size)
        self.target_side = int(target_side)
        self.input_norm = str(input_norm or "batch_robust").lower()
        self.rgb_indices = [int(i) for i in (rgb_indices or [3, 2, 1])]
        self.sar_reduce = str(sar_reduce or "mean").lower()
        self.joint_mode = str(joint_mode or "average").lower()
        self.rgb_encoder = _MaRSSwinEncoder(model_name, resolve_path(rgb_checkpoint), in_chans=3, image_size=self.image_size)
        self.sar_encoder = _MaRSSwinEncoder(model_name, resolve_path(sar_checkpoint), in_chans=1, image_size=self.image_size)
        self.output_norm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False) if token_norm else nn.Identity()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(f"MaRS S1 channel mismatch: got {s1.shape[1]}, expected 2")
        if s2.shape[1] <= max(self.rgb_indices):
            raise ValueError(f"MaRS S2 channel mismatch: got {s2.shape[1]}, need indices {self.rgb_indices}")
        s1_in, s2_in = self._normalize_inputs(s1, s2)
        sar_img = self._sar_image(s1_in)
        rgb_img = s2_in[:, self.rgb_indices]
        sar_feat = self.sar_encoder(resize_image(sar_img, self.image_size))
        rgb_feat = self.rgb_encoder(resize_image(rgb_img, self.image_size))
        if sar_feat.shape[-2:] != (self.target_side, self.target_side):
            sar_feat = F.interpolate(sar_feat, size=(self.target_side, self.target_side), mode="bilinear", align_corners=False)
        if rgb_feat.shape[-2:] != (self.target_side, self.target_side):
            rgb_feat = F.interpolate(rgb_feat, size=(self.target_side, self.target_side), mode="bilinear", align_corners=False)
        sar = self.output_norm(map_to_tokens(sar_feat))
        optical = self.output_norm(map_to_tokens(rgb_feat))
        if self.joint_mode in {"average", "mean", "avg"}:
            joint = 0.5 * (sar + optical)
        elif self.joint_mode == "optical":
            joint = optical
        elif self.joint_mode == "sar":
            joint = sar
        else:
            raise ValueError(f"Unknown MaRS joint_mode: {self.joint_mode}")
        return {"SAR_encodings": sar, "optical_encodings": optical, "joint_encodings": joint}

    def _sar_image(self, s1: torch.Tensor) -> torch.Tensor:
        if self.sar_reduce in {"mean", "avg"}:
            return s1.mean(dim=1, keepdim=True)
        if self.sar_reduce == "first":
            return s1[:, :1]
        if self.sar_reduce == "second":
            return s1[:, 1:2]
        raise ValueError(f"Unknown MaRS sar_reduce: {self.sar_reduce}")

    def _normalize_inputs(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return batch_robust_norm(s1), batch_robust_norm(s2)
        raise ValueError(f"Unknown MaRS input_norm: {self.input_norm}")


def build_mars_backbone(config: dict[str, Any]) -> MaRSBackbone:
    return MaRSBackbone(
        rgb_checkpoint=config["rgb_checkpoint"],
        sar_checkpoint=config["sar_checkpoint"],
        size=config.get("size", "base"),
        image_size=int(config.get("image_size", 256)),
        target_side=int(config.get("target_side", 12)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        rgb_indices=config.get("rgb_indices", [3, 2, 1]),
        sar_reduce=config.get("sar_reduce", "mean"),
        joint_mode=config.get("joint_mode", "average"),
    )
