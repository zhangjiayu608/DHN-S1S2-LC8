from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .eo_backbone_utils import align_tokens, batch_robust_norm, load_encoder_weights, resize_image, resolve_path, torch_load


class SoftConBackbone(nn.Module):
    """SoftCon ViT-B/14 SAR and multispectral encoders."""

    def __init__(
        self,
        repo_dir: str | Path,
        sar_checkpoint: str | Path,
        optical_checkpoint: str | Path,
        image_size: int = 224,
        target_side: int = 12,
        raw_patch_size: int = 8,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        optical_channels: int = 13,
        joint_mode: str = "average",
    ) -> None:
        super().__init__()
        repo_dir = resolve_path(repo_dir)
        sar_checkpoint = resolve_path(sar_checkpoint)
        optical_checkpoint = resolve_path(optical_checkpoint)
        if not sar_checkpoint.exists():
            raise FileNotFoundError(f"Cannot find SoftCon SAR checkpoint: {sar_checkpoint}")
        if not optical_checkpoint.exists():
            raise FileNotFoundError(f"Cannot find SoftCon optical checkpoint: {optical_checkpoint}")
        self.encoder_dim = 768
        self.patch_size = int(raw_patch_size)
        self.image_size = int(image_size)
        self.target_side = int(target_side)
        self.input_norm = str(input_norm or "batch_robust").lower()
        self.optical_channels = int(optical_channels)
        self.joint_mode = str(joint_mode or "average").lower()

        sys.path.insert(0, str(repo_dir))
        try:
            from models.dinov2.vision_transformer import vit_base
        finally:
            try:
                sys.path.remove(str(repo_dir))
            except ValueError:
                pass

        self.sar_model = vit_base(patch_size=14, img_size=self.image_size, in_chans=2, block_chunks=0, init_values=1e-5)
        self.optical_model = vit_base(
            patch_size=14,
            img_size=self.image_size,
            in_chans=self.optical_channels,
            block_chunks=0,
            init_values=1e-5,
        )
        unused = ("mask_token", "head.", "classifier.")
        self.sar_load_message = load_encoder_weights(self.sar_model, torch_load(sar_checkpoint), ignored_prefixes=unused)
        self.optical_load_message = load_encoder_weights(self.optical_model, torch_load(optical_checkpoint), ignored_prefixes=unused)
        self.output_norm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False) if token_norm else nn.Identity()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(f"SoftCon S1 channel mismatch: got {s1.shape[1]}, expected 2")
        s1_in, s2_in = self._normalize_inputs(s1, s2)
        s1_in = resize_image(s1_in, self.image_size)
        s2_in = resize_image(self._optical_bands(s2_in), self.image_size)
        sar = self.output_norm(
            align_tokens(self.sar_model.forward_features(s1_in.float())["x_norm_patchtokens"].contiguous(), self.target_side)
        )
        optical = self.output_norm(
            align_tokens(self.optical_model.forward_features(s2_in.float())["x_norm_patchtokens"].contiguous(), self.target_side)
        )
        if self.joint_mode in {"average", "mean", "avg"}:
            joint = 0.5 * (sar + optical)
        elif self.joint_mode in {"optical", "ms"}:
            joint = optical
        else:
            raise ValueError(f"Unknown SoftCon joint_mode: {self.joint_mode}")
        return {"SAR_encodings": sar, "optical_encodings": optical, "joint_encodings": joint}

    def _optical_bands(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != 12:
            raise ValueError("SoftCon expects Sentinel-2 bands in canonical order, with B10 removed.")
        if self.optical_channels == 12:
            return image
        if self.optical_channels == 13:
            return torch.cat((image[:, :10], torch.zeros_like(image[:, :1]), image[:, 10:]), dim=1)
        raise ValueError("SoftCon optical checkpoints must use 12 or 13 Sentinel-2 bands.")

    def _normalize_inputs(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return batch_robust_norm(s1), batch_robust_norm(s2)
        raise ValueError(f"Unknown SoftCon input_norm: {self.input_norm}")


def build_softcon_backbone(config: dict[str, Any]) -> SoftConBackbone:
    return SoftConBackbone(
        repo_dir=config["repo_dir"],
        sar_checkpoint=config["sar_checkpoint"],
        optical_checkpoint=config["optical_checkpoint"],
        image_size=int(config.get("image_size", 224)),
        target_side=int(config.get("target_side", 12)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        optical_channels=int(config.get("optical_channels", 13)),
        joint_mode=config.get("joint_mode", "average"),
    )
