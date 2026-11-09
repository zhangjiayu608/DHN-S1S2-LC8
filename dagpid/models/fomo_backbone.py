from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .eo_backbone_utils import (
    align_tokens,
    batch_robust_norm,
    resize_image,
    resolve_path,
    torch_load,
)


def _load_json5_like(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"//.*", "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


class FoMoBackbone(nn.Module):
    """FoMo-Net encoder adapted to the common S1/S2 token interface."""

    _S2_10_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    _S1_KEYS = [4, 5]
    _S2_KEYS = [7, 8, 9, 10, 11, 12, 13, 14, 17, 18]

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        image_size: int = 64,
        target_image_size: int = 96,
        model_patch_size: int = 16,
        raw_patch_size: int = 8,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        s2_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        repo_dir = resolve_path(repo_dir)
        checkpoint = resolve_path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot find FoMo checkpoint: {checkpoint}")

        sys.path.insert(0, str(repo_dir))
        try:
            from model_zoo import multimodal_mae
        finally:
            try:
                sys.path.remove(str(repo_dir))
            except ValueError:
                pass

        configs = _load_json5_like(
            repo_dir / "configs" / "datasets" / "fomo_pretraining_datasets.json"
        )
        configs["modality_channels"] = {
            int(k): v for k, v in configs["modality_channels"].items()
        }
        configs["single_embedding_layer"] = True
        self.model = multimodal_mae.MultiSpectralViT(
            image_size=int(image_size),
            patch_size=int(model_patch_size),
            channels=1,
            num_classes=1000,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=2048,
            configs=configs,
        )
        self.load_message = self.model.load_state_dict(
            torch_load(checkpoint), strict=True
        )

        self.encoder_dim = 768
        self.patch_size = int(raw_patch_size)
        self.model_patch_size = int(model_patch_size)
        self.image_size = int(image_size)
        self.target_image_size = int(target_image_size)
        self.input_norm = str(input_norm or "batch_robust").lower()
        self.s2_indices = [int(i) for i in (s2_indices or self._S2_10_INDICES)]
        self.output_norm = (
            nn.LayerNorm(self.encoder_dim, elementwise_affine=False)
            if token_norm
            else nn.Identity()
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(f"FoMo S1 channel mismatch: got {s1.shape[1]}, expected 2")
        if s2.shape[1] <= max(self.s2_indices):
            raise ValueError(
                f"FoMo S2 channel mismatch: got {s2.shape[1]}, need indices {self.s2_indices}"
            )
        s1_in, s2_in = self._normalize_inputs(s1, s2[:, self.s2_indices])
        s1_in = resize_image(s1_in, self.image_size)
        s2_in = resize_image(s2_in, self.image_size)
        target_side = self.target_image_size // self.patch_size

        sar = align_tokens(self._encode(s1_in, self._S1_KEYS), target_side)
        optical = align_tokens(self._encode(s2_in, self._S2_KEYS), target_side)
        joint = align_tokens(
            self._encode(
                torch.cat([s1_in, s2_in], dim=1), self._S1_KEYS + self._S2_KEYS
            ),
            target_side,
        )
        return {
            "SAR_encodings": self.output_norm(sar),
            "optical_encodings": self.output_norm(optical),
            "joint_encodings": self.output_norm(joint),
        }

    def _encode(self, x: torch.Tensor, keys: list[int]) -> torch.Tensor:
        tokens = self.model((x, keys), pool=False)
        bsz, count, dim = tokens.shape
        bands = len(keys)
        spatial = count // bands
        if spatial * bands != count:
            raise RuntimeError(
                f"Unexpected FoMo token count {count} for {bands} spectral keys"
            )
        tokens = tokens.view(bsz, bands, spatial, dim).mean(dim=1)
        side = int(spatial**0.5)
        if side * side != spatial:
            raise RuntimeError(f"Unexpected FoMo spatial token count: {spatial}")
        return tokens

    def _normalize_inputs(
        self, s1: torch.Tensor, s2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return batch_robust_norm(s1), batch_robust_norm(s2)
        if self.input_norm in {"batch_zscore", "zscore"}:
            dims = (2, 3)
            s1 = (
                s1.float() - s1.float().mean(dim=dims, keepdim=True)
            ) / s1.float().std(dim=dims, keepdim=True).clamp_min(1e-6)
            s2 = (
                s2.float() - s2.float().mean(dim=dims, keepdim=True)
            ) / s2.float().std(dim=dims, keepdim=True).clamp_min(1e-6)
            return s1, s2
        raise ValueError(f"Unknown FoMo input_norm: {self.input_norm}")


def build_fomo_backbone(config: dict[str, Any]) -> FoMoBackbone:
    return FoMoBackbone(
        repo_dir=config["repo_dir"],
        checkpoint=config["checkpoint"],
        image_size=int(config.get("image_size", 64)),
        target_image_size=int(config.get("target_image_size", 96)),
        model_patch_size=int(config.get("model_patch_size", 16)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        s2_indices=config.get("s2_indices"),
    )
