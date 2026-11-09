from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .eo_backbone_utils import (
    align_tokens,
    batch_robust_norm,
    load_module_from_file,
    map_to_tokens,
    resize_image,
    resolve_path,
    torch_load,
)


class AnySatBackbone(nn.Module):
    """AnySat release encoder adapted to the common S1/S2 token interface."""

    _S2_10_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        image_size: int = 96,
        patch_size_m: int = 80,
        raw_patch_size: int = 8,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        doy: int = 182,
        s2_indices: list[int] | None = None,
        model_size: str = "base",
    ) -> None:
        super().__init__()
        repo_dir = resolve_path(repo_dir)
        checkpoint = resolve_path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot find AnySat checkpoint: {checkpoint}")

        module = load_module_from_file(
            "_dagpid_anysat_hubconf", repo_dir / "hubconf.py"
        )
        sys.path.insert(0, str(repo_dir))
        try:
            self.model = module.AnySat(model_size=model_size, flash_attn=False)
        finally:
            try:
                sys.path.remove(str(repo_dir))
            except ValueError:
                pass
        state_obj = torch_load(checkpoint)
        state = (
            state_obj.get("state_dict", state_obj)
            if isinstance(state_obj, dict)
            else state_obj
        )
        self.load_message = self.model.model.load_state_dict(state, strict=True)

        self.encoder_dim = int(self.model.model.embed_dim)
        self.patch_size = int(raw_patch_size)
        self.image_size = int(image_size)
        self.patch_size_m = int(patch_size_m)
        self.input_norm = str(input_norm or "batch_robust").lower()
        self.doy = int(doy)
        self.s2_indices = [int(i) for i in (s2_indices or self._S2_10_INDICES)]
        self.output_norm = (
            nn.LayerNorm(self.encoder_dim, elementwise_affine=False)
            if token_norm
            else nn.Identity()
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(
                f"AnySat S1 channel mismatch: got {s1.shape[1]}, expected 2"
            )
        if s2.shape[1] <= max(self.s2_indices):
            raise ValueError(
                f"AnySat S2 channel mismatch: got {s2.shape[1]}, need indices {self.s2_indices}"
            )
        s1_in, s2_in = self._normalize_inputs(s1, s2)
        s1_in = resize_image(s1_in, self.image_size)
        s2_in = resize_image(s2_in[:, self.s2_indices], self.image_size)
        target_side = self.image_size // self.patch_size
        sar = align_tokens(self._encode({"s1-asc": s1_in.unsqueeze(1)}), target_side)
        optical = align_tokens(self._encode({"s2": s2_in.unsqueeze(1)}), target_side)
        joint = align_tokens(
            self._encode({"s1-asc": s1_in.unsqueeze(1), "s2": s2_in.unsqueeze(1)}),
            target_side,
        )
        return {
            "SAR_encodings": self.output_norm(sar),
            "optical_encodings": self.output_norm(optical),
            "joint_encodings": self.output_norm(joint),
        }

    def _encode(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        bsz = next(iter(data.values())).shape[0]
        device = next(iter(data.values())).device
        dates = torch.full((bsz, 1), self.doy, dtype=torch.long, device=device)
        if "s1-asc" in data:
            data["s1-asc_dates"] = dates
        if "s2" in data:
            data["s2_dates"] = dates
        out = self.model(data, patch_size=self.patch_size_m, output="patch")
        if out.dim() != 4:
            raise RuntimeError(
                f"Unexpected AnySat patch output shape: {tuple(out.shape)}"
            )
        if out.shape[1] == self.encoder_dim:
            return map_to_tokens(out.contiguous())
        if out.shape[-1] == self.encoder_dim:
            return out.flatten(1, 2).contiguous()
        raise RuntimeError(
            f"Cannot infer AnySat channel dimension from shape: {tuple(out.shape)}"
        )

    def _normalize_inputs(
        self, s1: torch.Tensor, s2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return batch_robust_norm(s1), batch_robust_norm(s2)
        raise ValueError(f"Unknown AnySat input_norm: {self.input_norm}")


def build_anysat_backbone(config: dict[str, Any]) -> AnySatBackbone:
    return AnySatBackbone(
        repo_dir=config["repo_dir"],
        checkpoint=config["checkpoint"],
        image_size=int(config.get("image_size", 96)),
        patch_size_m=int(config.get("patch_size_m", 80)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        doy=int(config.get("doy", 182)),
        s2_indices=config.get("s2_indices"),
        model_size=config.get("model_size", "base"),
    )
