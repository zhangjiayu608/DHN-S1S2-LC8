from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .eo_backbone_utils import align_tokens, batch_robust_norm, load_module_from_file, resize_image, resolve_path


class GalileoBackbone(nn.Module):
    """Galileo encoder adapted to the common S1/S2 token interface."""

    _S2_10_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

    def __init__(
        self,
        repo_dir: str | Path,
        weights_dir: str | Path,
        image_size: int = 96,
        raw_patch_size: int = 8,
        input_norm: str = "official",
        token_norm: bool = True,
        month: int = 6,
        input_resolution_m: int = 10,
        s2_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        repo_dir = resolve_path(repo_dir)
        weights_dir = resolve_path(weights_dir)
        model_file = repo_dir / "single_file_galileo.py"
        if not model_file.exists():
            raise FileNotFoundError(f"Cannot find Galileo single_file_galileo.py: {model_file}")
        if not weights_dir.exists():
            raise FileNotFoundError(f"Cannot find Galileo weights directory: {weights_dir}")

        module = load_module_from_file("_dagpid_galileo_single_file", model_file)
        self.encoder = module.Encoder.load_from_folder(weights_dir, torch.device("cpu"))
        self.encoder_dim = int(self.encoder.embedding_size)
        self.patch_size = int(raw_patch_size)
        self.image_size = int(image_size)
        self.input_norm = str(input_norm or "official").lower()
        self.month = int(month)
        self.input_resolution_m = int(input_resolution_m)
        self.s2_indices = [int(i) for i in (s2_indices or self._S2_10_INDICES)]
        self.output_norm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False) if token_norm else nn.Identity()

        self.space_time_band_count = len(module.SPACE_TIME_BANDS)
        self.space_time_group_count = len(module.SPACE_TIME_BANDS_GROUPS_IDX)
        self.space_band_count = len(module.SPACE_BANDS)
        self.time_band_count = len(module.TIME_BANDS)
        self.static_band_count = len(module.STATIC_BANDS)
        self.space_group_count = len(module.SPACE_BAND_GROUPS_IDX)
        self.time_group_count = len(module.TIME_BAND_GROUPS_IDX)
        self.static_group_count = len(module.STATIC_BAND_GROUPS_IDX)
        self.s1_group = [idx for idx, key in enumerate(module.SPACE_TIME_BANDS_GROUPS_IDX) if key == "S1"]
        self.s2_groups = [idx for idx, key in enumerate(module.SPACE_TIME_BANDS_GROUPS_IDX) if key.startswith("S2")]
        self.s1_band_indices = [module.SPACE_TIME_BANDS.index(name) for name in ("VV", "VH")]
        self.s2_band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
        self.s2_band_indices = [module.SPACE_TIME_BANDS.index(name) for name in self.s2_band_names]

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(f"Galileo S1 channel mismatch: got {s1.shape[1]}, expected 2")
        if s2.shape[1] <= max(self.s2_indices):
            raise ValueError(f"Galileo S2 channel mismatch: got {s2.shape[1]}, need indices {self.s2_indices}")

        s1_in, s2_in = self._normalize_inputs(s1, s2)
        s1_in = resize_image(s1_in, self.image_size)
        s2_in = resize_image(s2_in[:, self.s2_indices], self.image_size)
        sar = self._encode(s1=s1_in, s2=None)
        optical = self._encode(s1=None, s2=s2_in)
        joint = self._encode(s1=s1_in, s2=s2_in)
        target_side = self.image_size // self.patch_size
        return {
            "SAR_encodings": self.output_norm(align_tokens(sar, target_side)),
            "optical_encodings": self.output_norm(align_tokens(optical, target_side)),
            "joint_encodings": self.output_norm(align_tokens(joint, target_side)),
        }

    def _encode(self, s1: torch.Tensor | None, s2: torch.Tensor | None) -> torch.Tensor:
        ref = s1 if s1 is not None else s2
        if ref is None:
            raise ValueError("Galileo requires at least one modality")
        bsz, _, height, width = ref.shape
        dtype, device = ref.dtype, ref.device
        t_steps = 1

        space_time = torch.zeros(
            bsz, height, width, t_steps, self.space_time_band_count, dtype=dtype, device=device
        )
        mask = torch.ones(
            bsz, height, width, t_steps, self.space_time_group_count, dtype=dtype, device=device
        )
        if s1 is not None:
            space_time[:, :, :, 0, self.s1_band_indices] = s1.permute(0, 2, 3, 1)
            mask[:, :, :, :, self.s1_group] = 0
        if s2 is not None:
            space_time[:, :, :, 0, self.s2_band_indices] = s2.permute(0, 2, 3, 1)
            mask[:, :, :, :, self.s2_groups] = 0

        space = torch.zeros(bsz, height, width, self.space_band_count, dtype=dtype, device=device)
        time = torch.zeros(bsz, t_steps, self.time_band_count, dtype=dtype, device=device)
        static = torch.zeros(bsz, self.static_band_count, dtype=dtype, device=device)
        space_mask = torch.ones(bsz, height, width, self.space_group_count, dtype=dtype, device=device)
        time_mask = torch.ones(bsz, t_steps, self.time_group_count, dtype=dtype, device=device)
        static_mask = torch.ones(bsz, self.static_group_count, dtype=dtype, device=device)
        months = torch.full((bsz, t_steps), self.month, dtype=torch.long, device=device)

        out = self.encoder(
            space_time,
            space,
            time,
            static,
            mask,
            space_mask,
            time_mask,
            static_mask,
            months,
            patch_size=self.patch_size,
            input_resolution_m=self.input_resolution_m,
        )
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = out
        return self.encoder.apply_mask_and_average_tokens_per_patch(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
        ).contiguous()

    def _normalize_inputs(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return batch_robust_norm(s1), batch_robust_norm(s2)
        if self.input_norm in {"official", "galileo"}:
            return (s1.float() + 25.0) / 25.0, s2.float() / 10000.0
        raise ValueError(f"Unknown Galileo input_norm: {self.input_norm}")


def build_galileo_backbone(config: dict[str, Any]) -> GalileoBackbone:
    return GalileoBackbone(
        repo_dir=config["repo_dir"],
        weights_dir=config["weights_dir"],
        image_size=int(config.get("image_size", 96)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "official"),
        token_norm=bool(config.get("token_norm", True)),
        month=int(config.get("month", 6)),
        input_resolution_m=int(config.get("input_resolution_m", 10)),
        s2_indices=config.get("s2_indices"),
    )
