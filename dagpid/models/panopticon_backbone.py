from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


_PANOPTICON_MODULE = "_dagpid_panopticon_hubconf"


def _load_panopticon_hubconf(repo_dir: Path) -> Any:
    hubconf_file = repo_dir / "hubconf.py"
    if not hubconf_file.exists():
        raise FileNotFoundError(f"Cannot find Panopticon hubconf.py under {repo_dir}")

    module = sys.modules.get(_PANOPTICON_MODULE)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(_PANOPTICON_MODULE, hubconf_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Panopticon hubconf from {hubconf_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PANOPTICON_MODULE] = module
    sys.path.insert(0, str(repo_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(repo_dir))
        except ValueError:
            pass
    return module


def _torch_load_weights(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_panopticon_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    state_dict = checkpoint_obj.get("model", checkpoint_obj)
    for prefix in ("teacher.backbone.", "student.backbone.", "backbone."):
        selected = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if selected:
            return selected
    return state_dict


def _as_float_list(value: Any, *, name: str) -> list[float]:
    if value is None:
        raise ValueError(f"Missing required Panopticon list: {name}")
    return [float(item) for item in value]


class PanopticonBackbone(nn.Module):
    """Panopticon ViT-B/14 encoder adapted to the CROMA token interface."""

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        target_image_size: int = 168,
        raw_patch_size: int = 8,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        s1_channel_ids: list[float] | None = None,
        s2_channel_ids: list[float] | None = None,
        joint_mode: str = "multimodal",
    ) -> None:
        super().__init__()
        repo_dir = Path(repo_dir).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot find Panopticon checkpoint: {checkpoint}")

        self.encoder_dim = 768
        self.patch_size = int(raw_patch_size)
        self.model_patch_size = 14
        self.target_image_size = int(target_image_size)
        self.input_norm = str(input_norm or "batch_robust").lower()
        self.joint_mode = str(joint_mode or "multimodal").lower()
        self.s1_channel_ids = _as_float_list(s1_channel_ids, name="s1_channel_ids")
        self.s2_channel_ids = _as_float_list(s2_channel_ids, name="s2_channel_ids")

        module = _load_panopticon_hubconf(repo_dir)
        self.model = module._panopticon_vitb14()
        checkpoint_obj = _torch_load_weights(checkpoint)
        state_dict = _extract_panopticon_state_dict(checkpoint_obj)
        self.load_message = self.model.load_state_dict(state_dict, strict=True)
        self.output_norm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False) if token_norm else nn.Identity()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != len(self.s1_channel_ids):
            raise ValueError(f"Panopticon S1 channel mismatch: got {s1.shape[1]}, expected {len(self.s1_channel_ids)}")
        if s2.shape[1] != len(self.s2_channel_ids):
            raise ValueError(f"Panopticon S2 channel mismatch: got {s2.shape[1]}, expected {len(self.s2_channel_ids)}")

        s1_in, s2_in = self._normalize_inputs(s1, s2)
        sar = self._encode(s1_in, self.s1_channel_ids)
        optical = self._encode(s2_in, self.s2_channel_ids)
        if sar.shape[1] != optical.shape[1]:
            optical = self._align_tokens(optical, int(math.sqrt(sar.shape[1])))
        joint = self._joint_tokens(s1_in, s2_in, sar, optical)
        return {
            "SAR_encodings": self.output_norm(sar),
            "optical_encodings": self.output_norm(optical),
            "joint_encodings": self.output_norm(joint),
        }

    def _encode(self, x: torch.Tensor, channel_ids: list[float]) -> torch.Tensor:
        x = self._resize_input(x)
        ids = torch.tensor(channel_ids, device=x.device, dtype=torch.float32).view(1, -1).repeat(x.shape[0], 1)
        out = self.model.forward_features({"imgs": x.float(), "chn_ids": ids})
        return out["x_norm_patchtokens"].contiguous()

    def _joint_tokens(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> torch.Tensor:
        if self.joint_mode in {"multimodal", "joint", "concat"}:
            return self._encode(torch.cat([s1, s2], dim=1), self.s1_channel_ids + self.s2_channel_ids)
        if self.joint_mode in {"average", "mean", "avg"}:
            return 0.5 * (sar + optical)
        raise ValueError(f"Unknown Panopticon joint_mode: {self.joint_mode}")

    def _normalize_inputs(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return self._batch_robust_norm(s1), self._batch_robust_norm(s2)
        if self.input_norm in {"batch_zscore", "zscore"}:
            return self._batch_zscore(s1), self._batch_zscore(s2)
        raise ValueError(f"Unknown Panopticon input_norm: {self.input_norm}")

    def _resize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_image_size <= 0 or x.shape[-2:] == (self.target_image_size, self.target_image_size):
            return x
        return F.interpolate(x, size=(self.target_image_size, self.target_image_size), mode="bilinear", align_corners=False)

    @staticmethod
    def _batch_robust_norm(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        dims = (2, 3)
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True).clamp_min(1e-6)
        lo = mean - 2 * std
        hi = mean + 2 * std
        return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

    @staticmethod
    def _batch_zscore(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        dims = (2, 3)
        return (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _align_tokens(tokens: torch.Tensor, target_side: int) -> torch.Tensor:
        bsz, count, dim = tokens.shape
        side = int(math.sqrt(count))
        if side == target_side:
            return tokens
        fmap = tokens.transpose(1, 2).reshape(bsz, dim, side, side)
        fmap = F.interpolate(fmap, size=(target_side, target_side), mode="bilinear", align_corners=False)
        return fmap.flatten(2).transpose(1, 2).contiguous()


def build_panopticon_backbone(config: dict[str, Any]) -> PanopticonBackbone:
    return PanopticonBackbone(
        repo_dir=config["repo_dir"],
        checkpoint=config["checkpoint"],
        target_image_size=int(config.get("target_image_size", 168)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        s1_channel_ids=config.get("s1_channel_ids", [-1, -2]),
        s2_channel_ids=config.get(
            "s2_channel_ids",
            [442.923, 492.997, 559.599, 664.630, 704.006, 740.552, 782.419, 827.539, 864.780, 945.029, 1613.862, 2203.618],
        ),
        joint_mode=config.get("joint_mode", "multimodal"),
    )
