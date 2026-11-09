from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .eo_backbone_utils import load_encoder_weights

_CFM_PACKAGE = "_dagpid_copernicusfm_official"


def _load_copernicus_model_module(repo_dir: Path) -> Any:
    src_dir = repo_dir / "src"
    model_file = src_dir / "model_vit.py"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Cannot find Copernicus-FM model_vit.py under {src_dir}"
        )

    if _CFM_PACKAGE not in sys.modules:
        package = types.ModuleType(_CFM_PACKAGE)
        package.__path__ = [str(src_dir)]  # type: ignore[attr-defined]
        sys.modules[_CFM_PACKAGE] = package

    module_name = f"{_CFM_PACKAGE}.model_vit"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Copernicus-FM model module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(src_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(src_dir))
        except ValueError:
            pass
    return module


def _as_float_list(value: Any, *, name: str) -> list[float]:
    if value is None:
        raise ValueError(f"Missing required Copernicus-FM list: {name}")
    return [float(item) for item in value]


class CopernicusFMBackbone(nn.Module):
    """Copernicus-FM encoder adapted to the CROMA token interface."""

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        size: str = "base",
        kernel_size: int = 16,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        s1_wavelengths: list[float] | None = None,
        s1_bandwidths: list[float] | None = None,
        s2_wavelengths: list[float] | None = None,
        s2_bandwidths: list[float] | None = None,
    ) -> None:
        super().__init__()
        repo_dir = Path(repo_dir).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Cannot find Copernicus-FM checkpoint: {checkpoint}"
            )

        self.kernel_size = int(kernel_size)
        self.input_norm = str(input_norm or "none").lower()
        self.patch_size = self.kernel_size
        self.s1_wavelengths = _as_float_list(s1_wavelengths, name="s1_wavelengths")
        self.s1_bandwidths = _as_float_list(s1_bandwidths, name="s1_bandwidths")
        self.s2_wavelengths = _as_float_list(s2_wavelengths, name="s2_wavelengths")
        self.s2_bandwidths = _as_float_list(s2_bandwidths, name="s2_bandwidths")

        size = str(size or "base").lower()
        dims = {"small": 384, "base": 768, "large": 1024}
        if size not in dims:
            raise ValueError(f"Unsupported Copernicus-FM size: {size}")
        self.encoder_dim = dims[size]

        module = _load_copernicus_model_module(repo_dir)
        builder = getattr(module, f"vit_{size}_patch16")
        last_block = {"small": 11, "base": 11, "large": 23}[size]
        self.model = builder(
            num_classes=0,
            global_pool=False,
            return_intermediate=True,
            intermediate_indices=[last_block],
        )
        self.model.norm = nn.Identity()

        checkpoint_obj = torch.load(checkpoint, map_location="cpu")
        state_dict = (
            checkpoint_obj.get("model", checkpoint_obj)
            if isinstance(checkpoint_obj, dict)
            else checkpoint_obj
        )
        self.load_message = load_encoder_weights(self.model, state_dict)
        self.output_norm = (
            nn.LayerNorm(self.encoder_dim, elementwise_affine=False)
            if token_norm
            else nn.Identity()
        )

        self.register_buffer(
            "s1_mean",
            torch.tensor([-12.548, -20.192]).view(1, 2, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "s1_std", torch.tensor([5.257, 5.912]).view(1, 2, 1, 1), persistent=False
        )
        self.register_buffer(
            "s2_mean",
            torch.tensor(
                [
                    1353.7,
                    1117.2,
                    1041.8,
                    946.5,
                    1199.1,
                    2003.0,
                    2374.0,
                    2301.2,
                    2599.7,
                    732.1,
                    1820.6,
                    1118.2,
                ]
            ).view(1, 12, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "s2_std",
            torch.tensor(
                [
                    897.3,
                    736.0,
                    684.8,
                    620.0,
                    791.9,
                    1341.3,
                    1595.4,
                    1545.5,
                    1750.1,
                    475.1,
                    1216.5,
                    736.7,
                ]
            ).view(1, 12, 1, 1),
            persistent=False,
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != len(self.s1_wavelengths):
            raise ValueError(
                f"Copernicus-FM S1 channel mismatch: got {s1.shape[1]}, expected {len(self.s1_wavelengths)}"
            )
        if s2.shape[1] != len(self.s2_wavelengths):
            raise ValueError(
                f"Copernicus-FM S2 channel mismatch: got {s2.shape[1]}, expected {len(self.s2_wavelengths)}"
            )

        s1_in, s2_in = self._normalize_inputs(s1, s2)
        sar = self._encode_spectral(s1_in, self.s1_wavelengths, self.s1_bandwidths)
        optical = self._encode_spectral(s2_in, self.s2_wavelengths, self.s2_bandwidths)
        if sar.shape[1] != optical.shape[1]:
            optical = self._align_tokens(optical, int(math.sqrt(sar.shape[1])))
        joint = 0.5 * (sar + optical)
        return {
            "SAR_encodings": self.output_norm(sar),
            "optical_encodings": self.output_norm(optical),
            "joint_encodings": self.output_norm(joint),
        }

    def _encode_spectral(
        self, x: torch.Tensor, wavelengths: list[float], bandwidths: list[float]
    ) -> torch.Tensor:
        meta = torch.full(
            (x.shape[0], 4), float("nan"), device=x.device, dtype=torch.float32
        )
        _, features = self.model(
            x.float(),
            meta,
            wavelengths,
            bandwidths,
            None,
            "spectral",
            self.kernel_size,
        )
        if not features:
            raise RuntimeError("Copernicus-FM did not return intermediate features")
        fmap = features[-1]
        return fmap.flatten(2).transpose(1, 2).contiguous()

    def _normalize_inputs(
        self, s1: torch.Tensor, s2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return self._batch_robust_norm(s1), self._batch_robust_norm(s2)
        if self.input_norm in {"copernicus", "copernicus_bench", "fixed"}:
            return self._fixed_norm(
                s1.float(), self.s1_mean, self.s1_std
            ), self._fixed_norm(s2.float(), self.s2_mean, self.s2_std)
        raise ValueError(f"Unknown Copernicus-FM input_norm: {self.input_norm}")

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
    def _fixed_norm(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        lo = mean.to(device=x.device, dtype=x.dtype) - 2 * std.to(
            device=x.device, dtype=x.dtype
        )
        hi = mean.to(device=x.device, dtype=x.dtype) + 2 * std.to(
            device=x.device, dtype=x.dtype
        )
        return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

    @staticmethod
    def _align_tokens(tokens: torch.Tensor, target_side: int) -> torch.Tensor:
        bsz, count, dim = tokens.shape
        side = int(math.sqrt(count))
        if side == target_side:
            return tokens
        fmap = tokens.transpose(1, 2).reshape(bsz, dim, side, side)
        fmap = torch.nn.functional.interpolate(
            fmap, size=(target_side, target_side), mode="bilinear", align_corners=False
        )
        return fmap.flatten(2).transpose(1, 2).contiguous()


def build_copernicusfm_backbone(config: dict[str, Any]) -> CopernicusFMBackbone:
    return CopernicusFMBackbone(
        repo_dir=config["repo_dir"],
        checkpoint=config["checkpoint"],
        size=config.get("size", "base"),
        kernel_size=int(config.get("kernel_size", 16)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        s1_wavelengths=config.get("s1_wavelengths", [50000000, 50000000]),
        s1_bandwidths=config.get("s1_bandwidths", [1e9, 1e9]),
        s2_wavelengths=config.get(
            "s2_wavelengths",
            [440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1610, 2190],
        ),
        s2_bandwidths=config.get(
            "s2_bandwidths", [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 90, 180]
        ),
    )
