from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import types
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .eo_backbone_utils import load_encoder_weights


_TM_PACKAGE = "_dagpid_terramind_official"
_TM_UTILS_STUB = "terratorch.models.backbones.terramind.utils"


def _ensure_package(name: str, path: Path | None = None) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if path is not None:
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
    return module


def _install_terramind_stubs() -> None:
    # TerraMind's modality_info imports this utility through the terratorch
    # package path. Stubbing the narrow module avoids importing all TerraTorch.
    _ensure_package("terratorch")
    _ensure_package("terratorch.models")
    _ensure_package("terratorch.models.backbones")
    _ensure_package("terratorch.models.backbones.terramind")
    if _TM_UTILS_STUB not in sys.modules:
        utils = types.ModuleType(_TM_UTILS_STUB)

        def generate_uint15_hash(seed_str: str) -> int:
            return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (2**15)

        utils.generate_uint15_hash = generate_uint15_hash  # type: ignore[attr-defined]
        sys.modules[_TM_UTILS_STUB] = utils


def _load_terramind_vit_module(repo_dir: Path) -> Any:
    terramind_dir = repo_dir / "terratorch" / "models" / "backbones" / "terramind"
    model_dir = terramind_dir / "model"
    model_file = model_dir / "terramind_vit.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Cannot find TerraMind terramind_vit.py under {model_dir}")

    _install_terramind_stubs()
    _ensure_package(_TM_PACKAGE, terramind_dir)
    _ensure_package(f"{_TM_PACKAGE}.model", model_dir)

    module_name = f"{_TM_PACKAGE}.model.terramind_vit"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import TerraMind ViT module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _torch_load_weights(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TerraMindBackbone(nn.Module):
    """TerraMind ViT encoder adapted to the CROMA token interface."""

    _SIZE_CONFIGS: dict[str, dict[str, Any]] = {
        "tiny": {
            "encoder_depth": 12,
            "dim": 192,
            "num_heads": 3,
            "qkv_bias": True,
            "proj_bias": True,
            "mlp_bias": True,
            "gated_mlp": False,
            "act": "gelu",
        },
        "small": {
            "encoder_depth": 12,
            "dim": 384,
            "num_heads": 6,
            "qkv_bias": True,
            "proj_bias": True,
            "mlp_bias": True,
            "gated_mlp": False,
            "act": "gelu",
        },
        "base": {
            "encoder_depth": 12,
            "dim": 768,
            "num_heads": 12,
            "qkv_bias": False,
            "proj_bias": False,
            "mlp_bias": False,
            "gated_mlp": True,
            "act": "silu",
        },
        "large": {
            "encoder_depth": 24,
            "dim": 1024,
            "num_heads": 16,
            "qkv_bias": False,
            "proj_bias": False,
            "mlp_bias": False,
            "gated_mlp": True,
            "act": "silu",
        },
    }

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        size: str = "base",
        image_size: int = 224,
        target_image_size: int = 192,
        model_patch_size: int = 16,
        raw_patch_size: int = 8,
        input_norm: str = "terramind",
        token_norm: bool = True,
        sar_modality: str = "S1GRD",
        optical_modality: str = "S2L2A",
        joint_mode: str = "average",
    ) -> None:
        super().__init__()
        repo_dir = Path(repo_dir).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot find TerraMind checkpoint: {checkpoint}")

        size = str(size or "base").lower()
        if size not in self._SIZE_CONFIGS:
            raise ValueError(f"Unsupported TerraMind size: {size}")
        size_cfg = dict(self._SIZE_CONFIGS[size])
        self.encoder_dim = int(size_cfg.pop("dim"))
        self.patch_size = int(raw_patch_size)
        self.model_patch_size = int(model_patch_size)
        self.target_image_size = int(target_image_size)
        self.input_norm = str(input_norm or "terramind").lower()
        self.sar_modality = str(sar_modality)
        if self.sar_modality not in {"S1GRD", "S1RTC"} or optical_modality != "S2L2A":
            raise ValueError("The 12-band interface supports S2L2A with S1GRD or S1RTC.")
        self.optical_modality = str(optical_modality)
        self.joint_mode = str(joint_mode or "average").lower()

        module = _load_terramind_vit_module(repo_dir)
        TerraMindViT = module.TerraMindViT
        LayerNorm = module.LayerNorm
        act_layer = nn.GELU if size_cfg.pop("act") == "gelu" else nn.SiLU
        self.model = TerraMindViT(
            img_size=int(image_size),
            modalities=[self.optical_modality, self.sar_modality],
            merge_method="dict",
            patch_size=self.model_patch_size,
            dim=self.encoder_dim,
            mlp_ratio=4,
            num_register_tokens=0,
            drop_path_rate=0.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            modality_drop_rate=0.0,
            act_layer=act_layer,
            norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
            encoder_norm=True,
            tokenizer_dict=None,
            pretrained=False,
            **size_cfg,
        )
        checkpoint_obj = _torch_load_weights(checkpoint)
        state_dict = checkpoint_obj.get("model", checkpoint_obj) if isinstance(checkpoint_obj, dict) else checkpoint_obj
        self.load_message = load_encoder_weights(
            self.model, state_dict, ignored_prefixes=("tokenizer",),
            ignored_unexpected_prefixes=("decoder.", "decoder_embeddings.", "decoder_norm.", "decoder_proj_context.", "mask_token", "encoder_embeddings."),
        )
        self.output_norm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False) if token_norm else nn.Identity()

        sar_mean = [-10.930, -17.329] if self.sar_modality == "S1RTC" else [-12.599, -20.293]
        sar_std = [4.391, 4.459] if self.sar_modality == "S1RTC" else [5.195, 5.890]
        self.register_buffer("s1_mean", torch.tensor(sar_mean).view(1, 2, 1, 1))
        self.register_buffer("s1_std", torch.tensor(sar_std).view(1, 2, 1, 1))
        self.register_buffer(
            "s2_mean",
            torch.tensor(
                [
                    1390.458,
                    1503.317,
                    1718.197,
                    1853.910,
                    2199.100,
                    2779.975,
                    2987.011,
                    3083.234,
                    3132.220,
                    3162.988,
                    2424.884,
                    1857.648,
                ]
            ).view(1, 12, 1, 1),
        )
        self.register_buffer(
            "s2_std",
            torch.tensor(
                [
                    2106.761,
                    2141.107,
                    2038.973,
                    2134.138,
                    2085.321,
                    1889.926,
                    1820.257,
                    1871.918,
                    1753.829,
                    1797.379,
                    1434.261,
                    1334.311,
                ]
            ).view(1, 12, 1, 1),
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(f"TerraMind S1 channel mismatch: got {s1.shape[1]}, expected 2")
        if s2.shape[1] != 12:
            raise ValueError(f"TerraMind S2 channel mismatch: got {s2.shape[1]}, expected 12")

        s1_in, s2_in = self._normalize_inputs(s1, s2)
        s1_in = self._resize_input(s1_in)
        s2_in = self._resize_input(s2_in)
        sar = self._encode_one(self.sar_modality, s1_in)
        optical = self._encode_one(self.optical_modality, s2_in)
        if sar.shape[1] != optical.shape[1]:
            optical = self._align_tokens(optical, int(math.sqrt(sar.shape[1])))
        joint = self._joint_tokens(sar, optical, s1_in, s2_in)
        return {
            "SAR_encodings": self.output_norm(sar),
            "optical_encodings": self.output_norm(optical),
            "joint_encodings": self.output_norm(joint),
        }

    def _encode_one(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model({modality: x})
        last = outputs[-1]
        if not isinstance(last, dict) or modality not in last:
            raise RuntimeError(f"TerraMind did not return tokens for modality {modality}")
        return last[modality].contiguous()

    def _joint_tokens(self, sar: torch.Tensor, optical: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        if self.joint_mode in {"average", "mean", "avg"}:
            return 0.5 * (sar + optical)
        if self.joint_mode in {"multimodal", "cross", "joint"}:
            outputs = self.model({self.sar_modality: s1, self.optical_modality: s2})
            last = outputs[-1]
            if not isinstance(last, dict) or self.sar_modality not in last or self.optical_modality not in last:
                raise RuntimeError("TerraMind did not return multimodal joint tokens")
            return 0.5 * (last[self.sar_modality] + last[self.optical_modality])
        raise ValueError(f"Unknown TerraMind joint_mode: {self.joint_mode}")

    def _normalize_inputs(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"terramind", "fixed", "official"}:
            return self._fixed_zscore(s1.float(), self.s1_mean, self.s1_std), self._fixed_zscore(
                s2.float(), self.s2_mean, self.s2_std
            )
        if self.input_norm in {"batch", "batch_zscore", "zscore"}:
            return self._batch_zscore(s1), self._batch_zscore(s2)
        if self.input_norm in {"batch_robust", "robust"}:
            return self._batch_robust_norm(s1), self._batch_robust_norm(s2)
        raise ValueError(f"Unknown TerraMind input_norm: {self.input_norm}")

    def _resize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_image_size <= 0 or x.shape[-2:] == (self.target_image_size, self.target_image_size):
            return x
        return F.interpolate(x, size=(self.target_image_size, self.target_image_size), mode="bilinear", align_corners=False)

    @staticmethod
    def _fixed_zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype).clamp_min(1e-6)

    @staticmethod
    def _batch_zscore(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        dims = (2, 3)
        return (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True).clamp_min(1e-6)

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
    def _align_tokens(tokens: torch.Tensor, target_side: int) -> torch.Tensor:
        bsz, count, dim = tokens.shape
        side = int(math.sqrt(count))
        if side == target_side:
            return tokens
        fmap = tokens.transpose(1, 2).reshape(bsz, dim, side, side)
        fmap = F.interpolate(fmap, size=(target_side, target_side), mode="bilinear", align_corners=False)
        return fmap.flatten(2).transpose(1, 2).contiguous()


def build_terramind_backbone(config: dict[str, Any]) -> TerraMindBackbone:
    return TerraMindBackbone(
        repo_dir=config["repo_dir"],
        checkpoint=config["checkpoint"],
        size=config.get("size", "base"),
        image_size=int(config.get("image_size", 224)),
        target_image_size=int(config.get("target_image_size", 192)),
        model_patch_size=int(config.get("model_patch_size", 16)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "terramind"),
        token_norm=bool(config.get("token_norm", True)),
        sar_modality=config.get("sar_modality", "S1GRD"),
        optical_modality=config.get("optical_modality", "S2L2A"),
        joint_mode=config.get("joint_mode", "average"),
    )
