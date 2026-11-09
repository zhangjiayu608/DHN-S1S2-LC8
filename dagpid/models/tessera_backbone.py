from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .eo_backbone_utils import batch_robust_norm, resolve_path


_TESSERA_PACKAGE = "_dagpid_tessera_v11"


def _load_tessera_module(repo_dir: Path) -> Any:
    src_dir = repo_dir / "tessera-master" / "tessera_infer_QAT" / "src"
    model_dir = src_dir / "models"
    module_file = model_dir / "ssl_model_v1_1.py"
    if not module_file.exists():
        raise FileNotFoundError(f"Cannot find Tessera ssl_model_v1_1.py: {module_file}")
    pkg = sys.modules.get(_TESSERA_PACKAGE)
    if pkg is None:
        pkg = types.ModuleType(_TESSERA_PACKAGE)
        pkg.__path__ = [str(src_dir)]  # type: ignore[attr-defined]
        sys.modules[_TESSERA_PACKAGE] = pkg
    models_pkg_name = f"{_TESSERA_PACKAGE}.models"
    models_pkg = sys.modules.get(models_pkg_name)
    if models_pkg is None:
        models_pkg = types.ModuleType(models_pkg_name)
        models_pkg.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
        sys.modules[models_pkg_name] = models_pkg

    for name in ("modules", "ssl_model_v1_1"):
        full_name = f"{models_pkg_name}.{name}"
        if full_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(full_name, model_dir / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import Tessera module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{models_pkg_name}.ssl_model_v1_1"]


class TesseraBackbone(nn.Module):
    """TESSERA v1.1 tile representation adapted as 8x8-patch tokens."""

    _S2_10_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

    _CONFIG = {
        "fusion_method": "concat",
        "latent_dim": 192,
        "representation_dim": 192,
        "s2_num_heads": 4,
        "s2_num_layers": 4,
        "s2_dim_feedforward": 2048,
        "s1_num_heads": 4,
        "s1_num_layers": 4,
        "s1_dim_feedforward": 2048,
        "split_s1_modalities": False,
        "num_obs_checkpoints": [8, 16, 24, 32, 40, 48, 56, 64],
    }

    def __init__(
        self,
        repo_dir: str | Path,
        checkpoint: str | Path,
        image_size: int = 96,
        raw_patch_size: int = 8,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        doy: int = 182,
        s2_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        repo_dir = resolve_path(repo_dir)
        checkpoint = resolve_path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot find Tessera checkpoint: {checkpoint}")
        module = _load_tessera_module(repo_dir)
        self.model = module.build_v1_1_inference_model(dict(self._CONFIG), torch.device("cpu"))
        self.load_message = module.load_v1_1_checkpoint(self.model, str(checkpoint))
        missing, unexpected = self.load_message[:2]
        if missing or unexpected:
            raise RuntimeError(f"Tessera encoder state mismatch: missing={missing[:12]}, unexpected={unexpected[:12]}")
        self.encoder_dim = int(self._CONFIG["representation_dim"])
        self.patch_size = int(raw_patch_size)
        self.image_size = int(image_size)
        self.input_norm = str(input_norm or "batch_robust").lower()
        self.doy = int(doy)
        self.s2_indices = [int(i) for i in (s2_indices or self._S2_10_INDICES)]
        self.output_norm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False) if token_norm else nn.Identity()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(f"Tessera S1 channel mismatch: got {s1.shape[1]}, expected 2")
        if s2.shape[1] <= max(self.s2_indices):
            raise ValueError(f"Tessera S2 channel mismatch: got {s2.shape[1]}, need indices {self.s2_indices}")
        s1_in, s2_in = self._normalize_inputs(s1, s2[:, self.s2_indices])
        s1_seq = self._to_sequences(s1_in)
        s2_seq = self._to_sequences(s2_in)
        zeros_s1 = torch.zeros_like(s1_seq)
        zeros_s2 = torch.zeros_like(s2_seq)
        sar = self.output_norm(self._encode(s2_seq=zeros_s2, s1_seq=s1_seq, bsz=s1.shape[0]))
        optical = self.output_norm(self._encode(s2_seq=s2_seq, s1_seq=zeros_s1, bsz=s1.shape[0]))
        joint = self.output_norm(self._encode(s2_seq=s2_seq, s1_seq=s1_seq, bsz=s1.shape[0]))
        return {"SAR_encodings": sar, "optical_encodings": optical, "joint_encodings": joint}

    def _encode(self, s2_seq: torch.Tensor, s1_seq: torch.Tensor, bsz: int) -> torch.Tensor:
        s2_repr = self.model.s2_backbone(s2_seq)
        s1_repr = self.model.s1_backbone(s1_seq)
        fused = self.model.dim_reducer(torch.cat([s2_repr, s1_repr], dim=-1))
        return fused.reshape(bsz, -1, self.encoder_dim).contiguous()

    def _to_sequences(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        pooled = F.avg_pool2d(x, kernel_size=self.patch_size, stride=self.patch_size)
        tokens = pooled.permute(0, 2, 3, 1).reshape(-1, pooled.shape[1])
        doy = torch.full((tokens.shape[0], 1), float(self.doy), dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, doy], dim=-1).unsqueeze(1)

    def _normalize_inputs(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return batch_robust_norm(s1), batch_robust_norm(s2)
        raise ValueError(f"Unknown Tessera input_norm: {self.input_norm}")


def build_tessera_backbone(config: dict[str, Any]) -> TesseraBackbone:
    return TesseraBackbone(
        repo_dir=config["repo_dir"],
        checkpoint=config["checkpoint"],
        image_size=int(config.get("image_size", 96)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        doy=int(config.get("doy", 182)),
        s2_indices=config.get("s2_indices"),
    )
