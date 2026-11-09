from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .priors import robust_s2_rgb


class LearnedPseudoRGB(nn.Module):
    """Stable 12-channel optical to pseudo-RGB adapter.

    The fixed RGB band selection keeps the foundation encoder sane at step 0.
    By default the residual is disabled/frozen because frozen foundation
    encoders do not backpropagate a useful signal into the spectral adapter.
    """

    def __init__(
        self,
        in_channels: int = 12,
        rgb_indices: list[int] | tuple[int, int, int] = (3, 2, 1),
        residual_scale: float = 0.05,
        train_residual: bool = False,
    ) -> None:
        super().__init__()
        self.rgb_indices = tuple(int(i) for i in rgb_indices)
        self.residual_scale = float(residual_scale)
        self.train_residual = bool(train_residual)
        self.residual = nn.Conv2d(in_channels, 3, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        if not self.train_residual:
            for parameter in self.residual.parameters():
                parameter.requires_grad_(False)

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        base = robust_s2_rgb(s2, self.rgb_indices)
        if self.residual_scale == 0.0:
            return base
        x = s2.float()
        dims = (2, 3)
        x = (x - x.mean(dim=dims, keepdim=True)) / x.std(
            dim=dims, keepdim=True
        ).clamp_min(1e-6)
        delta = torch.tanh(self.residual(x)) * self.residual_scale
        return (base + delta).clamp(0.0, 1.0)


class DINOv3TokenEncoder(nn.Module):
    """Frozen DINOv3-SAT image encoder that returns trainable projected tokens."""

    def __init__(
        self,
        model_dir: str,
        repo_dir: str | None = None,
        input_size: int = 224,
        out_dim: int = 768,
        rgb_indices: list[int] | tuple[int, int, int] = (3, 2, 1),
        freeze: bool = True,
        use_checkpoint: bool = False,
        spectral_residual_scale: float = 0.05,
        train_spectral_residual: bool = False,
    ) -> None:
        super().__init__()
        self.model_dir = str(model_dir)
        self.repo_dir = str(repo_dir or "")
        self.input_size = int(input_size)
        self.freeze = bool(freeze)
        self.use_checkpoint = bool(use_checkpoint)
        self.hidden_size = self._read_hidden_size(Path(model_dir))
        self.num_register_tokens = self._read_num_register_tokens(Path(model_dir))
        self.rgb = LearnedPseudoRGB(
            rgb_indices=rgb_indices,
            residual_scale=spectral_residual_scale,
            train_residual=train_spectral_residual,
        )
        self.model = self._load_model()
        self.model.requires_grad_(not self.freeze)
        if self.freeze:
            self.model.eval()
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, out_dim),
        )
        self.register_buffer(
            "mean", torch.tensor([0.430, 0.411, 0.296]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.213, 0.156, 0.143]).view(1, 3, 1, 1)
        )

    @staticmethod
    def _read_hidden_size(model_dir: Path) -> int:
        import json

        with (model_dir / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("hidden_size", 1024))

    @staticmethod
    def _read_num_register_tokens(model_dir: Path) -> int:
        import json

        with (model_dir / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("num_register_tokens", 0))

    def _load_model(self) -> nn.Module:
        try:
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover - server dependency
            raise ImportError(
                "DINOv3TokenEncoder requires transformers and local DINOv3 weights. "
                "Disable model.external_priors.dinov3.enabled if the environment is not ready."
            ) from exc
        return AutoModel.from_pretrained(self.model_dir, local_files_only=True)

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            self.model.eval()
        rgb = self.rgb(s2)
        rgb = F.interpolate(
            rgb,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        rgb = (rgb - self.mean.to(rgb.dtype)) / self.std.to(rgb.dtype)
        ctx = torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze)
        with ctx:
            if self.use_checkpoint and not self.freeze:

                def run_model(pixel_values: torch.Tensor) -> torch.Tensor:
                    return self.model(pixel_values=pixel_values).last_hidden_state

                hidden = torch.utils.checkpoint.checkpoint(
                    run_model, rgb, use_reentrant=False
                )
            else:
                outputs = self.model(pixel_values=rgb)
                hidden = getattr(outputs, "last_hidden_state", outputs[0])
        patch_tokens = hidden[:, 1 + self.num_register_tokens :, :]
        side = int(math.sqrt(patch_tokens.shape[1]))
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError(
                f"DINOv3 patch token count is not square: {patch_tokens.shape[1]}"
            )
        if self.freeze:
            patch_tokens = patch_tokens.detach()
        return self.proj(patch_tokens.float()).to(dtype=s2.dtype)
