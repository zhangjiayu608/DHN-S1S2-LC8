from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .eo_backbone_utils import batch_robust_norm, extract_state_dict, load_encoder_weights, resize_image, resolve_path, torch_load


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dims: int, patch_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2).contiguous()


class _Attention(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class _FFN(nn.Module):
    def __init__(self, embed_dims: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden = int(embed_dims * mlp_ratio)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(embed_dims, hidden), nn.GELU()), nn.Linear(hidden, embed_dims)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[1](self.layers[0](x))


class _Layer(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dims)
        self.attn = _Attention(embed_dims, num_heads)
        self.ln2 = nn.LayerNorm(embed_dims)
        self.ffn = _FFN(embed_dims, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _SkySenseViT(nn.Module):
    def __init__(self, in_channels: int, patch_size: int = 4, embed_dims: int = 1024, num_layers: int = 24, num_heads: int = 16, position_tokens: int = 257, with_cls_token: bool = True) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dims = int(embed_dims)
        self.with_cls_token = bool(with_cls_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, position_tokens, embed_dims))
        self.patch_embed = _PatchEmbed(in_channels, embed_dims, self.patch_size)
        self.layers = nn.ModuleList([_Layer(embed_dims, num_heads, 4) for _ in range(num_layers)])

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        tokens = self.patch_embed(x)
        if self.with_cls_token:
            tokens = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        tokens = tokens + self._interpolate_pos(tokens, height, width)
        for layer in self.layers:
            tokens = layer(tokens)
        return (tokens[:, 1:] if self.with_cls_token else tokens).contiguous()

    def _interpolate_pos(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        target_h = height // self.patch_size
        target_w = width // self.patch_size
        pos = self.pos_embed
        if pos.shape[1] == tokens.shape[1]:
            return pos
        has_cls_position = int(math.sqrt(pos.shape[1] - 1)) ** 2 == pos.shape[1] - 1
        cls_pos = pos[:, :1] if has_cls_position else torch.zeros_like(pos[:, :1])
        patch_pos = pos[:, 1:] if has_cls_position else pos
        side = int(math.sqrt(patch_pos.shape[1]))
        if side * side != patch_pos.shape[1]:
            raise RuntimeError(f"Cannot interpolate SkySense pos_embed with shape {tuple(pos.shape)}")
        patch_pos = patch_pos.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(target_h, target_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, -1)
        return (torch.cat([cls_pos, patch_pos], dim=1) if self.with_cls_token else patch_pos).to(tokens.dtype)


class SkySenseBackbone(nn.Module):
    """SkySense S1/S2 ViT backbone pair adapted without mmcv/mmseg runtime deps."""

    def __init__(
        self,
        sar_checkpoint: str | Path,
        optical_checkpoint: str | Path,
        target_image_size: int = 48,
        raw_patch_size: int = 8,
        input_norm: str = "batch_robust",
        token_norm: bool = True,
        s2_indices: list[int] | None = None,
        joint_mode: str = "average",
        with_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_dim = 1024
        self.patch_size = int(raw_patch_size)
        self.target_image_size = int(target_image_size)
        self.input_norm = str(input_norm or "batch_robust").lower()
        self.s2_indices = [int(i) for i in (s2_indices or [1, 2, 3, 4, 5, 6, 7, 8, 10, 11])]
        self.joint_mode = str(joint_mode or "average").lower()
        self.sar_model, self.sar_load_message = self._build_encoder(2, sar_checkpoint, with_cls_token)
        self.optical_model, self.optical_load_message = self._build_encoder(len(self.s2_indices), optical_checkpoint, with_cls_token)
        self.output_norm = nn.LayerNorm(self.encoder_dim, elementwise_affine=False) if token_norm else nn.Identity()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> dict[str, torch.Tensor]:
        if s1.shape[1] != 2:
            raise ValueError(f"SkySense S1 channel mismatch: got {s1.shape[1]}, expected 2")
        if s2.shape[1] <= max(self.s2_indices):
            raise ValueError(f"SkySense S2 channel mismatch: got {s2.shape[1]}, need indices {self.s2_indices}")
        s1_in, s2_in = self._normalize_inputs(s1, s2[:, self.s2_indices])
        s1_in = resize_image(s1_in, self.target_image_size)
        s2_in = resize_image(s2_in, self.target_image_size)
        sar = self.output_norm(self.sar_model.forward_tokens(s1_in.float()))
        optical = self.output_norm(self.optical_model.forward_tokens(s2_in.float()))
        if self.joint_mode in {"average", "mean", "avg"}:
            joint = 0.5 * (sar + optical)
        elif self.joint_mode == "sar":
            joint = sar
        elif self.joint_mode == "optical":
            joint = optical
        else:
            raise ValueError(f"Unknown SkySense joint_mode: {self.joint_mode}")
        return {"SAR_encodings": sar, "optical_encodings": optical, "joint_encodings": joint}

    def _normalize_inputs(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_norm in {"none", "off", "false"}:
            return s1.float(), s2.float()
        if self.input_norm in {"batch", "batch_robust", "robust"}:
            return batch_robust_norm(s1), batch_robust_norm(s2)
        raise ValueError(f"Unknown SkySense input_norm: {self.input_norm}")

    @staticmethod
    def _build_encoder(in_channels: int, checkpoint: str | Path, with_cls_token: bool):
        state = extract_state_dict(torch_load(resolve_path(checkpoint)))
        model = _SkySenseViT(in_channels=in_channels, position_tokens=state["pos_embed"].shape[1], with_cls_token=with_cls_token)
        result = load_encoder_weights(model, state, ignored_unexpected_prefixes=("mask_token", "vocabulary_token", "vocabulary_weight"))
        return model, result


def build_skysense_backbone(config: dict[str, Any]) -> SkySenseBackbone:
    return SkySenseBackbone(
        sar_checkpoint=config["sar_checkpoint"],
        optical_checkpoint=config["optical_checkpoint"],
        target_image_size=int(config.get("target_image_size", 48)),
        raw_patch_size=int(config.get("raw_patch_size", 8)),
        input_norm=config.get("input_norm", "batch_robust"),
        token_norm=bool(config.get("token_norm", True)),
        s2_indices=config.get("s2_indices", [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]),
        joint_mode=config.get("joint_mode", "average"),
        with_cls_token=bool(config.get("with_cls_token", True)),
    )
