from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class CrossModalResidualMappings(nn.Module):
    """Construct modality-complementary features from paired optical-SAR observations."""

    def __init__(
        self, dim: int, hidden_dim: int | None = None, scale_init: float = 0.35
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or dim)
        self.sar_to_optical = self._make_mapper(dim, hidden_dim)
        self.optical_to_sar = self._make_mapper(dim, hidden_dim)
        self.sar_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.optical_scale = nn.Parameter(torch.tensor(float(scale_init)))

    @staticmethod
    def _make_mapper(dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        target_side: int,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        sar = self._align(tokens.get("sar"), target_side)
        optical = self._align(tokens.get("optical"), target_side)
        dinov3 = self._align(tokens.get("dinov3"), target_side)
        local = self._align(tokens.get("local"), target_side)

        if sar is not None:
            sar_context = self._context_or_zero(local, sar)
            delta = torch.tanh(
                self.sar_to_optical(torch.cat([sar, sar_context], dim=-1))
            )
            out["cf_optical"] = sar + torch.tanh(self.sar_scale).to(
                dtype=sar.dtype
            ) * delta.to(dtype=sar.dtype)

        optical_source = optical if optical is not None else dinov3
        if optical_source is not None:
            opt_context = (
                dinov3
                if dinov3 is not None
                else self._context_or_zero(local, optical_source)
            )
            opt_context = self._align(opt_context, target_side)
            assert opt_context is not None
            delta = torch.tanh(
                self.optical_to_sar(torch.cat([optical_source, opt_context], dim=-1))
            )
            out["cf_sar"] = optical_source + torch.tanh(self.optical_scale).to(
                dtype=optical_source.dtype
            ) * delta.to(dtype=optical_source.dtype)
        return out

    @staticmethod
    def _context_or_zero(
        context: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor:
        if context is None:
            return torch.zeros_like(reference)
        return context

    @staticmethod
    def _align(tokens: torch.Tensor | None, target_side: int) -> torch.Tensor | None:
        if tokens is None:
            return None
        bsz, count, dim = tokens.shape
        if count == target_side * target_side:
            return tokens
        side = int(math.sqrt(count))
        if side * side != count:
            if count == 1:
                return tokens.expand(-1, target_side * target_side, -1).contiguous()
            raise ValueError(
                f"CrossModalResidualMappings expects square token count, got {count}"
            )
        fmap = tokens.transpose(1, 2).reshape(bsz, dim, side, side)
        fmap = F.interpolate(
            fmap, size=(target_side, target_side), mode="bilinear", align_corners=False
        )
        return fmap.flatten(2).transpose(1, 2).contiguous()
