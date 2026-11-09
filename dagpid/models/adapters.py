from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SensorConditionedAdapter(nn.Module):
    """Sensor-conditioned residual bottleneck adapter with a magnitude gate."""

    def __init__(self, dim: int, rank: int = 64, magnitude_init: float = 1.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.sensor_prompt = nn.Parameter(torch.zeros(1, 1, rank))
        self.magnitude = nn.Parameter(torch.full((1, 1, dim), float(magnitude_init)))
        nn.init.trunc_normal_(self.sensor_prompt, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(tokens)) + self.sensor_prompt))
        return tokens + self.magnitude.to(dtype=residual.dtype) * residual
