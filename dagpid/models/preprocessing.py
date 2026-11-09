from __future__ import annotations

import torch
from torch import nn


def normalize_local(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    mean = image.mean(dim=(2, 3), keepdim=True)
    std = image.std(dim=(2, 3), keepdim=True, correction=0).clamp_min(1e-6)
    return ((image - mean) / std).clamp(-5, 5)


class CromaNormalizer(nn.Module):
    """CROMA's mean ± 2 std clipping with fixed training-pool statistics."""

    def __init__(self, statistics: dict | None = None) -> None:
        super().__init__()
        statistics = statistics or {}
        if statistics and not {"s1", "s2"}.issubset(statistics):
            raise ValueError("Normalization statistics must cover both sensors.")
        for sensor, channels in (("s1", 2), ("s2", 12)):
            values = statistics.get(sensor, {})
            mean = torch.tensor(values.get("mean", [0.0] * channels), dtype=torch.float32)
            std = torch.tensor(values.get("std", [1.0] * channels), dtype=torch.float32)
            if mean.shape != (channels,) or std.shape != (channels,):
                raise ValueError(f"Invalid {sensor} normalization channel count.")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError(f"Invalid {sensor} normalization statistics.")
            self.register_buffer(f"{sensor}_mean", mean.view(1, channels, 1, 1))
            self.register_buffer(f"{sensor}_std", std.view(1, channels, 1, 1))
        self.register_buffer("ready", torch.tensor(bool(statistics), dtype=torch.bool))
        self._ready = bool(statistics)
        self.register_load_state_dict_post_hook(self._refresh_ready)

    def _refresh_ready(self, module: nn.Module, incompatible_keys) -> None:
        self._ready = bool(self.ready.cpu())

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._ready:
            raise RuntimeError("CROMA requires training-pool preprocessing statistics or a checkpoint containing them.")
        normalized = []
        for sensor, image in (("s1", s1), ("s2", s2)):
            mean, std = getattr(self, f"{sensor}_mean"), getattr(self, f"{sensor}_std")
            if image.ndim != 4 or image.shape[1] != mean.shape[1]:
                raise ValueError(f"Invalid {sensor} input shape: {tuple(image.shape)}")
            normalized.append(((image.float() - mean) / (4 * std) + 0.5).clamp(0, 1))
        return normalized[0], normalized[1]
