from __future__ import annotations

import torch


def robust_s2_rgb(
    s2: torch.Tensor,
    indices: list[int] | tuple[int, int, int] = (3, 2, 1),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert 12-channel S2 to pseudo-RGB and normalize per sample/channel."""
    idx = torch.as_tensor(indices, device=s2.device, dtype=torch.long)
    rgb = s2.index_select(1, idx).float()
    dims = (2, 3)
    mean = rgb.mean(dim=dims, keepdim=True)
    std = rgb.std(dim=dims, keepdim=True).clamp_min(eps)
    lo = mean - 2.5 * std
    hi = mean + 2.5 * std
    return ((rgb - lo) / (hi - lo).clamp_min(eps)).clamp(0.0, 1.0)
