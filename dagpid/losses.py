from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DAGPIDLoss(nn.Module):
    """Weighted segmentation CE, complementary regression, and two graph priors."""

    def __init__(
        self,
        ignore_index: int = 255,
        class_weights: list[float] | None = None,
        complementary_weight: float = 0.02,
        graph_prior_weight: float = 3e-4,
        graph_sparsity_weight: float = 1e-5,
        schedule: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        weights = (
            None
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32)
        )
        if weights is not None and (
            weights.ndim != 1
            or not torch.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("Class weights must be a finite, positive vector.")
        self.register_buffer("class_weights", weights)
        self.weights = {
            "complementary": float(complementary_weight),
            "graph_prior": float(graph_prior_weight),
            "graph_sparsity": float(graph_sparsity_weight),
        }
        if any(
            not math.isfinite(value) or value < 0 for value in self.weights.values()
        ):
            raise ValueError("Loss weights must be finite and nonnegative.")
        self.schedule = dict(schedule or {})
        allowed = {
            f"{name}_{suffix}"
            for name in self.weights
            for suffix in ("start_epoch", "ramp_epochs")
        }
        if set(self.schedule) - allowed:
            raise ValueError(
                f"Unknown loss schedule keys: {sorted(set(self.schedule) - allowed)}"
            )
        if any(int(value) < 1 for value in self.schedule.values()):
            raise ValueError("Loss schedule epochs must be positive.")
        self.current_epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    @property
    def requires_aux(self) -> bool:
        return any(self.weights.values())

    def _scheduled_weight(self, name: str) -> float:
        start = int(self.schedule.get(f"{name}_start_epoch", 1))
        ramp = int(self.schedule.get(f"{name}_ramp_epochs", 1))
        fraction = min(1.0, max(0.0, (self.current_epoch - start + 1) / ramp))
        return self.weights[name] * fraction

    def _ce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        pixel_loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        denominator = (
            valid.sum()
            if self.class_weights is None
            else (self.class_weights[safe_target] * valid).sum()
        )
        return pixel_loss.sum() / denominator.clamp_min(1e-12)

    @staticmethod
    def _complementary_loss(aux: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        for name in ("cf_sar", "cf_optical"):
            prediction, target = aux.get(f"{name}_tokens"), aux.get(f"{name}_target")
            if prediction is None and target is None:
                continue
            if prediction is None or target is None or prediction.shape != target.shape:
                raise ValueError(
                    f"Invalid complementary prediction/target pair: {name}"
                )
            losses.append(F.smooth_l1_loss(prediction.float(), target.detach().float()))
        if not losses:
            raise ValueError(
                "Active complementary loss requires complementary features and detached targets."
            )
        return torch.stack(losses).mean()

    def forward(
        self,
        outputs: dict[str, Any],
        target: torch.Tensor,
        source_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del source_ids
        logits = outputs["logits"]
        if logits.ndim != 4 or target.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError(
                "Segmentation logits and targets have incompatible shapes."
            )
        if (
            self.class_weights is not None
            and self.class_weights.numel() != logits.shape[1]
        ):
            raise ValueError("Class weight count must match the segmentation head.")
        aux = outputs.get("aux_both", {})
        total = self._ce_loss(logits, target)
        parts = {"loss_ce": total.detach()}
        for name in self.weights:
            weight = self._scheduled_weight(name)
            if not weight:
                continue
            if name == "complementary":
                value = self._complementary_loss(aux)
            else:
                value = aux.get(f"{name}_loss")
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    raise ValueError(
                        f"Active {name} loss requires its scalar model auxiliary."
                    )
            total = total + weight * value
            parts[f"loss_{name}"] = value.detach()
        parts["loss"] = total.detach()
        return total, parts
