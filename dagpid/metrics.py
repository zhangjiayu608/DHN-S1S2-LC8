from __future__ import annotations

import math

import cv2
import numpy as np
import torch


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    nan = torch.full_like(numerator, float("nan"), dtype=torch.float64)
    return torch.where(denominator > 0, numerator / denominator, nan)


def _to_float(value: torch.Tensor) -> float:
    if torch.isnan(value):
        return 0.0
    return float(value.detach().cpu())


def _to_list(values: torch.Tensor) -> list[float]:
    return [
        float(x) if not math.isnan(float(x)) else float("nan")
        for x in values.detach().cpu()
    ]


def _disk_footprint(radius: float) -> np.ndarray:
    radius_int = max(0, int(math.ceil(float(radius))))
    y, x = np.ogrid[-radius_int : radius_int + 1, -radius_int : radius_int + 1]
    return ((x * x + y * y) <= radius_int * radius_int).astype(np.uint8)


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """OpenSeg/GSCNN boundary map helper, adapted for same-size masks.

    Source lineage:
    - OpenSeg lib/metrics/F1_running_score.py
    - GSCNN utils/f_boundary.py
    - Berkeley boundary benchmark seg2bmap routine
    """
    seg = seg.astype(bool)
    east = np.zeros_like(seg)
    south = np.zeros_like(seg)
    southeast = np.zeros_like(seg)

    east[:, :-1] = seg[:, 1:]
    south[:-1, :] = seg[1:, :]
    southeast[:-1, :-1] = seg[1:, 1:]

    boundary = (seg ^ east) | (seg ^ south) | (seg ^ southeast)
    boundary[-1, :] = seg[-1, :] ^ east[-1, :]
    boundary[:, -1] = seg[:, -1] ^ south[:, -1]
    boundary[-1, -1] = False
    return boundary


def _official_boundary_f1(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    ignore_mask: np.ndarray,
    boundary_threshold: float,
) -> float:
    """Boundary F-measure matching OpenSeg/GSCNN db_eval_boundary.

    The default threshold 0.00088 follows OpenSeg/GSCNN; for 96x96 tiles it
    becomes a 1-pixel tolerance after the official diagonal-length scaling.
    """
    pred_mask = pred_mask.astype(np.uint8, copy=True)
    gt_mask = gt_mask.astype(np.uint8, copy=True)
    pred_mask[ignore_mask] = 0
    gt_mask[ignore_mask] = 0

    pred_boundary = _seg2bmap(pred_mask)
    gt_boundary = _seg2bmap(gt_mask)

    bound_pix = (
        boundary_threshold
        if boundary_threshold >= 1
        else np.ceil(boundary_threshold * np.linalg.norm(pred_mask.shape))
    )
    selem = _disk_footprint(bound_pix)
    pred_dil = cv2.dilate(pred_boundary.astype(np.uint8), selem).astype(bool)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), selem).astype(bool)

    gt_match = gt_boundary & pred_dil
    pred_match = pred_boundary & gt_dil

    n_pred = np.sum(pred_boundary)
    n_gt = np.sum(gt_boundary)
    if n_pred == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_pred > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_pred == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = float(np.sum(pred_match)) / float(n_pred)
        recall = float(np.sum(gt_match)) / float(n_gt)

    return (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )


class SegmentationMetrics:
    """Official-style semantic segmentation metrics.

    Region metrics follow the DINOv3/MMsegmentation convention:
    mIoU, aAcc/OA, mAcc, Dice/F-score, precision and recall are computed from
    global intersections/unions with NaN-aware class averaging.

    Boundary-F1 follows the OpenSeg/GSCNN `db_eval_boundary` implementation.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: torch.device | str = "cpu",
        boundary_threshold: float = 0.00088,
        compute_boundary: bool = True,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.boundary_threshold = float(boundary_threshold)
        self.compute_boundary = bool(compute_boundary)
        self.confmat = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.float64, device=device
        )
        self.boundary_f_sum = torch.zeros(
            self.num_classes, dtype=torch.float64, device=device
        )
        self.boundary_count = torch.zeros(
            self.num_classes, dtype=torch.float64, device=device
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1) if logits.dim() == 4 else logits
        pred = pred.to(dtype=torch.long)
        target = target.to(dtype=torch.long)
        self._update_confusion(pred, target)
        if self.compute_boundary:
            self._update_boundary_f1(pred, target)

    def compute(self) -> dict[str, float | list[float]]:
        tp = torch.diag(self.confmat)
        gt_area = self.confmat.sum(dim=1)
        pred_area = self.confmat.sum(dim=0)
        union = gt_area + pred_area - tp
        total = self.confmat.sum()

        iou = _safe_div(tp, union)
        acc = _safe_div(tp, gt_area)
        precision = _safe_div(tp, pred_area)
        recall = _safe_div(tp, gt_area)
        dice = _safe_div(2.0 * tp, pred_area + gt_area)
        f1 = dice

        oa = tp.sum() / total.clamp_min(1.0)
        expected = (gt_area * pred_area).sum() / (total * total).clamp_min(1.0)
        kappa = (oa - expected) / (1.0 - expected).clamp_min(1e-12)
        freq = gt_area / total.clamp_min(1.0)
        fwiou = (
            (freq[gt_area > 0] * iou[gt_area > 0]).sum()
            if bool((gt_area > 0).any())
            else torch.tensor(0.0, device=tp.device)
        )

        boundary_per_class = _safe_div(self.boundary_f_sum, self.boundary_count)
        boundary_f1 = (
            torch.nanmean(boundary_per_class)
            if bool((self.boundary_count > 0).any())
            else torch.tensor(0.0, device=tp.device)
        )

        miou = torch.nanmean(iou)
        macc = torch.nanmean(acc)
        mprecision = torch.nanmean(precision)
        mrecall = torch.nanmean(recall)
        mf1 = torch.nanmean(f1)
        mdice = torch.nanmean(dice)
        return {
            "mIoU": _to_float(miou),
            "mIoU_all": _to_float(miou),
            "OA": _to_float(oa),
            "aAcc": _to_float(oa),
            "mAcc": _to_float(macc),
            "mPrecision": _to_float(mprecision),
            "mRecall": _to_float(mrecall),
            "mF1": _to_float(mf1),
            "macro_F1": _to_float(mf1),
            "mDice": _to_float(mdice),
            "FWIoU": _to_float(fwiou),
            "Kappa": _to_float(kappa),
            "BoundaryF1": _to_float(boundary_f1),
            "BF1": _to_float(boundary_f1),
            "per_class_IoU": _to_list(iou),
            "per_class_Acc": _to_list(acc),
            "per_class_Dice": _to_list(dice),
            "per_class_F1": _to_list(f1),
            "per_class_precision": _to_list(precision),
            "per_class_recall": _to_list(recall),
            "per_class_BoundaryF1": _to_list(boundary_per_class),
        }

    def reset(self) -> None:
        self.confmat.zero_()
        self.boundary_f_sum.zero_()
        self.boundary_count.zero_()

    def _update_confusion(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        valid = target != self.ignore_index
        pred = pred[valid].view(-1)
        target = target[valid].view(-1)
        if target.numel() == 0:
            return
        indices = target * self.num_classes + pred
        bins = torch.bincount(indices, minlength=self.num_classes**2).to(
            device=self.confmat.device, dtype=self.confmat.dtype
        )
        self.confmat += bins.reshape(self.num_classes, self.num_classes)

    def _update_boundary_f1(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        sums = np.zeros(self.num_classes, dtype=np.float64)
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for pred_item, target_item in zip(pred_np, target_np):
            ignore_mask = target_item == self.ignore_index
            for class_id in range(self.num_classes):
                score = _official_boundary_f1(
                    pred_item == class_id,
                    target_item == class_id,
                    ignore_mask,
                    self.boundary_threshold,
                )
                sums[class_id] += score
                counts[class_id] += 1.0
        self.boundary_f_sum += torch.as_tensor(
            sums, dtype=self.boundary_f_sum.dtype, device=self.boundary_f_sum.device
        )
        self.boundary_count += torch.as_tensor(
            counts, dtype=self.boundary_count.dtype, device=self.boundary_count.device
        )
