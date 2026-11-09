from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import S1S2LC8TiffDataset


def row_fingerprint(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for key in ("sample_id", "s1_source_array_sha256", "s2_source_array_sha256"):
            digest.update(str(row.get(key, "")).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def class_weights(rows: list[dict], num_classes: int) -> tuple[list[float], list[int]]:
    counts = np.array(
        [
            sum(int(row[f"class_hist_{class_id}"]) for row in rows)
            for class_id in range(num_classes)
        ],
        dtype=np.int64,
    )
    if (counts < 0).any() or not counts.any():
        raise ValueError("Training-label histograms must contain valid class pixels.")
    present = counts > 0
    weights = np.ones(num_classes, dtype=np.float64)
    inverse = 1 / np.sqrt(counts[present] / counts.sum())
    weights[present] = np.clip(inverse / inverse.mean(), 0.25, 4.0)
    return weights.tolist(), counts.tolist()


def prepare_training_metadata(cfg: dict, dataset: S1S2LC8TiffDataset) -> None:
    weights, counts = class_weights(dataset.rows, int(cfg["model"]["num_classes"]))
    cfg["loss"]["class_weights"] = weights
    cfg["data"]["training_class_pixels"] = counts
    if not cfg["model"].get("croma_input_norm", False):
        return
    pool = (
        dataset
        if len(dataset) == 45000
        else S1S2LC8TiffDataset(
            dataset.root, "train", metadata_root=dataset.protocol_root
        )
    )
    with (dataset.protocol_root / "preprocessing_main.json").open(
        encoding="utf-8"
    ) as handle:
        statistics = json.load(handle)
    if statistics.get("pool_fingerprint") != row_fingerprint(
        pool.rows
    ) or statistics.get("samples") != len(pool):
        raise ValueError(
            "Preprocessing statistics differ from the fixed main training pool."
        )
    cfg["model"]["preprocessing"] = statistics
