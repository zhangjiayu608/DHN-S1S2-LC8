from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class S1S2LC8TiffDataset(Dataset):
    """Paired 96x96 Sentinel-1/2 patches with a fixed LC8 manifest."""

    def __init__(
        self,
        root,
        split: str,
        *,
        metadata_root,
        manifest_path=None,
        augment: bool = False,
        seed: int = 13,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("Only the main train, val and test splits are supported.")
        self.root = Path(root)
        self.protocol_root = Path(metadata_root)
        self.split = split
        self.seed = int(seed)
        self.augment = augment
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.manifest_path = self.protocol_root / (manifest_path or f"{split}.csv.gz")
        if not self.manifest_path.resolve().is_relative_to(
            self.protocol_root.resolve()
        ):
            raise ValueError("Manifest must be inside metadata_root.")
        with (self.protocol_root / "manifest_sha256.json").open(
            encoding="utf-8"
        ) as handle:
            hashes = json.load(handle)
        name = self.manifest_path.relative_to(self.protocol_root).as_posix()
        actual_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        if hashes.get(name) != actual_hash:
            raise ValueError(f"Manifest checksum mismatch: {self.manifest_path}")
        with gzip.open(
            self.manifest_path, "rt", encoding="utf-8", newline=""
        ) as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows or any(row["split"] != split for row in self.rows):
            raise ValueError(f"Empty or incorrect split: {self.manifest_path}")
        if len({row["sample_id"] for row in self.rows}) != len(self.rows):
            raise ValueError(f"Duplicate samples: {self.manifest_path}")
        self.s2_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch: int):
        self.epoch.fill_(int(epoch))

    def __getitem__(self, index):
        row = self.rows[index]
        s1 = tifffile.imread(self.root / row["s1_path"]).astype(np.float32)
        s2 = tifffile.imread(self.root / row["s2_path"]).astype(np.float32)
        mask = tifffile.imread(self.root / row["lc_path"])
        if not np.isfinite(s1).all() or not np.isfinite(s2).all():
            raise RuntimeError(f"Nonfinite sensor pixels: {row['sample_id']}")
        if not np.issubdtype(mask.dtype, np.integer) or not np.all(
            ((mask >= 0) & (mask < 8)) | (mask == 255)
        ):
            raise RuntimeError(f"Invalid LC8 labels: {row['sample_id']}")
        if s2.ndim == 3 and s2.shape[-1] == 13:
            s2 = s2[:, :, self.s2_keep]
        if (
            s1.shape != (96, 96, 2)
            or s2.shape != (96, 96, 12)
            or mask.shape != (96, 96)
        ):
            raise RuntimeError(f"Invalid paired image shapes: {row['sample_id']}")
        s1 = torch.from_numpy(np.ascontiguousarray(s1)).permute(2, 0, 1)
        s2 = torch.from_numpy(np.ascontiguousarray(s2)).permute(2, 0, 1)
        mask = torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64)))
        if self.augment:
            key = f"{self.seed}|{int(self.epoch)}|{row['sample_id']}"
            sample_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (
                2**63 - 1
            )
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(sample_seed)
                s1, s2, mask = self._augment(s1, s2, mask)
        return {
            "s1": s1.contiguous(),
            "s2": s2.contiguous(),
            "mask": mask.contiguous(),
            "id": row["sample_id"],
            "source_dataset": row["source_dataset"],
        }

    @staticmethod
    def _augment(s1, s2, mask):
        if torch.rand(()) < 0.5:
            s1, s2, mask = (
                torch.flip(s1, [2]),
                torch.flip(s2, [2]),
                torch.flip(mask, [1]),
            )
        if torch.rand(()) < 0.5:
            s1, s2, mask = (
                torch.flip(s1, [1]),
                torch.flip(s2, [1]),
                torch.flip(mask, [0]),
            )
        k = int(torch.randint(0, 4, ()).item())
        if k:
            s1, s2, mask = (
                torch.rot90(s1, k, [1, 2]),
                torch.rot90(s2, k, [1, 2]),
                torch.rot90(mask, k, [0, 1]),
            )
        return s1, s2, mask
