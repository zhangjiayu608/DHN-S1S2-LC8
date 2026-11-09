from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]


def merge_config(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_catalog(path: str | Path = PROJECT / "config.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, dict) or len(catalog.get("models", {})) != 18:
        raise ValueError("config.yaml must define the 18 main-experiment variants.")
    return catalog


def resolve_assets(value, root: Path):
    if isinstance(value, dict):
        return {key: resolve_assets(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_assets(child, root) for child in value]
    if isinstance(value, str) and value.startswith("${ASSETS_ROOT}/"):
        return str(root / value.removeprefix("${ASSETS_ROOT}/"))
    return value


def load_config(
    path=PROJECT / "config.yaml",
    *,
    model: str,
    labels: int,
    seed: int = 13,
    data_root=None,
    assets_root=None,
    output_dir=None,
    workers: int | None = None,
) -> dict:
    catalog = read_catalog(path)
    if model not in catalog["models"]:
        raise ValueError(f"Unknown model {model!r}; use --list-models.")
    if labels not in (10, 100):
        raise ValueError("--labels must be 10 or 100.")
    if labels == 10 and seed != 13:
        raise ValueError(
            "The released 10% subset reproduces the seed-13 main experiment."
        )
    paths = catalog["paths"]
    absolute = lambda value: Path(value).expanduser().resolve()
    assets = absolute(
        assets_root or os.environ.get("DAGPID_ASSETS") or PROJECT / paths["assets_root"]
    )
    cfg = deepcopy(catalog["shared"])
    cfg["model"] = resolve_assets(
        merge_config(cfg["model"], catalog["models"][model]), assets
    )
    cfg["data"] = merge_config(cfg["data"], catalog["labels"][labels])
    cfg["data"].update(
        root=str(absolute(data_root or PROJECT / paths["data_root"])),
        metadata_root=str(PROJECT / "metadata"),
        val_manifest="val.csv.gz",
        test_manifest="test.csv.gz",
        expected_train_samples=4500 if labels == 10 else 45000,
    )
    if workers is not None:
        if workers < 0:
            raise ValueError("--workers must be nonnegative.")
        cfg["data"]["num_workers"] = workers
    cfg["seed"] = seed
    cfg["model"]["head_init_seed"] = seed
    cfg["output_dir"] = str(
        absolute(
            output_dir
            or PROJECT
            / paths["output_root"]
            / model
            / f"labels_{labels}"
            / f"seed_{seed}"
        )
    )
    cfg["experiment"] = {"model": model, "labels": labels}
    cfg["assets_root"] = str(assets)
    if cfg["model"].pop("scan_backend", "mamba") != "mamba":
        raise ValueError("Main experiments require the Mamba backend.")
    return cfg


def relocate_config(
    cfg: dict, *, data_root=None, assets_root=None, metadata_root=None
) -> dict:
    cfg = deepcopy(cfg)
    if data_root:
        cfg["data"]["root"] = str(Path(data_root).expanduser().resolve())
    cfg["data"]["metadata_root"] = str(
        Path(metadata_root or PROJECT / "metadata").resolve()
    )
    if assets_root:
        old_root = Path(cfg["assets_root"])
        new_root = Path(assets_root).expanduser().resolve()

        def move(value):
            if isinstance(value, dict):
                return {key: move(child) for key, child in value.items()}
            if isinstance(value, list):
                return [move(child) for child in value]
            if isinstance(value, str) and Path(value).is_relative_to(old_root):
                return str(new_root / Path(value).relative_to(old_root))
            return value

        cfg["model"] = move(cfg["model"])
        cfg["assets_root"] = str(new_root)
    return cfg
