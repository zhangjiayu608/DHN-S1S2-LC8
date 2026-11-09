from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_manifest(configuration: dict[str, Any]) -> dict[str, str]:
    assets = {}

    def visit(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}.{index}")
        elif isinstance(value, str):
            path = Path(value)
            if path.is_file() and path.suffix.lower() in {
                ".pt",
                ".pth",
                ".bin",
                ".safetensors",
                ".ckpt",
            }:
                assets[location] = file_sha256(path)
            elif location.endswith("model_dir") and path.is_dir():
                for child in sorted(path.glob("*.safetensors")) + sorted(
                    path.glob("pytorch_model*.bin")
                ):
                    assets[f"{location}/{child.name}"] = file_sha256(child)

    visit(configuration, "model")
    return assets


def configuration_fingerprint(cfg: dict) -> str:
    excluded = {"output_dir", "_resolved", "_runtime", "asset_manifest"}
    comparable = {key: value for key, value in cfg.items() if key not in excluded}
    return hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()


def model_payload(model: nn.Module, cfg: dict) -> dict[str, Any]:
    base = getattr(model, "module", model)
    assets = cfg.get("asset_manifest", {})
    prefixes = []
    if any(key.startswith("model.backbone.") for key in assets):
        prefixes.extend(
            f"backbone.{prefix}"
            for prefix in getattr(base.backbone, "pretrained_parameter_prefixes", ())
        )
    if any(key.startswith("model.external_priors.dinov3.") for key in assets):
        prefixes.append("dinov3_encoder.model.")
    omitted = {
        name
        for name, parameter in base.named_parameters()
        if not parameter.requires_grad
        and any(name.startswith(prefix) for prefix in prefixes)
    }
    state = {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in base.state_dict().items()
        if name not in omitted
    }
    return {
        "model": state,
        "omitted_parameters": sorted(omitted),
        "asset_manifest": assets,
        "config": cfg,
        "format_version": 2,
    }


def load_model_state(model: nn.Module, payload: dict, cfg: dict | None = None) -> None:
    base = getattr(model, "module", model)
    state = payload.get("model", payload)
    omitted = set(payload.get("omitted_parameters", []))
    if omitted:
        expected_assets = payload.get("asset_manifest")
        if cfg is None:
            raise ValueError(
                "Frozen pretrained assets differ from the checkpoint references."
            )
        if getattr(base, "_verified_checkpoint_assets", None) != expected_assets:
            if asset_manifest(cfg["model"]) != expected_assets:
                raise ValueError(
                    "Frozen pretrained assets differ from the checkpoint references."
                )
            base._verified_checkpoint_assets = expected_assets
        parameters = dict(base.named_parameters())
        if any(
            name not in parameters or parameters[name].requires_grad for name in omitted
        ):
            raise ValueError(
                "Checkpoint omits parameters that are not frozen in this model."
            )
        missing, unexpected = base.load_state_dict(state, strict=False)
        if set(missing) != omitted or unexpected:
            raise RuntimeError(
                f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
            )
    else:
        base.load_state_dict(state, strict=True)


def save_payload(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def random_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = list(state["numpy"])
    numpy_state[1] = np.asarray(numpy_state[1], dtype=np.uint32)
    np.random.set_state(tuple(numpy_state))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
