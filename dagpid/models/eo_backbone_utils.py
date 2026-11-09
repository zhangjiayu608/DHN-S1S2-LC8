from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def torch_load(path: str | Path) -> Any:
    try:
        return torch.load(resolve_path(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(resolve_path(path), map_location="cpu")
    except Exception:
        return torch.load(resolve_path(path), map_location="cpu", weights_only=False)


def load_encoder_weights(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    ignored_prefixes: tuple[str, ...] = ("head.", "classifier."),
    ignored_unexpected_prefixes: tuple[str, ...] = (),
) -> Any:
    state = extract_state_dict(checkpoint)
    if state and all(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    expected = model.state_dict()
    required = {name for name in expected if not name.startswith(ignored_prefixes)}
    missing = sorted(required - state.keys())
    unexpected = sorted(
        name
        for name in state.keys() - expected.keys()
        if not name.startswith(ignored_prefixes + ignored_unexpected_prefixes)
    )
    mismatched = sorted(
        name
        for name in required & state.keys()
        if not isinstance(state[name], torch.Tensor)
        or state[name].shape != expected[name].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"{type(model).__name__} pretrained state mismatch: "
            f"missing={missing[:12]}, unexpected={unexpected[:12]}, shape_mismatch={mismatched[:12]}"
        )
    compatible = {
        name: value
        for name, value in state.items()
        if name in expected
        and isinstance(value, torch.Tensor)
        and value.shape == expected[name].shape
    }
    result = model.load_state_dict(compatible, strict=False)
    parameters = dict(model.named_parameters())
    core_parameters = sum(
        parameter.numel() for name, parameter in parameters.items() if name in required
    )
    model.pretrained_load_report = {
        "required_state_keys": len(required),
        "loaded_core_parameters": core_parameters,
        "core_parameter_coverage": 1.0,
        "ignored_prefixes": list(ignored_prefixes),
        "ignored_unexpected_prefixes": list(ignored_unexpected_prefixes),
    }
    return result


def load_module_from_file(
    module_name: str, file_path: str | Path, package_path: str | Path | None = None
) -> Any:
    file_path = resolve_path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find module file: {file_path}")
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    if package_path is not None and module_name not in sys.modules:
        package = types.ModuleType(module_name.rpartition(".")[0])
        package.__path__ = [str(resolve_path(package_path))]  # type: ignore[attr-defined]
        sys.modules.setdefault(module_name.rpartition(".")[0], package)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(file_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(file_path.parent))
        except ValueError:
            pass
    return module


def extract_state_dict(
    obj: Any, prefixes: tuple[str, ...] = ()
) -> dict[str, torch.Tensor]:
    state = obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module", "encoder", "backbone"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {type(state)!r}")
    if prefixes:
        for prefix in prefixes:
            selected = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if selected:
                return selected
    return state


def batch_robust_norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    dims = (2, 3)
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True).clamp_min(1e-6)
    lo = mean - 2 * std
    hi = mean + 2 * std
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def batch_zscore(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    dims = (2, 3)
    return (x - x.mean(dim=dims, keepdim=True)) / x.std(
        dim=dims, keepdim=True
    ).clamp_min(1e-6)


def resize_image(x: torch.Tensor, target_size: int) -> torch.Tensor:
    if target_size <= 0 or x.shape[-2:] == (target_size, target_size):
        return x
    return F.interpolate(
        x, size=(target_size, target_size), mode="bilinear", align_corners=False
    )


def tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    bsz, count, dim = tokens.shape
    side = int(math.sqrt(count))
    if side * side != count:
        raise RuntimeError(f"Token count is not square: {count}")
    return tokens.transpose(1, 2).reshape(bsz, dim, side, side)


def map_to_tokens(feature: torch.Tensor) -> torch.Tensor:
    return feature.flatten(2).transpose(1, 2).contiguous()


def align_tokens(tokens: torch.Tensor, target_side: int) -> torch.Tensor:
    bsz, count, dim = tokens.shape
    side = int(math.sqrt(count))
    if side == target_side:
        return tokens
    fmap = tokens.transpose(1, 2).reshape(bsz, dim, side, side)
    fmap = F.interpolate(
        fmap, size=(target_side, target_side), mode="bilinear", align_corners=False
    )
    return fmap.flatten(2).transpose(1, 2).contiguous()
