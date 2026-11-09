from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

from .dataset import S1S2LC8TiffDataset
from .statistics import prepare_training_metadata
from .losses import DAGPIDLoss
from .metrics import SegmentationMetrics
from .models import build_segmentor
from .seed import seed_everything, seed_worker
from .checkpoint import (
    asset_manifest,
    configuration_fingerprint,
    load_model_state,
    model_payload,
    random_state,
    restore_random_state,
    save_payload,
)


def fit(cfg: dict, resume: Path | None = None) -> None:
    seed_everything(int(cfg["seed"]))
    cfg["model"]["head_init_seed"] = int(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Fresh training requires an empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_loaders(cfg)
    prepare_training_metadata(cfg, train_loader.dataset)
    cfg["asset_manifest"] = asset_manifest(cfg["model"])
    model = build_segmentor(cfg["model"]).to(device)
    criterion = DAGPIDLoss(**cfg["loss"]).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg["train"].get("amp") == "fp16" and device.type == "cuda"
    )
    runtime = {"best_miou": -1.0, "best_epoch": 0, "epochs_without_improvement": 0}
    first_epoch = 1
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if "training_state" not in checkpoint:
            raise ValueError(
                "Resume requires last.pt with training state; best.pt contains inference state only."
            )
        if checkpoint["configuration_fingerprint"] != configuration_fingerprint(cfg):
            raise ValueError(
                "Resume configuration, seed, labels, or preprocessing differ from the saved run."
            )
        load_model_state(model, checkpoint, cfg)
        state = checkpoint["training_state"]
        if len(state["random_states"]) != 1:
            raise ValueError("Resume requires a checkpoint from a single-GPU run.")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        criterion.load_state_dict(state["criterion"])
        train_loader.generator.set_state(state["loader_generator"].cpu())
        runtime.update(state["runtime"])
        first_epoch = int(checkpoint["epoch"]) + 1
        restore_random_state(state["random_states"][0])
    (output_dir / "config_resolved.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    tqdm.write(
        f"DAGPID-Adapter | {cfg.get('experiment', {})} | train={len(train_loader.dataset)} val={len(val_loader.dataset)}"
    )
    parameters = list(model.parameters())
    tqdm.write(
        f"Parameters: {sum((value.numel() for value in parameters)):,} total, {sum((value.numel() for value in parameters if value.requires_grad)):,} trainable"
    )
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stopping_patience", 0))
    for epoch in range(first_epoch, epochs + 1):
        start = time.perf_counter()
        train_loader.dataset.set_epoch(epoch)
        train_loader.generator.manual_seed(int(cfg["seed"]) + epoch)
        train_loader.sampler.generator.manual_seed(int(cfg["seed"]) + epoch)
        val_loader.generator.manual_seed(int(cfg["seed"]) + 100000 + epoch)
        scheduler.step_epoch(epoch)
        with tqdm(
            total=len(train_loader) + len(val_loader),
            desc=f"Epoch {epoch:03d}/{epochs} train",
            disable=not sys.stderr.isatty(),
            ncols=80,
            ascii=True,
            mininterval=1.0,
            file=sys.stderr,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as progress:
            train_stats = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                cfg,
                epoch,
                epochs,
                progress=progress,
            )
            progress.set_description(f"Epoch {epoch:03d}/{epochs} val", refresh=False)
            val_stats = evaluate(
                model,
                val_loader,
                criterion,
                device,
                cfg,
                epoch,
                epochs,
                progress=progress,
            )
        miou = float(val_stats["metrics"]["mIoU"])
        improved = miou > runtime["best_miou"] + float(
            cfg["train"].get("early_stopping_min_delta", 0)
        )
        if improved:
            runtime.update(
                best_miou=miou, best_epoch=epoch, epochs_without_improvement=0
            )
        else:
            runtime["epochs_without_improvement"] += 1
        states = [random_state()]
        row = {
            "epoch": epoch,
            "lr": current_primary_lr(optimizer),
            "seconds": round(time.perf_counter() - start, 3),
            "best_mIoU": runtime["best_miou"],
            "best_epoch": runtime["best_epoch"],
            "epochs_without_improvement": runtime["epochs_without_improvement"],
            "train": train_stats,
            "val": val_stats,
        }
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        append_csv(output_dir / "metrics.csv", row)
        payload = dict(model_payload(model, cfg), epoch=epoch, val_stats=val_stats)
        if improved and cfg["train"].get("checkpoint_selection", "best") == "best":
            save_payload(output_dir / "best.pt", payload)
        payload["configuration_fingerprint"] = configuration_fingerprint(cfg)
        payload["training_state"] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "criterion": criterion.state_dict(),
            "runtime": runtime,
            "random_states": states,
            "loader_generator": train_loader.generator.get_state(),
        }
        save_payload(output_dir / "last.pt", payload)
        tqdm.write(format_epoch_summary(row, epochs))
        if patience > 0 and runtime["epochs_without_improvement"] >= patience:
            break


def build_loaders(cfg):
    data = cfg["data"]
    common = {"root": data["root"], "metadata_root": data["metadata_root"]}
    train_set = S1S2LC8TiffDataset(
        **common,
        split="train",
        manifest_path=data["train_manifest"],
        augment=bool(data.get("augment", True)),
        seed=int(cfg["seed"]),
    )
    val_set = S1S2LC8TiffDataset(
        **common, split="val", manifest_path=data["val_manifest"]
    )
    if len(train_set) != int(data["expected_train_samples"]) or len(val_set) != 9994:
        raise ValueError(
            "Unexpected number of main-experiment training or validation samples."
        )
    data.update(
        train_fraction_full_count=45000,
        train_fraction_effective_count=len(train_set),
        actual_label_ratio=len(train_set) / 45000,
    )
    kwargs = dataloader_kwargs(data)
    train_loader = DataLoader(
        train_set,
        batch_size=int(data["batch_size"]),
        sampler=RandomSampler(
            train_set, generator=torch.Generator().manual_seed(int(cfg["seed"]))
        ),
        drop_last=bool(data.get("drop_last", True)),
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
        worker_init_fn=seed_worker,
        **kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(data.get("val_batch_size", data["batch_size"])),
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(cfg["seed"]) + 100000),
        worker_init_fn=seed_worker,
        **kwargs,
    )
    return train_loader, val_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    cfg,
    epoch: int,
    max_epochs: int,
    progress=None,
) -> dict[str, Any]:
    model.train()
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)
    set_model_epoch(model, epoch)
    totals: dict[str, float] = {}
    steps = 0
    amp = cfg["train"].get("amp", "bf16")
    show_metrics = bool(cfg["train"].get("progress_train_metrics", True))
    accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    metrics = SegmentationMetrics(
        cfg["model"]["num_classes"],
        cfg["loss"]["ignore_index"],
        device=device,
        compute_boundary=bool(
            cfg["train"].get("progress_train_boundary_metrics", False)
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        s1 = batch["s1"].to(device, non_blocking=True)
        s2 = batch["s2"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        s1, s2, mask = maybe_cutmix(s1, s2, mask, cfg)
        with autocast_context(device, amp):
            outputs = model(s1, s2, return_aux=criterion.requires_aux)
            loss, parts = criterion(outputs, mask)
            group_start = ((batch_idx - 1) // accum_steps) * accum_steps
            group_size = min(accum_steps, num_batches - group_start)
            backward_loss = loss / group_size
        if scaler.is_enabled():
            scaler.scale(backward_loss).backward()
            if batch_idx % accum_steps == 0 or batch_idx == num_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"].get("grad_clip", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            backward_loss.backward()
            if batch_idx % accum_steps == 0 or batch_idx == num_batches:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"].get("grad_clip", 1.0)
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        if show_metrics:
            metrics.update(outputs["logits"].detach(), mask)
        steps += 1
        if progress is not None:
            progress.update(1)

    avg = reduce_average_dict(totals, steps, device)
    if show_metrics:
        avg["metrics"] = metrics.compute()
    return avg


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    cfg,
    epoch: int,
    max_epochs: int,
    desc_suffix: str = "",
    progress=None,
) -> dict[str, Any]:
    model.eval()
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)
    set_model_epoch(model, epoch)
    metrics = SegmentationMetrics(
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["loss"]["ignore_index"],
        device=device,
        compute_boundary=bool(cfg["train"].get("validation_boundary", False)),
    )
    totals: dict[str, float] = {}
    steps = 0
    amp = cfg["train"].get("amp", "bf16")
    for batch in loader:
        s1 = batch["s1"].to(device, non_blocking=True)
        s2 = batch["s2"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            outputs = model(s1, s2, return_aux=criterion.requires_aux)
            _, parts = criterion(outputs, mask)
        metrics.update(outputs["logits"], mask)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
        if progress is not None:
            progress.update(1)

    result = {
        "losses": reduce_average_dict(totals, steps, device),
        "metrics": metrics.compute(),
    }
    return result


def format_epoch_summary(row: dict[str, Any], epochs: int) -> str:
    metrics = row["val"]["metrics"]
    return (
        f"Epoch {row['epoch']:03d}/{epochs} | loss={row['train']['loss']:.4f} "
        f"val_loss={row['val']['losses']['loss']:.4f} lr={row['lr']:.2e} {row['seconds']:.1f}s\n"
        f"  Val mIoU={metrics['mIoU']:.2%} OA={metrics['OA']:.2%} F1={metrics['mF1']:.2%} "
        f"| best={row['best_mIoU']:.2%} (epoch {row['best_epoch']})"
    )


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    groups = {True: [], False: []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[is_no_decay_parameter(name, parameter)].append(parameter)
    if not any(groups.values()):
        raise ValueError("The model has no trainable parameters.")
    options = cfg["optimizer"]
    parameters = [
        {
            "params": values,
            "weight_decay": (
                float(options.get("no_decay_weight_decay", 0))
                if no_decay
                else float(options.get("weight_decay", 0.05))
            ),
        }
        for no_decay, values in groups.items()
        if values
    ]
    return torch.optim.AdamW(parameters, lr=float(options["lr"]))


def reduce_average_dict(
    totals: dict[str, float], steps: int, device: torch.device
) -> dict[str, float]:
    keys = sorted(totals)
    values = [totals[key] for key in keys] + [float(steps)]
    tensor = torch.stack(
        [torch.as_tensor(value, dtype=torch.float64, device=device) for value in values]
    )
    total_steps = max(1.0, float(tensor[-1].item()))
    return {key: float(tensor[i].item() / total_steps) for i, key in enumerate(keys)}


def dataloader_kwargs(data_cfg: dict[str, Any]) -> dict[str, Any]:
    num_workers = int(data_cfg.get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True))
        and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return kwargs


def is_no_decay_parameter(name: str, parameter: nn.Parameter) -> bool:
    if getattr(parameter, "_no_weight_decay", False):
        return True
    lname = name.lower()
    if parameter.ndim <= 1 or lname.endswith(".bias"):
        return True
    no_decay_terms = (
        "norm",
        "pos_embed",
        "position",
        "class_embedding",
        "cls_token",
        "register",
        "relative_position",
        "rope",
    )
    return any(term in lname for term in no_decay_terms)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def set_model_epoch(model: nn.Module, epoch: int) -> None:
    base_model = getattr(model, "module", model)
    if hasattr(base_model, "set_epoch"):
        base_model.set_epoch(epoch)


def maybe_cutmix(
    s1: torch.Tensor,
    s2: torch.Tensor,
    mask: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cfg = cfg.get("train", {})
    prob = float(train_cfg.get("cutmix_prob", 0.0))
    if prob <= 0 or s1.shape[0] < 2 or torch.rand((), device=s1.device) >= prob:
        return s1, s2, mask
    alpha = max(1e-6, float(train_cfg.get("cutmix_alpha", 1.0)))
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    height, width = mask.shape[-2:]
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_h = max(1, int(height * cut_ratio))
    cut_w = max(1, int(width * cut_ratio))
    cy = int(torch.randint(0, height, (), device=s1.device).item())
    cx = int(torch.randint(0, width, (), device=s1.device).item())
    y1 = max(0, cy - cut_h // 2)
    y2 = min(height, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(width, cx + cut_w // 2)
    if y2 <= y1 or x2 <= x1:
        return s1, s2, mask
    perm = torch.randperm(s1.shape[0], device=s1.device)
    if torch.equal(perm, torch.arange(s1.shape[0], device=s1.device)):
        return s1, s2, mask
    s1 = s1.clone()
    s2 = s2.clone()
    mask = mask.clone()
    s1[:, :, y1:y2, x1:x2] = s1[perm, :, y1:y2, x1:x2]
    s2[:, :, y1:y2, x1:x2] = s2[perm, :, y1:y2, x1:x2]
    mask[:, y1:y2, x1:x2] = mask[perm, y1:y2, x1:x2]
    return s1, s2, mask


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, cfg: dict) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.epochs = int(cfg["train"]["epochs"])
        self.warmup = int(cfg.get("scheduler", {}).get("warmup_epochs", 0))
        self.minimum = float(cfg.get("scheduler", {}).get("min_lr_ratio", 0.02))
        self.epoch = 0
        if self.warmup < 0 or self.warmup >= self.epochs or not 0 <= self.minimum <= 1:
            raise ValueError("Invalid warmup/cosine scheduler settings.")

    def step_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.warmup and epoch <= self.warmup:
            scale = epoch / self.warmup
        else:
            progress = min(
                1.0, max(0.0, (epoch - self.warmup) / (self.epochs - self.warmup))
            )
            scale = self.minimum + (1 - self.minimum) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        for group, base_lr in zip(
            self.optimizer.param_groups, self.base_lrs, strict=True
        ):
            group["lr"] = base_lr * scale

    def state_dict(self) -> dict:
        return {"base_lrs": self.base_lrs, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.base_lrs = list(state["base_lrs"])
        self.step_epoch(int(state["epoch"]))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    del steps_per_epoch
    if cfg.get("scheduler", {}).get("name", "warmup_cosine") != "warmup_cosine":
        raise ValueError("Only the reviewed warmup_cosine schedule is supported.")
    return WarmupCosineScheduler(optimizer, cfg)


def current_primary_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def append_csv(path, row):
    values = {
        key: row[key] for key in ("epoch", "lr", "seconds", "best_mIoU", "best_epoch")
    }
    for split in ("train", "val"):
        stats = row[split]
        losses = stats if split == "train" else stats["losses"]
        values.update(
            {
                f"{split}_{key}": value
                for key, value in losses.items()
                if key.startswith("loss")
            }
        )
        values.update(
            {
                f"{split}_{key}": value
                for key, value in stats.get("metrics", {}).items()
                if key in ("mIoU", "OA", "mAcc", "mF1", "mPrecision", "Kappa", "BF1")
            }
        )
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values))
        if not exists:
            writer.writeheader()
        writer.writerow(values)
