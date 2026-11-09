from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dagpid.config import relocate_config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DAGPID-Adapter checkpoint on the fixed main split."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.gpu is not None:
        if args.gpu < 0:
            parser.error("--gpu must be nonnegative.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from dagpid.checkpoint import load_model_state
    from dagpid.dataset import S1S2LC8TiffDataset
    from dagpid.engine import autocast_context, dataloader_kwargs, set_model_epoch
    from dagpid.metrics import SegmentationMetrics
    from dagpid.models import build_segmentor
    from dagpid.seed import seed_everything, seed_worker

    if not torch.cuda.is_available():
        parser.error(
            "Evaluation requires the CUDA Mamba environment used for training."
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Checkpoint must contain its resolved training configuration.")
    cfg = relocate_config(
        checkpoint["config"], data_root=args.data_root, assets_root=args.assets_root
    )
    if args.workers is not None:
        if args.workers < 0:
            parser.error("--workers must be nonnegative.")
        cfg["data"]["num_workers"] = args.workers
    data = cfg["data"]
    seed_everything(int(cfg["seed"]))
    dataset = S1S2LC8TiffDataset(
        data["root"],
        args.split,
        metadata_root=data["metadata_root"],
        manifest_path=data[f"{args.split}_manifest"],
    )
    if len(dataset) != {"val": 9994, "test": 21399}[args.split]:
        raise ValueError(
            "Evaluation must use the complete fixed validation or test split."
        )
    loader = DataLoader(
        dataset,
        batch_size=int(data["val_batch_size"]),
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
        **dataloader_kwargs(data),
    )
    device = torch.device("cuda:0")
    model = build_segmentor(cfg["model"]).to(device)
    load_model_state(model, checkpoint, cfg)
    model.eval()
    set_model_epoch(model, int(checkpoint["epoch"]))
    metrics = SegmentationMetrics(
        cfg["model"]["num_classes"],
        cfg["loss"]["ignore_index"],
        device=device,
        compute_boundary=True,
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.split):
            s1, s2 = (batch[key].to(device, non_blocking=True) for key in ("s1", "s2"))
            with autocast_context(device, cfg["train"]["amp"]):
                outputs = model(s1, s2)
            metrics.update(
                outputs["logits"], batch["mask"].to(device, non_blocking=True)
            )
    result = {
        "split": args.split,
        "samples": len(dataset),
        "epoch": int(checkpoint["epoch"]),
        "metrics": metrics.compute(),
        "confusion_matrix": metrics.confmat.cpu().tolist(),
    }
    output = args.output_dir or args.checkpoint.resolve().parent / args.split
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
