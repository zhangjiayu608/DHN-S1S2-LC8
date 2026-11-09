from __future__ import annotations

import argparse
import os
from pathlib import Path

from dagpid.config import PROJECT, load_config, read_catalog


def main():
    parser = argparse.ArgumentParser(
        description="Train DAGPID-Adapter with one of the 18 frozen RSFM variants."
    )
    parser.add_argument("--config", type=Path, default=PROJECT / "config.yaml")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--labels", type=int, choices=(10, 100), default=100)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--resume", type=Path, help="Resume an epoch-boundary last.pt checkpoint."
    )
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(read_catalog(args.config)["models"]))
        return
    if not args.model:
        parser.error(
            "--model is required; use --list-models for the available variants."
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error(
            "Each training run uses one GPU. Launch separate runs to compare models in parallel."
        )
    if args.gpu is not None:
        if args.gpu < 0:
            parser.error("--gpu must be nonnegative.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(
        args.config,
        model=args.model,
        labels=args.labels,
        seed=args.seed,
        data_root=args.data_root,
        assets_root=args.assets_root,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    import torch

    if not torch.cuda.is_available():
        parser.error(
            "Training requires a CUDA GPU and the mamba-ssm/causal-conv1d extensions."
        )
    from dagpid.engine import fit

    fit(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
