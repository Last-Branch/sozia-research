"""
CLI for the lip-reading pipeline.

Usage::

    python -m lip_reading train                              # full training on BosphorusSign22k
    python -m lip_reading train --dataset autsl              # full training on AUTSL
    python -m lip_reading train --mouth-only                 # mouth-only landmarks (120-dim)
    python -m lip_reading train --dataset autsl --mouth-only
    python -m lip_reading train --test                       # 10-class smoke test
"""

from __future__ import annotations

import argparse

from tsl_recognition.dataset.registry import DATASET_CHOICES

from .config import LipTrainConfig


def cmd_train(args: argparse.Namespace) -> None:
    """Execute the lip-reading training command."""
    from .train import train

    dataset = args.dataset or "bosphorus"
    if args.test:
        cfg = LipTrainConfig.test(dataset=dataset)
    else:
        cfg = LipTrainConfig.full(dataset=dataset)
    cfg.use_mouth_only = args.mouth_only
    train(cfg)


def main() -> None:
    """Main CLI entry point for the lip-reading pipeline."""
    parser = argparse.ArgumentParser(
        prog="lip_reading",
        description="Lip-reading word classifier — trained on TSL sign datasets",
    )

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--test",
        action="store_true",
        help="Use 10-class test config instead of full training",
    )
    shared.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="bosphorus",
        help="Dataset to use (default: bosphorus)",
    )
    shared.add_argument(
        "--mouth-only",
        action="store_true",
        help="Use only the 40 mouth/lip landmarks (120-dim) instead of full face (249-dim)",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", parents=[shared], help="Train the lip-reading classifier")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
