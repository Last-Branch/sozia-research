"""
CLI for the lip-reading pipeline.

Usage::

    python -m lip_reading train                   # full training on BosphorusSign22k
    python -m lip_reading train --dataset autsl   # full training on AUTSL
    python -m lip_reading train --test            # 10-class smoke test (BosphorusSign22k)
    python -m lip_reading train --dataset autsl --test
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

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", parents=[shared], help="Train the lip-reading classifier")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
