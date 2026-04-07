"""
Data Preparation: Raw JSONL → Processed Train/Valid Split

Reads the raw gloss-Turkish sentence pairs, applies Turkish-aware
lowercasing, wraps glosses in <gloss> tags, and splits into
train/valid sets.

If processed files already exist the script skips regeneration to
preserve the exact split used in experiments. Use --force to overwrite.

Usage:
    python scripts/prepare_data.py          # safe: skips if files exist
    python scripts/prepare_data.py --force  # regenerate from scratch
"""

import argparse
import json
import random
from pathlib import Path

from gloss_to_text.utils import turkish_lower

# ---------------------------------------------------------------------------
# Paths — relative to the project root (one level above this script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "slr_gloss_tr_cleaned.jsonl"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
VALID_PATH = PROJECT_ROOT / "data" / "processed" / "valid.jsonl"

VALID_RATIO = 0.1
SEED = 42


def _process_entry(entry: dict) -> dict:
    gloss = turkish_lower(entry["input"]).strip()
    return {"input": f"<gloss> {gloss} </gloss>", "output": entry["output"]}


def _write_jsonl(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/valid splits from raw data.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed files.")
    args = parser.parse_args()

    if TRAIN_PATH.exists() and VALID_PATH.exists() and not args.force:
        print("[SKIP] Processed files already exist. Use --force to regenerate.")
        print(f"       {TRAIN_PATH}")
        print(f"       {VALID_PATH}")
        return

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Loaded {len(data)} samples from {RAW_PATH}")

    processed = [_process_entry(entry) for entry in data]

    random.seed(SEED)
    random.shuffle(processed)

    split_idx = len(processed) - int(len(processed) * VALID_RATIO)
    train_data = processed[:split_idx]
    valid_data = processed[split_idx:]

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(TRAIN_PATH, train_data)
    _write_jsonl(VALID_PATH, valid_data)

    print(f"[INFO] Train: {len(train_data)} samples → {TRAIN_PATH}")
    print(f"[INFO] Valid: {len(valid_data)} samples → {VALID_PATH}")


if __name__ == "__main__":
    main()
