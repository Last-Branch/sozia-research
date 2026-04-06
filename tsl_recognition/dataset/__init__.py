"""
Dataset subpackage for the TSL recognition pipeline.

Modules:
    registry      - DATASET_REGISTRY and get_dataset_info() factory
    base          - DatasetInfo abstract base class
    bosphorus     - BosphorusSign22k dataset config
    autsl         - AUTSL dataset config
    loader        - LazySignDataset, scan_dataset, get_or_compute_scaler, build_loaders
    augmentation  - Temporal and spatial augmentation pipeline
    split         - Train/val/test split generation and persistence
    interpolation - Linear interpolation of missing keypoint frames
"""
