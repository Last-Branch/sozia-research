"""
Turkish Sign Language Recognition (TSL-R) pipeline.

This package implements a complete pipeline for sign language recognition using
MediaPipe landmarks and deep learning sequence models.

Pipeline stages:
    1. Extraction  -- Extract face, hand, and pose landmarks from sign videos
    2. Split       -- Generate persistent train/val/test split manifests
    3. Training    -- Train a sequence model to classify signs from landmark sequences
    4. Inference   -- Recognize signs in real-time from webcam or video files
    5. Validation  -- Verify preprocessing consistency between training and inference

Package structure:
    config          - Configuration constants, paths, hyperparameters
    models/         - Model registry and architecture implementations (GRU, ...)
    dataset/        - Dataset registry, LazySignDataset, scaler, DataLoaders, augmentation
    extraction/     - Video processing, MediaPipe landmarker setup, keypoint extraction
    evaluation/     - Training loop, evaluation, inference, validation, visualization
    cli             - CLI entry point for all pipeline commands
"""
