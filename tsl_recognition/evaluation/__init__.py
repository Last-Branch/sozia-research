"""
Evaluation subpackage — training, inference, and validation for the TSL recognition pipeline.

Modules:
    train         - Training loop, LR scheduling, checkpointing, metric logging
    inference     - Real-time and video-file inference (TRIGGER/MOTION/CONTINUOUS/VIDEO_FILE)
    evaluate      - Re-evaluate a saved model on the held-out test set
    validate      - Check that inference preprocessing matches training preprocessing
    visualization - MediaPipe landmark drawing and inference HUD overlay
"""
