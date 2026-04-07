# sozia-research

Training and evaluation pipelines for the Sozia sign language recognition and translation system.

## Pipelines

| Package | Description |
|---------|-------------|
| `tsl_recognition` | GRU-based Turkish Sign Language recognition (MediaPipe landmarks → gloss) |
| `gloss_to_text` | LoRA fine-tuning and evaluation of LLMs for gloss-to-text translation |

## Installation

```bash
conda env create -f configs/environment.yml
conda activate sozia-research

# Recognition pipeline only
pip install -e ".[recognition]"

# Translation pipeline only
pip install -e ".[translation]"

# Both
pip install -e ".[recognition,translation]"
```

## Project structure

```
sozia-research/
├── configs/                    # Config templates and conda environment
│   ├── environment.yml
│   ├── recognition_train.yml
│   └── translation_bench.yml
├── tsl_recognition/            # TSL recognition pipeline
│   ├── models/                 # GRU model
│   ├── dataset/                # Dataset loading, augmentation, splitting
│   ├── extraction/             # MediaPipe landmark extraction
│   └── evaluation/             # Training, evaluation, inference, validation
├── gloss_to_text/              # Gloss-to-text translation pipeline
│   ├── prompts/                # Prompt strategies (P1–P3 × EN/TR)
│   ├── fine_tuning/            # LoRA fine-tuning
│   └── evaluation/             # Baseline, RAG, and Gemini judge
├── scripts/                    # Data preparation utilities
│   └── prepare_data.py
└── notebooks/                  # Exploratory notebooks
```

## Usage

### TSL Recognition

```bash
# Prepare data splits
python -m tsl_recognition split

# Extract MediaPipe landmarks
python -m tsl_recognition extract

# Train
python -m tsl_recognition train

# Evaluate
python -m tsl_recognition evaluate
```

### Gloss-to-Text

```bash
# Prepare train/valid splits
python scripts/prepare_data.py

# Baseline evaluation (no fine-tuning)
python -m gloss_to_text.evaluation.base_model_bench --model_id google/gemma-2-9b-it

# Fine-tune and evaluate (EN prompt)
python -m gloss_to_text.fine_tuning.unified_bench \
    --model_id google/gemma-2-9b-it --strategy P3_EN

# Fine-tune and evaluate (TR prompt, with autocast)
python -m gloss_to_text.fine_tuning.unified_bench \
    --model_id google/gemma-2-9b-it --strategy P3_TR \
    --use_autocast --no_optim --no_grad_checkpointing

# Score predictions with Gemini judge
python -m gloss_to_text.evaluation.gemini_judge
```

## Environment variables

| Variable | Required by |
|----------|-------------|
| `HF_TOKEN` | Any Hugging Face model download |
| `GEMINI_API_KEY` | `gemini_judge.py` |

Place them in a `.env` file at the project root or export them in your shell.
