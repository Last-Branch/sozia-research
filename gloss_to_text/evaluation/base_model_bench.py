"""
Baseline evaluation — no fine-tuning.

Runs inference on the validation set for a given model and the two
prompt strategies that best suit it (one EN, one TR), saving results
to ``experiments/baselines/<model_slug>/``.

Usage:
    python -m gloss_to_text.evaluation.base_model_bench --model_id <hf_model_id>
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)

# Compatibility patch: some Unsloth-patched environments lack int1-int16 attrs.
for _bit in range(1, 17):
    if not hasattr(torch, f"int{_bit}"):
        setattr(torch, f"int{_bit}", torch.int8)
os.environ["UNSLOTH_ENABLE_PATCHES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Strategy mapping: model family → (EN strategy, TR strategy)
_MODEL_STRATEGY_MAP = {
    "gemma": ("P3_EN", "P3_TR"),
    "llama": ("P2_EN", "P2_TR"),
    "trendyol": ("P2_EN", "P2_TR"),
}


def _run_inference(
    model: Any,
    tokenizer: Any,
    model_id: str,
    strategy_key: str,
    output_dir: Path,
    valid_samples: list[dict],
    chrf: Any,
) -> None:
    from ..prompts.strategies import PROMPT_STRATEGIES
    from ..utils import get_chat_template, polish_turkish, turkish_lower

    instruction = PROMPT_STRATEGIES[strategy_key]
    res_path = output_dir / "result.json"

    results: list[dict] = []
    processed_glosses: set[str] = set()
    if res_path.exists():
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed_glosses = {r["gloss"] for r in results}
            print(f"[INFO] Resuming {strategy_key}: {len(results)} entries already done.")
        except (json.JSONDecodeError, KeyError):
            results = []

    print(f"\n[INFO] Running inference: {model_id} | {strategy_key}")

    for sample in tqdm(valid_samples, desc=f"Eval {strategy_key}"):
        if sample["input"] in processed_glosses:
            continue

        g_in = turkish_lower(sample["input"])
        prompt = get_chat_template(model_id, instruction, g_in)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, num_beams=5,
                repetition_penalty=1.15, do_sample=False,
            )
        duration = time.perf_counter() - t0

        raw_pred = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ).strip()
        prediction = polish_turkish(raw_pred)

        score = chrf.compute(
            predictions=[prediction],
            references=[[sample["output"]]],
            word_order=2,
        )["score"]

        results.append({
            "gloss": sample["input"],
            "expected": sample["output"],
            "prediction": prediction,
            "latency_sec": round(duration, 4),
            "chrf": round(score, 2),
        })

        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"[DONE] {strategy_key} complete.")


def main() -> None:
    import evaluate as eval_lib

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    args = parser.parse_args()

    model_slug = args.model_id.split("/")[-1].lower().replace("-", "_")

    is_trendyol = "trendyol" in args.model_id.lower()
    if is_trendyol:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_id, legacy=False, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval()

    chrf = eval_lib.load("chrf")
    valid_path = PROJECT_ROOT / "data" / "processed" / "valid.jsonl"
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_samples = [json.loads(line) for line in f]

    family = "gemma"
    if "llama" in args.model_id.lower():
        family = "llama"
    elif "trendyol" in args.model_id.lower():
        family = "trendyol"

    en_strat, tr_strat = _MODEL_STRATEGY_MAP[family]
    for strat in (en_strat, tr_strat):
        run_name = f"{model_slug}_base_{strat.lower()}"
        output_dir = PROJECT_ROOT / "models" / "gloss_to_text" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        _run_inference(model, tokenizer, args.model_id, strat, output_dir, valid_samples, chrf)


if __name__ == "__main__":
    main()
