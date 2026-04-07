"""
Unified fine-tuning and post-fine-tuning benchmark.

Merges the original ``unified_bench.py`` (EN) and ``unified_bench_tr.py``
(TR) into a single script. The only behavioural difference between the two
was:
- ``--use_autocast``: wraps generation in ``torch.amp.autocast`` (required
  for some TR-prompt / dtype combinations, e.g. Trendyol on bfloat16).
- Training-only: the TR version omits ``optim`` and ``gradient_checkpointing``
  in ``SFTConfig`` (kept as booleans controlled by ``--no_optim`` /
  ``--no_grad_checkpointing``).

Skips training when ``final_adapter`` already exists, skips the whole run
when ``result.json`` already exists.

Usage:
    python -m gloss_to_text.fine_tuning.unified_bench \\
        --model_id google/gemma-2-9b-it --strategy P3_EN

    # Turkish-style run (autocast on, paged optimizer off):
    python -m gloss_to_text.fine_tuning.unified_bench \\
        --model_id google/gemma-2-9b-it --strategy P3_TR \\
        --use_autocast --no_optim --no_grad_checkpointing
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    LlamaTokenizer,
)
from trl import SFTConfig, SFTTrainer

for _bit in range(1, 17):
    if not hasattr(torch, f"int{_bit}"):
        setattr(torch, f"int{_bit}", torch.int8)
os.environ["UNSLOTH_ENABLE_PATCHES"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _train(
    base_model,
    tokenizer,
    model_id: str,
    strategy: str,
    output_dir: Path,
    adapter_path: Path,
    use_optim: bool,
    use_grad_checkpointing: bool,
) -> PeftModel:
    from gloss_to_text.prompts.strategies import PROMPT_STRATEGIES
    from gloss_to_text.utils import get_chat_template, turkish_lower

    instruction = PROMPT_STRATEGIES[strategy]

    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(
        base_model,
        LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )

    data_files = {
        "train": str(PROJECT_ROOT / "data" / "processed" / "train.jsonl"),
        "test": str(PROJECT_ROOT / "data" / "processed" / "valid.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    def _map_fn(x: dict) -> dict:
        text = get_chat_template(
            model_id, instruction,
            turkish_lower(x["input"]),
            turkish_lower(x["output"]) + tokenizer.eos_token,
        )
        return {"text": text}

    sft_kwargs: dict = dict(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=1024,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=5,
        bf16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=100,
        report_to="none",
    )
    if use_optim:
        sft_kwargs["optim"] = "paged_adamw_8bit"
    if use_grad_checkpointing:
        sft_kwargs["gradient_checkpointing"] = True

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"].map(_map_fn),
        eval_dataset=dataset["test"].map(_map_fn),
        args=SFTConfig(**sft_kwargs),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    return model


def _evaluate(
    model,
    tokenizer,
    model_id: str,
    strategy: str,
    result_path: Path,
    use_autocast: bool,
) -> None:
    import evaluate as eval_lib

    from gloss_to_text.prompts.strategies import PROMPT_STRATEGIES
    from gloss_to_text.utils import get_chat_template, polish_turkish, turkish_lower

    instruction = PROMPT_STRATEGIES[strategy]
    chrf = eval_lib.load("chrf")

    valid_path = str(PROJECT_ROOT / "data" / "processed" / "valid.jsonl")
    dataset = load_dataset("json", data_files={"test": valid_path})["test"]

    model.eval()
    results: list[dict] = []
    total_latency = 0.0

    for sample in tqdm(dataset, desc="Inference"):
        g_in = turkish_lower(sample["input"])
        prompt = get_chat_template(model_id, instruction, g_in)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        t0 = time.perf_counter()
        with torch.no_grad():
            if use_autocast:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs, max_new_tokens=128, num_beams=5,
                        repetition_penalty=1.15, do_sample=False,
                    )
            else:
                outputs = model.generate(
                    **inputs, max_new_tokens=128, num_beams=5,
                    repetition_penalty=1.15, do_sample=False,
                )
        dur = time.perf_counter() - t0
        total_latency += dur

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
            "gloss": g_in,
            "expected": sample["output"],
            "prediction": prediction,
            "latency_sec": round(dur, 4),
            "chrf": round(score, 2),
        })

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    model_slug = result_path.parts[-3]
    print(f"[SUCCESS] {model_slug} avg latency: {total_latency / len(dataset):.4f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True,
                        help="Prompt strategy key, e.g. P3_EN or P3_TR.")
    parser.add_argument("--use_autocast", action="store_true",
                        help="Wrap generation in torch.amp.autocast (needed for some TR configs).")
    parser.add_argument("--no_optim", action="store_true",
                        help="Skip paged_adamw_8bit optimiser during training.")
    parser.add_argument("--no_grad_checkpointing", action="store_true",
                        help="Disable gradient checkpointing during training.")
    args = parser.parse_args()

    model_slug = args.model_id.split("/")[-1].lower().replace("-", "_")
    output_dir = PROJECT_ROOT / "experiments" / "benchmarks" / model_slug / args.strategy.lower()
    adapter_path = output_dir / "final_adapter"
    result_path = output_dir / "result.json"

    if result_path.exists():
        print(f"[SKIP] {result_path} already exists.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    is_trendyol = "trendyol" in args.model_id.lower()
    if is_trendyol:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_id, legacy=False, use_fast=False)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if is_trendyol:
        base_model.resize_token_embeddings(len(tokenizer))

    if adapter_path.exists():
        print(f"[INFO] Adapter found at {adapter_path}. Skipping training.")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("[INFO] No adapter found. Starting training.")
        model = _train(
            base_model, tokenizer, args.model_id, args.strategy,
            output_dir, adapter_path,
            use_optim=not args.no_optim,
            use_grad_checkpointing=not args.no_grad_checkpointing,
        )

    _evaluate(model, tokenizer, args.model_id, args.strategy, result_path, args.use_autocast)

    del model, base_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
