"""
RAG baseline evaluation.

Uses the fine-tuned Gemma-2-9B-it + P3_EN adapter as the generation
model, augmenting each prompt with the two most similar training
examples retrieved via semantic search.

Usage:
    python -m gloss_to_text.evaluation.rag_bench
"""

import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

for _bit in range(1, 17):
    if not hasattr(torch, f"int{_bit}"):
        setattr(torch, f"int{_bit}", torch.int8)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_MODEL_ID = "google/gemma-2-9b-it"
_MODEL_DIR = PROJECT_ROOT / "models" / "gloss_to_text" / "gemma_2_9b_it_ft_p3_en"
_OUTPUT_DIR = PROJECT_ROOT / "models" / "gloss_to_text" / "gemma_2_9b_it_ft_rag_p3_en"


def _build_rag_prompt(instruction: str, gloss: str, context: str) -> str:
    full_instruction = f"{instruction}\n\n{context}" if context else instruction
    return (
        f"<start_of_turn>user\n{full_instruction}\n\nGloss: {gloss}"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )


def main() -> None:
    import evaluate as eval_lib

    from ..prompts.strategies import PROMPT_STRATEGIES
    from ..utils import polish_turkish, turkish_lower

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_DIR, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval()

    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]

    train_glosses = [turkish_lower(item["input"]) for item in train_data]
    train_embeddings = embedder.encode(train_glosses, convert_to_tensor=True)

    base_instruction = PROMPT_STRATEGIES["P3_EN"]

    chrf = eval_lib.load("chrf")
    valid_path = PROJECT_ROOT / "data" / "processed" / "valid.jsonl"
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_samples = [json.loads(line) for line in f]

    results: list[dict] = []
    total_latency = 0.0

    print("[INFO] Starting RAG evaluation...")
    for sample in tqdm(valid_samples):
        query_gloss = turkish_lower(sample["input"])

        query_emb = embedder.encode(query_gloss, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, train_embeddings, top_k=2)[0]

        context = "Reference Examples for Morphology:\n" + "".join(
            f"- Gloss: {train_data[int(h['corpus_id'])]['input']} "
            f"-> Turkish: {train_data[int(h['corpus_id'])]['output']}\n"
            for h in hits
        )

        prompt = _build_rag_prompt(base_instruction, query_gloss, context)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
            references=[[sample["output"] if isinstance(sample["output"], str) else sample["output"][0]]],
            word_order=2,
        )["score"]

        results.append({
            "gloss": sample["input"],
            "expected": sample["output"],
            "prediction": prediction,
            "latency_sec": round(dur, 4),
            "chrf": round(score, 2),
            "rag_context": context,
        })

    out_path = _OUTPUT_DIR / "result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"[SUCCESS] Avg latency: {total_latency / len(valid_samples):.4f}s")
    print(f"[SUCCESS] Results saved to {out_path}")


if __name__ == "__main__":
    main()
