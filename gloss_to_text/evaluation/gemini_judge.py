"""
Gemini 2.5 Flash judge for gloss-to-text evaluation.

Scores each prediction on a 1–10 rubric using Gemini as a linguistics
expert. Scans the experiments directory for result files automatically
instead of relying on a hardcoded list.

Reads GEMINI_API_KEY from the environment (or a .env file at the project
root). Intended to be run locally — not on HPC.

Usage:
    python -m gloss_to_text.evaluation.gemini_judge
    python -m gloss_to_text.evaluation.gemini_judge --experiments_dir /path/to/experiments
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import google.generativeai as genai
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MODEL_NAME = "gemini-2.5-flash"
_SUCCESS_THRESHOLD = 5
_CHECKPOINT_INTERVAL = 10


def _load_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def _discover_result_files(experiments_dir: Path) -> list[Path]:
    """Walk the experiments tree and collect all result JSON files."""
    return sorted(
        p for p in experiments_dir.rglob("*.json") if p.name != "REPORT.json"
    )


def _score_entry(judge_model, gloss: str, expected: str, predicted: str) -> int | None:
    """Call Gemini to score a single translation on a 1–10 scale."""
    prompt = f"""[Expert Identity]: You are a linguistics professor specialising in Turkish Sign Language (TSL) and Turkish translation.

[Task]: Evaluate the quality of the "Model Prediction" compared to the "Ground Truth", using the "Input Gloss" for context.

[Data]:
- Gloss: "{gloss}"
- Ground Truth: "{expected}"
- Model Prediction: "{predicted}"

[Scoring Rubric (1-10)]:
- 10: Perfect translation. Natural, accurate, and grammatically flawless.
- 8-9: Excellent. Accurate meaning, very minor stylistic or punctuation issues.
- 6-7: Good/Acceptable. Meaning is clear, but has slight grammatical errors or awkward phrasing.
- 4-5: Poor. Meaning is partially preserved but difficult to understand.
- 1-3: Failure. Gibberish or completely wrong meaning.

[Rules]: Focus on Semantic Preservation. Return ONLY the integer score."""
    try:
        response = judge_model.generate_content(prompt)
        match = re.search(r"\b([1-9]|10)\b", response.text.strip())
        return int(match.group(0)) if match else None
    except Exception as exc:
        print(f"\n[ERROR] API call failed: {exc}")
        return None


def _process_file(judge_model, file_path: Path) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    path_parts = file_path.parts
    if "benchmarks" in path_parts:
        idx = list(path_parts).index("benchmarks")
        label = " | ".join(path_parts[idx + 1 : idx + 3]).replace("_", " ").upper()
    else:
        label = file_path.parent.name

    print(f"\n[JUDGE] Assessing: {label}")

    updated = 0
    scores = []

    for entry in tqdm(data, desc="Scoring"):
        if "gemini_score_10" in entry and entry["gemini_score_10"] is not None:
            scores.append(entry["gemini_score_10"])
            continue

        score = _score_entry(judge_model, entry["gloss"], entry["expected"], entry["prediction"])
        if score is not None:
            entry["gemini_score_10"] = score
            scores.append(score)
            updated += 1

            if updated % _CHECKPOINT_INTERVAL == 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

        time.sleep(0.5)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    if scores:
        print(f"[INFO] Done — avg score: {sum(scores) / len(scores):.2f}")


def _generate_report(experiments_dir: Path) -> None:
    result_files = _discover_result_files(experiments_dir)
    final_report: dict[str, dict] = {}

    for file_path in result_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scores = [e["gemini_score_10"] for e in data if "gemini_score_10" in e]
        if not scores:
            continue

        success_rate = len([s for s in scores if s >= _SUCCESS_THRESHOLD]) / len(data) * 100
        avg_chrf = sum(e.get("chrf", 0) for e in data) / len(data)

        key = "_".join(file_path.relative_to(experiments_dir).with_suffix("").parts)
        final_report[key] = {
            "Avg Score": round(sum(scores) / len(scores), 2),
            "Success Rate (%)": round(success_rate, 2),
            "Avg chrF": round(avg_chrf, 2),
            "Samples": len(data),
        }

    report_path = experiments_dir / "REPORT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 50)
    print(f"REPORT GENERATED: {report_path}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments_dir",
        type=Path,
        default=PROJECT_ROOT / "experiments",
        help="Root experiments directory to scan for result files.",
    )
    args = parser.parse_args()

    _load_env(PROJECT_ROOT)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env or export it in your shell.")

    genai.configure(api_key=api_key)
    judge_model = genai.GenerativeModel(_MODEL_NAME)

    result_files = _discover_result_files(args.experiments_dir)
    print(f"[INFO] Found {len(result_files)} result file(s) to judge.")

    for file_path in result_files:
        _process_file(judge_model, file_path)

    _generate_report(args.experiments_dir)


if __name__ == "__main__":
    main()
