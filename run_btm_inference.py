"""
Beyond the Mean: Data Collection Script
========================================
Collects K=10 stochastic samples per item per model at T=0.7
for Reliable Change Index analysis.

Pre-registration: OSF [insert DOI after registration]
Hardware: AMD RX 7900 GRE 16GB, Vulkan backend, llama-cpp-python
Author: Jon-Paul Cacioli (ORCID: 0009-0000-7054-2014)

Setup:
    1. Add these entries to your local_config.py MODEL_PATHS:
       "llama3.1-8b": r"D:\\bcb_pilot\\models\\Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
       "qwen3-8b":    r"<path after download>",

    2. Place sampled_items.jsonl in the same directory as this script,
       or pass --items <path>.

Usage:
    python run_btm_inference.py --model llama3-8b
    python run_btm_inference.py --model llama3.1-8b
    python run_btm_inference.py --model qwen2.5-7b
    python run_btm_inference.py --model qwen3-8b
    python run_btm_inference.py --model all
    python run_btm_inference.py --model llama3-8b --verify
    python run_btm_inference.py --model llama3-8b --resume 500
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from llama_cpp import Llama

# Import model paths from your existing config
from local_config import MODEL_PATHS

# ============================================================
# Configuration
# ============================================================

K = 10          # samples per item (pre-registered)
T = 0.7         # temperature (pre-registered)
MAX_TOKENS = 8  # short response expected (single letter)

# Pre-registered execution order
RUN_ORDER = ["llama3-8b", "llama3.1-8b", "qwen2.5-7b", "qwen3-8b"]

# Human-readable model names for logging
MODEL_NAMES = {
    "llama3-8b":   "Meta-Llama-3-8B-Instruct",
    "llama3.1-8b": "Meta-Llama-3.1-8B-Instruct",
    "qwen2.5-7b":  "Qwen2.5-7B-Instruct",
    "qwen3-8b":    "Qwen3-8B",
}

# Default paths
DEFAULT_ITEMS = r"D:\bcb_pilot\data\sampled_items.jsonl"
DEFAULT_OUTPUT = r"D:\beyond_the_mean\data"

# System prompts (pre-registered, Section 2.3)
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following multiple-choice "
    "question by responding with a single letter (A through J). "
    "Do not explain your reasoning."
)

SYSTEM_PROMPT_QWEN3 = (
    "/no_think You are a helpful assistant. Answer the following multiple-choice "
    "question by responding with a single letter (A through J). "
    "Do not explain your reasoning."
)

VALID_LETTERS = set("ABCDEFGHIJ")


# ============================================================
# Helpers
# ============================================================

def load_items(path: str) -> list[dict]:
    """Load items from JSONL file."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Loaded {len(items)} items from {path}")
    return items


def extract_answer(response: str) -> str | None:
    """Extract first valid letter A-J from model response.
    Pre-registered rule (Section 2.5): first valid letter A-J.
    Strips empty <think></think> tags from Qwen3 before extraction."""
    if not response:
        return None
    import re
    # Remove empty think blocks (Qwen3 chat template artefact)
    cleaned = re.sub(r"<think>\s*</think>\s*", "", response, flags=re.IGNORECASE)
    for char in cleaned.strip():
        if char.upper() in VALID_LETTERS:
            return char.upper()
    return None


def check_thinking_traces(response: str) -> bool:
    """Check if Qwen3 response contains substantive thinking traces.
    Pre-registered verification (Section 2.3).
    Empty <think></think> tags are a chat-template artefact in Qwen3
    and do not constitute reasoning. Only flag responses where the
    think block contains non-whitespace content."""
    import re
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        if len(content) > 0:
            return True  # actual reasoning content
    return False


def format_question(item: dict) -> str:
    """Format item into user message."""
    return f"{item['question']}\n\n{item['options_formatted']}"


def get_system_prompt(model_key: str) -> str:
    """Return appropriate system prompt for model."""
    if model_key == "qwen3-8b":
        return SYSTEM_PROMPT_QWEN3
    return SYSTEM_PROMPT


def compute_seed(item_index: int, k: int) -> int:
    """Deterministic seed: seed_k = item_index * 10 + k.
    Pre-registered (Section 2.4)."""
    return item_index * 10 + k


# ============================================================
# Inference
# ============================================================

def run_single_trial(
    model: Llama,
    model_key: str,
    item: dict,
    item_index: int,
    k: int,
) -> dict:
    """Run one inference trial. Returns trial record."""
    seed = compute_seed(item_index, k)
    system_prompt = get_system_prompt(model_key)
    user_message = format_question(item)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    t0 = time.time()
    response = model.create_chat_completion(
        messages=messages,
        temperature=T,
        max_tokens=MAX_TOKENS,
        seed=seed,
    )
    elapsed = time.time() - t0

    raw_response = response["choices"][0]["message"]["content"]
    extracted = extract_answer(raw_response)
    correct_answer = item["answer"]
    is_correct = (extracted == correct_answer) if extracted else None

    record = {
        "item_id": item["item_id"],
        "item_index": item_index,
        "domain": item["domain"],
        "k": k,
        "seed": seed,
        "model": model_key,
        "raw_response": raw_response,
        "extracted_answer": extracted,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "is_missing": extracted is None,
        "elapsed_s": round(elapsed, 3),
    }

    if model_key == "qwen3-8b":
        record["has_thinking_trace"] = check_thinking_traces(raw_response)

    return record


def run_model(model_key: str, items: list[dict], output_dir: str, resume_from: int = 0):
    """Run all K=10 samples for all items for one model."""
    if model_key not in MODEL_PATHS:
        print(f"ERROR: '{model_key}' not found in local_config.py MODEL_PATHS.")
        print(f"Add it to local_config.py and re-run.")
        return

    model_path = MODEL_PATHS[model_key]
    model_name = MODEL_NAMES.get(model_key, model_key)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"trials_{model_key.replace('.', '_')}.jsonl")

    print(f"\n{'='*60}")
    print(f"Model:  {model_name} ({model_key})")
    print(f"Path:   {model_path}")
    print(f"Output: {output_path}")
    print(f"Items:  {len(items)}, K={K}, T={T}")
    print(f"Total trials: {len(items) * K}")
    if resume_from > 0:
        print(f"Resuming from item index {resume_from}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    t_load = time.time()
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    print(f"Model loaded in {time.time() - t_load:.1f}s.\n")

    # Resume logic
    mode = "a" if resume_from > 0 else "w"
    total_trials = 0
    missing_count = 0
    thinking_trace_count = 0
    start_time = time.time()

    with open(output_path, mode, encoding="utf-8") as out_f:
        for idx in range(resume_from, len(items)):
            item = items[idx]

            for k in range(K):
                record = run_single_trial(model, model_key, item, idx, k)
                out_f.write(json.dumps(record) + "\n")

                total_trials += 1
                if record["is_missing"]:
                    missing_count += 1
                if model_key == "qwen3-8b" and record.get("has_thinking_trace"):
                    thinking_trace_count += 1

            # Progress every 100 items
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                items_done = idx + 1 - resume_from
                rate = items_done / elapsed if elapsed > 0 else 0
                remaining = (len(items) - idx - 1) / rate if rate > 0 else 0
                missing_pct = (missing_count / total_trials * 100) if total_trials > 0 else 0

                print(
                    f"  [{idx+1}/{len(items)}] "
                    f"{total_trials} trials | "
                    f"{missing_count} missing ({missing_pct:.1f}%) | "
                    f"{rate:.1f} items/s | "
                    f"ETA {remaining/60:.0f} min"
                )
                out_f.flush()

    elapsed_total = time.time() - start_time
    missing_pct = (missing_count / total_trials * 100) if total_trials > 0 else 0

    print(f"\n{'='*60}")
    print(f"COMPLETE: {model_name}")
    print(f"  Trials:  {total_trials}")
    print(f"  Missing: {missing_count} ({missing_pct:.1f}%)")
    if model_key == "qwen3-8b":
        think_pct = (thinking_trace_count / total_trials * 100) if total_trials > 0 else 0
        print(f"  Thinking traces: {thinking_trace_count} ({think_pct:.1f}%)")
        if think_pct > 5:
            print(f"  *** WARNING: Thinking trace rate exceeds 5% pre-registered threshold ***")
    print(f"  Time:    {elapsed_total/60:.1f} min")
    print(f"  Output:  {output_path}")
    print(f"{'='*60}\n")

    del model


# ============================================================
# Verification
# ============================================================

def verify_output(model_key: str, output_dir: str):
    """Quick verification of output file."""
    output_path = os.path.join(output_dir, f"trials_{model_key.replace('.', '_')}.jsonl")
    if not os.path.exists(output_path):
        print(f"No output file found: {output_path}")
        return

    records = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_records = len(records)
    n_items = len(set(r["item_index"] for r in records))
    n_missing = sum(1 for r in records if r["is_missing"])
    n_valid = n_records - n_missing
    n_correct = sum(1 for r in records if r["is_correct"] is True)
    accuracy = n_correct / n_valid if n_valid > 0 else 0

    print(f"\nVerification: {MODEL_NAMES.get(model_key, model_key)}")
    print(f"  Records:  {n_records} (expected {2000 * K})")
    print(f"  Items:    {n_items} (expected 2000)")
    print(f"  Missing:  {n_missing} ({n_missing/n_records*100:.1f}%)")
    print(f"  Accuracy: {accuracy*100:.1f}%")

    if model_key == "qwen3-8b":
        n_think = sum(1 for r in records if r.get("has_thinking_trace"))
        print(f"  Thinking traces: {n_think} ({n_think/n_records*100:.1f}%)")

    # Check all K samples present per item
    from collections import Counter
    item_counts = Counter(r["item_index"] for r in records)
    incomplete = {idx: c for idx, c in item_counts.items() if c != K}
    if incomplete:
        print(f"  WARNING: {len(incomplete)} items with != {K} samples")
        for idx, c in sorted(incomplete.items())[:5]:
            print(f"    item_index={idx}: {c} samples")
    else:
        print(f"  All items have exactly {K} samples.")

    # Domain breakdown
    from collections import defaultdict
    domain_correct = defaultdict(int)
    domain_valid = defaultdict(int)
    for r in records:
        if not r["is_missing"]:
            domain_valid[r["domain"]] += 1
            if r["is_correct"]:
                domain_correct[r["domain"]] += 1
    print(f"  Domain accuracy:")
    for d in sorted(domain_valid.keys()):
        acc = domain_correct[d] / domain_valid[d] if domain_valid[d] > 0 else 0
        print(f"    {d}: {acc*100:.1f}% ({domain_valid[d]} valid trials)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Beyond the Mean: Data Collection (K=10, T=0.7)"
    )
    parser.add_argument(
        "--model",
        choices=RUN_ORDER + ["all"],
        required=True,
        help="Which model to run (or 'all' for pre-registered sequential order)",
    )
    parser.add_argument(
        "--items",
        default=DEFAULT_ITEMS,
        help=f"Path to sampled_items.jsonl (default: {DEFAULT_ITEMS})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Resume from item index (0-indexed). Default: 0 (start fresh)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output file(s) without running inference",
    )
    args = parser.parse_args()

    models_to_run = RUN_ORDER if args.model == "all" else [args.model]

    items = load_items(args.items)
    assert len(items) == 2000, f"Expected 2000 items, got {len(items)}"

    for model_key in models_to_run:
        if args.verify:
            verify_output(model_key, args.output)
        else:
            run_model(model_key, items, args.output, resume_from=args.resume)
            verify_output(model_key, args.output)


if __name__ == "__main__":
    main()
