"""Compute prompt token counts for Phase B: 3 format × 2 specificity on Python domain.

Usage:
    python -m src.analyze_token_counts \
        --tokenizer /root/autodl-tmp/models/qwen2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .data.prepare_python import load_python_dataset
from .feedback.templates import FeedbackFormat, FeedbackSpecificity, render_feedback
from .inference.prompts import build_correction_prompt
from .verifiers.python_static import PythonStaticVerifier


def count_prompt_tokens(conversation: list[dict[str, str]], tokenizer) -> int:
    """Count tokens in the full prompt (system + user messages)."""
    # Use apply_chat_template for accurate token counting
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="/root/autodl-tmp/models/qwen2.5-7b-instruct")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load data + verifier
    samples = load_python_dataset(dataset_name="s2e-lab/SecurityEval", n_samples=9999, seed=42)
    verifier = PythonStaticVerifier()

    # Filter samples with diagnostics (same as Phase B)
    filtered = []
    for sample in samples:
        diags = verifier.verify(sample.code)
        if len(diags) > 0:
            filtered.append((sample, diags))
    print(f"Filtered: {len(filtered)} / {len(samples)} samples with diagnostics")

    formats = [FeedbackFormat.NL, FeedbackFormat.RAW_JSON, FeedbackFormat.HYBRID]
    specificities = [FeedbackSpecificity.PRECISE, FeedbackSpecificity.GENERIC]

    results = {}
    for fmt in formats:
        for spec in specificities:
            key = f"{fmt.value}_{spec.value}"
            token_counts = []
            feedback_only_counts = []
            for sample, diags in filtered:
                fb = render_feedback(diags, fmt, spec)
                conv = build_correction_prompt(sample.code, fb, "python", iteration=0)
                n_tokens = count_prompt_tokens(conv, tokenizer)
                token_counts.append(n_tokens)
                # Also count feedback-only tokens
                fb_tokens = len(tokenizer.encode(fb))
                feedback_only_counts.append(fb_tokens)

            results[key] = {
                "mean": statistics.mean(token_counts),
                "std": statistics.stdev(token_counts) if len(token_counts) > 1 else 0,
                "min": min(token_counts),
                "max": max(token_counts),
                "fb_mean": statistics.mean(feedback_only_counts),
                "fb_std": statistics.stdev(feedback_only_counts) if len(feedback_only_counts) > 1 else 0,
                "n": len(token_counts),
            }

    # Print table
    print(f"\nFormat × Specificity Token Statistics (n={len(filtered)}, Qwen tokenizer):\n")
    print("Full prompt tokens:")
    print(f"| {'Format':<10} | {'Specificity':<12} | {'Mean':>10} | {'Std':>8} | {'Min':>6} | {'Max':>6} |")
    print(f"|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*10}|{'-'*8}|{'-'*8}|")
    for fmt in formats:
        for spec in specificities:
            key = f"{fmt.value}_{spec.value}"
            r = results[key]
            print(f"| {fmt.value:<10} | {spec.value:<12} | {r['mean']:>10.1f} | {r['std']:>8.1f} | {r['min']:>6d} | {r['max']:>6d} |")

    print(f"\nFeedback-only tokens:")
    print(f"| {'Format':<10} | {'Specificity':<12} | {'Mean':>10} | {'Std':>8} |")
    print(f"|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*10}|")
    for fmt in formats:
        for spec in specificities:
            key = f"{fmt.value}_{spec.value}"
            r = results[key]
            print(f"| {fmt.value:<10} | {spec.value:<12} | {r['fb_mean']:>10.1f} | {r['fb_std']:>8.1f} |")

    # Pairwise differences
    print("\nPairwise differences (mean full-prompt tokens):")
    nl_precise = results["nl_precise"]["mean"]
    nl_generic = results["nl_generic"]["mean"]
    for fmt in [FeedbackFormat.HYBRID, FeedbackFormat.RAW_JSON]:
        for spec in specificities:
            key = f"{fmt.value}_{spec.value}"
            base_key = f"nl_{spec.value}"
            diff = results[key]["mean"] - results[base_key]["mean"]
            pct = diff / results[base_key]["mean"] * 100
            print(f"  {key} - {base_key} = {diff:+.1f} tokens ({pct:+.1f}%)")

    # precise vs generic
    print("\nPrecise vs Generic differences (mean full-prompt tokens):")
    for fmt in formats:
        p = results[f"{fmt.value}_precise"]["mean"]
        g = results[f"{fmt.value}_generic"]["mean"]
        diff = p - g
        pct = diff / g * 100
        print(f"  {fmt.value}: precise - generic = {diff:+.1f} tokens ({pct:+.1f}%)")

    # Save JSON
    out_path = Path("results/token_count_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
