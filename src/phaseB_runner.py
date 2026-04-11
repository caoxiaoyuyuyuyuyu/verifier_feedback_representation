"""Phase B — format ablation: 3 format × 2 specificity × 2 domain × 1 model (Qwen).

12 cells total. Qwen2.5-7B-Instruct only (D012 pre-registered).
Reuses run_cell from phaseA_runner with format parameterization.

Usage:
    python -m src.phaseB_runner \
        --model-path /root/autodl-tmp/models/qwen2.5-7b-instruct \
        --device cuda:0 \
        --output results/phaseB_qwen.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_code_block(text: str, lang: str = "") -> str:
    pattern = rf"```{lang}\s*\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def run_cell(runner, gen_config, samples, verifier, domain, specificity, fmt_enum,
             code_key, min_samples: int = 30):
    """Run one cell with parameterized format (unlike Phase A's hardcoded NL)."""
    from .feedback.templates import FeedbackSpecificity, render_feedback
    from .inference.prompts import build_correction_prompt
    from .verifiers.diagnostic import Severity

    spec = FeedbackSpecificity(specificity)

    filtered: list[tuple[object, str, list]] = []
    for sample in samples:
        code = getattr(sample, code_key)
        diags_before = verifier.verify(code)
        if len(diags_before) > 0:
            filtered.append((sample, code, diags_before))
    n_filtered = len(filtered)
    logger.info("[run_cell %s/%s/%s] kept %d/%d samples",
                domain, fmt_enum.value, specificity, n_filtered, len(samples))

    if n_filtered < min_samples:
        return {
            "domain": domain,
            "format": fmt_enum.value,
            "specificity": specificity,
            "status": "INSUFFICIENT_SAMPLES",
            "n_filtered": n_filtered,
            "n_total": len(samples),
            "min_samples": min_samples,
            "note": f"only {n_filtered} samples with any-diag; need >={min_samples}",
        }

    codes = [c for _, c, _ in filtered]
    all_diags_before = [d for _, _, d in filtered]
    feedbacks = [render_feedback(d, fmt_enum, spec) for d in all_diags_before]

    conversations = [
        build_correction_prompt(code, fb, domain, iteration=0)
        for code, fb in zip(codes, feedbacks)
    ]

    t0 = time.time()
    gen_results = runner.generate_chat(conversations, gen_config)
    gen_time = time.time() - t0

    passed = 0
    total_diags_before = 0
    total_diags_after_effective = 0
    total_diags_after_raw = 0
    total_errors_before = 0
    total_errors_after = 0
    n_parse_survived = 0
    predicate_fix_ratios: list[float] = []
    per_sample = []

    for i, (gen_res, diags_before) in enumerate(zip(gen_results, all_diags_before)):
        corrected = extract_code_block(gen_res.output, "svg" if domain == "svg" else "python")
        diags_after = verifier.verify(corrected)
        errors_before = [d for d in diags_before if d.severity == Severity.ERROR]
        errors_after = [d for d in diags_after if d.severity == Severity.ERROR]

        had_parse_err_before = any(d.rule_id == "parse_error" for d in diags_before)
        has_parse_err_after = any(d.rule_id == "parse_error" for d in diags_after)
        newly_broken = has_parse_err_after and not had_parse_err_before
        if newly_broken:
            effective_diags_after = max(len(diags_before) + 10, len(diags_after))
        else:
            effective_diags_after = len(diags_after)

        sample_passed = len(errors_after) == 0
        if sample_passed:
            passed += 1

        total_diags_before += len(diags_before)
        total_diags_after_raw += len(diags_after)
        total_diags_after_effective += effective_diags_after
        total_errors_before += len(errors_before)
        total_errors_after += len(errors_after)

        if domain == "svg":
            if not has_parse_err_after:
                n_parse_survived += 1
                warns_before = sum(1 for d in diags_before if d.severity == Severity.WARNING)
                warns_after = sum(1 for d in diags_after if d.severity == Severity.WARNING)
                if warns_before > 0:
                    predicate_fix_ratios.append((warns_before - warns_after) / warns_before)

        per_sample.append({
            "idx": i,
            "diags_before": len(diags_before),
            "diags_after": len(diags_after),
            "diags_after_effective": effective_diags_after,
            "errors_before": len(errors_before),
            "errors_after": len(errors_after),
            "newly_broken": newly_broken,
            "had_parse_err_before": had_parse_err_before,
            "has_parse_err_after": has_parse_err_after,
            "passed": sample_passed,
            "tokens": gen_res.token_count,
        })

    per_sample_reduction = [
        1 - s["diags_after_effective"] / max(s["diags_before"], 1)
        for s in per_sample
    ]
    diag_reduction_macro = (
        sum(per_sample_reduction) / len(per_sample_reduction)
        if per_sample_reduction else 0.0
    )
    diag_reduction_micro = (
        1 - total_diags_after_effective / total_diags_before
        if total_diags_before > 0 else 0.0
    )
    error_reduction = (
        1 - total_errors_after / total_errors_before
        if total_errors_before > 0 else 0.0
    )

    cell_out: dict = {
        "domain": domain,
        "format": fmt_enum.value,
        "specificity": specificity,
        "status": "OK",
        "n_filtered": n_filtered,
        "n_total": len(samples),
        "diagnostic_reduction_rate_macro": round(diag_reduction_macro, 4),
        "diagnostic_reduction_rate_micro": round(diag_reduction_micro, 4),
        "per_sample_reduction": [round(r, 4) for r in per_sample_reduction],
        "avg_diagnostics_before": round(total_diags_before / n_filtered, 2),
        "avg_diagnostics_after_raw": round(total_diags_after_raw / n_filtered, 2),
        "avg_diagnostics_after_effective": round(total_diags_after_effective / n_filtered, 2),
        "gen_time_seconds": round(gen_time, 1),
        "avg_tokens": round(
            sum(r.token_count for r in gen_results) / len(gen_results), 1
        ) if gen_results else 0,
        "per_sample": per_sample,
    }

    if domain == "svg":
        cell_out["parse_survival_rate"] = round(n_parse_survived / n_filtered, 4)
        cell_out["predicate_fix_rate_among_survivors"] = round(
            (sum(predicate_fix_ratios) / len(predicate_fix_ratios))
            if predicate_fix_ratios else 0.0,
            4,
        )
        cell_out["n_survivors_with_warn_before"] = len(predicate_fix_ratios)
    else:
        cell_out["pass_rate"] = round(passed / n_filtered, 4)
        cell_out["passed"] = passed
        cell_out["error_reduction_rate"] = round(error_reduction, 4)
        cell_out["avg_errors_before"] = round(total_errors_before / n_filtered, 2)
        cell_out["avg_errors_after"] = round(total_errors_after / n_filtered, 2)

    return cell_out


def main():
    parser = argparse.ArgumentParser(description="Phase B: format ablation (3 fmt × 2 spec × 2 domain)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--svg-split", default="medium")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--output", default="results/phaseB_qwen.json")
    parser.add_argument("--domains", default="svg,python")
    parser.add_argument("--formats", default="nl,raw_json,hybrid",
                        help="comma-separated formats to run")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for data loading and generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="generation temperature (0.0 = greedy)")
    args = parser.parse_args()

    from .data.prepare_svg import load_svg_dataset
    from .data.prepare_python import load_python_dataset
    from .feedback.templates import FeedbackFormat
    from .inference.vllm_runner import GenerationConfig, VLLMRunner
    from .verifiers.python_static import PythonStaticVerifier
    from .verifiers.svg_geometric import SVGGeometricVerifier

    logger.info("Loading SVG dataset (split=%s)...", args.svg_split)
    svg_samples = load_svg_dataset(
        dataset_name="xiaoooobai/SVGenius",
        n_samples=9999,
        seed=args.seed,
        split=args.svg_split,
    )
    logger.info("Loaded %d SVG samples", len(svg_samples))

    logger.info("Loading Python dataset...")
    python_samples = load_python_dataset(
        dataset_name="s2e-lab/SecurityEval",
        n_samples=9999,
        seed=args.seed,
    )
    logger.info("Loaded %d Python samples", len(python_samples))

    svg_verifier = SVGGeometricVerifier()
    python_verifier = PythonStaticVerifier()

    logger.info("Loading model %s on %s...", args.model_path, args.device)
    runner = VLLMRunner(
        model_name=args.model_path,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
        device=args.device,
        seed=args.seed,
    )
    gen_config = GenerationConfig(max_tokens=args.max_tokens, temperature=args.temperature)

    results: dict = {}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _dump() -> None:
        summary = {
            "model": args.model_path,
            "device": args.device,
            "svg_split": args.svg_split,
            "phase": "B",
            "cells": results,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    active_domains = [d.strip() for d in args.domains.split(",")]
    active_formats = [FeedbackFormat(f.strip()) for f in args.formats.split(",")]

    all_cells = [
        ("svg", svg_samples, svg_verifier, "svg_string"),
        ("python", python_samples, python_verifier, "code"),
    ]

    for domain, samples, verifier, code_key in all_cells:
        if domain not in active_domains:
            logger.info("=== Skipping domain: %s ===", domain)
            continue
        for fmt in active_formats:
            for specificity in ["precise", "generic"]:
                cell_key = f"{domain}_{fmt.value}_{specificity}"
                logger.info("=== Cell: %s (%d samples) ===", cell_key, len(samples))
                cell = run_cell(runner, gen_config, samples, verifier, domain,
                               specificity, fmt, code_key, min_samples=args.min_samples)
                results[cell_key] = cell
                _dump()

                if cell.get("status") == "INSUFFICIENT_SAMPLES":
                    logger.info("  SKIPPED: insufficient samples (n=%d)", cell["n_filtered"])
                    continue

                logger.info(
                    "  drr_macro=%.1f%% micro=%.1f%%, n=%d, time=%.1fs",
                    cell["diagnostic_reduction_rate_macro"] * 100,
                    cell["diagnostic_reduction_rate_micro"] * 100,
                    cell["n_filtered"],
                    cell["gen_time_seconds"],
                )

    runner.shutdown()
    _dump()
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
