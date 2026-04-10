"""Phase A — 8-cell headline claim: precise vs generic × 2 model × 2 domain.

Fixed NL format. Runs 4 cells per model (2 domain × 2 specificity).
Each model runs on its own GPU.

Usage:
    # Qwen on cuda:0
    python -m src.phaseA_runner \
        --model-path /root/autodl-tmp/models/qwen2.5-7b-instruct \
        --device cuda:0 \
        --svg-split medium \
        --output results/phaseA_qwen.json

    # Llama on cuda:1
    python -m src.phaseA_runner \
        --model-path /root/autodl-tmp/models/llama-3.1-8b-instruct \
        --device cuda:1 \
        --svg-split medium \
        --output results/phaseA_llama.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
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


def run_cell(runner, gen_config, samples, verifier, domain, specificity, code_key,
             min_samples: int = 30):
    """Run one cell: filter any-diag -> feedback -> generate -> re-verify.

    D015 redesign:
    - Filter domain expanded from errors_before>0 to total_diagnostics_before>0
      (any severity counts as "bug signal")
    - Primary metric is diagnostic_reduction_rate with parse_error penalty (Bug C):
      if model newly introduces parse_error, effective diags_after is inflated to
      max(diags_before + 10, diags_after) so reduction becomes strongly negative
    - SVG cells emit parse_survival_rate + predicate_fix_rate_among_survivors
    - Python cells keep pass_rate + error_reduction_rate as secondary
    - first_attempt_pass_rate deleted (degenerate under single-turn architecture)
    - INSUFFICIENT_SAMPLES sentinel if filtered < min_samples
    """
    from .feedback.templates import FeedbackFormat, FeedbackSpecificity, render_feedback
    from .inference.prompts import build_correction_prompt
    from .verifiers.diagnostic import Severity

    spec = FeedbackSpecificity(specificity)
    fmt = FeedbackFormat.NL

    # D015 change 1: filter on any-diagnostic (not ERROR-only)
    filtered: list[tuple[object, str, list]] = []
    for sample in samples:
        code = getattr(sample, code_key)
        diags_before = verifier.verify(code)
        if len(diags_before) > 0:
            filtered.append((sample, code, diags_before))
    n_filtered = len(filtered)
    logger.info("[run_cell %s/%s] kept %d/%d samples with any diagnostic before",
                domain, specificity, n_filtered, len(samples))

    if n_filtered < min_samples:
        return {
            "domain": domain,
            "format": "nl",
            "specificity": specificity,
            "status": "INSUFFICIENT_SAMPLES",
            "n_filtered": n_filtered,
            "n_total": len(samples),
            "min_samples": min_samples,
            "note": f"only {n_filtered} samples with any-diag; need >={min_samples}",
        }

    codes = [c for _, c, _ in filtered]
    all_diags_before = [d for _, _, d in filtered]
    feedbacks = [render_feedback(d, fmt, spec) for d in all_diags_before]

    conversations = [
        build_correction_prompt(code, fb, domain, iteration=0)
        for code, fb in zip(codes, feedbacks)
    ]

    # Generate corrections
    t0 = time.time()
    gen_results = runner.generate_chat(conversations, gen_config)
    gen_time = time.time() - t0

    # Re-verify + per-sample analysis
    passed = 0
    total_diags_before = 0
    total_diags_after_effective = 0  # with parse_error penalty
    total_diags_after_raw = 0        # raw count, for diagnostic
    total_errors_before = 0
    total_errors_after = 0
    n_parse_survived = 0                       # SVG only
    predicate_fix_ratios: list[float] = []     # SVG only
    per_sample = []

    for i, (gen_res, diags_before) in enumerate(zip(gen_results, all_diags_before)):
        corrected = extract_code_block(gen_res.output, "svg" if domain == "svg" else "python")
        diags_after = verifier.verify(corrected)
        errors_before = [d for d in diags_before if d.severity == Severity.ERROR]
        errors_after = [d for d in diags_after if d.severity == Severity.ERROR]

        # D015 change 4: parse_error penalty (Bug C strict)
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

        # SVG-specific decomposition
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

    # D015 补丁 2: macro (per-sample mean) + micro (total-based) dual reporting
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
        "format": "nl",
        "specificity": specificity,
        "status": "OK",
        "n_filtered": n_filtered,
        "n_total": len(samples),
        "diagnostic_reduction_rate_macro": round(diag_reduction_macro, 4),  # PRIMARY
        "diagnostic_reduction_rate_micro": round(diag_reduction_micro, 4),  # SECONDARY
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
    else:  # python
        cell_out["pass_rate"] = round(passed / n_filtered, 4)
        cell_out["passed"] = passed
        cell_out["error_reduction_rate"] = round(error_reduction, 4)
        cell_out["avg_errors_before"] = round(total_errors_before / n_filtered, 2)
        cell_out["avg_errors_after"] = round(total_errors_after / n_filtered, 2)

    return cell_out


def _env_sanity_check() -> None:
    """D015 补丁 2 Fix 3: verify runtime env before heavy model loading."""
    import shutil
    import subprocess
    import sys

    logger.info("=== Env Sanity Check ===")
    logger.info("  python: %s", sys.executable)

    # torch + CUDA
    try:
        import torch
        logger.info("  torch=%s cuda=%s gpus=%d", torch.__version__,
                     torch.cuda.is_available(), torch.cuda.device_count())
        if not torch.cuda.is_available():
            logger.error("  FATAL: torch.cuda not available")
            sys.exit(1)
    except ImportError:
        logger.error("  FATAL: torch not installed")
        sys.exit(1)

    # bandit / pylint reachable from this python's venv
    from .verifiers.python_static import _tool_path
    for tool in ("bandit", "pylint"):
        p = _tool_path(tool)
        found = shutil.which(p) is not None or Path(p).exists()
        logger.info("  %s: %s (%s)", tool, p, "OK" if found else "MISSING")
        if not found:
            logger.warning("  %s not found — Python cells may get 0 diagnostics", tool)

    logger.info("=== Env Sanity OK ===")


def main():
    parser = argparse.ArgumentParser(description="Phase A: precise vs generic × 2 domain")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--svg-split", default="medium")
    parser.add_argument("--n-svg", type=int, default=0, help="0 = use all samples")
    parser.add_argument("--n-python", type=int, default=0, help="0 = use all samples")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--output", default="results/phaseA.json")
    args = parser.parse_args()

    _env_sanity_check()

    from .data.prepare_svg import load_svg_dataset
    from .data.prepare_python import load_python_dataset
    from .inference.vllm_runner import GenerationConfig, VLLMRunner
    from .verifiers.python_static import PythonStaticVerifier
    from .verifiers.svg_geometric import SVGGeometricVerifier

    # Load data — use all samples (n_samples=0 or very large)
    n_svg = args.n_svg if args.n_svg > 0 else 9999
    n_python = args.n_python if args.n_python > 0 else 9999

    logger.info("Loading SVG dataset (split=%s)...", args.svg_split)
    svg_samples = load_svg_dataset(
        dataset_name="xiaoooobai/SVGenius",
        n_samples=n_svg,
        seed=42,
        split=args.svg_split,
    )
    logger.info("Loaded %d SVG samples", len(svg_samples))

    logger.info("Loading Python dataset...")
    python_samples = load_python_dataset(
        dataset_name="s2e-lab/SecurityEval",
        n_samples=n_python,
        seed=42,
    )
    logger.info("Loaded %d Python samples", len(python_samples))

    # Init
    svg_verifier = SVGGeometricVerifier()
    python_verifier = PythonStaticVerifier()

    logger.info("Loading model %s on %s...", args.model_path, args.device)
    runner = VLLMRunner(
        model_name=args.model_path,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
        device=args.device,
    )
    gen_config = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0)

    # Run 4 cells: 2 domain × 2 specificity
    # D015 Step 7: incremental dump after every cell so a mid-run kill
    # doesn't lose completed cells.
    results: dict = {}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _dump() -> None:
        summary = {
            "model": args.model_path,
            "device": args.device,
            "svg_split": args.svg_split,
            "cells": results,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    for domain, samples, verifier, code_key in [
        ("svg", svg_samples, svg_verifier, "svg_string"),
        ("python", python_samples, python_verifier, "code"),
    ]:
        for specificity in ["precise", "generic"]:
            cell_key = f"{domain}_{specificity}"
            logger.info("=== Cell: %s (%d samples in split) ===", cell_key, len(samples))
            cell = run_cell(runner, gen_config, samples, verifier, domain, specificity, code_key)
            results[cell_key] = cell
            _dump()  # persist after every cell (D015 incremental dump)

            if cell.get("status") == "INSUFFICIENT_SAMPLES":
                logger.info("  SKIPPED: insufficient samples (n_filtered=%d)", cell["n_filtered"])
                continue

            if domain == "svg":
                logger.info(
                    "  diag_reduction_macro=%.1f%% micro=%.1f%%, parse_survival=%.1f%%, pred_fix=%.1f%%, n=%d, time=%.1fs",
                    cell["diagnostic_reduction_rate_macro"] * 100,
                    cell["diagnostic_reduction_rate_micro"] * 100,
                    cell["parse_survival_rate"] * 100,
                    cell["predicate_fix_rate_among_survivors"] * 100,
                    cell["n_filtered"],
                    cell["gen_time_seconds"],
                )
            else:
                logger.info(
                    "  diag_reduction_macro=%.1f%% micro=%.1f%%, pass_rate=%.1f%%, error_reduction=%.1f%%, n=%d, time=%.1fs",
                    cell["diagnostic_reduction_rate_macro"] * 100,
                    cell["diagnostic_reduction_rate_micro"] * 100,
                    cell["pass_rate"] * 100,
                    cell["error_reduction_rate"] * 100,
                    cell["n_filtered"],
                    cell["gen_time_seconds"],
                )

    runner.shutdown()
    _dump()  # final dump (redundant but safe)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
