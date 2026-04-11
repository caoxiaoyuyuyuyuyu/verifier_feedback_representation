"""No-Feedback baseline — measure self-correction ability without any verifier signal.

Gives the model only the task prompt + initial code (no verifier feedback), asks
for a generic "review and improve" pass, then re-verifies and computes DRR using
exactly the same logic as phaseA_runner.py. Produces one cell per domain:
"svg_no_feedback" and "python_no_feedback".

Usage:
    python -m src.no_feedback_runner \
        --model-path /root/autodl-tmp/models/qwen2.5-7b-instruct \
        --device cuda:0 \
        --svg-split medium \
        --output results/no_feedback_qwen.json
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


# ---------------------------------------------------------------------------
# No-feedback prompt construction
# ---------------------------------------------------------------------------

_NO_FEEDBACK_SYSTEM = {
    "svg": (
        "You are an SVG code review assistant. "
        "Given an SVG, review it and improve its quality while preserving the "
        "original design intent. Output only the improved SVG code."
    ),
    "python": (
        "You are a Python code review assistant. "
        "Given Python code, review it and improve its quality while preserving "
        "the original functionality. Output only the improved Python code."
    ),
}

_NO_FEEDBACK_USER_TEMPLATE = """\
## Original Code

```{lang}
{code}
```

## Task

Review the code above and improve it. Output only the improved code in a ```{lang}``` block."""


def build_no_feedback_prompt(code: str, domain: str) -> list[dict[str, str]]:
    """Build a correction prompt with NO verifier feedback.

    The model sees only the initial code and a generic "review and improve"
    instruction. This measures self-correction ability with zero external signal.
    """
    lang = "svg" if domain == "svg" else "python"
    system = _NO_FEEDBACK_SYSTEM[domain]
    user_content = _NO_FEEDBACK_USER_TEMPLATE.format(lang=lang, code=code)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def extract_code_block(text: str, lang: str = "") -> str:
    pattern = rf"```{lang}\s*\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Cell runner — mirrors phaseA_runner.run_cell but without feedback rendering
# ---------------------------------------------------------------------------

def run_cell(runner, gen_config, samples, verifier, domain, code_key,
             min_samples: int = 30):
    """Run one no-feedback cell for a single domain.

    Mirrors phaseA_runner.run_cell evaluation exactly (any-diag filter,
    parse_error penalty, macro+micro DRR, domain secondary metrics), differing
    only in prompt construction (no feedback is passed to the model).
    """
    from .verifiers.diagnostic import Severity

    # Filter on any-diagnostic (same as phase A)
    filtered: list[tuple[object, str, list]] = []
    for sample in samples:
        code = getattr(sample, code_key)
        diags_before = verifier.verify(code)
        if len(diags_before) > 0:
            filtered.append((sample, code, diags_before))
    n_filtered = len(filtered)
    logger.info("[run_cell %s/no_feedback] kept %d/%d samples with any diagnostic before",
                domain, n_filtered, len(samples))

    if n_filtered < min_samples:
        return {
            "domain": domain,
            "format": "none",
            "specificity": "no_feedback",
            "status": "INSUFFICIENT_SAMPLES",
            "n_filtered": n_filtered,
            "n_total": len(samples),
            "min_samples": min_samples,
            "note": f"only {n_filtered} samples with any-diag; need >={min_samples}",
        }

    codes = [c for _, c, _ in filtered]
    all_diags_before = [d for _, _, d in filtered]

    conversations = [build_no_feedback_prompt(code, domain) for code in codes]

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
        "format": "none",
        "specificity": "no_feedback",
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


def _env_sanity_check() -> None:
    import shutil
    import sys

    logger.info("=== Env Sanity Check ===")
    logger.info("  python: %s", sys.executable)
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

    from .verifiers.python_static import _tool_path
    for tool in ("bandit", "pylint"):
        p = _tool_path(tool)
        found = shutil.which(p) is not None or Path(p).exists()
        logger.info("  %s: %s (%s)", tool, p, "OK" if found else "MISSING")
        if not found:
            logger.warning("  %s not found — Python cells may get 0 diagnostics", tool)

    logger.info("=== Env Sanity OK ===")


def main():
    parser = argparse.ArgumentParser(
        description="No-feedback baseline: review+improve with zero verifier signal")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--svg-split", default="medium")
    parser.add_argument("--n-svg", type=int, default=100)
    parser.add_argument("--n-python", type=int, default=0, help="0 = use all samples")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--output", default="results/no_feedback.json")
    parser.add_argument("--domains", default="svg,python",
                        help="comma-separated domains to run")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="minimum filtered samples to run a cell (default 30)")
    args = parser.parse_args()

    _env_sanity_check()

    from .data.prepare_svg import load_svg_dataset
    from .data.prepare_python import load_python_dataset
    from .inference.vllm_runner import GenerationConfig, VLLMRunner
    from .verifiers.python_static import PythonStaticVerifier
    from .verifiers.svg_geometric import SVGGeometricVerifier

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

    results: dict = {}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _dump() -> None:
        summary = {
            "model": args.model_path,
            "device": args.device,
            "svg_split": args.svg_split,
            "condition": "no_feedback",
            "cells": results,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    active_domains = [d.strip() for d in args.domains.split(",")]
    all_cells = [
        ("svg", svg_samples, svg_verifier, "svg_string"),
        ("python", python_samples, python_verifier, "code"),
    ]
    for domain, samples, verifier, code_key in all_cells:
        if domain not in active_domains:
            logger.info("=== Skipping domain: %s (not in --domains=%s) ===", domain, args.domains)
            continue
        cell_key = f"{domain}_no_feedback"
        logger.info("=== Cell: %s (%d samples in split) ===", cell_key, len(samples))
        cell = run_cell(runner, gen_config, samples, verifier, domain, code_key,
                       min_samples=args.min_samples)
        results[cell_key] = cell
        _dump()

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
    _dump()
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
