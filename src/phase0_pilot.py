"""Phase 0 — H3 Directional Pilot.

50-sample NL vs hybrid on each domain, Qwen3.5-9B only.
Decision matrix for H3 (format × domain interaction) framing.

Usage:
    python -m src.phase0_pilot \
        --model-path /root/autodl-tmp/models/qwen3.5-9b \
        --svg-cache /root/autodl-tmp/data/svgenius \
        --python-cache /root/autodl-tmp/data/security_eval \
        --output results/phase0_pilot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def extract_code_block(text: str, lang: str = "") -> str:
    """从 LLM 输出中提取代码块。"""
    pattern = rf"```{lang}\s*\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: 如果没有代码块标记，返回整个输出
    return text.strip()


def run_pilot(args: argparse.Namespace) -> dict:
    """执行 Phase 0 pilot 实验。"""
    from .data.prepare_svg import load_svg_dataset
    from .data.prepare_python import load_python_dataset
    from .feedback.templates import (
        FeedbackFormat, FeedbackSpecificity, render_feedback,
    )
    from .inference.prompts import build_correction_prompt
    from .inference.vllm_runner import GenerationConfig, VLLMRunner
    from .verifiers.python_static import PythonStaticVerifier
    from .verifiers.svg_geometric import SVGGeometricVerifier
    from .verifiers.diagnostic import Severity

    n_samples = args.n_samples
    results = {}

    # --- Load data ---
    logger.info("Loading SVG dataset (%d samples)...", n_samples)
    svg_samples = load_svg_dataset(
        dataset_name="xiaoooobai/SVGenius",
        n_samples=n_samples,
        seed=42,
        split=args.svg_split,
    )

    logger.info("Loading Python dataset (%d samples)...", n_samples)
    python_samples = load_python_dataset(
        dataset_name="s2e-lab/SecurityEval",
        n_samples=n_samples,
        seed=42,
    )

    # --- Init verifiers ---
    svg_verifier = SVGGeometricVerifier()
    python_verifier = PythonStaticVerifier()

    # --- Init vLLM ---
    logger.info("Initializing vLLM with %s...", args.model_path)
    runner = VLLMRunner(
        model_name=args.model_path,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
        device=args.device,
    )

    gen_config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    # --- Pilot: NL vs Hybrid × SVG vs Python ---
    formats_to_test = [FeedbackFormat.NL, FeedbackFormat.HYBRID]
    specificity = FeedbackSpecificity.PRECISE  # Phase 0 uses precise only

    for domain, samples, verifier, code_key in [
        ("svg", svg_samples, svg_verifier, "svg_string"),
        ("python", python_samples, python_verifier, "code"),
    ]:
        for fmt in formats_to_test:
            cell_key = f"{domain}_{fmt.value}"
            logger.info("=== Running cell: %s (%d samples) ===", cell_key, len(samples))

            # Step 1: Verify original code
            all_diags_before = []
            feedbacks = []
            codes = []

            for sample in samples:
                code = getattr(sample, code_key)
                codes.append(code)
                diags = verifier.verify(code)
                all_diags_before.append(diags)
                fb = render_feedback(diags, fmt, specificity)
                feedbacks.append(fb)

            # Step 2: Build correction prompts
            conversations = []
            for code, fb in zip(codes, feedbacks):
                conv = build_correction_prompt(code, fb, domain, iteration=0)
                conversations.append(conv)

            # Step 3: Generate corrections
            t0 = time.time()
            gen_results = runner.generate_chat(conversations, gen_config)
            gen_time = time.time() - t0

            # Step 4: Re-verify corrected code
            passed = 0
            total_diags_after = 0
            for i, (gen_res, diags_before) in enumerate(zip(gen_results, all_diags_before)):
                corrected = extract_code_block(gen_res.output, "svg" if domain == "svg" else "python")
                diags_after = verifier.verify(corrected)
                errors_after = [d for d in diags_after if d.severity == Severity.ERROR]
                if not errors_after:
                    passed += 1
                total_diags_after += len(diags_after)

            pass_rate = passed / len(samples) if samples else 0
            avg_diags = total_diags_after / len(samples) if samples else 0

            cell_result = {
                "domain": domain,
                "format": fmt.value,
                "specificity": "precise",
                "n_samples": len(samples),
                "pass_rate": round(pass_rate, 4),
                "passed": passed,
                "avg_diagnostics_after": round(avg_diags, 2),
                "gen_time_seconds": round(gen_time, 1),
                "avg_tokens": round(
                    sum(r.token_count for r in gen_results) / len(gen_results), 1
                ) if gen_results else 0,
            }
            results[cell_key] = cell_result
            logger.info("Cell %s: pass_rate=%.1f%% (%d/%d), time=%.1fs",
                        cell_key, pass_rate * 100, passed, len(samples), gen_time)

    runner.shutdown()

    # --- H3 Decision Matrix ---
    svg_nl = results.get("svg_nl", {}).get("pass_rate", 0)
    svg_hybrid = results.get("svg_hybrid", {}).get("pass_rate", 0)
    py_nl = results.get("python_nl", {}).get("pass_rate", 0)
    py_hybrid = results.get("python_hybrid", {}).get("pass_rate", 0)

    svg_delta = svg_nl - svg_hybrid  # positive = NL better
    py_delta = py_hybrid - py_nl      # positive = hybrid better

    h3_decision = _h3_decision_matrix(svg_delta, py_delta)

    summary = {
        "pilot_results": results,
        "h3_analysis": {
            "svg_nl_pass_rate": svg_nl,
            "svg_hybrid_pass_rate": svg_hybrid,
            "svg_delta_nl_minus_hybrid": round(svg_delta, 4),
            "python_nl_pass_rate": py_nl,
            "python_hybrid_pass_rate": py_hybrid,
            "python_delta_hybrid_minus_nl": round(py_delta, 4),
            "h3_decision": h3_decision,
        },
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)
    logger.info("H3 decision: %s", h3_decision)

    return summary


def _h3_decision_matrix(svg_delta: float, py_delta: float) -> str:
    """Pre-registered H3 decision matrix.

    Args:
        svg_delta: NL - hybrid pass rate on SVG (positive = NL better)
        py_delta: hybrid - NL pass rate on Python (positive = hybrid better)

    Returns:
        H3 framing decision string.
    """
    threshold = 0.03  # 3pp

    svg_sig = abs(svg_delta) >= threshold
    py_sig = abs(py_delta) >= threshold

    # Check expected directions
    svg_expected = svg_delta > 0      # NL > hybrid on SVG
    py_expected = py_delta > 0        # hybrid > NL on Python

    if svg_sig and py_sig and svg_expected and py_expected:
        return "H3_CONFIRMED: format×domain interaction (NL>hybrid on SVG, hybrid>NL on Python)"
    elif (svg_sig and svg_expected and not py_sig):
        return "H3_PARTIAL_SVG: NL>hybrid on SVG confirmed, Python flat — SVG confirmatory, Python exploratory"
    elif (py_sig and py_expected and not svg_sig):
        return "H3_PARTIAL_PYTHON: hybrid>NL on Python confirmed, SVG flat — Python confirmatory, SVG exploratory"
    elif svg_sig and py_sig and not (svg_expected and py_expected):
        return "H3_INTERACTION_REVERSED: significant but unexpected direction — reframe as 'interaction, direction TBD'"
    else:
        return "H3_DROPPED: no significant format×domain interaction — drop from confirmatory hypotheses"


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Pilot: NL vs Hybrid")
    parser.add_argument("--model-path", required=True, help="Local model path")
    parser.add_argument("--svg-cache", default="/root/autodl-tmp/data/svgenius",
                        help="SVGenius cache dir")
    parser.add_argument("--python-cache", default="/root/autodl-tmp/data/security_eval",
                        help="SecurityEval cache dir")
    parser.add_argument("--output", default="results/phase0_pilot.json",
                        help="Output JSON path")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--svg-split", default="easy")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_pilot(args)


if __name__ == "__main__":
    main()
