"""主实验入口 — Factorial Design Orchestration.

遍历所有实验 cell: 2 model × 2 domain × 2 specificity × 3 format × 2 iterations = 48 cells
每个 cell 运行 300 queries，共 14.4K inferences。

额外运行:
  - Counterfactual ablation (Phase 4): CF-A + CF-B on SVG domain
  - Baseline comparisons: no-correction, self-refine, scalar-only
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .feedback.templates import FeedbackFormat, FeedbackSpecificity

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """实验全局配置。"""
    # Models
    models: list[str] = field(default_factory=lambda: [
        "Qwen/Qwen3-Coder-Next-7B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ])

    # Domains
    domains: list[str] = field(default_factory=lambda: ["svg", "python"])

    # Factorial dimensions
    formats: list[FeedbackFormat] = field(default_factory=lambda: [
        FeedbackFormat.RAW_JSON,
        FeedbackFormat.NL,
        FeedbackFormat.HYBRID,
    ])
    specificities: list[FeedbackSpecificity] = field(default_factory=lambda: [
        FeedbackSpecificity.GENERIC,
        FeedbackSpecificity.PRECISE,
    ])
    max_iterations: int = 2  # {0, 1}

    # Data
    n_samples_per_domain: int = 300

    # Infrastructure
    output_dir: str = "results"
    seed: int = 42
    vllm_tensor_parallel: int = 1
    vllm_gpu_memory: float = 0.9


@dataclass
class CellResult:
    """单个 factorial cell 的结果。"""
    model: str
    domain: str
    fmt: str
    specificity: str
    iteration: int
    pass_rate: float
    feto: float
    n_samples: int
    diagnostics_path: str  # 详细结果保存路径


def run_factorial(config: ExperimentConfig) -> list[CellResult]:
    """运行主 factorial 实验。

    遍历 itertools.product(models, domains, formats, specificities, iterations),
    对每个 cell 执行: load data → verify → render feedback → generate correction → re-verify → metrics

    Args:
        config: 实验配置。

    Returns:
        所有 cell 的结果列表。
    """
    cells = list(itertools.product(
        config.models,
        config.domains,
        config.formats,
        config.specificities,
        range(config.max_iterations),
    ))
    logger.info(f"Factorial design: {len(cells)} cells × {config.n_samples_per_domain} samples")

    results: list[CellResult] = []

    for model, domain, fmt, spec, iteration in cells:
        logger.info(f"Running cell: {model} / {domain} / {fmt.value} / {spec.value} / iter={iteration}")

        # TODO: Step 1 — Load data (cached per domain)
        # TODO: Step 2 — Run verifier on original code → diagnostics_before
        # TODO: Step 3 — Render feedback (fmt × spec)
        # TODO: Step 4 — Build correction prompts
        # TODO: Step 5 — Generate corrections via VLLMRunner
        # TODO: Step 6 — Re-verify corrected code → diagnostics_after
        # TODO: Step 7 — Compute metrics (pass_rate, FETO)
        # TODO: Step 8 — Save cell results

    return results


def run_counterfactual(config: ExperimentConfig) -> dict:
    """Phase 4: 2×2 counterfactual ablation (SVG domain only).

    对 SVG domain 的 precise feedback:
      - Original (control)
      - CF-A: plausible-wrong substitutes
      - CF-B: shuffled element mapping

    比较 pass_rate delta 证明因果使用。

    Args:
        config: 实验配置。

    Returns:
        Counterfactual 结果字典: {condition → pass_rate}
    """
    # TODO: 加载 SVG data + run verifier
    # TODO: 生成 3 个条件的 feedback (original, CF-A, CF-B)
    # TODO: 对每个条件运行 correction + re-verify
    # TODO: 计算 delta pass_rate
    raise NotImplementedError


def run_baselines(config: ExperimentConfig) -> dict:
    """Phase 5: Baseline comparisons.

    - no_correction: 不给 feedback，直接评估原始代码
    - self_refine: 不给外部 feedback，让模型自我反思
    - scalar_only: 只告诉 pass/fail，不给详细 diagnostic

    Args:
        config: 实验配置。

    Returns:
        Baseline 结果字典。
    """
    # TODO: no_correction baseline
    # TODO: self_refine baseline (prompt: "review and fix your code")
    # TODO: scalar_only baseline (feedback = "FAIL" / "PASS")
    raise NotImplementedError


def main():
    """CLI 入口。"""
    # TODO: argparse 解析配置
    # TODO: 设置 logging
    # TODO: 创建 output_dir
    # TODO: run_factorial → run_counterfactual → run_baselines
    # TODO: 汇总结果，生成 summary JSON
    raise NotImplementedError


if __name__ == "__main__":
    main()
