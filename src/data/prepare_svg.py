"""SVG Dataset Preparation — SVGenius 数据加载 + sanitizer dry-run.

加载 SVGenius 数据集的 SVG 样本，执行 verifier dry-run 确认 predicate 适用率。
目标: <5% predicate drop rate (即 >95% 的样本能被至少 1 个 predicate 检出问题)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SVGSample:
    """单个 SVG 数据样本。"""
    sample_id: str
    svg_string: str
    task_description: str       # 原始任务描述
    metadata: dict               # 数据集原始 metadata


@dataclass
class SanitizerReport:
    """Dry-run 报告。"""
    total_samples: int
    parseable: int               # svgelements 能解析的样本数
    has_diagnostics: int         # 至少 1 个 predicate 触发的样本数
    predicate_coverage: dict[str, int]  # predicate_name → 触发次数
    drop_rate: float             # 1 - has_diagnostics / total


def load_svg_dataset(
    dataset_name: str = "svg-genius",
    split: str = "test",
    n_samples: int = 300,
    seed: int = 42,
) -> list[SVGSample]:
    """从 HuggingFace 加载 SVGenius 数据集。

    Args:
        dataset_name: HF 数据集名称。
        split: 数据集 split。
        n_samples: 采样数量（实验用 300）。
        seed: 采样随机种子。

    Returns:
        SVGSample 列表。
    """
    # TODO: datasets.load_dataset(dataset_name, split=split)
    # TODO: 随机采样 n_samples
    # TODO: 包装为 SVGSample
    raise NotImplementedError


def run_sanitizer_dry_run(
    samples: list[SVGSample],
    verifier: object,  # SVGGeometricVerifier
) -> SanitizerReport:
    """对数据集执行 verifier dry-run，检查 predicate 适用率。

    Args:
        samples: SVG 样本列表。
        verifier: SVGGeometricVerifier 实例。

    Returns:
        SanitizerReport，包含 drop rate 和逐 predicate 覆盖统计。
    """
    # TODO: 对每个样本运行 verifier.verify()
    # TODO: 统计可解析率、predicate 触发率
    # TODO: 报告 drop rate，目标 <5%
    raise NotImplementedError
