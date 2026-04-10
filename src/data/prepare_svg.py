"""SVG Dataset Preparation — SVGenius 数据加载 + sanitizer dry-run.

加载 SVGenius 数据集的 SVG 样本，执行 verifier dry-run 确认 predicate 适用率。
目标: <5% predicate drop rate (即 >95% 的样本能被至少 1 个 predicate 检出问题)。
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass

from ..verifiers.diagnostic import Severity
from ..verifiers.svg_geometric import SVGGeometricVerifier

logger = logging.getLogger(__name__)


@dataclass
class SVGSample:
    """单个 SVG 数据样本。"""
    sample_id: str
    svg_string: str
    task_description: str
    metadata: dict


@dataclass
class SanitizerReport:
    """Dry-run 报告。"""
    total_samples: int
    parseable: int
    has_diagnostics: int
    predicate_coverage: dict[str, int]  # predicate_name → 触发次数
    severity_dist: dict[str, int]       # severity → count
    drop_rate: float                    # 1 - has_diagnostics / parseable
    parse_fail_rate: float
    samples_with_errors: int            # ERROR severity 的样本数


def load_svg_dataset(
    dataset_name: str = "xiaoooobai/SVGenius",
    split: str = "test",
    n_samples: int = 300,
    seed: int = 42,
    svg_column: str = "svg",
    task_column: str = "instruction",
) -> list[SVGSample]:
    """从 HuggingFace 加载 SVGenius 数据集。

    Args:
        dataset_name: HF 数据集名称。
        split: 数据集 split。
        n_samples: 采样数量。
        seed: 采样随机种子。
        svg_column: SVG 内容所在列名。
        task_column: 任务描述所在列名。

    Returns:
        SVGSample 列表。
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)

    # Sample if dataset larger than needed
    if len(ds) > n_samples:
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), n_samples)
        ds = ds.select(indices)

    samples = []
    for i, row in enumerate(ds):
        svg = row.get(svg_column, "") or ""
        task = row.get(task_column, "") or ""
        sample_id = row.get("id", f"svg_{i:04d}")
        metadata = {k: v for k, v in row.items()
                    if k not in {svg_column, task_column, "id"}}
        samples.append(SVGSample(
            sample_id=str(sample_id),
            svg_string=svg,
            task_description=task,
            metadata=metadata,
        ))

    logger.info("Loaded %d SVG samples from %s/%s", len(samples), dataset_name, split)
    return samples


def run_sanitizer_dry_run(
    samples: list[SVGSample],
    verifier: SVGGeometricVerifier | None = None,
) -> SanitizerReport:
    """对数据集执行 verifier dry-run，检查 predicate 适用率。

    Args:
        samples: SVG 样本列表。
        verifier: SVGGeometricVerifier 实例，None 时使用默认配置。

    Returns:
        SanitizerReport。
    """
    if verifier is None:
        verifier = SVGGeometricVerifier()

    total = len(samples)
    parseable = 0
    has_diag = 0
    has_error = 0
    pred_counter: Counter = Counter()
    sev_counter: Counter = Counter()

    for sample in samples:
        diags = verifier.verify(sample.svg_string)

        # Check if parse succeeded (parse_error means failure)
        if any(d.rule_id == "parse_error" for d in diags):
            continue
        parseable += 1

        if diags:
            has_diag += 1
            for d in diags:
                pred_counter[d.rule_id] += 1
                sev_counter[d.severity.value] += 1
            if any(d.severity == Severity.ERROR for d in diags):
                has_error += 1

    drop_rate = 1 - (has_diag / parseable) if parseable > 0 else 1.0
    parse_fail_rate = 1 - (parseable / total) if total > 0 else 1.0

    report = SanitizerReport(
        total_samples=total,
        parseable=parseable,
        has_diagnostics=has_diag,
        predicate_coverage=dict(pred_counter),
        severity_dist=dict(sev_counter),
        drop_rate=round(drop_rate, 4),
        parse_fail_rate=round(parse_fail_rate, 4),
        samples_with_errors=has_error,
    )

    logger.info(
        "Sanitizer dry-run: %d total, %d parseable (%.1f%%), %d with diagnostics, "
        "drop_rate=%.1f%%, predicates=%s",
        total, parseable, (1 - parse_fail_rate) * 100,
        has_diag, drop_rate * 100, dict(pred_counter),
    )
    return report
