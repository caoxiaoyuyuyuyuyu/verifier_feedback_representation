"""Python Dataset Preparation — SecurityEval 数据加载。

加载 s2e-lab/SecurityEval 数据集，确保 Bandit/Pylint 有足够检出。
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PythonSample:
    """单个 Python 数据样本。"""
    sample_id: str
    code: str
    task_description: str
    expected_issues: list[str]
    metadata: dict


def load_python_dataset(
    dataset_name: str = "s2e-lab/SecurityEval",
    split: str = "train",
    n_samples: int = 300,
    seed: int = 42,
    code_column: str = "Insecure_code",
    prompt_column: str = "Prompt",
) -> list[PythonSample]:
    """加载 SecurityEval 数据集。

    Args:
        dataset_name: HF 数据集名称。
        split: 数据集 split。
        n_samples: 目标样本数。
        seed: 随机种子。
        code_column: 代码内容列名。
        prompt_column: prompt 列名。

    Returns:
        PythonSample 列表。
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)

    # Filter samples that have non-empty code
    valid_indices = [
        i for i, row in enumerate(ds)
        if (row.get(code_column) or "").strip()
    ]

    if len(valid_indices) > n_samples:
        rng = random.Random(seed)
        valid_indices = rng.sample(valid_indices, n_samples)

    ds_subset = ds.select(valid_indices)

    samples = []
    for i, row in enumerate(ds_subset):
        code = row.get(code_column, "") or ""
        prompt = row.get(prompt_column, "") or ""
        cwe = row.get("ID", "") or row.get("cwe", "") or ""
        sample_id = f"py_{i:04d}"

        samples.append(PythonSample(
            sample_id=sample_id,
            code=code,
            task_description=prompt,
            expected_issues=[cwe] if cwe else [],
            metadata={k: v for k, v in row.items()
                      if k not in {code_column, prompt_column}},
        ))

    logger.info("Loaded %d Python samples from %s/%s", len(samples), dataset_name, split)
    return samples
