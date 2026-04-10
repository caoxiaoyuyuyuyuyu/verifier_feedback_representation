"""Python Dataset Preparation — PythonSecurityEval + Pylint-targeted subset.

加载安全相关 Python 代码样本，确保 Bandit/Pylint 有足够检出。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PythonSample:
    """单个 Python 数据样本。"""
    sample_id: str
    code: str
    task_description: str
    expected_issues: list[str]   # 预期的 issue types (用于验证 verifier 覆盖)
    metadata: dict


def load_python_dataset(
    dataset_name: str = "python-security-eval",
    n_samples: int = 300,
    seed: int = 42,
    min_issues: int = 1,
) -> list[PythonSample]:
    """加载 Python 安全/质量评估数据集。

    策略: PythonSecurityEval 提供安全漏洞样本 (Bandit 覆盖)，
    补充 Pylint-targeted subset 确保代码质量 issue 的多样性。

    Args:
        dataset_name: HF 数据集名称。
        n_samples: 目标样本数。
        seed: 随机种子。
        min_issues: 每个样本最少需要的 issue 数（过滤太简单的样本）。

    Returns:
        PythonSample 列表。
    """
    # TODO: 加载 PythonSecurityEval
    # TODO: 用 PythonStaticVerifier 预筛选，确保每个样本有 ≥min_issues 个 diagnostic
    # TODO: 如果安全样本不够 300，补充 Pylint-targeted 代码
    raise NotImplementedError
