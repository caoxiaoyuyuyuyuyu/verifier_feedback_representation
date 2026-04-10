"""Evaluation Metrics.

核心指标:
  - verifier_pass_rate: 纠错后代码通过 verifier 的比例
  - FETO (BPE-Jaccard): Feedback-Edit Token Overlap, 衡量 feedback 对 edit 的影响
  - attention_concentration: feedback token 位置的注意力集中度
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PassRateResult:
    """Pass rate 计算结果。"""
    total: int
    passed: int
    rate: float
    per_rule: dict[str, float]  # rule_id → pass rate


@dataclass
class FETOResult:
    """FETO 分数结果。"""
    score: float                # 平均 BPE-Jaccard
    per_sample: list[float]     # 逐样本分数


@dataclass
class AttentionResult:
    """注意力集中度结果。"""
    concentration: float        # 平均集中度
    per_layer: list[float]      # 逐层集中度
    per_sample: list[float]     # 逐样本集中度


def verifier_pass_rate(
    diagnostics_before: list[list[Any]],
    diagnostics_after: list[list[Any]],
) -> PassRateResult:
    """计算纠错前后的 verifier pass rate。

    Pass = 纠错后 diagnostics 中 ERROR severity 数量为 0。

    Args:
        diagnostics_before: 纠错前每个样本的 Diagnostic 列表。
        diagnostics_after: 纠错后每个样本的 Diagnostic 列表。

    Returns:
        PassRateResult，包含总体和逐规则 pass rate。
    """
    # TODO: 统计 ERROR severity diagnostics
    # TODO: 计算总体 pass rate
    # TODO: 按 rule_id 分组计算逐规则 pass rate
    raise NotImplementedError


def feto_score(
    feedback_tokens: list[list[str]],
    edit_tokens: list[list[str]],
) -> FETOResult:
    """计算 FETO (Feedback-Edit Token Overlap) — BPE-Jaccard 相似度。

    衡量 feedback 中的 token 在 edit diff 中出现的比例。
    高 FETO = 模型的修改与 feedback 内容高度相关。

    Args:
        feedback_tokens: 每个样本 feedback 的 BPE token 列表。
        edit_tokens: 每个样本 code edit diff 的 BPE token 列表。

    Returns:
        FETOResult，包含平均和逐样本 Jaccard 分数。
    """
    # TODO: 对每个样本计算 Jaccard(set(feedback), set(edit))
    # TODO: 汇总平均
    raise NotImplementedError


def attention_concentration(
    attention_weights: np.ndarray,
    feedback_token_positions: list[list[int]],
) -> AttentionResult:
    """计算 feedback token 位置的注意力集中度。

    在生成 edit token 时，模型对 feedback token 位置分配的注意力比例。

    Args:
        attention_weights: shape (n_samples, n_layers, n_heads, seq_len, seq_len)
        feedback_token_positions: 每个样本中 feedback token 的位置索引。

    Returns:
        AttentionResult，包含平均、逐层、逐样本集中度。
    """
    # TODO: 提取 generation token 对 feedback positions 的注意力
    # TODO: 对 heads 取平均，计算 feedback 位置的注意力占比
    # TODO: 按层聚合
    raise NotImplementedError
