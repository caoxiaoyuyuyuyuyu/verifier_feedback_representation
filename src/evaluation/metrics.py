"""Evaluation Metrics.

核心指标:
  - verifier_pass_rate: 纠错后代码通过 verifier 的比例
  - FETO (BPE-Jaccard): Feedback-Edit Token Overlap
  - attention_concentration: feedback token 位置的注意力集中度
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..verifiers.diagnostic import Diagnostic, Severity


@dataclass
class PassRateResult:
    """Pass rate 计算结果。"""
    total: int
    passed: int
    rate: float
    per_rule: dict[str, float]


@dataclass
class FETOResult:
    """FETO 分数结果。"""
    score: float
    per_sample: list[float]


@dataclass
class AttentionResult:
    """注意力集中度结果。"""
    concentration: float
    per_layer: list[float]
    per_sample: list[float]


def verifier_pass_rate(
    diagnostics_before: list[list[Diagnostic]],
    diagnostics_after: list[list[Diagnostic]],
) -> PassRateResult:
    """计算纠错前后的 verifier pass rate。

    Pass = 纠错后 ERROR severity 数量为 0。
    """
    total = len(diagnostics_after)
    if total == 0:
        return PassRateResult(total=0, passed=0, rate=0.0, per_rule={})

    passed = 0
    # Track per-rule: how many samples had this rule_id error before → fixed after
    rule_before: dict[str, int] = defaultdict(int)
    rule_fixed: dict[str, int] = defaultdict(int)

    for before, after in zip(diagnostics_before, diagnostics_after):
        errors_after = [d for d in after if d.severity == Severity.ERROR]
        if not errors_after:
            passed += 1

        # Per-rule tracking
        rules_before = {d.rule_id for d in before if d.severity == Severity.ERROR}
        rules_after = {d.rule_id for d in after if d.severity == Severity.ERROR}
        for rule in rules_before:
            rule_before[rule] += 1
            if rule not in rules_after:
                rule_fixed[rule] += 1

    per_rule = {
        rule: rule_fixed[rule] / count
        for rule, count in rule_before.items()
        if count > 0
    }

    return PassRateResult(
        total=total,
        passed=passed,
        rate=passed / total,
        per_rule=per_rule,
    )


def feto_score(
    feedback_tokens: list[list[str]],
    edit_tokens: list[list[str]],
) -> FETOResult:
    """FETO (Feedback-Edit Token Overlap) — BPE-Jaccard similarity.

    Jaccard(set(feedback_tokens), set(edit_tokens)) per sample.
    """
    per_sample = []
    for fb, ed in zip(feedback_tokens, edit_tokens):
        fb_set = set(fb)
        ed_set = set(ed)
        if not fb_set and not ed_set:
            per_sample.append(0.0)
            continue
        intersection = len(fb_set & ed_set)
        union = len(fb_set | ed_set)
        per_sample.append(intersection / union if union > 0 else 0.0)

    avg = sum(per_sample) / len(per_sample) if per_sample else 0.0
    return FETOResult(score=avg, per_sample=per_sample)


def attention_concentration(
    attention_weights: np.ndarray,
    feedback_token_positions: list[list[int]],
) -> AttentionResult:
    """Feedback token 位置的注意力集中度。

    对每个样本/层，计算 generation tokens 对 feedback positions 的
    注意力占总注意力的比例。

    Args:
        attention_weights: (n_samples, n_layers, n_heads, gen_len, seq_len)
            已提取的 generation token 行的注意力权重。
        feedback_token_positions: 每个样本中 feedback token 的位置索引。
    """
    n_samples, n_layers, n_heads, gen_len, seq_len = attention_weights.shape

    per_sample = []
    per_layer_accum = np.zeros(n_layers)

    for i in range(n_samples):
        positions = feedback_token_positions[i]
        if not positions or gen_len == 0:
            per_sample.append(0.0)
            continue

        # (n_layers, n_heads, gen_len, seq_len)
        sample_attn = attention_weights[i]
        # Average over heads: (n_layers, gen_len, seq_len)
        head_avg = sample_attn.mean(axis=1)

        # Sum attention to feedback positions vs total
        total_attn = head_avg.sum(axis=-1)  # (n_layers, gen_len)
        fb_attn = head_avg[:, :, positions].sum(axis=-1)  # (n_layers, gen_len)

        # Concentration per layer: mean over gen tokens
        layer_conc = np.where(
            total_attn > 0,
            fb_attn / total_attn,
            0.0,
        ).mean(axis=1)  # (n_layers,)

        per_layer_accum += layer_conc
        per_sample.append(float(layer_conc.mean()))

    per_layer = (per_layer_accum / max(n_samples, 1)).tolist()
    avg = sum(per_sample) / len(per_sample) if per_sample else 0.0

    return AttentionResult(
        concentration=avg,
        per_layer=per_layer,
        per_sample=per_sample,
    )
