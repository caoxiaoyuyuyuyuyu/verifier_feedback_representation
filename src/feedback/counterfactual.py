"""Counterfactual Feedback Generator — Phase 4 causal probe.

2×2 counterfactual design:
  - CF-A (plausible-wrong-in-format): 保留正确格式，替换内容为 plausible 但错误的诊断
  - CF-B (shuffled): 打乱 diagnostic 与 element 的对应关系

用于验证 H4: 模型因果地使用反馈内容，而非仅依赖格式信号。
"""

from __future__ import annotations

import random
from enum import Enum

from ..verifiers.diagnostic import Diagnostic


class CFType(Enum):
    CF_A = "plausible_wrong"  # 格式正确，内容错误
    CF_B = "shuffled"          # 诊断-元素映射打乱


# 每个 verifier 的 fix vocabulary (20 templates each)
# 用于 CF-A: 从中采样生成 plausible-but-wrong fix_direction
SVG_FIX_VOCABULARY: list[str] = [
    # TODO: 20 个 SVG 相关的 plausible fix templates
    # e.g. "move element {id} {direction} by {amount}px"
    # e.g. "scale element {id} to {pct}% of current size"
    # e.g. "adjust viewBox to include element {id}"
]

PYTHON_FIX_VOCABULARY: list[str] = [
    # TODO: 20 个 Python 安全/质量相关的 plausible fix templates
    # e.g. "replace exec() with ast.literal_eval()"
    # e.g. "add input validation for parameter {param}"
    # e.g. "use parameterized query instead of string formatting"
]


def generate_counterfactual(
    diagnostics: list[Diagnostic],
    cf_type: CFType,
    domain: str,
    rng: random.Random | None = None,
) -> list[Diagnostic]:
    """生成 counterfactual 版本的诊断列表。

    Args:
        diagnostics: 原始（正确的）诊断列表。
        cf_type: CF-A (plausible wrong) 或 CF-B (shuffled)。
        domain: "svg" 或 "python"，决定使用哪个 fix vocabulary。
        rng: 随机数生成器（用于可复现性）。

    Returns:
        修改后的 Diagnostic 列表，保持相同数量和格式。
    """
    rng = rng or random.Random()

    if cf_type == CFType.CF_A:
        return _generate_plausible_wrong(diagnostics, domain, rng)
    elif cf_type == CFType.CF_B:
        return _generate_shuffled(diagnostics, rng)
    else:
        raise ValueError(f"Unknown CF type: {cf_type}")


def _generate_plausible_wrong(
    diagnostics: list[Diagnostic],
    domain: str,
    rng: random.Random,
) -> list[Diagnostic]:
    """CF-A: 替换 fix_direction 和 message_precise 为 plausible 但错误的内容。

    保持: rule_id, severity, element_ids, metric_name, format 结构
    替换: fix_direction (从 vocabulary 采样), message_precise, metric_value (扰动)
    """
    # TODO: 根据 domain 选择 vocabulary
    # TODO: 对每条 diagnostic 生成替代内容
    # TODO: metric_value 做合理扰动 (e.g. ±30% 但方向相反)
    raise NotImplementedError


def _generate_shuffled(
    diagnostics: list[Diagnostic],
    rng: random.Random,
) -> list[Diagnostic]:
    """CF-B: 打乱 diagnostic 与 element_ids 的映射。

    每条 diagnostic 的 element_ids 随机重新分配，
    使得 fix_direction 与实际问题元素不对应。
    """
    # TODO: 收集所有 element_ids，随机 shuffle
    # TODO: 重新分配给各 diagnostic
    raise NotImplementedError
