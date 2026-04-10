"""Feedback Templates — 3 format × 2 specificity = 6 representations.

Format:
  - RAW_JSON: Diagnostic.to_dict() 直接序列化
  - NL: 自然语言动词化描述
  - HYBRID: JSON 结构 + NL 注释

Specificity:
  - GENERIC: 只说"有问题"，不给具体数值和修复方向
  - PRECISE: 包含 metric_value、fix_direction、element_ids
"""

from __future__ import annotations

import json
from enum import Enum

from ..verifiers.diagnostic import Diagnostic


class FeedbackFormat(Enum):
    RAW_JSON = "raw_json"
    NL = "nl"
    HYBRID = "hybrid"


class FeedbackSpecificity(Enum):
    GENERIC = "generic"
    PRECISE = "precise"


def render_feedback(
    diagnostics: list[Diagnostic],
    fmt: FeedbackFormat,
    specificity: FeedbackSpecificity,
) -> str:
    """将 Diagnostic 列表渲染为指定 format × specificity 的反馈字符串。

    Args:
        diagnostics: verifier 输出的诊断列表。
        fmt: 输出格式 (RAW_JSON / NL / HYBRID)。
        specificity: 精确度 (GENERIC / PRECISE)。

    Returns:
        渲染后的反馈字符串，可直接插入 correction prompt。
    """
    renderers = {
        FeedbackFormat.RAW_JSON: _render_raw_json,
        FeedbackFormat.NL: _render_nl,
        FeedbackFormat.HYBRID: _render_hybrid,
    }
    return renderers[fmt](diagnostics, specificity)


def _render_raw_json(
    diagnostics: list[Diagnostic], specificity: FeedbackSpecificity
) -> str:
    """RAW_JSON: Diagnostic dict 的 JSON 序列化。

    GENERIC 模式下剥离 metric_value, fix_direction, message_precise。
    """
    # TODO: 根据 specificity 过滤字段
    # TODO: json.dumps with indent=2
    raise NotImplementedError


def _render_nl(
    diagnostics: list[Diagnostic], specificity: FeedbackSpecificity
) -> str:
    """NL: 自然语言逐条描述。

    GENERIC: "Issue found: {message_generic}"
    PRECISE: "Issue in elements {element_ids}: {message_precise}. Metric {metric_name}={metric_value}. Suggested fix: {fix_direction}"
    """
    # TODO: 遍历 diagnostics，按 specificity 选择 message 模板
    # TODO: 用 severity emoji/prefix 区分严重程度
    raise NotImplementedError


def _render_hybrid(
    diagnostics: list[Diagnostic], specificity: FeedbackSpecificity
) -> str:
    """HYBRID: JSON 结构体 + 内嵌 NL 注释。

    每条 diagnostic 输出为:
    ```
    // {NL description}
    {JSON object}
    ```
    """
    # TODO: 结合 _render_raw_json 和 _render_nl 的逻辑
    raise NotImplementedError
