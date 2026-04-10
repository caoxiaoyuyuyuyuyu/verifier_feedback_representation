"""Feedback Templates — 3 format × 2 specificity = 6 representations.

Format:
  - RAW_JSON: Diagnostic.to_dict() 直接序列化
  - NL: 自然语言描述
  - HYBRID: JSON 结构 + NL 注释

Specificity:
  - GENERIC: 只说"有问题"，不给具体数值和修复方向
  - PRECISE: 包含 metric_value、fix_direction、element_ids
"""

from __future__ import annotations

import json
from enum import Enum

from ..verifiers.diagnostic import Diagnostic, Severity


class FeedbackFormat(Enum):
    RAW_JSON = "raw_json"
    NL = "nl"
    HYBRID = "hybrid"


class FeedbackSpecificity(Enum):
    GENERIC = "generic"
    PRECISE = "precise"


_SEVERITY_PREFIX = {
    Severity.ERROR: "[ERROR]",
    Severity.WARNING: "[WARNING]",
    Severity.INFO: "[INFO]",
}

# Fields to strip in GENERIC mode
_PRECISE_ONLY_FIELDS = {
    "metric_value", "fix_direction", "message_precise", "metadata",
}


def render_feedback(
    diagnostics: list[Diagnostic],
    fmt: FeedbackFormat,
    specificity: FeedbackSpecificity,
) -> str:
    """将 Diagnostic 列表渲染为指定 format × specificity 的反馈字符串。"""
    if not diagnostics:
        return "No issues found. All checks passed."

    renderers = {
        FeedbackFormat.RAW_JSON: _render_raw_json,
        FeedbackFormat.NL: _render_nl,
        FeedbackFormat.HYBRID: _render_hybrid,
    }
    return renderers[fmt](diagnostics, specificity)


def _filter_dict(d: dict, specificity: FeedbackSpecificity) -> dict:
    """GENERIC 模式下剥离精确字段。"""
    if specificity == FeedbackSpecificity.PRECISE:
        return d
    return {k: v for k, v in d.items() if k not in _PRECISE_ONLY_FIELDS}


def _render_raw_json(
    diagnostics: list[Diagnostic], specificity: FeedbackSpecificity
) -> str:
    """RAW_JSON format."""
    items = [_filter_dict(d.to_dict(), specificity) for d in diagnostics]
    return json.dumps(items, indent=2, ensure_ascii=False)


def _nl_line(d: Diagnostic, specificity: FeedbackSpecificity) -> str:
    """渲染单条 Diagnostic 为 NL 行。"""
    prefix = _SEVERITY_PREFIX[d.severity]
    if specificity == FeedbackSpecificity.GENERIC:
        return f"{prefix} {d.message_generic}"

    parts = [f"{prefix} {d.message_precise or d.message_generic}"]
    if d.element_ids:
        parts.append(f"  Affected elements: {', '.join(d.element_ids)}")
    if d.metric_name and d.metric_value is not None:
        parts.append(f"  Metric: {d.metric_name} = {d.metric_value}")
    if d.fix_direction:
        parts.append(f"  Suggested fix: {d.fix_direction}")
    return "\n".join(parts)


def _render_nl(
    diagnostics: list[Diagnostic], specificity: FeedbackSpecificity
) -> str:
    """NL format: 自然语言逐条描述。"""
    header = f"Found {len(diagnostics)} issue(s):\n"
    lines = []
    for i, d in enumerate(diagnostics, 1):
        lines.append(f"{i}. {_nl_line(d, specificity)}")
    return header + "\n\n".join(lines)


def _render_hybrid(
    diagnostics: list[Diagnostic], specificity: FeedbackSpecificity
) -> str:
    """HYBRID format: NL 注释 + JSON 结构体。"""
    blocks = []
    for d in diagnostics:
        nl = _nl_line(d, specificity).replace("\n", "\n// ")
        j = json.dumps(_filter_dict(d.to_dict(), specificity),
                        indent=2, ensure_ascii=False)
        blocks.append(f"// {nl}\n{j}")
    return "\n\n".join(blocks)
