"""Diagnostic dataclass — 全项目核心数据结构。

所有 verifier 输出统一为 Diagnostic 列表，feedback templates 基于此结构
渲染 6 种 representation (3 format × 2 specificity)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(Enum):
    """诊断严重程度。"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic:
    """单条验证诊断结果。

    Attributes:
        rule_id: 规则标识符, e.g. "bbox_coverage", "B101_exec"
        severity: 严重程度
        element_ids: 涉及的元素标识 (SVG element id / Python AST node lineno)
        message_generic: 通用描述, e.g. "Elements overlap detected"
        message_precise: 精确描述, 包含具体数值和修复方向
        metric_name: 度量名称, e.g. "overlap_ratio", "coverage_pct"
        metric_value: 度量数值
        fix_direction: 修复方向提示, e.g. "reduce overlap by moving element X left"
        metadata: 额外信息 (predicate-specific)
    """
    rule_id: str
    severity: Severity
    element_ids: tuple[str, ...] = ()
    message_generic: str = ""
    message_precise: str = ""
    metric_name: str = ""
    metric_value: float | None = None
    fix_direction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON-safe dict，供 raw JSON feedback 使用。"""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "element_ids": list(self.element_ids),
            "message_generic": self.message_generic,
            "message_precise": self.message_precise,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "fix_direction": self.fix_direction,
            "metadata": self.metadata,
        }
