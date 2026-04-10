"""Claude-as-Judge Failure Taxonomy.

6-class failure rubric for analyzing correction failures:
  1. IGNORED: feedback 完全被忽略，edit 与原始代码无显著差异
  2. MISLOCATED: 修改了错误的元素/位置
  3. WRONG_DIRECTION: 修改方向与 feedback 相反
  4. PARTIAL: 部分修复，但未完全解决
  5. OVERCORRECTION: 过度修正导致新问题
  6. MISUNDERSTOOD: 对 feedback 的理解与实际含义不符
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureClass(Enum):
    IGNORED = "ignored"
    MISLOCATED = "mislocated"
    WRONG_DIRECTION = "wrong_direction"
    PARTIAL = "partial"
    OVERCORRECTION = "overcorrection"
    MISUNDERSTOOD = "misunderstood"


@dataclass
class JudgmentResult:
    """单样本的 judge 判定结果。"""
    sample_id: str
    failure_class: FailureClass
    confidence: float           # 0-1, judge 的置信度
    reasoning: str              # judge 的推理过程
    evidence: str               # 支持判定的具体证据


@dataclass
class TaxonomyReport:
    """整批样本的失败分类报告。"""
    total_failures: int
    distribution: dict[str, int]        # failure_class → count
    distribution_pct: dict[str, float]  # failure_class → percentage
    judgments: list[JudgmentResult]


# Judge prompt template
_JUDGE_SYSTEM = """\
You are an expert code reviewer analyzing why a code correction attempt failed.
Given the original code, verifier feedback, and the corrected code that still has issues,
classify the failure into exactly one of these categories:

1. IGNORED: The feedback was completely ignored - the edit is trivial or unrelated
2. MISLOCATED: The fix targeted the wrong element/location
3. WRONG_DIRECTION: The fix went in the opposite direction of what feedback suggested
4. PARTIAL: Some issues were fixed but others remain
5. OVERCORRECTION: The fix went too far and introduced new problems
6. MISUNDERSTOOD: The model misinterpreted what the feedback meant

Respond in JSON: {"class": "...", "confidence": 0.0-1.0, "reasoning": "...", "evidence": "..."}"""

_JUDGE_USER = """\
## Original Code
```
{original_code}
```

## Verifier Feedback
{feedback}

## Corrected Code (still failing)
```
{corrected_code}
```

## Remaining Issues
{remaining_diagnostics}

Classify this failure."""


def run_failure_taxonomy(
    failures: list[dict],
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_concurrent: int = 10,
) -> TaxonomyReport:
    """对一批失败样本运行 Claude-as-judge 分类。

    Args:
        failures: 失败样本列表，每个包含:
            - sample_id, original_code, feedback, corrected_code, remaining_diagnostics
        api_key: Anthropic API key。
        model: Judge 使用的 Claude 模型。
        max_concurrent: 最大并发请求数。

    Returns:
        TaxonomyReport，包含分布统计和逐样本判定。
    """
    # TODO: 构建 judge prompts
    # TODO: 并发调用 Claude API (asyncio + httpx 或 anthropic SDK)
    # TODO: 解析 JSON 响应为 JudgmentResult
    # TODO: 汇总分布统计
    raise NotImplementedError
