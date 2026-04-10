"""Correction Prompt Templates.

构建包含 verifier feedback 的纠错 prompt。
System prompt 设定角色，user prompt 包含原始代码 + feedback。
"""

from __future__ import annotations

# System prompt: 指示模型根据 verifier feedback 修正代码
SYSTEM_PROMPT_SVG = (
    "You are an SVG code correction assistant. "
    "Given an SVG that failed geometric verification, fix the issues "
    "described in the feedback while preserving the original design intent. "
    "Output only the corrected SVG code."
)

SYSTEM_PROMPT_PYTHON = (
    "You are a Python code correction assistant. "
    "Given Python code that failed static analysis checks, fix the issues "
    "described in the feedback while preserving the original functionality. "
    "Output only the corrected Python code."
)

_SYSTEM_PROMPTS = {
    "svg": SYSTEM_PROMPT_SVG,
    "python": SYSTEM_PROMPT_PYTHON,
}

# User prompt template
_USER_TEMPLATE = """\
## Original Code

```{lang}
{code}
```

## Verifier Feedback

{feedback}

## Task

Fix the issues identified in the feedback. Output only the corrected code in a ```{lang}``` block."""


def build_correction_prompt(
    code: str,
    feedback: str,
    domain: str,
    iteration: int = 0,
) -> list[dict[str, str]]:
    """构建纠错 prompt (chat format)。

    Args:
        code: 原始代码（SVG 或 Python）。
        feedback: 渲染后的 feedback 字符串（由 templates.render_feedback 生成）。
        domain: "svg" 或 "python"。
        iteration: 当前纠错轮次 (0=首次, 1=二次修正)。

    Returns:
        OpenAI chat format 的消息列表 [{"role": ..., "content": ...}, ...]
    """
    lang = "svg" if domain == "svg" else "python"
    system = _SYSTEM_PROMPTS[domain]

    if iteration > 0:
        system += (
            f"\nThis is correction attempt #{iteration + 1}. "
            "The previous fix did not fully resolve all issues."
        )

    user_content = _USER_TEMPLATE.format(
        lang=lang,
        code=code,
        feedback=feedback,
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
