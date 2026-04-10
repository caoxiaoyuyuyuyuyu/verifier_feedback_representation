"""Counterfactual Feedback Generator — Phase 4 causal probe.

2×2 counterfactual design:
  - CF-A (plausible-wrong-in-format): 保留正确格式，替换内容为 plausible 但错误的诊断
  - CF-B (shuffled): 打乱 diagnostic 与 element 的对应关系

用于验证 H4: 模型因果地使用反馈内容，而非仅依赖格式信号。
"""

from __future__ import annotations

import random
from dataclasses import replace
from enum import Enum

from ..verifiers.diagnostic import Diagnostic


class CFType(Enum):
    CF_A = "plausible_wrong"
    CF_B = "shuffled"


# Fix vocabulary: plausible but wrong suggestions per domain (20 each)
SVG_FIX_VOCABULARY: list[str] = [
    "move element {id} 50px to the right to fix alignment",
    "scale element {id} to 75% of current size",
    "adjust viewBox to include element {id} with 10px padding",
    "rotate element {id} by 15 degrees clockwise",
    "change element {id} fill opacity to 0.8",
    "translate element {id} by (20, -30) to center it",
    "reduce element {id} stroke-width to 1px",
    "set element {id} width to 120px",
    "move element {id} 40px upward to avoid overlap",
    "increase element {id} height by 25%",
    "align element {id} to the left edge of the viewBox",
    "set element {id} transform to scale(0.9)",
    "add 15px margin around element {id}",
    "change element {id} coordinates to (100, 50)",
    "reduce element {id} border-radius to 5px",
    "move element {id} to the center of its parent group",
    "set element {id} aspect ratio to 1:1",
    "scale element {id} uniformly by factor 1.2",
    "clip element {id} to the viewBox boundaries",
    "reposition element {id} relative to its sibling elements",
]

PYTHON_FIX_VOCABULARY: list[str] = [
    "replace exec() call with ast.literal_eval() for safety",
    "add input validation using isinstance() check",
    "use parameterized query instead of string formatting",
    "wrap the operation in a try-except block",
    "add os.path.realpath() to prevent path traversal",
    "replace pickle.loads() with json.loads()",
    "use subprocess.run() with shell=False instead of os.system()",
    "add rate limiting to prevent abuse",
    "use secrets.token_hex() instead of random.randint() for tokens",
    "add CSRF token validation before processing",
    "use hashlib.pbkdf2_hmac() instead of md5 for passwords",
    "sanitize user input with bleach.clean()",
    "add Content-Security-Policy header",
    "use tempfile.mkstemp() instead of predictable filenames",
    "replace eval() with a safe expression parser",
    "add authentication check before database access",
    "use cryptography.fernet instead of DES encryption",
    "validate URL scheme before making requests",
    "add file size limit check before processing upload",
    "use ssl.create_default_context() for HTTPS connections",
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
        domain: "svg" 或 "python"。
        rng: 随机数生成器。

    Returns:
        修改后的 Diagnostic 列表，保持相同数量和格式。
    """
    if not diagnostics:
        return []

    rng = rng or random.Random()

    if cf_type == CFType.CF_A:
        return _generate_plausible_wrong(diagnostics, domain, rng)
    else:
        return _generate_shuffled(diagnostics, rng)


def _generate_plausible_wrong(
    diagnostics: list[Diagnostic],
    domain: str,
    rng: random.Random,
) -> list[Diagnostic]:
    """CF-A: 替换 fix_direction 和 message_precise，保持格式结构。"""
    vocab = SVG_FIX_VOCABULARY if domain == "svg" else PYTHON_FIX_VOCABULARY
    result = []

    for d in diagnostics:
        candidates = [f for f in vocab if f != d.fix_direction]
        new_fix = rng.choice(candidates) if candidates else vocab[0]

        # Fill in {id} placeholder
        elem = d.element_ids[0] if d.element_ids else "target"
        new_fix = new_fix.replace("{id}", elem)

        # Perturb metric_value: shift in wrong direction
        new_metric = None
        if d.metric_value is not None:
            perturbation = rng.uniform(0.2, 0.5)
            new_metric = round(d.metric_value * (1 + perturbation), 4)

        new_precise = (
            f"Issue detected with rule {d.rule_id}: {new_fix}. "
            f"Current metric value: {new_metric}."
        )

        result.append(replace(
            d,
            fix_direction=new_fix,
            message_precise=new_precise,
            metric_value=new_metric,
        ))

    return result


def _generate_shuffled(
    diagnostics: list[Diagnostic],
    rng: random.Random,
) -> list[Diagnostic]:
    """CF-B: 打乱 element_ids 映射。"""
    if len(diagnostics) < 2:
        return list(diagnostics)

    all_elem_ids = [d.element_ids for d in diagnostics]

    # Shuffle until at least one mapping changes
    shuffled = list(all_elem_ids)
    for _ in range(100):
        rng.shuffle(shuffled)
        if any(s != o for s, o in zip(shuffled, all_elem_ids)):
            break

    result = []
    for d, new_ids in zip(diagnostics, shuffled):
        new_fix = d.fix_direction
        for old_id, new_id in zip(d.element_ids, new_ids):
            new_fix = new_fix.replace(f"'{old_id}'", f"'{new_id}'")
        result.append(replace(d, element_ids=new_ids, fix_direction=new_fix))

    return result
