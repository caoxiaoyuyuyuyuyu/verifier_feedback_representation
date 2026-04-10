"""Python Static Analysis Verifier — Bandit + Pylint wrapper.

将 Bandit (安全) 和 Pylint (代码质量) 的输出统一为 Diagnostic 格式。
支持对 Python 代码字符串或文件进行分析。
"""

from __future__ import annotations

from .diagnostic import Diagnostic, Severity

# Bandit severity → Diagnostic severity 映射
_BANDIT_SEVERITY_MAP = {
    "HIGH": Severity.ERROR,
    "MEDIUM": Severity.WARNING,
    "LOW": Severity.INFO,
}

# Pylint message type → Diagnostic severity 映射
_PYLINT_SEVERITY_MAP = {
    "error": Severity.ERROR,
    "warning": Severity.WARNING,
    "convention": Severity.INFO,
    "refactor": Severity.INFO,
}


class PythonStaticVerifier:
    """Python 静态分析验证器。

    Usage:
        verifier = PythonStaticVerifier(enable_bandit=True, enable_pylint=True)
        diagnostics = verifier.verify(code_string)
    """

    def __init__(
        self,
        enable_bandit: bool = True,
        enable_pylint: bool = True,
        pylint_checks: list[str] | None = None,
    ):
        """
        Args:
            enable_bandit: 是否启用 Bandit 安全检查。
            enable_pylint: 是否启用 Pylint 代码质量检查。
            pylint_checks: Pylint 检查项白名单，None = 全部启用。
        """
        self.enable_bandit = enable_bandit
        self.enable_pylint = enable_pylint
        self.pylint_checks = pylint_checks

    def verify(self, code: str) -> list[Diagnostic]:
        """对 Python 代码字符串执行静态分析。

        Args:
            code: Python 源代码字符串。

        Returns:
            Diagnostic 列表（Bandit + Pylint 结果合并）。
        """
        diagnostics: list[Diagnostic] = []
        if self.enable_bandit:
            diagnostics.extend(self._run_bandit(code))
        if self.enable_pylint:
            diagnostics.extend(self._run_pylint(code))
        return diagnostics

    def _run_bandit(self, code: str) -> list[Diagnostic]:
        """运行 Bandit 安全扫描，转换为 Diagnostic。

        Bandit 以 JSON 格式输出，每条 finding 映射为一个 Diagnostic:
        - rule_id: "B{test_id}_{test_name}" e.g. "B101_assert_used"
        - element_ids: (line_number,)
        - message_precise: 包含具体代码行和安全风险说明
        """
        # TODO: 写入临时文件，调用 bandit -f json
        # TODO: 解析 JSON 输出，映射 severity
        # TODO: 生成 message_generic (issue_text) 和 message_precise (含代码上下文+修复建议)
        raise NotImplementedError

    def _run_pylint(self, code: str) -> list[Diagnostic]:
        """运行 Pylint 检查，转换为 Diagnostic。

        Pylint 以 JSON 格式输出，每条 message 映射为一个 Diagnostic:
        - rule_id: "{msg_id}_{symbol}" e.g. "C0301_line_too_long"
        - element_ids: (line_number,)
        """
        # TODO: 写入临时文件，调用 pylint --output-format=json
        # TODO: 如果有 pylint_checks 白名单，用 --enable 过滤
        # TODO: 解析 JSON，映射 severity，生成 generic/precise messages
        raise NotImplementedError
