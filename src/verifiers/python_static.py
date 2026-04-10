"""Python Static Analysis Verifier — Bandit + Pylint wrapper.

将 Bandit (安全) 和 Pylint (代码质量) 的输出统一为 Diagnostic 格式。
支持对 Python 代码字符串进行分析。
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from .diagnostic import Diagnostic, Severity

logger = logging.getLogger(__name__)

_BANDIT_SEVERITY_MAP = {
    "HIGH": Severity.ERROR,
    "MEDIUM": Severity.WARNING,
    "LOW": Severity.INFO,
}

_BANDIT_CONFIDENCE_MAP = {
    "HIGH": 1.0,
    "MEDIUM": 0.7,
    "LOW": 0.4,
}

_PYLINT_SEVERITY_MAP = {
    "error": Severity.ERROR,
    "fatal": Severity.ERROR,
    "warning": Severity.WARNING,
    "convention": Severity.INFO,
    "refactor": Severity.INFO,
    "information": Severity.INFO,
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
        self.enable_bandit = enable_bandit
        self.enable_pylint = enable_pylint
        self.pylint_checks = pylint_checks

    def verify(self, code: str) -> list[Diagnostic]:
        """对 Python 代码字符串执行静态分析。"""
        diagnostics: list[Diagnostic] = []
        if self.enable_bandit:
            diagnostics.extend(self._run_bandit(code))
        if self.enable_pylint:
            diagnostics.extend(self._run_pylint(code))
        return diagnostics

    def _run_bandit(self, code: str) -> list[Diagnostic]:
        """运行 Bandit 安全扫描，转换为 Diagnostic。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["bandit", "-f", "json", "-q", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            # Bandit returns non-zero when it finds issues
            output = result.stdout
            if not output.strip():
                return []

            data = json.loads(output)
            results = data.get("results", [])
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning("Bandit execution failed: %s", e)
            return []
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        diagnostics = []
        code_lines = code.splitlines()

        for finding in results:
            test_id = finding.get("test_id", "B000")
            test_name = finding.get("test_name", "unknown")
            severity_str = finding.get("issue_severity", "LOW")
            confidence_str = finding.get("issue_confidence", "LOW")
            line_no = finding.get("line_number", 0)
            issue_text = finding.get("issue_text", "")
            more_info = finding.get("more_info", "")

            severity = _BANDIT_SEVERITY_MAP.get(severity_str, Severity.INFO)
            rule_id = f"{test_id}_{test_name}"

            # Get code context for precise message
            code_line = code_lines[line_no - 1].strip() if 0 < line_no <= len(code_lines) else ""
            precise_msg = (
                f"Line {line_no}: {issue_text} "
                f"(confidence: {confidence_str.lower()}). "
                f"Code: `{code_line}`"
            )
            if more_info:
                precise_msg += f" See: {more_info}"

            diagnostics.append(Diagnostic(
                rule_id=rule_id,
                severity=severity,
                element_ids=(str(line_no),),
                message_generic=issue_text,
                message_precise=precise_msg,
                metric_name="bandit_confidence",
                metric_value=_BANDIT_CONFIDENCE_MAP.get(confidence_str, 0.4),
                fix_direction=f"Fix {test_id} ({test_name}) security issue at line {line_no}",
                metadata={
                    "test_id": test_id,
                    "test_name": test_name,
                    "issue_severity": severity_str,
                    "issue_confidence": confidence_str,
                },
            ))

        return diagnostics

    def _run_pylint(self, code: str) -> list[Diagnostic]:
        """运行 Pylint 检查，转换为 Diagnostic。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        cmd = ["pylint", "--output-format=json2", "--disable=all"]

        if self.pylint_checks:
            cmd.append(f"--enable={','.join(self.pylint_checks)}")
        else:
            # Enable common useful checks
            cmd.append("--enable=all")
            cmd.append("--disable=C0114,C0115,C0116")  # skip missing docstrings

        cmd.append(tmp_path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            output = result.stdout
            if not output.strip():
                return []

            data = json.loads(output)
            messages = data.get("messages", [])
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning("Pylint execution failed: %s", e)
            return []
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        diagnostics = []
        code_lines = code.splitlines()

        for msg in messages:
            msg_id = msg.get("message-id", "")
            symbol = msg.get("symbol", "unknown")
            msg_type = msg.get("type", "convention")
            line_no = msg.get("line", 0)
            message = msg.get("message", "")

            severity = _PYLINT_SEVERITY_MAP.get(msg_type, Severity.INFO)
            rule_id = f"{msg_id}_{symbol}"

            code_line = code_lines[line_no - 1].strip() if 0 < line_no <= len(code_lines) else ""
            precise_msg = (
                f"Line {line_no}: [{msg_id}] {message}. "
                f"Code: `{code_line}`"
            )

            diagnostics.append(Diagnostic(
                rule_id=rule_id,
                severity=severity,
                element_ids=(str(line_no),),
                message_generic=f"{symbol}: {message}",
                message_precise=precise_msg,
                metric_name="pylint_type",
                metric_value={"error": 1.0, "warning": 0.7, "convention": 0.3,
                              "refactor": 0.3}.get(msg_type, 0.1),
                fix_direction=f"Fix {symbol} ({msg_id}) issue at line {line_no}",
                metadata={"msg_id": msg_id, "symbol": symbol, "type": msg_type},
            ))

        return diagnostics
