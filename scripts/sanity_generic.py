"""Generic feedback sanity check."""
import sys, importlib

# Direct imports to avoid __init__.py pulling in shapely
sys.path.insert(0, ".")
diagnostic_mod = importlib.import_module("src.verifiers.diagnostic")
Diagnostic = diagnostic_mod.Diagnostic
Severity = diagnostic_mod.Severity

# Patch verifiers __init__ to avoid svg_geometric import
import types
verifiers_pkg = types.ModuleType("src.verifiers")
verifiers_pkg.Diagnostic = Diagnostic
verifiers_pkg.Severity = Severity
sys.modules["src.verifiers"] = verifiers_pkg
sys.modules["src.verifiers.diagnostic"] = diagnostic_mod

templates_mod = importlib.import_module("src.feedback.templates")
render_feedback = templates_mod.render_feedback
FeedbackFormat = templates_mod.FeedbackFormat
FeedbackSpecificity = templates_mod.FeedbackSpecificity

print("=== Testing render_feedback GENERIC mode ===")

test_diag = Diagnostic(
    rule_id="overlap",
    severity=Severity.ERROR,
    element_ids=("circle_3",),
    message_generic="Elements overlap detected",
    message_precise="circle #3 cx=280 r=50 overflows viewport width=300",
    metric_name="overlap_area",
    metric_value=0.15,
    fix_direction="Reduce circle radius or move center left",
)

# GENERIC NL
generic_nl = render_feedback([test_diag], FeedbackFormat.NL, FeedbackSpecificity.GENERIC)
print(f"GENERIC NL:\n{generic_nl}\n")

# PRECISE NL
precise_nl = render_feedback([test_diag], FeedbackFormat.NL, FeedbackSpecificity.PRECISE)
print(f"PRECISE NL:\n{precise_nl}\n")

# Assertions
assert generic_nl.strip(), "GENERIC output is empty!"
assert "circle #3" not in generic_nl, "GENERIC leaks precise message!"
assert "overlap_area" not in generic_nl, "GENERIC leaks metric_name!"
assert "0.15" not in generic_nl, "GENERIC leaks metric_value!"
assert "Reduce circle" not in generic_nl, "GENERIC leaks fix_direction!"
print("OK: GENERIC NL passes all checks\n")

# GENERIC RAW_JSON
generic_json = render_feedback([test_diag], FeedbackFormat.RAW_JSON, FeedbackSpecificity.GENERIC)
print(f"GENERIC RAW_JSON:\n{generic_json}\n")
assert "metric_value" not in generic_json, "RAW_JSON leaks metric_value key"
assert "fix_direction" not in generic_json, "RAW_JSON leaks fix_direction key"
print("OK: GENERIC RAW_JSON passes\n")

# GENERIC HYBRID
generic_hybrid = render_feedback([test_diag], FeedbackFormat.HYBRID, FeedbackSpecificity.GENERIC)
print(f"GENERIC HYBRID:\n{generic_hybrid}\n")
print("OK: All 3 format x GENERIC checks passed")
