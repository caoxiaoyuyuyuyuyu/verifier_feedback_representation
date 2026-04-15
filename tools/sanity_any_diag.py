"""D015 Step 2b: sanity on expanded filter (any diagnostic > 0).

Counts samples where verifier produces >=1 diagnostic regardless of severity,
and contrasts against strict (ERROR-only) filter for comparison.
"""
from __future__ import annotations

from src.data.prepare_svg import load_svg_dataset
from src.data.prepare_python import load_python_dataset
from src.verifiers.svg_geometric import SVGGeometricVerifier
from src.verifiers.python_static import PythonStaticVerifier
from src.verifiers.diagnostic import Severity


def summarize(samples, verifier, code_attr: str, label: str):
    n_any = 0
    n_err = 0
    severity_counter = {}
    diag_count_hist = {"0": 0, "1-3": 0, "4-10": 0, "11+": 0}
    for s in samples:
        code = getattr(s, code_attr)
        diags = verifier.verify(code)
        k = len(diags)
        if k == 0:
            diag_count_hist["0"] += 1
        elif k <= 3:
            diag_count_hist["1-3"] += 1
        elif k <= 10:
            diag_count_hist["4-10"] += 1
        else:
            diag_count_hist["11+"] += 1
        if k > 0:
            n_any += 1
        if any(d.severity == Severity.ERROR for d in diags):
            n_err += 1
        for d in diags:
            sev_name = d.severity.name if hasattr(d.severity, "name") else str(d.severity)
            severity_counter[sev_name] = severity_counter.get(sev_name, 0) + 1
    total = len(samples)
    print(f"{label}: total={total}")
    print(f"  any-diag:    {n_any}/{total}  ({100*n_any/total:.1f}%)")
    print(f"  error-only:  {n_err}/{total}  ({100*n_err/total:.1f}%)")
    print(f"  severity distribution (total occurrences): {severity_counter}")
    print(f"  per-sample diag-count histogram: {diag_count_hist}")


def main():
    svg_verifier = SVGGeometricVerifier()
    print("=" * 60)
    print("SVG (xiaoooobai/SVGenius) with SVGGeometricVerifier")
    print("=" * 60)
    for split in ("easy", "medium", "hard"):
        try:
            svg_samples = load_svg_dataset(
                dataset_name="xiaoooobai/SVGenius",
                n_samples=9999,
                seed=42,
                split=split,
            )
            summarize(svg_samples, svg_verifier, "svg_string", f"SVG[{split}]")
        except Exception as e:
            print(f"SVG[{split}]: FAILED to load ({type(e).__name__}: {e})")
        print()

    print("=" * 60)
    print("Python (s2e-lab/SecurityEval) with PythonStaticVerifier")
    print("=" * 60)
    try:
        py_samples = load_python_dataset(
            dataset_name="s2e-lab/SecurityEval",
            n_samples=9999,
            seed=42,
        )
        summarize(py_samples, PythonStaticVerifier(), "code", "Python[SecurityEval]")
    except Exception as e:
        print(f"Python: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
