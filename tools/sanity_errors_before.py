"""Sanity: verify how many samples have errors_before>0 under current verifiers."""
from __future__ import annotations

from src.data.prepare_svg import load_svg_dataset
from src.data.prepare_python import load_python_dataset
from src.verifiers.svg_geometric import SVGGeometricVerifier
from src.verifiers.python_static import PythonStaticVerifier
from src.verifiers.diagnostic import Severity


def count_with_errors(samples, verifier, code_attr: str) -> int:
    n_err = 0
    for s in samples:
        code = getattr(s, code_attr)
        diags = verifier.verify(code)
        if any(d.severity == Severity.ERROR for d in diags):
            n_err += 1
    return n_err


def main():
    svg_verifier = SVGGeometricVerifier()
    for split in ("medium", "hard", "extreme"):
        try:
            svg_samples = load_svg_dataset(
                dataset_name="xiaoooobai/SVGenius",
                n_samples=9999,
                seed=42,
                split=split,
            )
            n_err = count_with_errors(svg_samples, svg_verifier, "svg_string")
            print(f"SVG[{split}]: {n_err}/{len(svg_samples)} samples have errors_before>0")
        except Exception as e:
            print(f"SVG[{split}]: FAILED to load ({type(e).__name__}: {e})")

    py_verifier = PythonStaticVerifier()
    try:
        py_samples = load_python_dataset(
            dataset_name="s2e-lab/SecurityEval",
            n_samples=9999,
            seed=42,
        )
        n_err = count_with_errors(py_samples, py_verifier, "code")
        print(f"Python[SecurityEval]: {n_err}/{len(py_samples)} samples have errors_before>0")
    except Exception as e:
        print(f"Python: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
