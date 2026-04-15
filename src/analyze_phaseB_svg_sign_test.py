#!/usr/bin/env python3
"""Phase B SVG: paired sign test + Wilcoxon for precise vs generic.

For each (model, format) cell, compares per-sample diagnostic reduction
(precise - generic) using a paired sign test (binomial) and Wilcoxon
signed-rank test. Uses Bonferroni alpha = 0.05 / 6 = 0.00833 across the
6 (model, format) comparisons.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

MODEL_FILES = {
    "qwen": RESULTS_DIR / "phaseB_qwen_svg.json",
    "llama": RESULTS_DIR / "phaseB_llama_svg_r2.json",
}
FORMATS = ["nl", "raw_json", "hybrid"]
N_COMPARISONS = 6  # 2 models x 3 formats
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS


def load_cell(data, fmt, spec):
    key = f"svg_{fmt}_{spec}"
    cell = data["cells"][key]
    if cell.get("status") != "OK":
        raise RuntimeError(f"cell {key} status={cell.get('status')}")
    return cell


def paired_test(precise_drr, generic_drr):
    diff = precise_drr - generic_drr
    n_pos = int(np.sum(diff > 0))
    n_neg = int(np.sum(diff < 0))
    n_tied = int(np.sum(diff == 0))
    n_nonzero = n_pos + n_neg

    sign_p = (
        binomtest(n_pos, n_nonzero, 0.5).pvalue if n_nonzero > 0 else 1.0
    )

    nonzero = diff[diff != 0]
    if len(nonzero) >= 10:
        _, wilcoxon_p = wilcoxon(nonzero)
        wilcoxon_p = float(wilcoxon_p)
    else:
        wilcoxon_p = None

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_tied": n_tied,
        "n_nonzero": n_nonzero,
        "sign_p": float(sign_p),
        "wilcoxon_p": wilcoxon_p,
    }


def analyse_model(model_name, path):
    data = json.loads(path.read_text())
    out = {}
    for fmt in FORMATS:
        precise = load_cell(data, fmt, "precise")
        generic = load_cell(data, fmt, "generic")
        p_drr = np.asarray(precise["per_sample_reduction"], dtype=float)
        g_drr = np.asarray(generic["per_sample_reduction"], dtype=float)
        if len(p_drr) != len(g_drr):
            raise RuntimeError(
                f"{model_name}/{fmt}: precise n={len(p_drr)} vs generic n={len(g_drr)}"
            )
        test = paired_test(p_drr, g_drr)
        test.update(
            {
                "n_samples": int(len(p_drr)),
                "macro_precise": float(precise["diagnostic_reduction_rate_macro"]),
                "macro_generic": float(generic["diagnostic_reduction_rate_macro"]),
                "micro_precise": float(precise["diagnostic_reduction_rate_micro"]),
                "micro_generic": float(generic["diagnostic_reduction_rate_micro"]),
            }
        )
        out[f"{model_name}_{fmt}"] = test
    return out


def fmt_p(p):
    if p is None:
        return "  n/a "
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def main():
    results = {}
    for model, path in MODEL_FILES.items():
        results.update(analyse_model(model, path))

    header = (
        f"{'cell':<18} {'n+':>4} {'n-':>4} {'n=':>4} "
        f"{'sign_p':>8} {'wilcox_p':>9} "
        f"{'macroP':>8} {'macroG':>8} {'microP':>8} {'microG':>8} sig"
    )
    print(header)
    print("-" * len(header))
    for key, r in results.items():
        sig = "*" if r["sign_p"] < BONFERRONI_ALPHA else " "
        print(
            f"{key:<18} {r['n_pos']:>4} {r['n_neg']:>4} {r['n_tied']:>4} "
            f"{fmt_p(r['sign_p']):>8} {fmt_p(r['wilcoxon_p']):>9} "
            f"{r['macro_precise']*100:>7.2f}% {r['macro_generic']*100:>7.2f}% "
            f"{r['micro_precise']*100:>7.2f}% {r['micro_generic']*100:>7.2f}% {sig}"
        )
    print(
        f"\nBonferroni alpha = 0.05 / {N_COMPARISONS} = {BONFERRONI_ALPHA:.4f}; "
        f"'*' = sign_p < alpha"
    )

    out_path = RESULTS_DIR / "phaseB_svg_sign_test.json"
    payload = {
        "description": "Phase B SVG paired sign test: precise vs generic per-sample DRR",
        "test": "binomial sign test (two-sided) + Wilcoxon signed-rank on nonzero diffs",
        "diff": "precise - generic (per sample)",
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "n_comparisons": N_COMPARISONS,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
