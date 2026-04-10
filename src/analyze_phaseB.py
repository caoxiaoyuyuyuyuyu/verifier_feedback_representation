#!/usr/bin/env python3
"""Phase B results analyzer: format × specificity ablation with bootstrap CI + paired tests."""

import json
import argparse
import numpy as np
from collections import OrderedDict
from scipy.stats import wilcoxon, binomtest


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(values, n_resamples=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lo), float(hi)


def bootstrap_delta_ci(p_vals, g_vals, n_resamples=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    p, g = np.array(p_vals), np.array(g_vals)
    deltas = []
    for _ in range(n_resamples):
        deltas.append(rng.choice(p, len(p), True).mean() - rng.choice(g, len(g), True).mean())
    deltas = np.array(deltas)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(deltas, [alpha * 100, (1 - alpha) * 100])
    return float(p.mean() - g.mean()), float(lo), float(hi)


def paired_tests(p_vals, g_vals):
    """Sign test + Wilcoxon for paired precise vs generic."""
    p, g = np.array(p_vals), np.array(g_vals)
    n = min(len(p), len(g))
    p, g = p[:n], g[:n]
    diffs = p - g

    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n_zero = int(np.sum(diffs == 0))
    n_nonzero = n_pos + n_neg

    sign_p = binomtest(n_pos, n_nonzero, 0.5, alternative='two-sided').pvalue if n_nonzero > 0 else 1.0

    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) >= 10:
        w_stat, w_p = wilcoxon(nonzero_diffs, alternative='two-sided')
    else:
        w_stat, w_p = float('nan'), float('nan')

    return {
        "n": n, "n_precise_better": n_pos, "n_generic_better": n_neg, "n_tied": n_zero,
        "sign_p": float(sign_p), "wilcoxon_W": float(w_stat), "wilcoxon_p": float(w_p),
        "median_diff": float(np.median(diffs)), "mean_diff": float(np.mean(diffs)),
    }


def format_specificity_interaction(cells_data, n_resamples=5000, seed=42):
    """Permutation-style bootstrap test for format × specificity interaction.

    Tests whether the precise-generic delta varies across formats.
    Interaction = max(delta_across_formats) - min(delta_across_formats).
    """
    rng = np.random.default_rng(seed)
    formats = list(cells_data.keys())
    if len(formats) < 2:
        return {"error": "Need ≥2 formats"}

    # Observed deltas per format
    observed_deltas = {}
    for fmt, (p_vals, g_vals) in cells_data.items():
        observed_deltas[fmt] = np.mean(p_vals) - np.mean(g_vals)

    observed_range = max(observed_deltas.values()) - min(observed_deltas.values())

    # Bootstrap
    boot_ranges = []
    for _ in range(n_resamples):
        boot_deltas = {}
        for fmt, (p_vals, g_vals) in cells_data.items():
            p, g = np.array(p_vals), np.array(g_vals)
            boot_deltas[fmt] = (rng.choice(p, len(p), True).mean() -
                                rng.choice(g, len(g), True).mean())
        boot_ranges.append(max(boot_deltas.values()) - min(boot_deltas.values()))

    boot_ranges = np.array(boot_ranges)
    # p-value: proportion of bootstrap samples where range ≥ observed under null
    # Actually, we test if the range is significantly > 0
    lo, hi = np.percentile(boot_ranges, [2.5, 97.5])

    return {
        "observed_deltas": {k: float(v) for k, v in observed_deltas.items()},
        "observed_range": float(observed_range),
        "range_ci_95": (float(lo), float(hi)),
        "significant": lo > 0,  # if entire CI > 0, range is significant
    }


def analyze(qwen_path: str, llama_path: str | None, output_path: str | None = None):
    qwen = load_results(qwen_path)
    llama = load_results(llama_path) if llama_path else None

    results = OrderedDict()
    models = {"qwen": qwen}
    if llama:
        models["llama"] = llama

    formats = ["nl", "raw_json", "hybrid"]
    specificities = ["precise", "generic"]

    # ── 1. Per-cell summary ──
    print("=" * 90)
    print("PHASE B RESULTS ANALYSIS")
    print("=" * 90)

    print(f"\n{'Cell':<35} {'n':>4} {'DRR_macro':>10} {'95% CI':>22} {'DRR_micro':>10}")
    print("-" * 85)

    results["cells"] = {}
    for model_name, data in models.items():
        for fmt in formats:
            for spec in specificities:
                cell_key = f"python_{fmt}_{spec}"
                cell = data["cells"].get(cell_key, {})
                full_key = f"{model_name}_{cell_key}"

                if cell.get("status") != "OK":
                    print(f"{full_key:<35} {'--':>4} {cell.get('status', 'MISSING'):>10}")
                    results["cells"][full_key] = {"status": cell.get("status", "MISSING")}
                    continue

                drr = cell["diagnostic_reduction_rate_macro"]
                drr_micro = cell["diagnostic_reduction_rate_micro"]
                n_f = cell["n_filtered"]

                ci_str = ""
                if "per_sample_reduction" in cell:
                    _, lo, hi = bootstrap_ci(cell["per_sample_reduction"])
                    ci_str = f"[{lo:.1%}, {hi:.1%}]"
                    results["cells"][full_key] = {
                        "drr_macro": drr, "drr_micro": drr_micro, "n": n_f,
                        "ci": (lo, hi),
                    }
                else:
                    results["cells"][full_key] = {"drr_macro": drr, "drr_micro": drr_micro, "n": n_f}

                print(f"{full_key:<35} {n_f:>4} {drr:>10.1%} {ci_str:>22} {drr_micro:>10.1%}")

    # ── 2. Precise vs Generic deltas per format ──
    print(f"\n{'='*90}")
    print("PRECISE vs GENERIC DELTAS (by format)")
    print("=" * 90)

    print(f"\n{'Cell':<25} {'Δ (p-g)':>10} {'95% CI':>22} {'Sign p':>10} {'Wilcoxon p':>12}")
    print("-" * 85)

    results["deltas"] = {}
    results["paired_tests"] = {}

    for model_name, data in models.items():
        for fmt in formats:
            p_cell = data["cells"].get(f"python_{fmt}_precise", {})
            g_cell = data["cells"].get(f"python_{fmt}_generic", {})
            key = f"{model_name}_{fmt}"

            if p_cell.get("status") != "OK" or g_cell.get("status") != "OK":
                print(f"{key:<25} {'SKIP':>10}")
                continue

            p_vals = p_cell.get("per_sample_reduction", [])
            g_vals = g_cell.get("per_sample_reduction", [])

            if not p_vals or not g_vals:
                print(f"{key:<25} {'NO DATA':>10}")
                continue

            d_mean, d_lo, d_hi = bootstrap_delta_ci(p_vals, g_vals)
            tests = paired_tests(p_vals, g_vals)

            results["deltas"][key] = {"delta": d_mean, "ci": (d_lo, d_hi)}
            results["paired_tests"][key] = tests

            ci_str = f"[{d_lo:+.1%}, {d_hi:+.1%}]"
            sign_str = f"{tests['sign_p']:.4f}" + (" *" if tests['sign_p'] < 0.05 else "")
            wilc_str = (f"{tests['wilcoxon_p']:.4f}" + (" *" if tests['wilcoxon_p'] < 0.05 else "")
                        if not np.isnan(tests['wilcoxon_p']) else "n<10")

            print(f"{key:<25} {d_mean:>+10.1%} {ci_str:>22} {sign_str:>10} {wilc_str:>12}")

    # ── 3. Format × Specificity interaction (per model) ──
    print(f"\n{'='*90}")
    print("FORMAT × SPECIFICITY INTERACTION TEST")
    print("=" * 90)

    results["interactions"] = {}
    for model_name, data in models.items():
        cells_data = {}
        for fmt in formats:
            p_cell = data["cells"].get(f"python_{fmt}_precise", {})
            g_cell = data["cells"].get(f"python_{fmt}_generic", {})
            if (p_cell.get("status") == "OK" and g_cell.get("status") == "OK"
                    and "per_sample_reduction" in p_cell and "per_sample_reduction" in g_cell):
                cells_data[fmt] = (p_cell["per_sample_reduction"], g_cell["per_sample_reduction"])

        if len(cells_data) >= 2:
            ix = format_specificity_interaction(cells_data)
            results["interactions"][model_name] = ix
            print(f"\n  {model_name}:")
            print(f"    Deltas: {', '.join(f'{k}={v:+.1%}' for k, v in ix['observed_deltas'].items())}")
            print(f"    Range: {ix['observed_range']:.1%}")
            print(f"    95% CI: [{ix['range_ci_95'][0]:.1%}, {ix['range_ci_95'][1]:.1%}]")
            print(f"    Significant: {'YES' if ix['significant'] else 'NO'}")

    # ── 4. Best configuration ──
    print(f"\n{'='*90}")
    print("BEST CONFIGURATIONS")
    print("=" * 90)

    all_cells = []
    for model_name, data in models.items():
        for fmt in formats:
            for spec in specificities:
                cell_key = f"python_{fmt}_{spec}"
                cell = data["cells"].get(cell_key, {})
                if cell.get("status") == "OK":
                    all_cells.append((model_name, fmt, spec, cell["diagnostic_reduction_rate_macro"]))

    all_cells.sort(key=lambda x: -x[3])
    print(f"\n{'Rank':<6} {'Model':<10} {'Format':<10} {'Spec':<10} {'DRR_macro':>10}")
    print("-" * 50)
    for i, (m, f, s, drr) in enumerate(all_cells[:10], 1):
        print(f"{i:<6} {m:<10} {f:<10} {s:<10} {drr:>10.1%}")

    # ── Save JSON ──
    if output_path:
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return obj

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"\nFull results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase B format ablation results")
    parser.add_argument("--qwen", required=True, help="Path to phaseB Qwen results JSON")
    parser.add_argument("--llama", default=None, help="Path to phaseB Llama results JSON")
    parser.add_argument("--output", default=None, help="Path to save analysis JSON")
    args = parser.parse_args()
    analyze(args.qwen, args.llama, args.output)


if __name__ == "__main__":
    main()
