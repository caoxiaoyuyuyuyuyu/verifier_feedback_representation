#!/usr/bin/env python3
"""Phase A results analyzer: 8-cell summary, decision gate, bootstrap CI, per-sample analysis."""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lo), float(hi)


def bootstrap_delta_ci(precise_vals: list[float], generic_vals: list[float],
                       n_resamples: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI for the difference (precise - generic) in means."""
    rng = np.random.default_rng(seed)
    p = np.array(precise_vals)
    g = np.array(generic_vals)
    deltas = []
    for _ in range(n_resamples):
        p_boot = rng.choice(p, size=len(p), replace=True).mean()
        g_boot = rng.choice(g, size=len(g), replace=True).mean()
        deltas.append(p_boot - g_boot)
    deltas = np.array(deltas)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(deltas, [alpha * 100, (1 - alpha) * 100])
    return float(p.mean() - g.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------

def trimmed_mean(values: list[float], trim_pct: float = 0.05) -> float:
    """Trimmed mean: drop top/bottom trim_pct of values."""
    arr = np.sort(values)
    n = len(arr)
    k = int(n * trim_pct)
    if k == 0:
        return float(arr.mean())
    return float(arr[k:-k].mean())


def winsorized_mean(values: list[float], trim_pct: float = 0.05) -> float:
    """Winsorized mean: clamp top/bottom trim_pct to boundary values."""
    arr = np.sort(values).copy()
    n = len(arr)
    k = int(n * trim_pct)
    if k > 0:
        arr[:k] = arr[k]
        arr[-k:] = arr[-k - 1]
    return float(arr.mean())


def interaction_test(qwen_svg_p, qwen_svg_g, qwen_py_p, qwen_py_g,
                     llama_svg_p, llama_svg_g, llama_py_p, llama_py_g,
                     n_resamples: int = 5000, seed: int = 42) -> dict:
    """Bootstrap test for domain × specificity interaction.

    Interaction = (precise-generic in Python) - (precise-generic in SVG),
    pooled across models. If CI excludes 0, interaction is significant.
    """
    rng = np.random.default_rng(seed)

    def _delta(p_vals, g_vals):
        return np.array(p_vals).mean() - np.array(g_vals).mean()

    # Observed interaction (pooled across models)
    py_delta = (_delta(qwen_py_p, qwen_py_g) + _delta(llama_py_p, llama_py_g)) / 2
    svg_delta = (_delta(qwen_svg_p, qwen_svg_g) + _delta(llama_svg_p, llama_svg_g)) / 2
    observed = py_delta - svg_delta

    boot_interactions = []
    for _ in range(n_resamples):
        def _boot_delta(p, g):
            p_arr, g_arr = np.array(p), np.array(g)
            return rng.choice(p_arr, len(p_arr), replace=True).mean() - \
                   rng.choice(g_arr, len(g_arr), replace=True).mean()

        py_d = (_boot_delta(qwen_py_p, qwen_py_g) + _boot_delta(llama_py_p, llama_py_g)) / 2
        svg_d = (_boot_delta(qwen_svg_p, qwen_svg_g) + _boot_delta(llama_svg_p, llama_svg_g)) / 2
        boot_interactions.append(py_d - svg_d)

    boot_interactions = np.array(boot_interactions)
    lo, hi = np.percentile(boot_interactions, [2.5, 97.5])
    significant = (lo > 0) or (hi < 0)

    return {
        "observed_interaction": float(observed),
        "ci_95": (float(lo), float(hi)),
        "significant": significant,
        "interpretation": "Python benefits more from precise than SVG" if observed > 0
                         else "SVG benefits more from precise than Python",
    }


# ---------------------------------------------------------------------------
# Cell analysis
# ---------------------------------------------------------------------------

def analyze_cell(cell: dict) -> dict:
    """Extract metrics from a single cell result."""
    if cell.get("status") != "OK":
        return {"status": cell.get("status", "MISSING"), "note": cell.get("note", "")}

    result = {
        "status": "OK",
        "n_filtered": cell["n_filtered"],
        "n_total": cell["n_total"],
        "drr_macro": cell["diagnostic_reduction_rate_macro"],
        "drr_micro": cell["diagnostic_reduction_rate_micro"],
        "avg_diag_before": cell["avg_diagnostics_before"],
        "avg_diag_after_eff": cell["avg_diagnostics_after_effective"],
        "avg_tokens": cell.get("avg_tokens", None),
    }

    # Bootstrap CI for DRR macro
    if "per_sample_reduction" in cell:
        mean, lo, hi = bootstrap_ci(cell["per_sample_reduction"])
        result["drr_macro_ci"] = (lo, hi)

    # SVG secondary
    if cell["domain"] == "svg":
        result["parse_survival"] = cell.get("parse_survival_rate")
        result["predicate_fix_rate"] = cell.get("predicate_fix_rate_among_survivors")
        result["n_survivors_with_warn"] = cell.get("n_survivors_with_warn_before")

    # Python secondary
    if cell["domain"] == "python":
        result["pass_rate"] = cell.get("pass_rate")
        result["passed"] = cell.get("passed")
        result["error_reduction"] = cell.get("error_reduction_rate")

    return result


# ---------------------------------------------------------------------------
# Per-sample comparison (precise vs generic)
# ---------------------------------------------------------------------------

def per_sample_comparison(precise_cell: dict, generic_cell: dict) -> dict:
    """Compare precise vs generic at per-sample level."""
    if precise_cell.get("status") != "OK" or generic_cell.get("status") != "OK":
        return {"error": "One or both cells not OK"}

    p_samples = precise_cell.get("per_sample", [])
    g_samples = generic_cell.get("per_sample", [])

    if not p_samples or not g_samples:
        return {"error": "No per_sample data"}

    # Build index by sample idx
    p_by_idx = {s["idx"]: s for s in p_samples}
    g_by_idx = {s["idx"]: s for s in g_samples}

    common_idxs = sorted(set(p_by_idx.keys()) & set(g_by_idx.keys()))

    precise_wins = 0
    generic_wins = 0
    ties = 0
    reversals = []  # samples where generic > precise

    for idx in common_idxs:
        p_eff = p_by_idx[idx]["diags_after_effective"]
        g_eff = g_by_idx[idx]["diags_after_effective"]
        p_before = p_by_idx[idx]["diags_before"]

        if p_eff < g_eff:
            precise_wins += 1
        elif g_eff < p_eff:
            generic_wins += 1
            reversals.append({
                "idx": idx,
                "diags_before": p_before,
                "precise_after": p_eff,
                "generic_after": g_eff,
                "newly_broken_precise": p_by_idx[idx].get("newly_broken", False),
                "newly_broken_generic": g_by_idx[idx].get("newly_broken", False),
            })
        else:
            ties += 1

    return {
        "n_common": len(common_idxs),
        "precise_wins": precise_wins,
        "generic_wins": generic_wins,
        "ties": ties,
        "win_rate_precise": precise_wins / len(common_idxs) if common_idxs else 0,
        "reversals": reversals[:10],  # top 10 for inspection
    }


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------

def evaluate_decision_gate(deltas: dict[str, float], n_filtered: dict[str, int]) -> dict:
    """
    Decision gate per ROADMAP:
    PASS = mean Δ ≥ 10pp AND ≥3/4 cells ≥ 10pp AND n ≥ 30 for all cells
    """
    cells = list(deltas.keys())
    mean_delta = np.mean(list(deltas.values()))
    cells_ge_10pp = [c for c in cells if deltas[c] >= 0.10]
    cells_insufficient = [c for c in cells if n_filtered[c] < 30]

    cond_a = mean_delta >= 0.10
    cond_b = len(cells_ge_10pp) >= 3
    cond_c = len(cells_insufficient) == 0

    verdict = "PASS" if (cond_a and cond_b and cond_c) else "FAIL"

    # Degradation analysis
    degradation = None
    if verdict == "FAIL":
        if len(cells_ge_10pp) >= 2:
            degradation = "some_domains"
        elif len(cells_ge_10pp) == 1 and any("svg" in c for c in cells_ge_10pp):
            degradation = "svg_only"
        elif mean_delta < 0.05:
            degradation = "negative"
        else:
            degradation = "exploratory"

    return {
        "verdict": verdict,
        "mean_delta": float(mean_delta),
        "cond_a_mean_ge_10pp": cond_a,
        "cond_b_3of4_ge_10pp": cond_b,
        "cond_c_all_n_ge_30": cond_c,
        "cells_ge_10pp": cells_ge_10pp,
        "cells_insufficient": cells_insufficient,
        "per_cell_delta": {c: float(deltas[c]) for c in cells},
        "degradation": degradation,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(qwen_path: str, llama_path: str, output_path: str | None = None):
    qwen = load_results(qwen_path)
    llama = load_results(llama_path)

    results = OrderedDict()

    # ── 1. Per-cell metrics ──
    print("=" * 80)
    print("PHASE A RESULTS ANALYSIS")
    print("=" * 80)

    cell_names = ["svg_precise", "svg_generic", "python_precise", "python_generic"]
    model_data = {"qwen": qwen, "llama": llama}

    results["cells"] = {}
    print(f"\n{'Cell':<30} {'n':>4} {'DRR_macro':>10} {'95% CI':>18} {'DRR_micro':>10}")
    print("-" * 80)

    for model_name, data in model_data.items():
        for cell_name in cell_names:
            cell = data["cells"].get(cell_name, {})
            analysis = analyze_cell(cell)
            key = f"{model_name}_{cell_name}"
            results["cells"][key] = analysis

            if analysis["status"] == "OK":
                ci_str = ""
                if "drr_macro_ci" in analysis:
                    ci_str = f"[{analysis['drr_macro_ci'][0]:.1%}, {analysis['drr_macro_ci'][1]:.1%}]"
                print(f"{key:<30} {analysis['n_filtered']:>4} {analysis['drr_macro']:>10.1%} {ci_str:>18} {analysis['drr_micro']:>10.1%}")
            else:
                print(f"{key:<30} {'--':>4} {analysis['status']:>10}")

    # ── 2. Precise vs Generic deltas ──
    print(f"\n{'='*80}")
    print("PRECISE vs GENERIC DELTAS (DRR_macro)")
    print("=" * 80)

    deltas = {}
    n_filtered = {}
    delta_cis = {}

    for model_name, data in model_data.items():
        for domain in ["svg", "python"]:
            p_cell = data["cells"].get(f"{domain}_precise", {})
            g_cell = data["cells"].get(f"{domain}_generic", {})

            cell_key = f"{model_name}_{domain}"

            if p_cell.get("status") == "OK" and g_cell.get("status") == "OK":
                delta = p_cell["diagnostic_reduction_rate_macro"] - g_cell["diagnostic_reduction_rate_macro"]
                deltas[cell_key] = delta
                n_filtered[cell_key] = min(p_cell["n_filtered"], g_cell["n_filtered"])

                # Bootstrap CI for delta
                if "per_sample_reduction" in p_cell and "per_sample_reduction" in g_cell:
                    d_mean, d_lo, d_hi = bootstrap_delta_ci(
                        p_cell["per_sample_reduction"],
                        g_cell["per_sample_reduction"]
                    )
                    delta_cis[cell_key] = (d_lo, d_hi)

    print(f"\n{'Cell':<20} {'Δ (precise-generic)':>20} {'95% CI':>22} {'≥10pp?':>8}")
    print("-" * 75)
    for cell_key in deltas:
        d = deltas[cell_key]
        ci_str = ""
        if cell_key in delta_cis:
            lo, hi = delta_cis[cell_key]
            ci_str = f"[{lo:+.1%}, {hi:+.1%}]"
        ge10 = "✓" if d >= 0.10 else "✗"
        print(f"{cell_key:<20} {d:>+20.1%} {ci_str:>22} {ge10:>8}")

    results["deltas"] = {k: float(v) for k, v in deltas.items()}
    results["delta_cis"] = {k: list(v) for k, v in delta_cis.items()}

    # ── 3. Decision gate ──
    print(f"\n{'='*80}")
    print("DECISION GATE")
    print("=" * 80)

    if len(deltas) == 4:
        gate = evaluate_decision_gate(deltas, n_filtered)
        results["decision_gate"] = gate

        print(f"\n  Verdict: {gate['verdict']}")
        print(f"  Mean Δ = {gate['mean_delta']:+.1%} (need ≥10%): {'✓' if gate['cond_a_mean_ge_10pp'] else '✗'}")
        print(f"  Cells ≥10pp: {len(gate['cells_ge_10pp'])}/4 (need ≥3): {'✓' if gate['cond_b_3of4_ge_10pp'] else '✗'}")
        print(f"  All n≥30: {'✓' if gate['cond_c_all_n_ge_30'] else '✗'} {gate['cells_insufficient'] if gate['cells_insufficient'] else ''}")
        if gate["degradation"]:
            print(f"  Degradation: {gate['degradation']}")
    else:
        print(f"\n  Cannot evaluate: only {len(deltas)}/4 cells available")
        results["decision_gate"] = {"verdict": "INCOMPLETE", "available_cells": len(deltas)}

    # ── 4. SVG secondary metrics ──
    print(f"\n{'='*80}")
    print("SVG SECONDARY METRICS")
    print("=" * 80)

    print(f"\n{'Cell':<30} {'parse_survival':>15} {'pred_fix_rate':>15} {'n_surv_warn':>12}")
    print("-" * 75)
    for model_name, data in model_data.items():
        for spec in ["precise", "generic"]:
            key = f"{model_name}_svg_{spec}"
            a = results["cells"].get(key, {})
            if a.get("status") == "OK":
                ps = a.get("parse_survival")
                pf = a.get("predicate_fix_rate")
                nw = a.get("n_survivors_with_warn")
                ps_str = f"{ps:.1%}" if ps is not None else "N/A"
                pf_str = f"{pf:.1%}" if pf is not None else "N/A"
                nw_str = str(nw) if nw is not None else "N/A"
                print(f"{key:<30} {ps_str:>15} {pf_str:>15} {nw_str:>12}")

    # ── 5. Python secondary metrics ──
    print(f"\n{'='*80}")
    print("PYTHON SECONDARY METRICS")
    print("=" * 80)

    print(f"\n{'Cell':<30} {'pass_rate':>10} {'passed':>8} {'err_reduction':>15}")
    print("-" * 65)
    for model_name, data in model_data.items():
        for spec in ["precise", "generic"]:
            key = f"{model_name}_python_{spec}"
            a = results["cells"].get(key, {})
            if a.get("status") == "OK":
                pr = a.get("pass_rate")
                pa = a.get("passed")
                er = a.get("error_reduction")
                pr_str = f"{pr:.1%}" if pr is not None else "N/A"
                er_str = f"{er:.1%}" if er is not None else "N/A"
                pa_str = str(pa) if pa is not None else "N/A"
                print(f"{key:<30} {pr_str:>10} {pa_str:>8} {er_str:>15}")

    # ── 6. Per-sample comparison ──
    print(f"\n{'='*80}")
    print("PER-SAMPLE COMPARISON (precise vs generic)")
    print("=" * 80)

    results["per_sample"] = {}
    for model_name, data in model_data.items():
        for domain in ["svg", "python"]:
            p_cell = data["cells"].get(f"{domain}_precise", {})
            g_cell = data["cells"].get(f"{domain}_generic", {})
            comp = per_sample_comparison(p_cell, g_cell)
            key = f"{model_name}_{domain}"
            results["per_sample"][key] = comp

            if "error" not in comp:
                print(f"\n  {key}: n={comp['n_common']}")
                print(f"    precise wins: {comp['precise_wins']} ({comp['precise_wins']/comp['n_common']:.0%})")
                print(f"    generic wins: {comp['generic_wins']} ({comp['generic_wins']/comp['n_common']:.0%})")
                print(f"    ties:         {comp['ties']} ({comp['ties']/comp['n_common']:.0%})")
                if comp["reversals"]:
                    print(f"    Top reversals (generic > precise):")
                    for r in comp["reversals"][:5]:
                        print(f"      idx={r['idx']}: before={r['diags_before']}, "
                              f"precise_after={r['precise_after']}, generic_after={r['generic_after']}"
                              f"{' [precise BROKE]' if r['newly_broken_precise'] else ''}")

    # ── 7. Robustness checks ──
    print(f"\n{'='*80}")
    print("ROBUSTNESS CHECKS")
    print("=" * 80)

    # 7a. Trimmed & winsorized mean for each cell
    print(f"\n{'Cell':<30} {'mean':>8} {'trimmed5%':>10} {'winsor5%':>10}")
    print("-" * 62)
    for model_name, data in model_data.items():
        for cell_name in cell_names:
            cell = data["cells"].get(cell_name, {})
            if cell.get("status") == "OK" and "per_sample_reduction" in cell:
                vals = cell["per_sample_reduction"]
                key = f"{model_name}_{cell_name}"
                tm = trimmed_mean(vals, 0.05)
                wm = winsorized_mean(vals, 0.05)
                print(f"{key:<30} {np.mean(vals):>8.1%} {tm:>10.1%} {wm:>10.1%}")

    results["robustness"] = {}

    # 7b. Trimmed/winsorized deltas
    print(f"\n{'Cell':<20} {'Δ_mean':>10} {'Δ_trimmed':>10} {'Δ_winsor':>10}")
    print("-" * 55)
    for model_name, data in model_data.items():
        for domain in ["svg", "python"]:
            p_cell = data["cells"].get(f"{domain}_precise", {})
            g_cell = data["cells"].get(f"{domain}_generic", {})
            if p_cell.get("status") == "OK" and g_cell.get("status") == "OK":
                p_vals = p_cell["per_sample_reduction"]
                g_vals = g_cell["per_sample_reduction"]
                d_mean = np.mean(p_vals) - np.mean(g_vals)
                d_trim = trimmed_mean(p_vals) - trimmed_mean(g_vals)
                d_wins = winsorized_mean(p_vals) - winsorized_mean(g_vals)
                key = f"{model_name}_{domain}"
                print(f"{key:<20} {d_mean:>+10.1%} {d_trim:>+10.1%} {d_wins:>+10.1%}")
                results["robustness"][key] = {
                    "delta_mean": float(d_mean),
                    "delta_trimmed_5pct": float(d_trim),
                    "delta_winsorized_5pct": float(d_wins),
                }

    # 7c. Domain × specificity interaction test
    print(f"\n--- Domain × Specificity Interaction Test ---")
    try:
        qwen_svg_p = qwen["cells"].get("svg_precise", {}).get("per_sample_reduction", [])
        qwen_svg_g = qwen["cells"].get("svg_generic", {}).get("per_sample_reduction", [])
        qwen_py_p = qwen["cells"].get("python_precise", {}).get("per_sample_reduction", [])
        qwen_py_g = qwen["cells"].get("python_generic", {}).get("per_sample_reduction", [])
        llama_svg_p = llama["cells"].get("svg_precise", {}).get("per_sample_reduction", [])
        llama_svg_g = llama["cells"].get("svg_generic", {}).get("per_sample_reduction", [])
        llama_py_p = llama["cells"].get("python_precise", {}).get("per_sample_reduction", [])
        llama_py_g = llama["cells"].get("python_generic", {}).get("per_sample_reduction", [])

        if all(len(v) > 0 for v in [qwen_svg_p, qwen_svg_g, qwen_py_p, qwen_py_g,
                                      llama_svg_p, llama_svg_g, llama_py_p, llama_py_g]):
            ix = interaction_test(qwen_svg_p, qwen_svg_g, qwen_py_p, qwen_py_g,
                                  llama_svg_p, llama_svg_g, llama_py_p, llama_py_g)
            results["interaction_test"] = ix
            print(f"  Observed interaction: {ix['observed_interaction']:+.1%}")
            print(f"  95% CI: [{ix['ci_95'][0]:+.1%}, {ix['ci_95'][1]:+.1%}]")
            print(f"  Significant: {'YES' if ix['significant'] else 'NO'}")
            print(f"  Interpretation: {ix['interpretation']}")
        else:
            print("  SKIPPED: missing per_sample_reduction data")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── 8. Sign test + Wilcoxon signed-rank test ──
    print(f"\n{'='*80}")
    print("SIGN TEST + WILCOXON SIGNED-RANK TEST (paired per-sample)")
    print("=" * 80)

    from scipy.stats import wilcoxon, binomtest

    results["paired_tests"] = {}
    for model_name, data in model_data.items():
        for domain in ["svg", "python"]:
            p_cell = data["cells"].get(f"{domain}_precise", {})
            g_cell = data["cells"].get(f"{domain}_generic", {})
            key = f"{model_name}_{domain}"

            if (p_cell.get("status") != "OK" or g_cell.get("status") != "OK"
                    or "per_sample_reduction" not in p_cell
                    or "per_sample_reduction" not in g_cell):
                print(f"\n  {key}: SKIPPED (missing data)")
                continue

            p_vals = np.array(p_cell["per_sample_reduction"])
            g_vals = np.array(g_cell["per_sample_reduction"])
            n = min(len(p_vals), len(g_vals))
            p_vals, g_vals = p_vals[:n], g_vals[:n]
            diffs = p_vals - g_vals

            # Sign test
            n_pos = int(np.sum(diffs > 0))
            n_neg = int(np.sum(diffs < 0))
            n_zero = int(np.sum(diffs == 0))
            n_nonzero = n_pos + n_neg
            if n_nonzero > 0:
                sign_result = binomtest(n_pos, n_nonzero, 0.5, alternative='two-sided')
                sign_p = sign_result.pvalue
            else:
                sign_p = 1.0

            # Wilcoxon signed-rank test (two-sided)
            nonzero_diffs = diffs[diffs != 0]
            if len(nonzero_diffs) >= 10:
                wilcox_stat, wilcox_p = wilcoxon(nonzero_diffs, alternative='two-sided')
            else:
                wilcox_stat, wilcox_p = float('nan'), float('nan')

            test_result = {
                "n": n,
                "sign_test": {
                    "n_precise_better": n_pos,
                    "n_generic_better": n_neg,
                    "n_tied": n_zero,
                    "p_value": float(sign_p),
                    "significant_05": sign_p < 0.05,
                },
                "wilcoxon": {
                    "statistic": float(wilcox_stat),
                    "p_value": float(wilcox_p),
                    "significant_05": wilcox_p < 0.05 if not np.isnan(wilcox_p) else False,
                },
                "median_diff": float(np.median(diffs)),
                "mean_diff": float(np.mean(diffs)),
            }
            results["paired_tests"][key] = test_result

            print(f"\n  {key} (n={n}):")
            print(f"    Sign test: precise>{n_pos}, generic>{n_neg}, tied={n_zero}, p={sign_p:.4f}"
                  f" {'*' if sign_p < 0.05 else ''}")
            if not np.isnan(wilcox_p):
                print(f"    Wilcoxon:  W={wilcox_stat:.0f}, p={wilcox_p:.4f}"
                      f" {'*' if wilcox_p < 0.05 else ''}")
            else:
                print(f"    Wilcoxon:  SKIPPED (n_nonzero={len(nonzero_diffs)} < 10)")
            print(f"    Median Δ={np.median(diffs):+.4f}, Mean Δ={np.mean(diffs):+.4f}")

    # ── Save JSON ──
    if output_path:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=convert)
        print(f"\n\nFull results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase A results")
    parser.add_argument("--qwen", required=True, help="Path to phaseA_v3_qwen.json")
    parser.add_argument("--llama", required=True, help="Path to phaseA_v3_llama.json")
    parser.add_argument("--output", default=None, help="Path to save full analysis JSON")
    parser.add_argument("--resamples", type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    args = parser.parse_args()

    analyze(args.qwen, args.llama, args.output)


if __name__ == "__main__":
    main()
