#!/usr/bin/env python3
"""B3 Power + B4 Wilcoxon + B5 Bonferroni analysis."""
import json, math, os, sys
import numpy as np
from scipy.stats import wilcoxon, norm

RESULTS = "/root/autodl-tmp/verifier_feedback_representation/results"

def load(fname):
    with open(os.path.join(RESULTS, fname)) as f:
        return json.load(f)

def extract_drr_pairs(cell):
    """Return arrays of (diags_before, diags_after_effective) per sample."""
    samples = cell["per_sample"]
    before = np.array([s["diags_before"] for s in samples], dtype=float)
    after  = np.array([s["diags_after_effective"] for s in samples], dtype=float)
    return before, after

def sample_drr(before, after):
    """Per-sample diagnostic reduction rate: (before - after) / before. 0/0 = 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        drr = np.where(before > 0, (before - after) / before, 0.0)
    return drr

def cohens_d(x, y):
    """Paired Cohen's d = mean(diff) / std(diff)."""
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return np.mean(diff) / sd

def achieved_power(d, n, alpha=0.05):
    """Power for one-sample/paired t-test (two-sided)."""
    z_alpha = norm.ppf(1 - alpha / 2)
    ncp = abs(d) * math.sqrt(n)
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return power

def wilcoxon_triplet(before, after):
    """Return (n_plus, n_minus, n_tie, W, p). 
    n_plus = correction improved (before > after), n_minus = worsened."""
    diff = before - after  # positive = improved
    n_plus  = int(np.sum(diff > 0))
    n_minus = int(np.sum(diff < 0))
    n_tie   = int(np.sum(diff == 0))
    # Remove ties for Wilcoxon
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        return n_plus, n_minus, n_tie, float('nan'), float('nan')
    try:
        stat, p = wilcoxon(nonzero, alternative='two-sided')
    except Exception:
        stat, p = float('nan'), float('nan')
    return n_plus, n_minus, n_tie, float(stat), float(p)

# ============================================================
# Load all data
# ============================================================
files = {
    "phaseA_qwen":  load("phaseA_v3_qwen.json"),
    "phaseA_llama": load("phaseA_v3_llama.json"),
    "phaseB_qwen_py":  load("phaseB_qwen_python_r2.json"),
    "phaseB_llama_py": load("phaseB_llama_python_r2.json"),
    "nofb_qwen":  load("no_feedback_qwen_r3.json"),
    "nofb_llama": load("no_feedback_llama.json"),
}

# ============================================================
# B3: Post-hoc power analysis (Phase B Python, 6 pairwise)
# ============================================================
print("=" * 70)
print("B3: POST-HOC POWER ANALYSIS (Phase B Python, precise vs generic)")
print("=" * 70)
print(f"| {'Comparison':<45} | {'n':>3} | {'Cohen d':>8} | {'Power':>7} |")
print(f"|{'-'*47}|{'-'*5}|{'-'*10}|{'-'*9}|")

b3_tests = []
for model_tag, key in [("Qwen", "phaseB_qwen_py"), ("Llama", "phaseB_llama_py")]:
    cells = files[key]["cells"]
    for fmt in ["nl", "raw_json", "hybrid"]:
        precise_key = f"python_{fmt}_precise"
        generic_key = f"python_{fmt}_generic"
        bp, ap = extract_drr_pairs(cells[precise_key])
        bg, ag = extract_drr_pairs(cells[generic_key])
        drr_p = sample_drr(bp, ap)
        drr_g = sample_drr(bg, ag)
        n = len(drr_p)
        d = cohens_d(drr_p, drr_g)
        pwr = achieved_power(d, n)
        label = f"{model_tag} {fmt}: precise vs generic"
        print(f"| {label:<45} | {n:>3} | {d:>+8.4f} | {pwr:>6.3f} |")
        b3_tests.append((label, n, d, pwr))

print()

# ============================================================
# B4: Wilcoxon signed-rank (n+, n-, n_tie) for ALL cells
# ============================================================
print("=" * 70)
print("B4: WILCOXON SIGNED-RANK TEST (before vs after correction)")
print("=" * 70)
print(f"| {'Experiment':<18} | {'Cell':<28} | {'n+':>3} | {'n-':>3} | {'tie':>3} | {'W':>8} | {'p':>10} |")
print(f"|{'-'*20}|{'-'*30}|{'-'*5}|{'-'*5}|{'-'*5}|{'-'*10}|{'-'*12}|")

all_tests = []  # for B5

# Phase A
for model_tag, key in [("Qwen", "phaseA_qwen"), ("Llama", "phaseA_llama")]:
    cells = files[key]["cells"]
    for cell_name in sorted(cells.keys()):
        cell = cells[cell_name]
        before, after = extract_drr_pairs(cell)
        np_, nm, nt, w, p = wilcoxon_triplet(before, after)
        exp = f"PhaseA {model_tag}"
        label = f"{cell_name} (n={len(before)})"
        print(f"| {exp:<18} | {label:<28} | {np_:>3} | {nm:>3} | {nt:>3} | {w:>8.1f} | {p:>10.6f} |")
        all_tests.append((f"A {model_tag} {cell_name}", p))

# Phase B Python
for model_tag, key in [("Qwen", "phaseB_qwen_py"), ("Llama", "phaseB_llama_py")]:
    cells = files[key]["cells"]
    for cell_name in sorted(cells.keys()):
        cell = cells[cell_name]
        before, after = extract_drr_pairs(cell)
        np_, nm, nt, w, p = wilcoxon_triplet(before, after)
        exp = f"PhaseB {model_tag}"
        label = f"{cell_name} (n={len(before)})"
        print(f"| {exp:<18} | {label:<28} | {np_:>3} | {nm:>3} | {nt:>3} | {w:>8.1f} | {p:>10.6f} |")
        all_tests.append((f"B {model_tag} {cell_name}", p))

# No-feedback
for model_tag, key in [("Qwen", "nofb_qwen"), ("Llama", "nofb_llama")]:
    cells = files[key]["cells"]
    for cell_name in sorted(cells.keys()):
        cell = cells[cell_name]
        before, after = extract_drr_pairs(cell)
        np_, nm, nt, w, p = wilcoxon_triplet(before, after)
        exp = f"NoFB {model_tag}"
        label = f"{cell_name} (n={len(before)})"
        print(f"| {exp:<18} | {label:<28} | {np_:>3} | {nm:>3} | {nt:>3} | {w:>8.1f} | {p:>10.6f} |")
        all_tests.append((f"NF {model_tag} {cell_name}", p))

print()

# ============================================================
# B5: Bonferroni correction
# ============================================================
k = len(all_tests)
alpha_adj = 0.05 / k

print("=" * 70)
print(f"B5: BONFERRONI CORRECTION (k={k}, α_adj={alpha_adj:.6f})")
print("=" * 70)
print(f"| {'#':>2} | {'Test':<40} | {'raw p':>10} | {'α_adj':>10} | {'Sig?':<5} |")
print(f"|{'-'*4}|{'-'*42}|{'-'*12}|{'-'*12}|{'-'*7}|")

for i, (name, p) in enumerate(all_tests, 1):
    sig = "YES" if (not math.isnan(p) and p < alpha_adj) else "no"
    p_str = f"{p:.6f}" if not math.isnan(p) else "NaN"
    print(f"| {i:>2} | {name:<40} | {p_str:>10} | {alpha_adj:>10.6f} | {sig:<5} |")

# Summary
n_sig = sum(1 for _, p in all_tests if not math.isnan(p) and p < alpha_adj)
print(f"\nSummary: {n_sig}/{k} tests significant after Bonferroni correction (α_adj={alpha_adj:.6f})")
