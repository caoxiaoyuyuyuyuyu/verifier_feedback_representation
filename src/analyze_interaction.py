"""Task 1-3 statistical analyses for verifier feedback representation."""
import json
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False

ROOT = Path("/root/autodl-tmp/verifier_feedback_representation")
RESULTS = ROOT / "results"
OUT = RESULTS / "interaction_analysis.json"


def load(name):
    return json.load(open(RESULTS / name))


def cells(d):
    return d["cells"]


# ---------- helpers ----------
def effective_after(sample, penalty):
    before = sample["diags_before"]
    after = sample["diags_after"]
    if sample.get("newly_broken", False):
        return max(before + penalty, after)
    return after


def drr_per_sample(sample, penalty):
    eff = effective_after(sample, penalty)
    base = max(sample["diags_before"], 1)
    return 1.0 - eff / base


def drr_macro(samples, penalty):
    return float(np.mean([drr_per_sample(s, penalty) for s in samples]))


def paired_diffs(ps_a, ps_b):
    # align by idx
    map_b = {s["idx"]: s for s in ps_b}
    out = []
    for sa in ps_a:
        sb = map_b.get(sa["idx"])
        if sb is None:
            continue
        ra = 1.0 - sa["diags_after_effective"] / max(sa["diags_before"], 1)
        rb = 1.0 - sb["diags_after_effective"] / max(sb["diags_before"], 1)
        out.append(ra - rb)
    return np.array(out)


def wilcoxon_report(diffs):
    n = len(diffs)
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n_tie = int(np.sum(diffs == 0))
    try:
        method = "exact" if n < 25 else "approx"
        res = stats.wilcoxon(diffs, zero_method="pratt", method=method, alternative="two-sided")
        stat = float(res.statistic)
        p = float(res.pvalue)
    except Exception as e:
        stat, p, method = None, None, f"error:{e}"
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_tied": n_tie,
        "statistic": stat,
        "p_value": p,
        "method": method,
        "zero_method": "pratt",
    }


def cohens_d_power(diffs, alpha):
    n = len(diffs)
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    d = m / sd if sd > 0 else float("nan")
    # Paired t-test post-hoc power, two-sided
    try:
        from statsmodels.stats.power import TTestPower
        power = float(TTestPower().power(effect_size=abs(d), nobs=n, alpha=alpha, alternative="two-sided"))
    except Exception:
        # Fallback: manual noncentral-t computation
        from scipy.stats import nct, t
        df = n - 1
        nc = abs(d) * math.sqrt(n)
        tcrit = t.ppf(1 - alpha / 2, df)
        power = float(1 - nct.cdf(tcrit, df, nc) + nct.cdf(-tcrit, df, nc))
    return {"mean_diff": m, "sd_diff": sd, "cohen_d": d, "power": power, "n": n, "alpha": alpha}


# ---------- load ----------
A_Q = cells(load("phaseA_v3_qwen.json"))
A_L = cells(load("phaseA_v3_llama.json"))
B_Q = cells(load("phaseB_qwen_python_r2.json"))
B_L = cells(load("phaseB_llama_python_r2.json"))

out = {"task1": {}, "task2": {}, "task3": {}}

# ---------- Task 1: mixed-effects ----------
def task1_phaseA(cells_d, label):
    rows = []
    for cell_name in ["svg_precise", "svg_generic", "python_precise", "python_generic"]:
        c = cells_d[cell_name]
        spec = 1 if c["specificity"] == "precise" else 0
        dom = 1 if c["domain"] == "python" else 0
        for s in c["per_sample"]:
            red = 1.0 - s["diags_after_effective"] / max(s["diags_before"], 1)
            sid = f"{c['domain']}_{s['idx']}"
            rows.append({"sample_id": sid, "specificity": spec, "domain": dom, "reduction": red})
    df = pd.DataFrame(rows)
    res = {"n_rows": len(df), "model": label}
    if not HAS_SM:
        res["error"] = "statsmodels not available"
        return res
    try:
        md = smf.mixedlm("reduction ~ specificity * domain", df, groups=df["sample_id"])
        fit = md.fit(reml=False, method="lbfgs")
        res["model_type"] = "mixedlm"
        res["converged"] = bool(fit.converged)
        res["params"] = {k: float(v) for k, v in fit.params.items()}
        res["pvalues"] = {k: float(v) for k, v in fit.pvalues.items()}
        res["bse"] = {k: float(v) for k, v in fit.bse.items()}
        if "specificity:domain" in fit.params:
            res["interaction"] = {
                "coef": float(fit.params["specificity:domain"]),
                "se": float(fit.bse["specificity:domain"]),
                "z": float(fit.tvalues["specificity:domain"]),
                "p": float(fit.pvalues["specificity:domain"]),
            }
    except Exception as e:
        # fallback to OLS
        res["mixedlm_error"] = str(e)
        ols = smf.ols("reduction ~ specificity * domain", df).fit()
        res["model_type"] = "ols_fallback"
        res["params"] = {k: float(v) for k, v in ols.params.items()}
        res["pvalues"] = {k: float(v) for k, v in ols.pvalues.items()}
        res["bse"] = {k: float(v) for k, v in ols.bse.items()}
        if "specificity:domain" in ols.params:
            res["interaction"] = {
                "coef": float(ols.params["specificity:domain"]),
                "se": float(ols.bse["specificity:domain"]),
                "t": float(ols.tvalues["specificity:domain"]),
                "p": float(ols.pvalues["specificity:domain"]),
            }
    return res


def task1_phaseB(cells_d, label):
    rows = []
    for cell_name, c in cells_d.items():
        spec = 1 if c["specificity"] == "precise" else 0
        fmt = c["format"]
        for s in c["per_sample"]:
            red = 1.0 - s["diags_after_effective"] / max(s["diags_before"], 1)
            sid = f"py_{s['idx']}"
            rows.append({"sample_id": sid, "format": fmt, "specificity": spec, "reduction": red})
    df = pd.DataFrame(rows)
    res = {"n_rows": len(df), "model": label}
    if not HAS_SM:
        res["error"] = "statsmodels not available"
        return res
    try:
        md = smf.mixedlm("reduction ~ C(format) * specificity", df, groups=df["sample_id"])
        fit = md.fit(reml=False, method="lbfgs")
        res["model_type"] = "mixedlm"
        res["converged"] = bool(fit.converged)
        res["params"] = {k: float(v) for k, v in fit.params.items()}
        res["pvalues"] = {k: float(v) for k, v in fit.pvalues.items()}
        res["bse"] = {k: float(v) for k, v in fit.bse.items()}
        inter = {k: {"coef": float(fit.params[k]), "se": float(fit.bse[k]),
                     "z": float(fit.tvalues[k]), "p": float(fit.pvalues[k])}
                 for k in fit.params.index if ":" in k}
        res["interaction_terms"] = inter
    except Exception as e:
        res["mixedlm_error"] = str(e)
        ols = smf.ols("reduction ~ C(format) * specificity", df).fit()
        res["model_type"] = "ols_fallback"
        res["params"] = {k: float(v) for k, v in ols.params.items()}
        res["pvalues"] = {k: float(v) for k, v in ols.pvalues.items()}
        res["bse"] = {k: float(v) for k, v in ols.bse.items()}
        inter = {k: {"coef": float(ols.params[k]), "se": float(ols.bse[k]),
                     "t": float(ols.tvalues[k]), "p": float(ols.pvalues[k])}
                 for k in ols.params.index if ":" in k}
        res["interaction_terms"] = inter
    return res


out["task1"]["phaseA_qwen"] = task1_phaseA(A_Q, "qwen")
out["task1"]["phaseA_llama"] = task1_phaseA(A_L, "llama")
out["task1"]["phaseB_qwen"] = task1_phaseB(B_Q, "qwen")
out["task1"]["phaseB_llama"] = task1_phaseB(B_L, "llama")

# ---------- Task 2: penalty sensitivity (Qwen SVG) ----------
svg_p = A_Q["svg_precise"]["per_sample"]
svg_g = A_Q["svg_generic"]["per_sample"]
t2 = {}
for pen in [0, 5, 10, 20]:
    dp = drr_macro(svg_p, pen)
    dg = drr_macro(svg_g, pen)
    t2[f"penalty_{pen}"] = {"DRR_precise": dp, "DRR_generic": dg, "delta": dp - dg}

svg_p_ex = [s for s in svg_p if s["idx"] != 54]
svg_g_ex = [s for s in svg_g if s["idx"] != 54]
dp = drr_macro(svg_p_ex, 10)
dg = drr_macro(svg_g_ex, 10)
t2["penalty_10_excl_54"] = {"DRR_precise": dp, "DRR_generic": dg, "delta": dp - dg,
                            "n_precise": len(svg_p_ex), "n_generic": len(svg_g_ex)}
out["task2"] = t2

# ---------- Task 3: Wilcoxon triplets + power ----------
t3 = {"phaseA": {}, "phaseB": {}, "power_phaseB": {}}

# Phase A: 4 model×domain cells
for model_name, cells_d in [("qwen", A_Q), ("llama", A_L)]:
    for dom in ["svg", "python"]:
        p = cells_d[f"{dom}_precise"]["per_sample"]
        g = cells_d[f"{dom}_generic"]["per_sample"]
        diffs = paired_diffs(p, g)
        t3["phaseA"][f"{model_name}_{dom}"] = wilcoxon_report(diffs)

# Phase B: 6 model×format cells
ALPHA_B = 0.05 / 6  # 0.00833
for model_name, cells_d in [("qwen", B_Q), ("llama", B_L)]:
    for fmt in ["nl", "raw_json", "hybrid"]:
        p = cells_d[f"python_{fmt}_precise"]["per_sample"]
        g = cells_d[f"python_{fmt}_generic"]["per_sample"]
        diffs = paired_diffs(p, g)
        key = f"{model_name}_{fmt}"
        t3["phaseB"][key] = wilcoxon_report(diffs)
        t3["power_phaseB"][key] = cohens_d_power(diffs, alpha=ALPHA_B)

out["task3"] = t3
out["_meta"] = {"alpha_bonferroni_phaseB": ALPHA_B, "has_statsmodels": HAS_SM}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)

# ---------- human-readable summary ----------
print("=" * 70)
print("TASK 1: Mixed-effects interaction test")
print("=" * 70)
for key in ["phaseA_qwen", "phaseA_llama"]:
    r = out["task1"][key]
    print(f"\n[{key}] model_type={r.get('model_type')} converged={r.get('converged')}")
    if "interaction" in r:
        i = r["interaction"]
        print(f"  specificity:domain  coef={i['coef']:.4f}  se={i['se']:.4f}  "
              f"stat={i.get('z', i.get('t')):.3f}  p={i['p']:.4f}")
for key in ["phaseB_qwen", "phaseB_llama"]:
    r = out["task1"][key]
    print(f"\n[{key}] model_type={r.get('model_type')} converged={r.get('converged')}")
    for k, v in r.get("interaction_terms", {}).items():
        stat = v.get("z", v.get("t"))
        print(f"  {k}  coef={v['coef']:.4f}  se={v['se']:.4f}  stat={stat:.3f}  p={v['p']:.4f}")

print("\n" + "=" * 70)
print("TASK 2: Penalty-free sensitivity grid (Qwen SVG)")
print("=" * 70)
print(f"{'Condition':<28} {'DRR(P)':>10} {'DRR(G)':>10} {'Delta':>10}")
labels = [("penalty_0", "penalty=0"),
          ("penalty_5", "penalty=5"),
          ("penalty_10", "penalty=10 (default)"),
          ("penalty_20", "penalty=20"),
          ("penalty_10_excl_54", "penalty=10, excl #54")]
for k, lbl in labels:
    r = out["task2"][k]
    print(f"{lbl:<28} {r['DRR_precise']:>10.4f} {r['DRR_generic']:>10.4f} {r['delta']:>10.4f}")

print("\n" + "=" * 70)
print("TASK 3: Wilcoxon paired tests")
print("=" * 70)
print("\n[Phase A]")
for k, r in out["task3"]["phaseA"].items():
    print(f"  {k:<20} n={r['n']} pos={r['n_pos']} neg={r['n_neg']} tied={r['n_tied']} "
          f"W={r['statistic']} p={r['p_value']:.4g} ({r['method']})")
print("\n[Phase B]")
for k, r in out["task3"]["phaseB"].items():
    print(f"  {k:<20} n={r['n']} pos={r['n_pos']} neg={r['n_neg']} tied={r['n_tied']} "
          f"W={r['statistic']} p={r['p_value']:.4g} ({r['method']})")
print(f"\n[Post-hoc power Phase B, alpha={ALPHA_B:.4f}]")
for k, r in out["task3"]["power_phaseB"].items():
    print(f"  {k:<20} d={r['cohen_d']:.4f} power={r['power']:.4f} (n={r['n']})")

print(f"\n[OK] Wrote {OUT}")
