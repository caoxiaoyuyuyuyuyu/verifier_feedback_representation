"""Joint LMM analyses for camera-ready R1 response.

Uses Phase A multiseed data (3 seeds × 2 models × 2 domains).
Each file has cells: {domain}_precise, {domain}_generic (NL format).

(a) Llama-only: DRR ~ specificity * domain + (1|sample_id)
(b) Qwen-only: DRR ~ specificity * domain + (1|sample_id)  [verification]
(c) Joint three-way: DRR ~ specificity * domain * model + (1|sample_id)

Outputs JSON with full model summaries for appendix inclusion.
"""
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("ERROR: statsmodels not installed", file=sys.stderr)
    sys.exit(1)


RESULTS_DIR = Path("/home/caoxiaoyu/verifier_feedback_representation/results")
MULTISEED_DIR = RESULTS_DIR / "multiseed"

# Cells in each file: {domain}_precise, {domain}_generic
CELL_MAP = {
    "svg_precise": ("svg", "precise"),
    "svg_generic": ("svg", "generic"),
    "python_precise": ("python", "precise"),
    "python_generic": ("python", "generic"),
}


def load_long_df():
    """Load all multiseed JSON files into a long-format DataFrame.

    Files: {model}_{domain}_seed{N}.json
    Each has cells like svg_precise, svg_generic (for SVG files)
    or python_precise, python_generic (for Python files).
    We average across seeds per sample for the LMM.
    """
    rows = []
    files_used = []

    for json_path in sorted(MULTISEED_DIR.glob("*.json")):
        stem = json_path.stem  # e.g. qwen_svg_seed42
        parts = stem.split("_")
        model_name = parts[0]  # qwen / llama
        domain_name = parts[1]  # svg / python
        seed = parts[2]  # seed42

        files_used.append(str(json_path))
        print(f"Loading {stem}")

        with open(json_path) as f:
            data = json.load(f)

        cells = data["cells"]
        for cell_name, (domain, spec) in CELL_MAP.items():
            if domain != domain_name:
                continue
            if cell_name not in cells:
                print(f"  WARNING: {cell_name} not in {stem}")
                continue
            cell = cells[cell_name]
            per_sample = cell["per_sample_reduction"]
            for idx, val in enumerate(per_sample):
                rows.append({
                    "sample_id": f"{domain}_{idx}",
                    "seed": seed,
                    "model": model_name,
                    "domain": domain,
                    "specificity": spec,
                    "spec_bin": 1 if spec == "precise" else 0,
                    "domain_bin": 1 if domain == "python" else 0,
                    "model_bin": 1 if model_name == "llama" else 0,
                    "reduction": float(val),
                })

    df = pd.DataFrame(rows)

    # Average across seeds per (sample_id, model, domain, specificity)
    print(f"\nRaw: {len(df)} rows from {len(files_used)} files, {df['seed'].nunique()} seeds")
    df_avg = (df.groupby(["sample_id", "model", "domain", "specificity",
                           "spec_bin", "domain_bin", "model_bin"])
              ["reduction"].mean().reset_index())
    print(f"After seed-averaging: {len(df_avg)} rows")
    return df_avg, files_used


def fit_lmm(df, formula, label):
    """Fit LMM with OLS fallback. Returns dict with full results."""
    out = {"label": label, "formula": formula, "n_obs": len(df),
           "n_samples": df["sample_id"].nunique()}

    try:
        md = smf.mixedlm(formula, data=df, groups=df["sample_id"])
        result = md.fit(method="lbfgs", reml=True)
        out["fit_method"] = "mixedlm_reml"
        out["summary"] = str(result.summary())
        params = result.params.to_dict()
        bse = result.bse.to_dict()
        tvals = result.tvalues.to_dict()
        pvals = result.pvalues.to_dict()
        out["converged"] = bool(getattr(result, "converged", True))
        try:
            out["random_effect_var"] = float(result.cov_re.iloc[0, 0])
        except Exception:
            pass
    except Exception as e:
        out["mixedlm_error"] = f"{type(e).__name__}: {e}"
        ols = smf.ols(formula, data=df).fit()
        out["fit_method"] = "ols_fallback"
        out["summary"] = str(ols.summary())
        params = ols.params.to_dict()
        bse = ols.bse.to_dict()
        tvals = ols.tvalues.to_dict()
        pvals = ols.pvalues.to_dict()
        out["converged"] = True

    coef_table = []
    for name in params.keys():
        coef_table.append({
            "term": name,
            "coef": float(params[name]),
            "se": float(bse.get(name, float("nan"))),
            "z": float(tvals.get(name, float("nan"))),
            "p": float(pvals.get(name, float("nan"))),
        })
    out["coef_table"] = coef_table

    # Cell means
    grp_cols = [c for c in ["model", "domain", "specificity"] if c in df.columns]
    cell_means = (df.groupby(grp_cols)["reduction"]
                  .agg(["mean", "std", "count"]).reset_index())
    out["cell_means"] = cell_means.to_dict(orient="records")
    return out


def main():
    df, files_used = load_long_df()
    print(f"\nLoaded {len(df)} rows, {df['sample_id'].nunique()} unique samples")
    print(df.groupby(["domain", "model", "specificity"]).size())

    results = {"data_source": "Phase A multiseed (seed-averaged)",
               "files_used": files_used}

    # (a) Llama-only: domain x specificity
    print("\n" + "=" * 60)
    print("(a) Llama-only: DRR ~ specificity * domain + (1|sample_id)")
    print("=" * 60)
    df_llama = df[df["model"] == "llama"].copy()
    fit_a = fit_lmm(
        df_llama,
        "reduction ~ spec_bin * domain_bin",
        "llama_domain_x_specificity"
    )
    print(fit_a["summary"])
    print("\nCoefficients:")
    for row in fit_a["coef_table"]:
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"  {row['term']:<30s}  coef={row['coef']:+.4f}  SE={row['se']:.4f}  z={row['z']:+.3f}  p={row['p']:.4g} {sig}")
    results["llama_domain_x_specificity"] = fit_a

    # (b) Qwen-only for verification
    print("\n" + "=" * 60)
    print("(b) Qwen-only: DRR ~ specificity * domain + (1|sample_id)")
    print("=" * 60)
    df_qwen = df[df["model"] == "qwen"].copy()
    fit_b = fit_lmm(
        df_qwen,
        "reduction ~ spec_bin * domain_bin",
        "qwen_domain_x_specificity"
    )
    print(fit_b["summary"])
    print("\nCoefficients:")
    for row in fit_b["coef_table"]:
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"  {row['term']:<30s}  coef={row['coef']:+.4f}  SE={row['se']:.4f}  z={row['z']:+.3f}  p={row['p']:.4g} {sig}")
    results["qwen_domain_x_specificity"] = fit_b

    # (c) Joint three-way: DRR ~ specificity * domain * model + (1|sample_id)
    print("\n" + "=" * 60)
    print("(c) Joint: DRR ~ specificity * domain * model + (1|sample_id)")
    print("=" * 60)
    fit_c = fit_lmm(
        df,
        "reduction ~ spec_bin * domain_bin * model_bin",
        "joint_three_way"
    )
    print(fit_c["summary"])
    print("\nCoefficients:")
    for row in fit_c["coef_table"]:
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        print(f"  {row['term']:<30s}  coef={row['coef']:+.4f}  SE={row['se']:.4f}  z={row['z']:+.3f}  p={row['p']:.4g} {sig}")
    for row in fit_c["coef_table"]:
        if "spec_bin:domain_bin:model_bin" in row["term"]:
            print(f"\n  >>> THREE-WAY INTERACTION: coef={row['coef']:+.4f}, "
                  f"SE={row['se']:.4f}, z={row['z']:+.3f}, p={row['p']:.4g}")
    results["joint_three_way"] = fit_c

    out_path = RESULTS_DIR / "joint_lmm_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
