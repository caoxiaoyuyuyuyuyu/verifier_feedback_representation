"""Joint LMM analyses for camera-ready R1 response.

Uses Phase B NL cells (corrected parser) as the data source:
- Python NL: from multiseed_phaseB/ (corrected parser, cells: python_nl_precise/generic)
- SVG: from multiseed/ (SVG only has NL format, no parser bug, cells: svg_precise/generic)

Multi-seed: averages per_sample_reduction across 3 seeds before fitting.

(a) Llama-only: DRR ~ specificity * domain + (1|sample_id)
(b) Qwen-only: DRR ~ specificity * domain + (1|sample_id)  [verification]
(c) Joint three-way: DRR ~ specificity * domain * model + (1|sample_id)

Outputs JSON with full model summaries for appendix inclusion.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("ERROR: statsmodels not installed", file=sys.stderr)
    sys.exit(1)


RESULTS_DIR = Path("/home/caoxiaoyu/verifier_feedback_representation/results")
SEEDS = [42, 123, 456]

# Data sources:
# Python NL → multiseed_phaseB/{model}_python_seed{s}.json  cells: python_nl_precise, python_nl_generic
# SVG NL    → multiseed/{model}_svg_seed{s}.json            cells: svg_precise, svg_generic
DATA_SPEC = {
    # (model, domain): (subdir, cell_precise, cell_generic)
    ("qwen", "python"):  ("multiseed_phaseB", "python_nl_precise", "python_nl_generic"),
    ("qwen", "svg"):     ("multiseed",        "svg_precise",       "svg_generic"),
    ("llama", "python"): ("multiseed_phaseB", "python_nl_precise", "python_nl_generic"),
    ("llama", "svg"):    ("multiseed",        "svg_precise",       "svg_generic"),
}


def load_long_df():
    """Load NL cells from multiseed files, average across seeds."""
    rows = []
    files_used = {}

    for (model, domain), (subdir, cell_precise, cell_generic) in DATA_SPEC.items():
        key = f"{model}_{domain}"
        seed_files = []
        for seed in SEEDS:
            path = RESULTS_DIR / subdir / f"{model}_{domain}_seed{seed}.json"
            if not path.exists():
                print(f"WARNING: {path} not found, skipping seed {seed} for {key}")
                continue
            seed_files.append(path)

        if not seed_files:
            print(f"ERROR: No seed files found for {key}")
            continue

        files_used[key] = [str(p) for p in seed_files]
        print(f"Loading {key} from {len(seed_files)} seed files ({subdir}/)")

        for spec_label, cell_name in [("precise", cell_precise), ("generic", cell_generic)]:
            seed_data = []
            for path in seed_files:
                with open(path) as f:
                    data = json.load(f)
                cells = data["cells"]
                if cell_name not in cells:
                    print(f"  WARNING: cell '{cell_name}' not in {path.name}. Available: {list(cells.keys())}")
                    continue
                seed_data.append(cells[cell_name]["per_sample_reduction"])

            if not seed_data:
                print(f"  WARNING: No data for {key} {spec_label}")
                continue

            # Average across seeds
            n_samples = min(len(s) for s in seed_data)
            avg = np.mean([s[:n_samples] for s in seed_data], axis=0)
            print(f"  {cell_name}: {n_samples} samples x {len(seed_data)} seeds -> averaged")

            for idx, val in enumerate(avg):
                rows.append({
                    "sample_id": f"{domain}_{idx}",
                    "model": model,
                    "domain": domain,
                    "specificity": spec_label,
                    "spec_bin": 1 if spec_label == "precise" else 0,
                    "domain_bin": 1 if domain == "python" else 0,
                    "model_bin": 1 if model == "llama" else 0,
                    "reduction": float(val),
                })

    df = pd.DataFrame(rows)
    print(f"\nTotal: {len(df)} rows from {len(files_used)} file groups")
    return df, files_used


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

    results = {"data_source": "Phase B NL cells (corrected parser) + multiseed SVG",
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

    out_path = RESULTS_DIR / "joint_lmm_phaseB_NL.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
