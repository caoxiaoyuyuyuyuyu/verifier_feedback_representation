"""Joint LMM analyses for camera-ready R1 response.

Uses Phase B NL cells (corrected parser) as the data source, NOT Phase A.
Phase B NL precise/generic = corrected-parser equivalent of Phase A.

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


# Phase B result files contain NL cells with corrected parser
RESULTS_DIR = Path("/root/autodl-tmp/verifier_feedback_representation/results")

# Phase B files: each contains cells like {domain}_{format}_{specificity}
# We extract only NL cells: svg_nl_precise, svg_nl_generic, python_nl_precise, python_nl_generic
PHASE_B_FILES = {
    "qwen_python": RESULTS_DIR / "phaseB_qwen_python_r2.json",
    "qwen_svg": RESULTS_DIR / "phaseB_qwen_svg.json",
    "llama_python": RESULTS_DIR / "phaseB_llama_python_r2.json",
    "llama_svg": RESULTS_DIR / "phaseB_llama_svg_r2.json",
}

# Fallback paths if r2 doesn't exist
PHASE_B_FALLBACKS = {
    "qwen_python": [RESULTS_DIR / "phaseB_qwen_python.json"],
    "qwen_svg": [RESULTS_DIR / "phaseB_qwen_svg.json"],
    "llama_python": [RESULTS_DIR / "phaseB_llama_python.json"],
    "llama_svg": [RESULTS_DIR / "phaseB_llama_svg.json",
                  RESULTS_DIR / "phaseB_llama_svg_r2.json"],
}

# NL cells to extract from each file
NL_CELLS = {
    "nl_precise": "precise",
    "nl_generic": "generic",
}


def find_file(key: str) -> Path:
    """Find the Phase B result file, trying primary then fallbacks."""
    primary = PHASE_B_FILES[key]
    if primary.exists():
        return primary
    for fb in PHASE_B_FALLBACKS.get(key, []):
        if fb.exists():
            return fb
    raise FileNotFoundError(f"No Phase B file found for {key}. Tried: {primary}, {PHASE_B_FALLBACKS.get(key, [])}")


def load_long_df() -> pd.DataFrame:
    """Load Phase B NL cells into a long-format DataFrame.

    Extracts only NL-format cells (nl_precise, nl_generic) from each
    model×domain Phase B file. This gives us corrected-parser data
    equivalent to Phase A but without the old parser bug.
    """
    rows = []
    files_used = {}

    for key, primary_path in PHASE_B_FILES.items():
        model_name, domain_name = key.split("_", 1)
        path = find_file(key)
        files_used[key] = str(path)
        print(f"Loading {key} from {path}")

        with open(path) as f:
            data = json.load(f)

        cells = data["cells"]
        available_cells = list(cells.keys())
        print(f"  Available cells: {available_cells}")

        for cell_name, spec in NL_CELLS.items():
            # Try different naming conventions
            actual_key = None
            candidates = [
                cell_name,                          # nl_precise
                f"{domain_name}_{cell_name}",       # svg_nl_precise / python_nl_precise
                f"{cell_name}_{domain_name}",       # nl_precise_svg
            ]
            for c in candidates:
                if c in cells:
                    actual_key = c
                    break

            if actual_key is None:
                print(f"  WARNING: Could not find NL cell for {cell_name} in {key}. "
                      f"Tried: {candidates}. Available: {available_cells}")
                continue

            cell = cells[actual_key]
            per_sample = cell["per_sample_reduction"]
            print(f"  {actual_key}: {len(per_sample)} samples")

            for idx, val in enumerate(per_sample):
                rows.append({
                    "sample_id": f"{domain_name}_{idx}",
                    "model": model_name,
                    "domain": domain_name,
                    "specificity": spec,
                    "spec_bin": 1 if spec == "precise" else 0,
                    "domain_bin": 1 if domain_name == "python" else 0,
                    "model_bin": 1 if model_name == "llama" else 0,
                    "reduction": float(val),
                })

    df = pd.DataFrame(rows)
    print(f"\nTotal: {len(df)} rows from {len(files_used)} files")
    print(f"Files used: {json.dumps(files_used, indent=2)}")
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

    results = {"data_source": "Phase B NL cells (corrected parser)",
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
