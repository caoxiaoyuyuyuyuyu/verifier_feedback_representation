"""Phase A LMM corrected: per-domain fit of reduction ~ specificity * model + (1|sample_id).

SVG domain and Python domain are fitted separately. Both models (Qwen, Llama)
are pooled within each domain. sample_id is a random intercept capturing
per-sample difficulty.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except ImportError:
    print("ERROR: statsmodels not installed", file=sys.stderr)
    sys.exit(1)


RESULTS_DIR = Path("/root/autodl-tmp/verifier_feedback_representation/results")
FILES = {
    "qwen": RESULTS_DIR / "phaseA_v3_qwen.json",
    "llama": RESULTS_DIR / "phaseA_v3_llama.json",
}
# Cell name -> (domain, specificity)
CELL_MAP = {
    "svg_precise": ("svg", "precise"),
    "svg_generic": ("svg", "generic"),
    "python_precise": ("python", "precise"),
    "python_generic": ("python", "generic"),
}


def load_long_df() -> pd.DataFrame:
    rows = []
    for model_name, path in FILES.items():
        with open(path) as f:
            data = json.load(f)
        cells = data["cells"]
        for cell_name, (domain, spec) in CELL_MAP.items():
            cell = cells[cell_name]
            per_sample = cell["per_sample_reduction"]
            for idx, val in enumerate(per_sample):
                rows.append({
                    "sample_id": f"{domain}_{idx}",
                    "model": model_name,
                    "domain": domain,
                    "specificity": spec,
                    "spec_bin": 1 if spec == "precise" else 0,
                    "reduction": float(val),
                })
    return pd.DataFrame(rows)


def fit_domain(df_domain: pd.DataFrame, domain: str) -> dict:
    out = {"domain": domain, "n_obs": len(df_domain),
           "n_samples": df_domain["sample_id"].nunique()}

    # Drop non-finite (e.g. -10.0 outlier? keep them — they are real data)
    df_domain = df_domain.dropna(subset=["reduction"]).copy()

    # Encode: specificity (precise=1, generic=0), model (Qwen=0, Llama=1 via C())
    formula = "reduction ~ spec_bin * C(model, Treatment('qwen'))"

    fit_method = None
    summary_text = None
    params = None
    try:
        md = smf.mixedlm(formula, data=df_domain, groups=df_domain["sample_id"])
        result = md.fit(method="lbfgs", reml=True)
        fit_method = "mixedlm_reml"
        summary_text = str(result.summary())
        params = result.params.to_dict()
        bse = result.bse.to_dict()
        tvals = result.tvalues.to_dict()
        pvals = result.pvalues.to_dict()
        converged = bool(result.converged) if hasattr(result, "converged") else True
        out["converged"] = converged
    except Exception as e:
        out["mixedlm_error"] = f"{type(e).__name__}: {e}"
        # Fallback OLS
        ols = smf.ols(formula, data=df_domain).fit()
        fit_method = "ols_fallback"
        summary_text = str(ols.summary())
        params = ols.params.to_dict()
        bse = ols.bse.to_dict()
        tvals = ols.tvalues.to_dict()
        pvals = ols.pvalues.to_dict()
        out["converged"] = True

    out["fit_method"] = fit_method
    out["summary"] = summary_text

    # Build coefficient table
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

    # Cell-level means for reference
    cell_means = (df_domain.groupby(["model", "specificity"])["reduction"]
                  .agg(["mean", "std", "count"]).reset_index())
    out["cell_means"] = cell_means.to_dict(orient="records")
    return out


def main():
    df = load_long_df()
    print(f"Loaded {len(df)} rows")
    print(df.groupby(["domain", "model", "specificity"]).size())

    results = {"data_summary": {
        "total_rows": int(len(df)),
        "by_cell": df.groupby(["domain", "model", "specificity"])
                      .size().reset_index(name="n").to_dict(orient="records"),
    }}

    for domain in ["svg", "python"]:
        sub = df[df["domain"] == domain].copy()
        print(f"\n===== Domain: {domain}  (n={len(sub)}) =====")
        fit = fit_domain(sub, domain)
        print(fit["summary"])
        results[domain] = fit

    out_path = RESULTS_DIR / "phaseA_lmm_corrected.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
