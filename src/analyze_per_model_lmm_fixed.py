"""Per-model LMM with domain-prefixed sample_id to avoid cross-domain collisions.

Model: reduction ~ specificity * domain + (1|sample_id), fit separately for
Qwen and Llama. The prior per-model script had a bug where SVG idx 0-99 and
Python idx 0-48 shared random effects; we fix it by prefixing sample_id with
the domain name.
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


RESULTS_DIR = Path("/root/autodl-tmp/verifier_feedback_representation/results")
FILES = {
    "qwen": RESULTS_DIR / "phaseA_v3_qwen.json",
    "llama": RESULTS_DIR / "phaseA_v3_llama.json",
}
CELL_MAP = {
    "svg_precise": ("svg", "precise"),
    "svg_generic": ("svg", "generic"),
    "python_precise": ("python", "precise"),
    "python_generic": ("python", "generic"),
}


def load_model_df(path: Path) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    cells = data["cells"]
    rows = []
    for cell_name, (domain, spec) in CELL_MAP.items():
        cell = cells[cell_name]
        per_sample = cell["per_sample_reduction"]
        for idx, val in enumerate(per_sample):
            rows.append({
                "sample_id": idx,
                "domain": domain,
                "specificity": spec,
                "spec_bin": 1 if spec == "precise" else 0,
                "reduction": float(val),
            })
    df = pd.DataFrame(rows)
    # ---- KEY FIX: domain-prefix sample_id to prevent cross-domain collisions.
    df["sample_id"] = df["domain"] + "_" + df["sample_id"].astype(str)
    return df


def fit_model(df: pd.DataFrame, model_name: str) -> dict:
    df = df.dropna(subset=["reduction"]).copy()
    out = {
        "model": model_name,
        "n_obs": int(len(df)),
        "n_samples": int(df["sample_id"].nunique()),
        "by_cell": df.groupby(["domain", "specificity"]).size()
                     .reset_index(name="n").to_dict(orient="records"),
    }

    formula = "reduction ~ spec_bin * C(domain, Treatment('svg'))"

    fit_method = None
    summary_text = None
    try:
        md = smf.mixedlm(formula, data=df, groups=df["sample_id"])
        result = md.fit(method="lbfgs", reml=True)
        fit_method = "mixedlm_reml"
        summary_text = str(result.summary())
        params = result.params.to_dict()
        bse = result.bse.to_dict()
        tvals = result.tvalues.to_dict()
        pvals = result.pvalues.to_dict()
        converged = bool(result.converged) if hasattr(result, "converged") else True
        singular = False
        # Detect near-singular random effect variance
        try:
            re_var = float(result.cov_re.iloc[0, 0])
            out["random_effect_var"] = re_var
            if re_var < 1e-8:
                singular = True
        except Exception:
            pass
        out["converged"] = converged
        out["singular"] = singular
    except Exception as e:
        out["mixedlm_error"] = f"{type(e).__name__}: {e}"
        ols = smf.ols(formula, data=df).fit()
        fit_method = "ols_fallback"
        summary_text = str(ols.summary())
        params = ols.params.to_dict()
        bse = ols.bse.to_dict()
        tvals = ols.tvalues.to_dict()
        pvals = ols.pvalues.to_dict()
        out["converged"] = True
        out["singular"] = True

    out["fit_method"] = fit_method
    out["summary"] = summary_text

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

    cell_means = (df.groupby(["domain", "specificity"])["reduction"]
                    .agg(["mean", "std", "count"]).reset_index())
    out["cell_means"] = cell_means.to_dict(orient="records")
    return out


def print_fit(fit: dict) -> None:
    name = fit["model"]
    print(f"\n=== {name} ===")
    print(f"n_obs={fit['n_obs']}, n_samples={fit['n_samples']}, "
          f"fit_method={fit['fit_method']}, converged={fit.get('converged')}, "
          f"singular={fit.get('singular')}")
    print(fit["summary"])
    print("Coefficient table:")
    for row in fit["coef_table"]:
        print(f"  {row['term']:<45s}  coef={row['coef']:+.4f}  "
              f"SE={row['se']:.4f}  z={row['z']:+.3f}  p={row['p']:.4g}")
    # Highlight interaction
    for row in fit["coef_table"]:
        if "spec_bin:" in row["term"]:
            print(f"\n  >>> specificity x domain interaction: "
                  f"coef={row['coef']:+.4f}, SE={row['se']:.4f}, "
                  f"z={row['z']:+.3f}, p={row['p']:.4g}")


def main():
    results = {}
    for model_name, path in FILES.items():
        df = load_model_df(path)
        print(f"\nLoaded {model_name}: {len(df)} rows, "
              f"{df['sample_id'].nunique()} unique sample_ids")
        print(df.groupby(["domain", "specificity"]).size())
        fit = fit_model(df, model_name)
        print_fit(fit)
        results[model_name] = fit

    out_path = RESULTS_DIR / "per_model_lmm_fixed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
