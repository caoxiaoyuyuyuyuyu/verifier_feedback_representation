"""Length-control regression: does token count confound format effects on DRR?

Fits two mixed-effects models per model (Qwen, Llama):
  M1: reduction ~ C(format, Treatment('nl')) + spec_bin + (1|sample_id)
  M2: reduction ~ C(format, Treatment('nl')) + spec_bin + fb_tokens + (1|sample_id)

If format coefficients stay significant after adding fb_tokens → length is NOT
the confound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from .data.prepare_python import load_python_dataset
from .feedback.templates import FeedbackFormat, FeedbackSpecificity, render_feedback
from .verifiers.python_static import PythonStaticVerifier


PROJ = Path(__file__).resolve().parent.parent
RESULTS = PROJ / "results"

FORMATS = [FeedbackFormat.NL, FeedbackFormat.RAW_JSON, FeedbackFormat.HYBRID]
SPECS = [FeedbackSpecificity.PRECISE, FeedbackSpecificity.GENERIC]

PHASE_B_FILES = {
    "qwen": RESULTS / "phaseB_qwen_python_r2.json",
    "llama": RESULTS / "phaseB_llama_python_r2.json",
}

CELL_KEY_MAP = {
    ("nl", "precise"): "python_nl_precise",
    ("nl", "generic"): "python_nl_generic",
    ("raw_json", "precise"): "python_raw_json_precise",
    ("raw_json", "generic"): "python_raw_json_generic",
    ("hybrid", "precise"): "python_hybrid_precise",
    ("hybrid", "generic"): "python_hybrid_generic",
}


def compute_per_sample_fb_tokens(
    filtered_samples: list,
    tokenizer,
) -> dict[tuple[str, str], list[int]]:
    """Return {(format_val, spec_val): [fb_tokens_per_sample]}."""
    result = {}
    for fmt in FORMATS:
        for spec in SPECS:
            counts = []
            for sample, diags in filtered_samples:
                fb = render_feedback(diags, fmt, spec)
                counts.append(len(tokenizer.encode(fb)))
            result[(fmt.value, spec.value)] = counts
    return result


def build_dataframe(
    model_name: str,
    phaseB: dict,
    fb_tokens_map: dict[tuple[str, str], list[int]],
) -> pd.DataFrame:
    """Build long-form DataFrame: one row per (sample, format, specificity)."""
    rows = []
    for (fmt_val, spec_val), cell_key in CELL_KEY_MAP.items():
        cell = phaseB["cells"][cell_key]
        reductions = cell["per_sample_reduction"]
        fb_toks = fb_tokens_map[(fmt_val, spec_val)]
        n = len(reductions)
        for i in range(n):
            rows.append({
                "model": model_name,
                "sample_id": f"s{i:03d}",
                "format": fmt_val,
                "spec_bin": 1 if spec_val == "precise" else 0,
                "reduction": reductions[i],
                "fb_tokens": fb_toks[i],
            })
    return pd.DataFrame(rows)


def fit_and_report(df: pd.DataFrame, label: str) -> None:
    """Fit M1 and M2, print comparison."""
    # Ensure 'nl' is reference level
    df["format"] = pd.Categorical(df["format"], categories=["nl", "raw_json", "hybrid"])

    formula1 = "reduction ~ C(format, Treatment('nl')) + spec_bin"
    formula2 = "reduction ~ C(format, Treatment('nl')) + spec_bin + fb_tokens"

    # Try mixed-effects; fallback to OLS if singular
    mixed = True
    try:
        md1 = smf.mixedlm(formula1, data=df, groups=df["sample_id"])
        res1 = md1.fit(reml=False)
        md2 = smf.mixedlm(formula2, data=df, groups=df["sample_id"])
        res2 = md2.fit(reml=False)
    except Exception as e:
        print(f"  [MixedLM failed: {e}; falling back to OLS]")
        mixed = False
        res1 = smf.ols(formula1, data=df).fit()
        res2 = smf.ols(formula2, data=df).fit()

    tag = "MixedLM" if mixed else "OLS (fallback)"

    # Extract coefficients
    def _coef_table(res):
        rows = {}
        for name in res.params.index:
            if name in ("Intercept", "Group Var"):
                continue
            rows[name] = (res.params[name], res.pvalues[name])
        return rows

    t1 = _coef_table(res1)
    t2 = _coef_table(res2)

    print(f"\n{'='*60}")
    print(f"=== {label} — {tag} ===")
    print(f"{'='*60}")

    print(f"\n--- Model 1 (no fb_tokens) ---")
    for name, (coef, p) in t1.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {name:<45s}  coef={coef:+.4f}  p={p:.4f}  {sig}")

    print(f"\n--- Model 2 (with fb_tokens) ---")
    for name, (coef, p) in t2.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        # Show change from M1 if applicable
        if name in t1:
            old_coef = t1[name][0]
            delta = coef - old_coef
            pct = (delta / abs(old_coef) * 100) if old_coef != 0 else float("inf")
            print(f"  {name:<45s}  coef={coef:+.4f}  p={p:.4f}  {sig}  (was {old_coef:+.4f}, Δ={delta:+.4f}, {pct:+.1f}%)")
        else:
            print(f"  {name:<45s}  coef={coef:+.4f}  p={p:.4f}  {sig}")

    # Coefficient change summary
    print(f"\n--- Coefficient change summary ---")
    for name in t1:
        c1, p1 = t1[name]
        if name in t2:
            c2, p2 = t2[name]
            delta = c2 - c1
            pct = (delta / abs(c1) * 100) if c1 != 0 else float("inf")
            was_sig = p1 < 0.05
            still_sig = p2 < 0.05
            status = "STILL SIG" if (was_sig and still_sig) else "LOST SIG" if (was_sig and not still_sig) else "GAINED SIG" if (not was_sig and still_sig) else "still n.s."
            print(f"  {name:<45s}  {c1:+.4f} → {c2:+.4f}  (Δ={delta:+.4f}, {pct:+.1f}%)  [{status}]")

    # Log-likelihood comparison if mixed
    if mixed:
        ll1 = res1.llf
        ll2 = res2.llf
        lr = 2 * (ll2 - ll1)
        from scipy.stats import chi2
        p_lr = chi2.sf(lr, df=1)
        print(f"\n  LR test (M2 vs M1): χ²={lr:.2f}, p={p_lr:.4f}")

    print()
    return res1, res2


def main():
    from transformers import AutoTokenizer

    tokenizer_path = "/root/autodl-tmp/models/qwen2.5-7b-instruct"
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load and filter samples (same pipeline as Phase B)
    print("Loading dataset and filtering...")
    samples = load_python_dataset(dataset_name="s2e-lab/SecurityEval", n_samples=9999, seed=42)
    verifier = PythonStaticVerifier()

    filtered = []
    for sample in samples:
        diags = verifier.verify(sample.code)
        if len(diags) > 0:
            filtered.append((sample, diags))
    print(f"Filtered: {len(filtered)} samples with diagnostics")

    # Compute per-sample feedback token counts
    print("Computing per-sample feedback token counts...")
    fb_tokens_map = compute_per_sample_fb_tokens(filtered, tokenizer)

    # Print token count summary
    print("\nFeedback token counts (mean ± std):")
    for (fmt_val, spec_val), counts in sorted(fb_tokens_map.items()):
        import statistics
        m = statistics.mean(counts)
        s = statistics.stdev(counts)
        print(f"  {fmt_val:>10s}_{spec_val:<8s}: {m:6.1f} ± {s:5.1f}")

    # Run analysis for each model
    all_results = {}
    for model_name, phaseB_path in PHASE_B_FILES.items():
        if not phaseB_path.exists():
            print(f"\nSkipping {model_name}: {phaseB_path} not found")
            continue

        phaseB = json.loads(phaseB_path.read_text())
        df = build_dataframe(model_name, phaseB, fb_tokens_map)

        print(f"\nDataFrame for {model_name}: {len(df)} rows, "
              f"formats={df['format'].unique().tolist()}, "
              f"samples={df['sample_id'].nunique()}")

        res1, res2 = fit_and_report(df, label=model_name.upper())
        all_results[model_name] = (res1, res2)

    # Final verdict
    print("=" * 60)
    print("=== CONCLUSION ===")
    print("=" * 60)
    for model_name, (res1, res2) in all_results.items():
        fmt_vars = [k for k in res2.params.index if "format" in k.lower() and "Intercept" not in k]
        any_lost = False
        for v in fmt_vars:
            if v in res1.pvalues and res1.pvalues[v] < 0.05 and res2.pvalues[v] >= 0.05:
                any_lost = True
                break
        if any_lost:
            print(f"  {model_name}: format effect LOST significance after controlling for token count → length IS a potential confound")
        else:
            all_still_sig = all(res2.pvalues.get(v, 1) < 0.05 for v in fmt_vars if res1.pvalues.get(v, 1) < 0.05)
            if all_still_sig:
                print(f"  {model_name}: format coefficients REMAIN significant after controlling for token count → length is NOT a confound")
            else:
                print(f"  {model_name}: mixed results — check individual coefficients above")


if __name__ == "__main__":
    main()
