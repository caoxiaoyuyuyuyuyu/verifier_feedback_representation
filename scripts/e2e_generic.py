"""End-to-end generic inference test — n=10, SVG easy, Qwen."""
from __future__ import annotations
import argparse, json, logging, re, time, sys
from pathlib import Path
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def extract_code_block(text, lang=""):
    pattern = rf"```{lang}\s*\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--svg-split", default="easy")
    parser.add_argument("--specificity", default="generic", choices=["generic", "precise"])
    parser.add_argument("--format", default="nl", choices=["nl", "hybrid", "raw_json"])
    parser.add_argument("--output", default="results/e2e_generic.json")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    from src.data.prepare_svg import load_svg_dataset
    from src.feedback.templates import FeedbackFormat, FeedbackSpecificity, render_feedback
    from src.inference.prompts import build_correction_prompt
    from src.inference.vllm_runner import GenerationConfig, VLLMRunner
    from src.verifiers.svg_geometric import SVGGeometricVerifier
    from src.verifiers.diagnostic import Severity

    specificity = FeedbackSpecificity(args.specificity)
    fmt = FeedbackFormat(args.format)

    logger.info("Loading SVG dataset (n=%d, split=%s)...", args.n_samples, args.svg_split)
    samples = load_svg_dataset("xiaoooobai/SVGenius", n_samples=args.n_samples, seed=42, split=args.svg_split)

    verifier = SVGGeometricVerifier()
    logger.info("Init model: %s on %s", args.model_path, args.device)
    runner = VLLMRunner(model_name=args.model_path, gpu_memory_utilization=0.85, max_model_len=args.max_model_len, device=args.device)
    gen_config = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0)

    codes, feedbacks, all_diags = [], [], []
    for s in samples:
        code = s.svg_string
        codes.append(code)
        diags = verifier.verify(code)
        all_diags.append(diags)
        fb = render_feedback(diags, fmt, specificity)
        feedbacks.append(fb)

    conversations = [build_correction_prompt(c, f, "svg", iteration=0) for c, f in zip(codes, feedbacks)]

    t0 = time.time()
    gen_results = runner.generate_chat(conversations, gen_config)
    gen_time = time.time() - t0

    passed = 0
    sample_outputs = []
    for i, (gr, diags_before) in enumerate(zip(gen_results, all_diags)):
        corrected = extract_code_block(gr.output, "svg")
        diags_after = verifier.verify(corrected)
        errors_after = [d for d in diags_after if d.severity == Severity.ERROR]
        ok = len(errors_after) == 0
        if ok:
            passed += 1
        sample_outputs.append({
            "index": i,
            "diags_before": len(diags_before),
            "diags_after": len(diags_after),
            "errors_after": len(errors_after),
            "passed": ok,
            "feedback_preview": feedbacks[i][:200],
            "output_preview": gr.output[:300],
        })

    runner.shutdown()

    pass_rate = passed / len(samples)
    result = {
        "specificity": args.specificity,
        "format": args.format,
        "n_samples": len(samples),
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "gen_time_s": round(gen_time, 1),
        "samples": sample_outputs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Done: pass_rate=%.1f%% (%d/%d), time=%.1fs", pass_rate*100, passed, len(samples), gen_time)
    print(json.dumps({"pass_rate": pass_rate, "passed": passed, "total": len(samples), "time_s": round(gen_time, 1)}, indent=2))

if __name__ == "__main__":
    main()
