"""Quick vLLM smoke test — 验证模型加载和推理正常。

Usage:
    python -m src.test_vllm --model-path /root/autodl-tmp/models/qwen3.5-9b
"""

import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    from .inference.vllm_runner import VLLMRunner, GenerationConfig

    print(f"Loading model from {args.model_path}...")
    runner = VLLMRunner(
        model_name=args.model_path,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
    )

    test_prompts = [
        "Fix this SVG to make the circle centered:\n```svg\n<svg viewBox='0 0 100 100'><circle cx='10' cy='10' r='50'/></svg>\n```\nFixed SVG:",
        "Fix this Python code to avoid command injection:\n```python\nimport os\ndef run(cmd):\n    os.system(cmd)\n```\nFixed code:",
        "What is 2+2? Answer with just the number:",
    ]

    config = GenerationConfig(max_tokens=512, temperature=0.0)

    print(f"\nRunning {len(test_prompts)} test prompts...")
    t0 = time.time()
    results = runner.generate(test_prompts, config)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Tokens: {r.token_count}, Finish: {r.finish_reason}")
        print(f"Output: {r.output[:300]}{'...' if len(r.output) > 300 else ''}")
    print("=" * 60)

    total_tokens = sum(r.token_count for r in results)
    print(f"\nTotal: {total_tokens} tokens, {total_tokens/elapsed:.0f} tok/s")
    print("vLLM smoke test PASSED")

    runner.shutdown()


if __name__ == "__main__":
    main()
