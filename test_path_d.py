"""Path D: transformers backend smoke test for VLLMRunner.generate_chat"""
import sys, time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

sys.path.insert(0, "/root/autodl-tmp/verifier_feedback_representation")
from src.inference.vllm_runner import VLLMRunner, GenerationConfig

MODEL = "/root/autodl-tmp/models/llama-3.1-8b-instruct/"

print("=" * 60)
print("PATH D TEST: transformers backend, generate_chat")
print("=" * 60)

# Force transformers backend
import os
os.environ["VFR_BACKEND"] = "transformers"

runner = VLLMRunner(
    model_name=MODEL,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

conversations = [
    [{"role": "user", "content": "Fix this SVG: <svg><rect width='100' height='100'/></svg>. Add a blue fill color."}],
    [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    [{"role": "user", "content": "Write a haiku about programming."}],
    [{"role": "user", "content": "Explain what SVG viewBox does in one sentence."}],
    [{"role": "user", "content": "Convert this CSS color to hex: rgb(255, 128, 0)"}],
    [{"role": "user", "content": "What are the basic SVG shapes? List them briefly."}],
    [{"role": "user", "content": "Fix this broken SVG path: M 10 10 L 20 Q 30 30"}],
    [{"role": "user", "content": "Write a Python one-liner to reverse a string."}],
]

config = GenerationConfig(max_tokens=512, temperature=0.7, top_p=0.9)

print(f"\nRunning generate_chat with {len(conversations)} conversations...")
print(f"Config: max_tokens={config.max_tokens}, temp={config.temperature}, top_p={config.top_p}")

t0 = time.time()
results = runner.generate_chat(conversations, config)
elapsed = time.time() - t0

total_tokens = sum(r.token_count for r in results)
tps = total_tokens / elapsed if elapsed > 0 else 0

print(f"\n{'=' * 60}")
print(f"RESULTS: {len(results)} outputs in {elapsed:.1f}s ({tps:.1f} tok/s, {total_tokens} total tokens)")
print(f"{'=' * 60}")

for i, r in enumerate(results):
    print(f"\n--- Result {i} ---")
    print(f"Tokens: {r.token_count}, Finish: {r.finish_reason}")
    print(f"Output: {r.output[:300]}")

runner.shutdown()
print("\n" + "=" * 60)
print("PATH_D TEST PASSED")
print("=" * 60)
