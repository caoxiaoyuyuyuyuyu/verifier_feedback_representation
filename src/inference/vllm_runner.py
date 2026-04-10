"""vLLM Continuous Batching Inference Runner.

支持参数化 model_name，使用 vLLM 的 offline batched inference API。
设计为实验主循环调用: 传入 prompt 列表，返回 generation 列表。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """生成参数配置。"""
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=lambda: ["```\n\n"])


@dataclass
class GenerationResult:
    """单条生成结果。"""
    prompt: str
    output: str
    token_count: int
    finish_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VLLMRunner:
    """vLLM 推理引擎封装。

    Usage:
        runner = VLLMRunner(model_name="Qwen/Qwen3.5-9B")
        results = runner.generate(prompts, config=GenerationConfig())
        runner.shutdown()
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        seed: int = 42,
        max_model_len: int | None = None,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self.max_model_len = max_model_len
        self._engine = None

    def _init_engine(self) -> None:
        """延迟初始化 vLLM engine。"""
        if self._engine is not None:
            return

        from vllm import LLM

        logger.info("Initializing vLLM engine: %s (tp=%d, gpu_mem=%.1f)",
                     self.model_name, self.tensor_parallel_size,
                     self.gpu_memory_utilization)

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "seed": self.seed,
            "trust_remote_code": True,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len

        t0 = time.time()
        self._engine = LLM(**kwargs)
        logger.info("vLLM engine ready in %.1fs", time.time() - t0)

    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        """批量生成。"""
        self._init_engine()
        config = config or GenerationConfig()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop,
        )

        logger.info("Generating %d prompts (max_tokens=%d, temp=%.1f)",
                     len(prompts), config.max_tokens, config.temperature)
        t0 = time.time()
        outputs = self._engine.generate(prompts, sampling_params)
        elapsed = time.time() - t0

        results = []
        total_tokens = 0
        for output in outputs:
            text = output.outputs[0].text
            n_tokens = len(output.outputs[0].token_ids)
            total_tokens += n_tokens
            results.append(GenerationResult(
                prompt=output.prompt,
                output=text,
                token_count=n_tokens,
                finish_reason=output.outputs[0].finish_reason or "unknown",
            ))

        tps = total_tokens / elapsed if elapsed > 0 else 0
        logger.info("Generated %d outputs in %.1fs (%.0f tok/s, %d total tokens)",
                     len(results), elapsed, tps, total_tokens)
        return results

    def generate_chat(
        self,
        conversations: list[list[dict[str, str]]],
        config: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        """Chat format 批量生成。

        Args:
            conversations: 每个元素是 [{"role": ..., "content": ...}, ...] 消息列表。
            config: 生成参数。
        """
        self._init_engine()

        tokenizer = self._engine.get_tokenizer()
        prompts = []
        for conv in conversations:
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(prompt)

        return self.generate(prompts, config)

    def shutdown(self) -> None:
        """释放 GPU 资源。"""
        if self._engine is not None:
            del self._engine
            self._engine = None
            logger.info("vLLM engine shut down")
