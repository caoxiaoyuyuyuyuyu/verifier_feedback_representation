"""vLLM Continuous Batching Inference Runner.

支持参数化 model_name，使用 vLLM 的 offline batched inference API。
设计为实验主循环调用: 传入 prompt 列表，返回 generation 列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        runner = VLLMRunner(model_name="Qwen/Qwen3-Coder-Next-7B")
        results = runner.generate(prompts, config=GenerationConfig())
        runner.shutdown()
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        seed: int = 42,
    ):
        """
        Args:
            model_name: HuggingFace model name or local path。
            tensor_parallel_size: 张量并行 GPU 数。
            gpu_memory_utilization: GPU 显存使用比例。
            seed: 随机种子。
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self._engine = None

    def _init_engine(self) -> None:
        """延迟初始化 vLLM engine（避免 import 时加载模型）。"""
        # TODO: from vllm import LLM, SamplingParams
        # TODO: self._engine = LLM(model=self.model_name, ...)
        raise NotImplementedError

    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        """批量生成。

        使用 vLLM continuous batching，自动处理大批量。

        Args:
            prompts: prompt 字符串列表。
            config: 生成参数，None 使用默认值。

        Returns:
            与 prompts 等长的 GenerationResult 列表。
        """
        # TODO: 延迟初始化 engine
        # TODO: 构建 SamplingParams
        # TODO: engine.generate(prompts, sampling_params)
        # TODO: 包装为 GenerationResult
        raise NotImplementedError

    def shutdown(self) -> None:
        """释放 GPU 资源。"""
        # TODO: 清理 vLLM engine
        self._engine = None
