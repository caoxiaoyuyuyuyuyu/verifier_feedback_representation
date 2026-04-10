"""vLLM / Transformers Inference Runner.

支持参数化 model_name，使用 vLLM 或 transformers 的 offline batched inference API。
设计为实验主循环调用: 传入 prompt 列表，返回 generation 列表。

Backend 选择:
  - VFR_BACKEND=transformers (默认): 用 HuggingFace transformers，无需 vllm
  - VFR_BACKEND=vllm: 用 vLLM continuous batching
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

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
    """推理引擎封装，支持 transformers 和 vllm 两种 backend。

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
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self.max_model_len = max_model_len
        self.device = device

        # Backend selection: env var, default to transformers
        self.backend = os.environ.get("VFR_BACKEND", "transformers").lower()
        if self.backend not in ("transformers", "vllm"):
            raise ValueError(f"Unknown VFR_BACKEND={self.backend!r}, expected 'transformers' or 'vllm'")

        # Lazy-init state
        self._engine = None        # vllm LLM instance
        self._model = None         # transformers model
        self._tokenizer = None     # transformers tokenizer
        self._model_loaded = False

    def _init_engine(self) -> None:
        """延迟初始化推理引擎。"""
        if self._model_loaded:
            return

        t0 = time.time()

        if self.backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Initializing transformers backend: %s", self.model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=False,
            )
            self._model.eval()
            # D015 Step 6: clear generation_config sampling params to suppress
            # "temperature/top_p/top_k will be ignored" warnings when do_sample=False.
            gc = getattr(self._model, "generation_config", None)
            if gc is not None:
                gc.temperature = None
                gc.top_p = None
                gc.top_k = None
            logger.info("transformers engine ready in %.1fs", time.time() - t0)

        elif self.backend == "vllm":
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

            self._engine = LLM(**kwargs)
            logger.info("vLLM engine ready in %.1fs", time.time() - t0)

        self._model_loaded = True

    def _generate_transformers(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        """transformers backend 的批量生成。"""
        batch_size = 8
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_model_len or 8192,
            ).to(self.device)

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": config.max_tokens,
                "do_sample": config.temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if config.temperature > 0:
                gen_kwargs["temperature"] = config.temperature
                gen_kwargs["top_p"] = config.top_p

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            # Slice off prompt tokens
            prompt_len = inputs.input_ids.shape[1]
            generated = outputs[:, prompt_len:]
            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)

            for j, text in enumerate(decoded):
                n_tokens = (generated[j] != self._tokenizer.pad_token_id).sum().item()
                results.append(GenerationResult(
                    prompt=batch[j],
                    output=text,
                    token_count=n_tokens,
                    finish_reason="stop",
                ))

        return results

    def _generate_vllm(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        """vllm backend 的批量生成。"""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop,
        )

        outputs = self._engine.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            text = output.outputs[0].text
            n_tokens = len(output.outputs[0].token_ids)
            results.append(GenerationResult(
                prompt=output.prompt,
                output=text,
                token_count=n_tokens,
                finish_reason=output.outputs[0].finish_reason or "unknown",
            ))
        return results

    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GenerationResult]:
        """批量生成。"""
        self._init_engine()
        config = config or GenerationConfig()

        logger.info("Generating %d prompts (max_tokens=%d, temp=%.1f, backend=%s)",
                     len(prompts), config.max_tokens, config.temperature, self.backend)
        t0 = time.time()

        if self.backend == "transformers":
            results = self._generate_transformers(prompts, config)
        else:
            results = self._generate_vllm(prompts, config)

        elapsed = time.time() - t0
        total_tokens = sum(r.token_count for r in results)
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

        if self.backend == "transformers":
            tokenizer = self._tokenizer
        else:
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
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._model_loaded = False
        logger.info("Engine shut down (backend=%s)", self.backend)
