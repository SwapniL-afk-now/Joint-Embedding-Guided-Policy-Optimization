from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SDCVLLMAdapterSpec:
    name: str
    int_id: int
    path: str


SDC_VLLM_ADAPTERS = {
    "success": SDCVLLMAdapterSpec("sdc_success", 124, "sdc_success_lora_path"),
    "failure": SDCVLLMAdapterSpec("sdc_failure", 125, "sdc_failure_lora_path"),
}


def sdc_vllm_adapter_spec(adapter: str) -> SDCVLLMAdapterSpec:
    try:
        return SDC_VLLM_ADAPTERS[adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown SDC vLLM adapter: {adapter}") from exc


def extract_response_logprobs_from_prompt_logprobs(*, prompt_logprobs: Sequence[Sequence[float]], prompt_len: int, response_len: int) -> list[float]:
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    if response_len < 0:
        raise ValueError("response_len must be non-negative")
    start, end = prompt_len - 1, prompt_len - 1 + response_len
    if end > len(prompt_logprobs):
        raise ValueError("prompt_logprobs does not contain the requested response span")
    return [float(row[0]) for row in prompt_logprobs[start:end]]
