# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sysconfig

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

from verl.utils.device import get_device_capability

_major, _ = get_device_capability()
# WAR: GB200 nodes without IMEX channel support raise ncclUnhandledCudaError 801 during
# Megatron all_gather (mbridge export_weights) when NCCL tries to use NVLS/MNNVL.
# Disable both on Blackwell (SM 10.x); non-Blackwell GPUs don't have MNNVL.
_gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"} if (_major or 0) >= 10 else {}
if (_major or 0) >= 10:
    # WAR: on RTX PRO 6000 Blackwell Max-Q (PCIe, no NVLink) the P2P and SHM NCCL
    # transports fault with "CUDA error: an illegal memory access" on the first
    # collective. Forcing the socket transport (both P2P and SHM disabled) is the
    # only combination that completes an all_reduce here. Honor explicit overrides
    # so NVLink-equipped Blackwell (e.g. GB200) can keep the fast transports.
    _gb200_nccl_env["NCCL_P2P_DISABLE"] = os.environ.get("NCCL_P2P_DISABLE", "1")
    _gb200_nccl_env["NCCL_SHM_DISABLE"] = os.environ.get("NCCL_SHM_DISABLE", "1")

# Use the site-packages of the interpreter actually running this process, so Ray
# workers inherit the same venv as the driver. Hardcoding a specific venv path
# (e.g. /venv/main) shadows the active env and can inject an incompatible numpy
# ahead of it on PYTHONPATH (breaks numba/scipy in the vLLM workers).
_venv_site_packages = sysconfig.get_paths().get("purelib", "")
_verl_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_extra_pythonpath = f"{_venv_site_packages}:{_verl_repo}"

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        # Degenerate rollouts can emit integers longer than CPython's default 4300-digit
        # int<->str conversion limit, which crashes the math reward parser with a
        # ValueError. Disable the limit for rollout/reward workers (0 = unlimited);
        # the reward path has its own timeout so pathological inputs can't hang it.
        "PYTHONINTMAXSTRDIGITS": os.environ.get("PYTHONINTMAXSTRDIGITS", "0"),
        # ABI shim: torch 2.11 renamed c10::cuda::c10_cuda_check_implementation's
        # line-number param from int -> unsigned int, breaking the prebuilt flash_attn
        # 2.8.3 .so (and vLLM's optional flash_attn rotary import). libabishim.so
        # defines the missing int-signature symbol, forwarding to torch's unsigned one.
        # Preload it into every worker so flash_attn loads on the torch-2.11 stack.
        **({"LD_PRELOAD": os.environ["LD_PRELOAD"]} if os.environ.get("LD_PRELOAD") else {}),
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # vllm 0.10+ defaults to V1 engine; set explicitly to avoid env conflict
        "VLLM_USE_V1": "1",
        # Limit rayon/tokenizers thread pools to avoid hitting container PID limits
        "RAYON_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        # Ensure Ray workers can find venv-installed packages
        "PYTHONPATH": _extra_pythonpath,
        **_gb200_nccl_env,
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    # Ray workers don't inherit the driver's os.environ; pass these explicitly.
    for key in ("WANDB_API_KEY", "HF_TOKEN", "LD_PRELOAD"):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"][key] = os.environ[key]
    return runtime_env
