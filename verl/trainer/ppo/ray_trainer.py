# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from collections import Counter, defaultdict
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.abstract_rl import trainer_hooks as abstract_hooks
from verl.experimental.abstract_rl.bootstrap import BootstrapState as AbstractRLBootstrapState
from verl.experimental.abstract_rl.config import validate_abstract_rl_config
from verl.experimental.gpi_ce import trainer_hooks as gpi_hooks
from verl.experimental.gpi_ce import validate_gpi_ce_config
from verl.experimental.sharpening_grpo.config import validate_sharpening_config
from verl.experimental.tafr_grpo.config import (
    should_checkpoint_and_refresh,
    should_run_failure_sft,
    validate_tafr_config,
)
from verl.experimental.tafr_grpo.dense_token_credit import DenseTokenCredit
from verl.experimental.tafr_grpo.failure_data_collector import FailureDataCollector
from verl.experimental.tafr_grpo.tafr_loss import compute_regularization_token_advantage
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__name__)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.gpi_ce_config = validate_gpi_ce_config(self.config)
        self.gpi_ce_enabled = bool(self.gpi_ce_config.enable)
        # The GPI-CE group softmax is only defined over a complete candidate group,
        # so groups must stay whole on one rank exactly as for prefix grouping.
        self.preserve_rollout_groups = bool(self.use_prefix_grouper or self.gpi_ce_enabled)
        self.gpi_ce_builder = None  # built lazily once the tokenizer is available
        self.sharpening_config = validate_sharpening_config(self.config)
        self.sharpening_enabled = bool(self.sharpening_config.enable)
        self.abstract_rl_config = validate_abstract_rl_config(self.config)
        self.abstract_rl_enabled = bool(self.abstract_rl_config.enable)
        self.abstract_rl_bootstrap = AbstractRLBootstrapState(self.abstract_rl_config)
        self.tafr_config = validate_tafr_config(self.config)
        self.tafr_enabled = bool(self.tafr_config.enable)
        self.tafr_failure_collector = (
            FailureDataCollector(
                max_size=self.tafr_config.failure_data_max_size,
                sampling=self.tafr_config.failure_data_sampling,
                seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            )
            if self.tafr_enabled
            else None
        )
        self.tafr_failure_model_updated_since_save = False
        self.tafr_vllm_adapters_ready = False
        self.tafr_llm_client = None
        self.tafr_last_logprob_metrics: dict[str, float] = {}
        # Persistent warm-start state: the failure advantage is forced to zero
        # until the failure model has been updated at least
        # `failure_min_updates_before_use` times (and, for the vLLM backend, the
        # scorer adapter has been synchronized). Persisted in tafr_state.pt.
        self.tafr_failure_model_update_count = 0
        self.tafr_failure_model_ready = False
        # Background thread that issues the fused anchor+replay log-prob
        # computation in parallel with reward scoring, old_log_prob,
        # ref_log_prob, and advantage computation. See
        # `_tafr_issue_anchor_replay_async` and `_tafr_join_anchor_replay`.
        self._tafr_logprob_thread: Optional[threading.Thread] = None
        self._tafr_logprob_error: Optional[BaseException] = None

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=str))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _print_generation_samples(self, inputs, outputs, gts, scores, *, label: str, sample_count: int = 3):
        """Print a compact rollout preview for console-only monitoring."""
        n = min(sample_count, len(outputs))
        if n <= 0:
            return

        correct = sum(float(score > 0) for score in scores)
        batch_acc = correct / len(scores) if scores else 0.0
        for i in range(n):
            response_preview = " ".join(str(outputs[i]).split())[:360]
            prompt_preview = " ".join(str(inputs[i]).split())[:180]
            print(
                f"{label} sample step={self.global_steps} sample_index={i} "
                f"batch_acc={batch_acc:.3f} score={float(scores[i]):.3f} "
                f"gt={gts[i]} prompt={prompt_preview} response={response_preview}",
                flush=True,
            )

    def _print_metric_summary(self, label: str, metrics: dict, step: int) -> None:
        """Print a compact progress line with the metrics humans watch live."""
        if not metrics:
            return
        preferred = [
            "actor/loss",
            "actor/pg_loss",
            "actor/kl_loss",
            "actor/grad_norm",
            "actor/entropy",
            "train/accuracy",
            "train/failure_rate",
            "train/pass_at_1",
            "train/pass_at_8",
            "train/unique_answer_ratio_at_k",
            "train/exploration_collapse_rate",
            "val/amc23/pass_at_1",
            "val/amc23/pass_at_8",
            "val/amc23/avg_at_k",
            "val/amc23/avg_at_8",
            "perf/time_per_step",
            "timing_s/step",
        ]
        parts = [f"progress {label} step={step}"]
        seen = set()
        for key in preferred:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float, np.floating, np.integer)):
                    parts.append(f"{key}={float(value):.4g}")
                else:
                    parts.append(f"{key}={value}")
                seen.add(key)

        val_suffixes = {"pass_at_1", "pass_at_8", "avg_at_k", "maj_at_8"}
        for key in sorted(metrics):
            key_parts = key.split("/")
            if len(key_parts) == 3 and key_parts[0] == "val" and key_parts[2] in val_suffixes and key not in seen:
                value = metrics[key]
                if isinstance(value, (int, float, np.floating, np.integer)):
                    parts.append(f"{key}={float(value):.4g}")
                    seen.add(key)
        print(" | ".join(parts), flush=True)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._print_generation_samples(inputs, outputs, sample_gts, scores, label="rollout")

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        if self.abstract_rl_enabled:
            # Serves training and validation alike, so the policy is evaluated on
            # the prompt format it was trained on.
            abstract_hooks.inject_instruction(self.abstract_rl_config, gen_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate_once(self, merged: bool = False, seed: int | None = None):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        # Exact prompt+response rows for optional representation probes.
        sample_rep_token_ids = []
        sample_rep_prompt_ids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            if seed is not None:
                test_gen_batch.meta_info["seed"] = int(seed)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            if "input_ids" in test_batch.batch and "attention_mask" in test_batch.batch:
                prompt_width = test_batch.batch["prompts"].shape[1]
                for ids, mask, prompt in zip(
                    test_batch.batch["input_ids"],
                    test_batch.batch["attention_mask"],
                    test_batch.batch["prompts"],
                    strict=False,
                ):
                    sample_rep_token_ids.append(ids[mask.bool()].detach().cpu())
                    sample_rep_prompt_ids.append(prompt[mask[:prompt_width].bool()].detach().cpu())

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._print_generation_samples(
                sample_inputs,
                sample_outputs,
                sample_gts,
                sample_scores,
                label="validation",
                sample_count=self.config.trainer.log_val_generations or 3,
            )
            # Preserve each validation seed.  Without a seed-specific directory,
            # repeated _validate_once calls overwrite the same ``{step}.jsonl`` and
            # only the last evaluation seed survives on disk.
            dump_dir = val_data_dir if seed is None else os.path.join(val_data_dir, f"seed_{seed}")
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=dump_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # The final validation seed remains available to the t-SNE callback.
        self._last_validation_rep_samples = {
            "token_ids": sample_rep_token_ids,
            "prompt_ids": sample_rep_prompt_ids,
            "data_sources": [str(x) for arr in data_source_lst for x in arr],
            "uids": [str(x) for x in sample_uids],
            "scores": [float(x) for x in sample_scores],
            "ground_truths": [str(x) for x in sample_gts],
        }

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns, sample_outputs
        )

    def _parse_validation_seeds(self) -> list[int]:
        seeds = self.config.trainer.get("validation_seeds", None)
        if not seeds:
            return []
        if isinstance(seeds, str):
            try:
                seeds = json.loads(seeds)
            except json.JSONDecodeError:
                seeds = [item.strip() for item in seeds.split(",") if item.strip()]
        return [int(seed) for seed in seeds]

    def _aggregate_seeded_validation_metrics(self, per_seed_metrics: dict[int, dict]) -> dict:
        metrics: dict[str, Any] = {}
        for seed, seed_metrics in per_seed_metrics.items():
            for key, value in seed_metrics.items():
                metrics[f"{key}/seed_{seed}"] = value
        all_keys = sorted({key for seed_metrics in per_seed_metrics.values() for key in seed_metrics})
        for key in all_keys:
            values = [seed_metrics[key] for seed_metrics in per_seed_metrics.values() if key in seed_metrics]
            if not values or not all(isinstance(v, (int, float, np.floating, np.integer)) for v in values):
                continue
            arr = np.asarray(values, dtype=float)
            metrics[key] = float(arr.mean())
            metrics[f"{key}/mean"] = float(arr.mean())
            metrics[f"{key}/std"] = float(arr.std())
        return metrics

    def _validate(self, merged: bool = False):
        if merged:
            return self._validate_once(merged=True)
        seeds = self._parse_validation_seeds()
        if not seeds:
            return self._validate_once()
        per_seed_metrics = {}
        seed_occurrences: dict[int, int] = {}
        for seed in seeds:
            # A repeated seed (validating the same seed multiple times) would otherwise
            # collide on this dict key; label repeats so each run's metrics survive.
            occurrence = seed_occurrences.get(seed, 0)
            seed_occurrences[seed] = occurrence + 1
            label = seed if occurrence == 0 else f"{seed}_rep{occurrence}"
            print(f"validation seed={seed} start", flush=True)
            per_seed_metrics[label] = self._validate_once(seed=seed)
            print(f"validation seed={seed} end", flush=True)
        metrics = self._aggregate_seeded_validation_metrics(per_seed_metrics)
        metrics["validation/seed_count"] = len(seeds)
        metrics["validation/seeds"] = ",".join(str(seed) for seed in seeds)
        return metrics

    def _text_repetition_rate(self, text: str, ngram: int = 4) -> float:
        tokens = re.findall(r"\S+", str(text).lower())
        if len(tokens) < ngram * 2:
            return 0.0
        ngrams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
        return float(1.0 - (len(set(ngrams)) / len(ngrams))) if ngrams else 0.0

    def _entropy_from_counts(self, counts: dict[str, int]) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count <= 0:
                continue
            p = count / total
            entropy -= p * math.log(p)
        return float(entropy)

    def _compute_grouped_accuracy_metrics(
        self, data_sources, sample_uids, reward_extra_infos_dict, prefix: str, outputs=None
    ):
        """Create FEPO-compatible pass@K/avg@K/maj@K and exploration metrics."""
        acc_values = reward_extra_infos_dict.get("acc") or reward_extra_infos_dict.get("score") or []
        pred_values = reward_extra_infos_dict.get("pred") or []
        outputs = outputs or []
        groups: dict[str, dict[str, list]] = {}
        for idx, uid in enumerate(sample_uids):
            source = str(data_sources[idx]) if idx < len(data_sources) else "unknown"
            key = f"{source}\t{uid}"
            group = groups.setdefault(key, {"source": source, "acc": [], "pred": [], "output": []})
            if idx < len(acc_values):
                acc = acc_values[idx]
                group["acc"].append(
                    float(acc > 0 if isinstance(acc, (int, float, np.floating, np.integer)) else bool(acc))
                )
            if idx < len(pred_values):
                pred = pred_values[idx]
                group["pred"].append(str(pred if pred is not None else "<unparsed>"))
            if idx < len(outputs):
                group["output"].append(str(outputs[idx]))

        by_source: dict[str, list[dict[str, list]]] = {}
        for group in groups.values():
            by_source.setdefault(group["source"], []).append(group)

        metrics = {}
        cutoffs = (1, 4, 8, 16)
        for source, source_groups in by_source.items():
            prompt_total = len(source_groups)
            generation_total = sum(len(group["acc"]) for group in source_groups)
            correct_total = sum(sum(group["acc"]) for group in source_groups)
            unique_response_rates = []
            unique_answer_rates = []
            answer_entropies = []
            majority_shares = []
            collapse_flags = []
            reward_stds = []
            repetition_rates = []
            response_lengths = []
            all_wrong = 0.0
            all_correct = 0.0
            mixed = 0.0
            pass_counts = {cutoff: 0.0 for cutoff in cutoffs}
            avg_counts = {cutoff: 0.0 for cutoff in cutoffs}
            for group in source_groups:
                vals = group["acc"]
                preds = group["pred"]
                outputs_group = group["output"]
                if vals:
                    all_wrong += float(not any(vals))
                    all_correct += float(all(vals))
                    mixed += float(any(vals) and not all(vals))
                    mean_val = sum(vals) / len(vals)
                    reward_stds.append((sum((val - mean_val) ** 2 for val in vals) / len(vals)) ** 0.5)
                if outputs_group:
                    unique_response_rates.append(len(set(outputs_group)) / len(outputs_group))
                    repetition_rates.extend(self._text_repetition_rate(output) for output in outputs_group)
                    response_lengths.extend(float(len(output.split())) for output in outputs_group)
                if preds:
                    counts = dict(Counter(preds))
                    unique_answer_rates.append(len(counts) / len(preds))
                    answer_entropies.append(self._entropy_from_counts(counts))
                    majority_shares.append(max(counts.values()) / len(preds))
                    collapse_flags.append(float(len(counts) <= 1))
                for cutoff in cutoffs:
                    subset_vals = vals[:cutoff]
                    subset_preds = preds[:cutoff]
                    pass_counts[cutoff] += float(any(value > 0.0 for value in subset_vals)) if subset_vals else 0.0
                    if subset_vals:
                        avg_counts[cutoff] += float(sum(1.0 for v in subset_vals if v > 0.0) / len(subset_vals))

            def mean(values):
                return float(sum(values) / len(values)) if values else 0.0

            def std(values):
                if not values:
                    return 0.0
                avg = mean(values)
                return float((sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5)

            base = f"{prefix}/{source}"
            metrics[f"{base}/sample_accuracy"] = float(correct_total / generation_total) if generation_total else 0.0
            metrics[f"{base}/total_prompts"] = int(prompt_total)
            metrics[f"{base}/total_generations"] = int(generation_total)
            for cutoff in cutoffs:
                metrics[f"{base}/pass_at_{cutoff}"] = float(pass_counts[cutoff] / prompt_total) if prompt_total else 0.0
                metrics[f"{base}/avg_at_{cutoff}"] = float(avg_counts[cutoff] / prompt_total) if prompt_total else 0.0
            metrics[f"{base}/pass_at_k"] = metrics[f"{base}/pass_at_{max(cutoffs)}"]
            metrics[f"{base}/avg_at_k"] = metrics[f"{base}/avg_at_{max(cutoffs)}"]
            metrics[f"{base}/unique_response_ratio_at_k"] = mean(unique_response_rates)
            metrics[f"{base}/unique_answer_ratio_at_k"] = mean(unique_answer_rates)
            metrics[f"{base}/answer_entropy_at_k"] = mean(answer_entropies)
            metrics[f"{base}/exploration_unique_prediction_rate"] = mean(unique_answer_rates)
            metrics[f"{base}/exploration_majority_prediction_share"] = mean(majority_shares)
            metrics[f"{base}/exploration_collapse_rate"] = mean(collapse_flags)
            metrics[f"{base}/exploration_reward_std_mean"] = mean(reward_stds)
            metrics[f"{base}/all_wrong_group_rate"] = float(all_wrong / prompt_total) if prompt_total else 0.0
            metrics[f"{base}/all_correct_group_rate"] = float(all_correct / prompt_total) if prompt_total else 0.0
            metrics[f"{base}/mixed_group_rate"] = float(mixed / prompt_total) if prompt_total else 0.0
            metrics[f"{base}/response_length_mean"] = mean(response_lengths)
            metrics[f"{base}/response_length_std"] = std(response_lengths)
            metrics[f"{base}/repetition_rate"] = mean(repetition_rates)
        return metrics

    def _filter_dapo_groups(
        self, batch: DataProto, reward_tensor: torch.Tensor, reward_extra_infos_dict: dict[str, list]
    ):
        filter_cfg = self.config.algorithm.get("filter_groups", None)
        if not filter_cfg or not filter_cfg.get("enable", False):
            return batch, reward_tensor, reward_extra_infos_dict, {}

        metric_name = filter_cfg.get("metric", "acc")
        if metric_name in reward_extra_infos_dict and len(reward_extra_infos_dict[metric_name]) == len(batch):
            metric_values = [float(v) for v in reward_extra_infos_dict[metric_name]]
        else:
            metric_values = reward_tensor.sum(-1).detach().cpu().float().tolist()

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(batch.non_tensor_batch.get("uid", [])):
            uid_to_indices[str(uid)].append(idx)

        selected_group_indices: list[list[int]] = []
        for indices in uid_to_indices.values():
            values = [metric_values[i] for i in indices if i < len(metric_values)]
            if not values:
                continue
            positives = [value > 0.0 for value in values]
            if any(positives) and not all(positives):
                selected_group_indices.append(indices)

        # Keep EVERY qualified group -- do not truncate to train_batch_size here.
        # Truncation is the caller's job, after accumulation across generation
        # batches (see _dapo_accumulate). Truncating here would discard qualified
        # groups that the next short batch needs to reach a full training batch.
        selected_indices = [idx for group in selected_group_indices for idx in group]
        metrics = {
            "dapo_filter/input_groups": float(len(uid_to_indices)),
            "dapo_filter/qualified_groups": float(len(selected_group_indices)),
        }
        if not selected_indices:
            # Zero qualified groups is a normal DAPO outcome, not an error: the caller
            # generates another batch. Returning the unfiltered batch here (the old
            # behaviour) would train on all-correct and all-wrong groups, whose
            # advantages are identically zero under GRPO -- a wasted update.
            metrics["dapo_filter/empty_batch"] = 1.0
            return batch[:0], reward_tensor[:0], {k: [] for k in reward_extra_infos_dict}, metrics

        batch = batch[selected_indices]
        reward_tensor = reward_tensor[selected_indices]
        filtered_extra = {}
        for key, values in reward_extra_infos_dict.items():
            if len(values) == len(metric_values):
                filtered_extra[key] = [values[i] for i in selected_indices]
            else:
                filtered_extra[key] = values
        return batch, reward_tensor, filtered_extra, metrics

    def _dapo_accumulate(self, batch: DataProto, reward_tensor: torch.Tensor, reward_extra_infos_dict: dict):
        """DAPO dynamic sampling: accumulate qualified groups until a full batch exists.

        Returns the assembled (batch, reward_tensor, extra_infos, metrics) once
        `train_batch_size` qualified groups have accumulated, or None to signal
        "generate another batch before updating".

        Why this is needed: filtering to mixed groups typically removes 30-50% of
        prompts (more as training converges and groups go all-correct), so simply
        training on whatever survives means the effective batch size silently drifts
        below the configured one and shrinks over the run. That confounds every
        comparison against a non-DAPO arm, which keeps its full batch throughout.

        No actor update happens between the accumulated generations, so every group
        in the assembled batch is on-policy with respect to the same weights -- this
        is the same guarantee upstream DAPO's inner resample loop provides.
        """
        filter_cfg = self.config.algorithm.get("filter_groups", None)
        if not filter_cfg or not filter_cfg.get("enable", False):
            return batch, reward_tensor, reward_extra_infos_dict, {}

        batch, reward_tensor, reward_extra_infos_dict, metrics = self._filter_dapo_groups(
            batch, reward_tensor, reward_extra_infos_dict
        )

        carry = getattr(self, "_dapo_carry", None)
        if carry is not None and len(batch) > 0:
            prev_batch, prev_reward, prev_extra = carry
            # global_token_num is a per-batch-size list (attention_mask.sum(-1)),
            # already stamped into meta_info by the caller before reward scoring.
            # DataProto.concat asserts non-metric meta_info keys are IDENTICAL across
            # inputs, so two differently-sized carried batches always conflict here.
            # It is stale the moment batches merge anyway -- the call site recomputes
            # it from the assembled batch once accumulation completes -- so drop it
            # from both sides rather than try to reconcile two different lists.
            prev_batch.meta_info.pop("global_token_num", None)
            batch.meta_info.pop("global_token_num", None)
            batch = DataProto.concat([prev_batch, batch])
            reward_tensor = torch.cat([prev_reward, reward_tensor], dim=0)
            reward_extra_infos_dict = {
                k: list(prev_extra.get(k, [])) + list(v) for k, v in reward_extra_infos_dict.items()
            }
        elif carry is not None:
            batch, reward_tensor, reward_extra_infos_dict = carry

        self._dapo_gen_batches = getattr(self, "_dapo_gen_batches", 0) + 1

        # Group boundaries survive the concat because uid is carried in non_tensor_batch.
        uids = [str(u) for u in batch.non_tensor_batch.get("uid", [])]
        group_order: list[str] = []
        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if uid not in uid_to_indices:
                group_order.append(uid)
            uid_to_indices[uid].append(idx)

        target_groups = int(self.config.data.train_batch_size)
        max_gen = int(filter_cfg.get("max_num_gen_batches", 0) or 0)
        exhausted = 0 < max_gen <= self._dapo_gen_batches

        if len(group_order) < target_groups and not exhausted:
            self._dapo_carry = (batch, reward_tensor, reward_extra_infos_dict)
            print(
                f"DAPO dynamic sampling: {len(group_order)}/{target_groups} qualified groups "
                f"after {self._dapo_gen_batches} generation batch(es); generating more.",
                flush=True,
            )
            return None

        keep = [i for uid in group_order[:target_groups] for i in uid_to_indices[uid]]
        metrics.update(
            {
                "dapo_filter/kept_groups": float(min(len(group_order), target_groups)),
                "dapo_filter/num_gen_batches": float(self._dapo_gen_batches),
                "dapo_filter/gen_budget_exhausted": 1.0 if exhausted and len(group_order) < target_groups else 0.0,
            }
        )
        self._dapo_carry = None
        self._dapo_gen_batches = 0

        if not keep:
            # Only reachable when the generation budget ran out with nothing qualified.
            return batch, reward_tensor, reward_extra_infos_dict, metrics

        extra = {
            k: ([v[i] for i in keep] if len(v) == len(uids) else v) for k, v in reward_extra_infos_dict.items()
        }
        return batch[keep], reward_tensor[keep], extra, metrics

    def _compute_train_comparison_metrics(self, batch: DataProto) -> dict[str, float | int]:
        scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().float().tolist()
        uids = [str(uid) for uid in batch.non_tensor_batch.get("uid", [])]
        data_sources = [str(src) for src in batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores))]
        preds = [str(pred if pred is not None else "<unparsed>") for pred in batch.non_tensor_batch.get("pred", [])]
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        reward_extra = {"score": scores, "acc": [float(score > 0.0) for score in scores], "pred": preds}
        grouped = self._compute_grouped_accuracy_metrics(
            data_sources, uids, reward_extra, prefix="train-dataset", outputs=outputs
        )
        correct_count = sum(float(score > 0.0) for score in scores)
        total = len(scores)
        metrics: dict[str, float | int] = {
            "train/accuracy": float(correct_count / total) if total else 0.0,
            "train/failure_rate": float(1.0 - (correct_count / total)) if total else 0.0,
            "train/correct_count": int(correct_count),
            "train/response_count": int(total),
        }
        groups = self._compute_grouped_accuracy_metrics(
            ["all"] * len(uids), uids, reward_extra, prefix="train", outputs=outputs
        )
        for key, value in groups.items():
            if key.startswith("train/all/"):
                metrics[key.replace("train/all/", "train/")] = value
        metrics.update(grouped)
        return metrics

    def _val_metrics_update(
        self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns, sample_outputs=None
    ):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = self._compute_grouped_accuracy_metrics(
            data_sources, sample_uids, reward_extra_infos_dict, prefix="val", outputs=sample_outputs
        )
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()
        if self.tafr_enabled:
            self.actor_rollout_wg.tafr_init(self._tafr_config_dict())

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )
        self.tafr_llm_client = self.llm_server_manager.get_client() if self.tafr_enabled else None

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _score_for_metric(self, metric: str, val_metrics: dict, best_ckpt_sources: list):
        """Compute the scalar selection score for one metric string.

        Returns (score, avg_at_k, pass_at_k) or (None, None, None) if the
        required keys are absent from val_metrics.
        """
        avg_scores = []
        pass_scores = []
        for source in best_ckpt_sources:
            # Exclude the per-seed (`/seed_*`) and dispersion (`/std`) variants: they share the
            # `acc/mean@` prefix, so an unfiltered match can pick a single seed's score.
            avg_matches = [
                v
                for k, v in val_metrics.items()
                if k.startswith(f"val-core/{source}/acc/mean@") and "/seed_" not in k and not k.endswith("/std")
            ]
            pass_matches = [
                v
                for k, v in val_metrics.items()
                if k.startswith(f"val-core/{source}/acc/best@") and k.endswith("/mean")
            ]
            if avg_matches:
                avg_scores.append(avg_matches[0])
            if pass_matches:
                pass_scores.append(pass_matches[0])

        avg_at_k = sum(avg_scores) / len(avg_scores) if avg_scores else None
        pass_at_k = sum(pass_scores) / len(pass_scores) if pass_scores else None

        if "+" in metric:
            # Equal-weight composite of explicit sub-budget metrics, e.g.
            # "pass+pass". Every component must be available.
            component_scores = []
            for component in metric.split("+"):
                kind, sep, budget = component.partition("@")
                if not sep or kind not in ("avg", "pass") or not budget:
                    return None, avg_at_k, pass_at_k
                field = f"{kind}_at_{budget}"
                values = [
                    val_metrics[f"val/{source}/{field}"]
                    for source in best_ckpt_sources
                    if f"val/{source}/{field}" in val_metrics
                ]
                if not values:
                    return None, avg_at_k, pass_at_k
                component_scores.append(sum(values) / len(values))
            return sum(component_scores) / len(component_scores), avg_at_k, pass_at_k
        elif "@" in metric and metric.split("@", 1)[0] in ("avg", "pass"):
            kind, _, budget = metric.partition("@")
            field = f"{kind}_at_{budget}"
            sub_scores = [
                val_metrics[f"val/{source}/{field}"]
                for source in best_ckpt_sources
                if f"val/{source}/{field}" in val_metrics
            ]
            if not sub_scores:
                return None, avg_at_k, pass_at_k
            return sum(sub_scores) / len(sub_scores), avg_at_k, pass_at_k
        elif metric == "avg":
            if avg_at_k is None:
                return None, avg_at_k, pass_at_k
            return avg_at_k, avg_at_k, pass_at_k
        elif metric == "pass":
            if pass_at_k is None:
                return None, avg_at_k, pass_at_k
            return pass_at_k, avg_at_k, pass_at_k
        else:
            # combined: needs both
            if avg_at_k is None or pass_at_k is None:
                return None, avg_at_k, pass_at_k
            return 0.5 * (avg_at_k + pass_at_k), avg_at_k, pass_at_k

    def _maybe_save_best_checkpoint(self, val_metrics: dict):
        """Save best/ checkpoint(s) whenever their selection score improves.

        Supports two config shapes:
          - trainer.best_ckpt_metric  (str, legacy): single metric, saves to best/
          - trainer.best_ckpt_metrics (list[str]):   one dir per metric, saves to
            best_{metric}/ (e.g. best_avg@8/ and best_pass@8/).  When this key is
            present it takes precedence over best_ckpt_metric.

        Selection metric strings:
          "combined"  = 0.5*(avg@k + pass@k)
          "avg"       = mean avg@k across sources
          "pass"      = mean pass@k across sources
          "avg@N"     = mean avg_at_N sub-budget score
          "pass@N"    = mean pass_at_N sub-budget score
        """
        best_ckpt_sources = self.config.trainer.get("best_ckpt_sources", None)
        if not best_ckpt_sources:
            return

        # Build list of (metric_str, dir_name) pairs to evaluate.
        metrics_list = self.config.trainer.get("best_ckpt_metrics", None)
        if metrics_list:
            # Sanitise metric string for use as a directory name (@ -> _at_).
            entries = [
                (str(m).lower(), f"best_{str(m).lower().replace('@', '_at_').replace('+', '_plus_')}")
                for m in metrics_list
            ]
        else:
            metric = str(self.config.trainer.get("best_ckpt_metric", "combined")).lower()
            entries = [(metric, "best")]

        improved = []
        for metric, dir_name in entries:
            combined, avg_at_k, pass_at_k = self._score_for_metric(metric, val_metrics, best_ckpt_sources)
            if combined is None:
                continue

            score_attr = f"_best_val_score_{dir_name}"
            if combined > getattr(self, score_attr, float("-inf")):
                setattr(self, score_attr, combined)
                avg_s = f"{avg_at_k:.4f}" if avg_at_k is not None else "n/a"
                pass_s = f"{pass_at_k:.4f}" if pass_at_k is not None else "n/a"
                print(
                    f"New best checkpoint [{dir_name}] at step {self.global_steps}: "
                    f"metric={metric} score={combined:.4f} (avg@k={avg_s}, pass@k={pass_s}) "
                    f"over {best_ckpt_sources}"
                )
                improved.append((metric, dir_name, combined, avg_at_k, pass_at_k))

        if not improved:
            return

        saved_actor_path = None
        self.checkpoint_manager.sleep_replicas()
        try:
            for idx, (metric, dir_name, _combined, _avg_at_k, _pass_at_k) in enumerate(improved):
                best_local_path = os.path.join(self.config.trainer.default_local_dir, dir_name, "actor")
                if idx == 0 or saved_actor_path is None or not os.path.exists(saved_actor_path):
                    self.actor_rollout_wg.save_checkpoint(
                        best_local_path,
                        None,
                        self.global_steps,
                        max_ckpt_to_keep=1,
                        tag=dir_name,
                        # full-parameter runs have no adapter to save; keep the full weights instead
                        save_contents=["peft_adapter", "extra"] if self.ref_in_actor else ["model", "extra"],
                    )
                    saved_actor_path = best_local_path
                else:
                    if os.path.exists(best_local_path):
                        shutil.rmtree(best_local_path)
                    shutil.copytree(saved_actor_path, best_local_path)
                    print(f"Copied best checkpoint actor payload to [{dir_name}] from [{improved[0][1]}]")
        finally:
            self.checkpoint_manager.update_weights(self.global_steps)

        for metric, dir_name, combined, avg_at_k, pass_at_k in improved:
            avg_w = f"{avg_at_k:.6f}" if avg_at_k is not None else "nan"
            pass_w = f"{pass_at_k:.6f}" if pass_at_k is not None else "nan"
            best_dir = os.path.join(self.config.trainer.default_local_dir, dir_name)
            os.makedirs(best_dir, exist_ok=True)
            with open(os.path.join(best_dir, "best_val_acc.txt"), "w") as f:
                f.write(
                    f"step={self.global_steps} metric={metric} score={combined:.6f} "
                    f"avg_at_k={avg_w} pass_at_k={pass_w} sources={best_ckpt_sources}\n"
                )

            seeds = self._parse_validation_seeds()
            field = None
            if "@" in metric and metric.split("@", 1)[0] in ("avg", "pass"):
                kind, _, budget = metric.partition("@")
                field = f"{kind}_at_{budget}"
            seed_values = {}
            if field and seeds:
                for seed in seeds:
                    values = [
                        val_metrics[f"val/{source}/{field}/seed_{seed}"]
                        for source in best_ckpt_sources
                        if f"val/{source}/{field}/seed_{seed}" in val_metrics
                    ]
                    if values:
                        seed_values[str(seed)] = float(sum(values) / len(values))
            metadata = {
                "selected_metric": metric,
                "global_step": int(self.global_steps),
                "benchmark_sources": list(best_ckpt_sources),
                "seed_list": seeds,
                "per_seed_values": seed_values,
                "mean_score": float(combined),
                "std_score": float(np.asarray(list(seed_values.values()), dtype=float).std()) if seed_values else 0.0,
                "avg_at_k": None if avg_at_k is None else float(avg_at_k),
                "pass_at_k": None if pass_at_k is None else float(pass_at_k),
            }
            with open(os.path.join(best_dir, "best_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        if self.tafr_enabled and should_checkpoint_and_refresh(self.global_steps, self.tafr_config):
            tafr_local_path = os.path.join(local_global_step_folder, "tafr")
            self.actor_rollout_wg.tafr_save_and_refresh(
                tafr_local_path,
                self.global_steps,
                getattr(self, "tafr_failure_model_updated_since_save", False),
                self._tafr_config_dict(),
            )
            self._tafr_sync_vllm_adapters()
            self.tafr_failure_model_updated_since_save = False

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        # Clean up old global_step directories to reclaim disk space.
        # The actor checkpoint manager only tracks actor/ subdirs, but
        # tafr/ and data.pt in parent dirs accumulate. Remove old
        # global_step dirs entirely if max_actor_ckpt_to_keep is set.
        if max_actor_ckpt_to_keep and isinstance(max_actor_ckpt_to_keep, int) and max_actor_ckpt_to_keep >= 1:
            ckpt_base = self.config.trainer.default_local_dir
            if not os.path.isabs(ckpt_base):
                ckpt_base = os.path.join(os.getcwd(), ckpt_base)
            if os.path.isdir(ckpt_base):
                for d in os.listdir(ckpt_base):
                    if not d.startswith("global_step_"):
                        continue
                    step = int(d.split("global_step_")[-1])
                    if step < self.global_steps:
                        old_path = os.path.join(ckpt_base, d)
                        print(f"Cleaning up old checkpoint dir: {old_path}")
                        shutil.rmtree(old_path, ignore_errors=True)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load TAFR state (EMA, failure model, failure SFT optimizer, readiness)
        tafr_local_path = os.path.join(global_step_folder, "tafr")
        if self.tafr_enabled and os.path.isdir(tafr_local_path):
            print(f"Loading TAFR state from {tafr_local_path}")
            tafr_load_output = self._tafr_rank0_result(
                self.actor_rollout_wg.tafr_load(tafr_local_path, self._tafr_config_dict()),
                operation="checkpoint load",
            )
            self.tafr_failure_model_update_count = int(
                tafr_load_output.get("tafr_failure_model_update_count", self.tafr_failure_model_update_count)
            )
            self.tafr_failure_model_ready = bool(
                tafr_load_output.get("tafr_failure_model_ready", self.tafr_failure_model_ready)
            )
            print(
                f"TAFR state loaded: failure_model_update_count={self.tafr_failure_model_update_count} "
                f"failure_model_ready={self.tafr_failure_model_ready}"
            )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper (or the representation loss) is enabled, uses group-level
        balancing to keep samples with the same uid together on the same rank -- for
        prefix sharing in the former case, and because the representation loss is only
        defined over a complete rollout group in the latter.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper / the representation loss to
        # keep same-uid samples together
        if getattr(self, "preserve_rollout_groups", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"Group-preserving batching with balance_batch requires num_uid_groups "
                    f"({num_groups}) % dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions when uid grouping must be maintained
        if not getattr(self, "preserve_rollout_groups", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics").get("mfu", 0.0)
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            or self.config.actor_rollout_ref.actor.get("exploration", {}).get("enabled", False)
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        # ppo_mini_batch_size is configured in PROMPTS; scale it to rows. Under GPI-CE a
        # prompt contributes K = online + offline rows, not rollout.n.
        rows_per_prompt = (
            self.gpi_ce_config.group_size if self.gpi_ce_enabled else self.config.actor_rollout_ref.rollout.n
        )
        ppo_mini_batch_size = ppo_mini_batch_size * rows_per_prompt
        batch_size = batch_td.batch_size[0]
        if batch_size % ppo_mini_batch_size != 0:
            rollout_n = self.config.actor_rollout_ref.rollout.n
            min_mini_batch_size = max(rollout_n, ppo_mini_batch_size // 2)
            adjusted_mini_batch_size = next(
                (
                    size
                    for size in range(ppo_mini_batch_size - rollout_n, min_mini_batch_size - 1, -rollout_n)
                    if batch_size % size == 0
                ),
                None,
            )
            if adjusted_mini_batch_size is None:
                adjusted_mini_batch_size = batch_size
            print(
                "Adjusted actor mini-batch size after filtering: "
                f"{ppo_mini_batch_size} -> {adjusted_mini_batch_size} "
                f"(batch_size={batch_size})",
                flush=True,
            )
            ppo_mini_batch_size = adjusted_mini_batch_size
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        # Opt-in escape hatch: a subclass (e.g. the JEPA trainer) can set this meta_info key to
        # defer the optimizer step here and fuse it with an auxiliary loss's gradient into one
        # combined step later. Absent for every other caller, so apply_step stays True (unchanged
        # behavior) unless explicitly requested.
        apply_step = not batch.meta_info.get("defer_optimizer_step", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            apply_step=apply_step,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
            # Forced on by the advantage-projection objective, which requires exactly
            # one optimizer step per logical batch regardless of the mini-batch split.
            accumulate_minibatch_grads=bool(
                self.config.actor_rollout_ref.actor.get("accumulate_minibatch_grads", False)
            ),
        )
        if self.tafr_enabled:
            tu.assign_non_tensor(batch_td, custom_tafr_grpo=self._tafr_config_dict())
        if self.gpi_ce_enabled:
            gpi_hooks.stamp_actor_batch(self, batch_td)
        if self.sharpening_enabled:
            tu.assign_non_tensor(
                batch_td,
                custom_sharpening_grpo=self._sharpening_config_dict(),
            )
        if self.abstract_rl_enabled:
            tu.assign_non_tensor(
                batch_td,
                custom_abstract_rl=abstract_hooks.config_dict(
                    self.abstract_rl_config, self.global_steps, self.abstract_rl_bootstrap
                ),
            )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        tafr_actor_output = {
            key: val
            for key, val in actor_output.items()
            if key.startswith("tafr_grpo/") or key.startswith("tafr_adv/")
        }
        sharpen_actor_output = {key: val for key, val in actor_output.items() if key.startswith("sharpen/")}
        gpi_actor_output = {key: val for key, val in actor_output.items() if key.startswith("gpi/")}
        abstract_actor_output = {key: val for key, val in actor_output.items() if key.startswith("abstract_rl/")}
        actor_output = rename_dict(
            {
                key: val
                for key, val in actor_output.items()
                if not key.startswith("tafr_grpo/")
                and not key.startswith("tafr_adv/")
                and not key.startswith("sharpen/")
                and not key.startswith("gpi/")
                and not key.startswith("abstract_rl/")
            },
            "actor/",
        )
        actor_output.update(gpi_actor_output)
        actor_output.update(tafr_actor_output)
        actor_output.update(sharpen_actor_output)
        actor_output.update(abstract_actor_output)
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _sharpening_config_dict(self) -> dict[str, Any]:
        return {
            "enable": bool(self.sharpening_config.enable),
            "use_grpo_reward": bool(self.sharpening_config.use_grpo_reward),
            "gamma": float(self.sharpening_config.gamma),
            "alpha": float(self.sharpening_config.alpha),
            "beta": float(self.sharpening_config.beta),
            "group_size": int(self.sharpening_config.group_size),
            "clip_ratio": float(self.sharpening_config.clip_ratio),
        }

    def _tafr_config_dict(self) -> dict[str, Any]:
        return {
            "enable": bool(self.tafr_config.enable),
            "loss_version": str(self.tafr_config.loss_version),
            "beta": float(self.tafr_config.beta),
            "beta_anchor": (
                float(self.tafr_config.beta_anchor) if self.tafr_config.beta_anchor is not None else None
            ),
            "beta_replay": (
                float(self.tafr_config.beta_replay) if self.tafr_config.beta_replay is not None else None
            ),
            "failure_advantage_clip": float(self.tafr_config.failure_advantage_clip),
            "failure_clip_ratio": (
                float(self.tafr_config.failure_clip_ratio) if self.tafr_config.failure_clip_ratio is not None else None
            ),
            "reuse_rollout_log_probs": bool(self.tafr_config.reuse_rollout_log_probs),
            "verify_rollout_log_probs": bool(self.tafr_config.verify_rollout_log_probs),
            "score_only_active_groups": bool(self.tafr_config.score_only_active_groups),
            "failure_min_updates_before_use": int(self.tafr_config.failure_min_updates_before_use),
            "ema_gamma": float(self.tafr_config.ema_gamma),
            "mix_eta": float(self.tafr_config.mix_eta),
            "sft_update_interval_grpo_steps": int(self.tafr_config.sft_update_interval_grpo_steps),
            "checkpoint_interval_grpo_steps": int(self.tafr_config.checkpoint_interval_grpo_steps),
            "failure_sft_lr": float(self.tafr_config.failure_sft_lr),
            "failure_sft_batch_size": int(self.tafr_config.failure_sft_batch_size),
            "failure_sft_max_token_len_per_gpu": int(self.tafr_config.failure_sft_max_token_len_per_gpu),
            "failure_sft_max_updates_per_interval": int(self.tafr_config.failure_sft_max_updates_per_interval),
            "save_to_disk_interval_grpo_steps": int(self.tafr_config.save_to_disk_interval_grpo_steps),
            "save_failure_sft_optimizer": bool(self.tafr_config.save_failure_sft_optimizer),
            "max_response_length": int(self.config.data.get("max_response_length", 2048)),
            "variant": str(self.tafr_config.variant),
            "logprob_backend": str(self.tafr_config.logprob_backend),
            "vllm_score_micro_batch_size": int(self.tafr_config.vllm_score_micro_batch_size),
            "tafr_failure_model_ready": bool(self.tafr_failure_model_ready),
            "tafr_failure_model_update_count": int(self.tafr_failure_model_update_count),
        }

    def _tafr_select_payload(self, output):
        candidates = output if isinstance(output, list) else [output]
        for item in candidates:
            if isinstance(item, dict) and item.get("enabled", False):
                return item
        return None


    @staticmethod
    def _tafr_rank0_result(output, *, operation: str) -> dict:
        """Normalize ONE_TO_ALL RPC output, which is a list even for one worker."""
        if isinstance(output, list):
            output = output[0] if output else None
        if not isinstance(output, dict):
            raise RuntimeError(f"TAFR {operation} returned no rank-0 metrics dictionary: {type(output).__name__}")
        return output

    def _tafr_refresh_ready(self) -> bool:
        count_ok = self.tafr_failure_model_update_count >= int(self.tafr_config.failure_min_updates_before_use)
        adapter_ok = self.tafr_config.logprob_backend != "vllm" or self.tafr_vllm_adapters_ready
        self.tafr_failure_model_ready = bool(count_ok and adapter_ok)
        return self.tafr_failure_model_ready

    def _tafr_sync_vllm_adapters(self) -> bool:
        if not self.tafr_enabled or self.tafr_config.logprob_backend != "vllm":
            logger.warning(
                "TAFR vLLM sync skipped: enabled=%s logprob_backend=%s (not 'vllm')",
                self.tafr_enabled,
                self.tafr_config.logprob_backend,
            )
            return False
        if self.config.actor_rollout_ref.rollout.name != "vllm" or self.llm_server_manager is None:
            logger.warning(
                "TAFR vLLM sync skipped: rollout.name=%s llm_server_manager=%s",
                self.config.actor_rollout_ref.rollout.name,
                self.llm_server_manager is not None,
            )
            return False
        try:
            output = self.actor_rollout_wg.tafr_export_vllm_adapters(self._tafr_config_dict())
            payload = self._tafr_select_payload(output)
            if payload is None:
                logger.warning(
                    "TAFR vLLM adapters NOT ready: export returned no enabled payload "
                    "(output=%s) -> scoring will abort rather than fall back to HF.",
                    output
                    if not isinstance(output, list)
                    else [{k: (v if k == "enabled" else "...") for k, v in (o or {}).items()} for o in output],
                )
                self.tafr_vllm_adapters_ready = False
                return False
            self.tafr_vllm_adapter_payload = payload
            self.tafr_vllm_adapters_ready = True
            # New payload => invalidate the "already loaded into vLLM" token so the next
            # scoring call reloads the refreshed adapters exactly once.
            self._tafr_payload_token = getattr(self, "_tafr_payload_token", 0) + 1
            logger.info(
                "TAFR vLLM adapters ready: %s will score via vLLM prefill.",
                "anchor/failure" if self.tafr_config.loss_version == "failure_token_adv" else "anchor/replay",
            )
            return True
        except Exception:
            logger.exception("TAFR vLLM adapter sync failed; falling back to HF logprob scoring")
            self.tafr_vllm_adapters_ready = False
            return False

    def _tafr_can_score_with_vllm(self) -> bool:
        return bool(
            self.tafr_enabled
            and self.tafr_config.logprob_backend == "vllm"
            and self.tafr_vllm_adapters_ready
            and self.tafr_llm_client is not None
        )

    def _tafr_group_reward_mean(self, batch: DataProto, reward_tensor: torch.Tensor) -> torch.Tensor:
        rewards = reward_tensor.sum(dim=-1).detach().float().clamp(0.0, 1.0)
        uids = [str(uid) for uid in batch.non_tensor_batch.get("uid", [])]
        if len(uids) != rewards.shape[0]:
            return rewards
        group_means = {}
        for uid in dict.fromkeys(uids):
            idx = torch.tensor([i for i, item in enumerate(uids) if item == uid], device=rewards.device)
            group_means[uid] = rewards.index_select(0, idx).mean()
        return torch.stack([group_means[uid] for uid in uids]).to(device=rewards.device)

    def _tafr_annotate_actor_batch(self, batch: DataProto, reward_tensor: torch.Tensor) -> None:
        if not self.tafr_enabled:
            return
        new_mode = self.tafr_config.loss_version == "failure_token_adv"
        self.tafr_last_logprob_metrics = {
            "tafr_grpo/logprob_backend": 1.0 if self._tafr_can_score_with_vllm() else 0.0,
            "tafr_grpo/vllm_score_tokens": 0.0,
            "tafr_grpo/vllm_score_failures": 0.0,
            "tafr_adv/failure_model_ready": 1.0 if self.tafr_failure_model_ready else 0.0,
            "tafr_adv/failure_model_update_count": float(self.tafr_failure_model_update_count),
        }
        if not new_mode:
            self.tafr_last_logprob_metrics["tafr_grpo/vllm_anchor_score_time"] = 0.0
            self.tafr_last_logprob_metrics["tafr_grpo/vllm_replay_score_time"] = 0.0
        group_reward_mean = self._tafr_group_reward_mean(batch, reward_tensor).to(batch.batch["responses"].device)
        batch.batch["tafr_group_reward_mean"] = group_reward_mean

    def _tafr_dump_token_advantage(self, batch: DataProto, advantage: torch.Tensor, stats: dict) -> None:
        """Write decoded tokens alongside their A_reg components, for inspection.

        Set custom_tafr_grpo.token_advantage_dump_dir. One file per dump step, with
        2 wrong + 2 correct rows. This is the only view that shows WHERE the signal
        lands -- aggregate stds cannot distinguish "localizes the error" from
        "varies per token but points nowhere in particular".
        """
        out_dir = self.tafr_config.token_advantage_dump_dir
        if not out_dir:
            return
        every = int(self.tafr_config.token_advantage_dump_every)
        if every <= 0 or self.global_steps % every != 0:
            return
        try:
            n_each = 2
            os.makedirs(out_dir, exist_ok=True)
            responses = batch.batch["responses"]
            mask = batch.batch["response_mask"].bool()
            rmean = batch.batch["tafr_group_reward_mean"].float().reshape(-1)
            stability = stats["stability_signal"]
            failure_c = stats["failure_component"]
            wrong = (rmean < 1.0).nonzero(as_tuple=False).squeeze(-1)[:n_each].tolist()
            right = (rmean >= 1.0).nonzero(as_tuple=False).squeeze(-1)[:n_each].tolist()

            path = os.path.join(out_dir, f"token_adv_step{self.global_steps}.txt")
            with open(path, "w") as fh:
                for row in wrong + right:
                    idx = mask[row].nonzero(as_tuple=False).squeeze(-1)
                    fh.write(f"\n=== row {row} group_reward_mean={rmean[row]:.3f} tokens={idx.numel()} ===\n")
                    fh.write(f"{'tok':>6} {'A_reg':>9} {'stability':>10} {'failure':>9}  text\n")
                    for t in idx.tolist():
                        piece = self.tokenizer.decode([int(responses[row, t])]).replace("\n", "\\n")
                        fh.write(
                            f"{t:>6} {float(advantage[row, t]):>9.4f} {float(stability[row, t]):>10.5f} "
                            f"{float(failure_c[row, t]):>9.5f}  {piece!r}\n"
                        )
            logger.info("TAFR token-advantage dump written: %s", path)
        except Exception:
            # Diagnostics must never take training down.
            logger.exception("TAFR token-advantage dump failed; continuing")

    def _tafr_reg_old_log_prob(self, batch: DataProto) -> torch.Tensor:
        """pi_old for A_reg: the copy scored on the same kernel path as anchor/failure.

        Falls back to the canonical old_log_probs (vLLM path, where anchor/failure
        are already scored by vLLM so all three are consistent).
        """
        return batch.batch.get("tafr_old_log_probs", batch.batch["old_log_probs"])

    def _tafr_build_dense_token_credit(self, batch: DataProto) -> None:
        """Build A_reg with the Dense Token Credit normalization.

        Same signals as the p99 path; the difference is that the residuals are
        zero-mean within each sequence (so the sequence-level GRPO preference is
        untouched) and scaled by each rollout's own |A_GRPO| rather than by one
        batch-wide scalar. See dense_token_credit.DenseTokenCredit.
        """
        cfg = self.tafr_config
        if cfg.adv_norm != "pbrs_pure" and getattr(self, "_tafr_dtc", None) is None:
            # Constructed once so the token-residual RMS EMA persists across steps.
            self._tafr_dtc = DenseTokenCredit(
                lambda_a=float(cfg.anchor_weight),
                lambda_f=float(cfg.failure_weight),
                dose=float(cfg.dtc_dose),
                tau=float(cfg.dtc_tau),
                eps=float(cfg.dtc_eps),
                ema_decay=float(cfg.dtc_ema_decay),
                eps_floor=float(cfg.dtc_eps_floor),
            )

        response_mask = batch.batch["response_mask"]
        mask = response_mask.to(dtype=torch.float32)
        counts = mask.sum(dim=-1).clamp_min(1.0)
        # advantages arrives broadcast to [B, T] but is constant across t; recover [B].
        grpo_advantage = batch.batch["advantages"]
        a_grpo = (grpo_advantage.float() * mask).sum(dim=-1) / counts

        old_log_prob = self._tafr_reg_old_log_prob(batch).float()
        anchor_log_prob = batch.batch["tafr_anchor_log_probs"].float()
        failure_log_prob = batch.batch["tafr_failure_log_prob"].float()
        # Warm start: until the failure model has been trained the repulsion half is
        # meaningless, so gate it to zero exactly as the p99 path does.
        c_x = 1.0 - batch.batch["tafr_group_reward_mean"].float().reshape(-1).clamp(0.0, 1.0)
        if not self.tafr_failure_model_ready:
            c_x = torch.zeros_like(c_x)

        stability_signal = (anchor_log_prob - old_log_prob) * mask
        failure_signal = (old_log_prob - failure_log_prob) * mask
        failure_component = c_x.reshape(-1, 1) * failure_signal

        if cfg.adv_norm == "pbrs_pure":
            # Potential-based reward shaping with coefficient 1, for the state potential
            # Phi(s_t) = log[pi_anchor(y_<t)/pi_old(y_<t)] (and its failure counterpart).
            # The per-token terms telescope to Phi(end) - Phi(start), so this is exactly
            # the theory's own answer: no scale, no normalization, no free parameter.
            # Requires beta == 1.0, since losses.py multiplies by beta on top.
            advantage = (stability_signal + failure_component).to(dtype=grpo_advantage.dtype)
        else:
            self._tafr_dtc.to(device=old_log_prob.device)
            advantage = self._tafr_dtc(
                A_grpo=a_grpo,
                logp_old=old_log_prob,
                logp_anchor=anchor_log_prob,
                logp_failure=failure_log_prob,
                c_x=c_x,
                mask=response_mask.bool(),
            ).to(dtype=grpo_advantage.dtype)
        stats = {"stability_signal": stability_signal, "failure_component": failure_component}

        batch.batch["tafr_regularization_advantage"] = advantage
        self._tafr_dump_token_advantage(batch, advantage, stats)
        batch.batch["tafr_stability_signal"] = stability_signal
        batch.batch["tafr_failure_signal"] = failure_signal
        batch.batch["tafr_failure_component"] = failure_component
        # No separate "raw" stage here: the dense delta IS the scaled quantity, and the
        # per-rollout scale is reported in tafr_regularization_scale below.
        batch.batch["tafr_raw_regularization_advantage"] = advantage
        batch.batch["tafr_scaled_regularization_advantage"] = advantage
        scale = torch.ones_like(a_grpo).reshape(-1, 1) if cfg.adv_norm == "pbrs_pure" \
            else a_grpo.abs().clamp_min(float(cfg.dtc_eps_floor)).reshape(-1, 1)
        batch.batch["tafr_regularization_scale"] = scale.to(dtype=advantage.dtype)

        if cfg.adv_norm == "pbrs_pure":
            # rms(s_a + s_f) -- the one number that pins down the s_a/s_f correlation,
            # and in this mode it is exactly the applied per-token perturbation.
            self.tafr_last_logprob_metrics["tafr_adv/pure_combined_rms"] = float(
                ((advantage.float().square() * mask).sum() / mask.sum().clamp_min(1.0)).sqrt()
            )
            self.tafr_last_logprob_metrics["tafr_adv/pure_influence_vs_abs_a"] = float(
                ((advantage.float().square() * mask).sum() / mask.sum().clamp_min(1.0)).sqrt()
                / a_grpo.abs().mean().clamp_min(1e-6)
            )
        else:
            self.tafr_last_logprob_metrics["tafr_adv/dtc_sigma_a"] = self._tafr_dtc.last_sigma_a
            self.tafr_last_logprob_metrics["tafr_adv/dtc_sigma_f"] = self._tafr_dtc.last_sigma_f
            self.tafr_last_logprob_metrics["tafr_adv/dtc_sigma_c"] = self._tafr_dtc.last_sigma_c
            self.tafr_last_logprob_metrics["tafr_adv/dtc_scale_mean"] = float(scale.mean())
        self.tafr_last_logprob_metrics["tafr_adv/raw_anchor_rms"] = float(
            ((stability_signal.square() * mask).sum() / mask.sum().clamp_min(1.0)).sqrt()
        )
        self.tafr_last_logprob_metrics["tafr_adv/raw_failure_rms"] = float(
            ((failure_component.square() * mask).sum() / mask.sum().clamp_min(1.0)).sqrt()
        )

    def _tafr_build_regularization_advantage(self, batch: DataProto) -> None:
        """Build paper A_reg once on the full logical batch before mini-batching."""
        if not self.tafr_enabled or self.tafr_config.loss_version != "failure_token_adv":
            return
        required = (
            "tafr_anchor_log_probs",
            "old_log_probs",
            "tafr_failure_log_prob",
            "advantages",
            "tafr_group_reward_mean",
            "response_mask",
        )
        missing = [key for key in required if key not in batch.batch]
        if missing:
            raise RuntimeError(f"TAFR regularization advantage is missing fields: {missing}")
        # Anything that is not the legacy p99 normalization is built by the dense builder,
        # which branches internally on adv_norm. Listing modes here instead is how
        # pbrs_pure silently fell through to the p99 path on its first launch.
        if self.tafr_config.adv_norm != "p99":
            self._tafr_build_dense_token_credit(batch)
            return
        advantage, stats = compute_regularization_token_advantage(
            anchor_log_prob=batch.batch["tafr_anchor_log_probs"],
            old_log_prob=self._tafr_reg_old_log_prob(batch),
            failure_log_prob=batch.batch["tafr_failure_log_prob"],
            grpo_advantage=batch.batch["advantages"],
            group_reward_mean=batch.batch["tafr_group_reward_mean"],
            response_mask=batch.batch["response_mask"],
            advantage_clip=float(self.tafr_config.failure_advantage_clip),
            failure_model_ready=bool(self.tafr_failure_model_ready),
            shuffle_tokens=bool(self.tafr_config.shuffle_advantage_tokens),
            anchor_weight=float(self.tafr_config.anchor_weight),
            failure_weight=float(self.tafr_config.failure_weight),
            normalize_terms_separately=bool(self.tafr_config.normalize_terms_separately),
        )
        batch.batch["tafr_regularization_advantage"] = advantage
        self._tafr_dump_token_advantage(batch, advantage, stats)
        batch.batch["tafr_stability_signal"] = stats["stability_signal"]
        batch.batch["tafr_failure_signal"] = stats["failure_signal"]
        batch.batch["tafr_failure_component"] = stats["failure_component"]
        batch.batch["tafr_raw_regularization_advantage"] = stats["raw_advantage"]
        batch.batch["tafr_scaled_regularization_advantage"] = stats["scaled_advantage"]
        self.tafr_last_logprob_metrics["tafr_adv/raw_anchor_rms"] = float(stats["raw_anchor_rms"])
        self.tafr_last_logprob_metrics["tafr_adv/raw_failure_rms"] = float(stats["raw_failure_rms"])
        a_rms = float(stats["anchor_contribution_rms"])
        f_rms = float(stats["failure_contribution_rms"])
        self.tafr_last_logprob_metrics["tafr_adv/anchor_contribution_rms"] = a_rms
        self.tafr_last_logprob_metrics["tafr_adv/failure_contribution_rms"] = f_rms
        self.tafr_last_logprob_metrics["tafr_adv/failure_over_anchor_ratio"] = f_rms / a_rms if a_rms > 1e-8 else 0.0
        scale = stats["normalization_scale"].to(device=advantage.device, dtype=advantage.dtype)
        batch.batch["tafr_regularization_scale"] = scale.expand(advantage.shape[0], 1)

    def _tafr_build_vllm_score_inputs(self, batch: DataProto):
        prompts = batch.batch["prompts"].detach().cpu()
        responses = batch.batch["responses"].detach().cpu()
        attention_mask = batch.batch["attention_mask"].detach().cpu()
        response_mask = batch.batch["response_mask"].detach().cpu()
        prompt_width = prompts.shape[-1]

        sequences: list[list[int]] = []
        prompt_lens: list[int] = []
        response_lens: list[int] = []
        for i in range(prompts.shape[0]):
            prompt_mask = attention_mask[i, :prompt_width].bool()
            prompt_ids = prompts[i][prompt_mask].tolist()
            response_len = int(response_mask[i].sum().item())
            response_ids = responses[i, :response_len].tolist()
            sequences.append([int(x) for x in prompt_ids + response_ids])
            prompt_lens.append(len(prompt_ids))
            response_lens.append(response_len)
        return sequences, prompt_lens, response_lens

    def _tafr_compute_vllm_log_probs(self, batch: DataProto, adapter: str) -> torch.Tensor:
        sequences, prompt_lens, response_lens = self._tafr_build_vllm_score_inputs(batch)
        micro_bsz = int(self.tafr_config.vllm_score_micro_batch_size)
        if not getattr(self, "tafr_vllm_adapter_payload", None):
            raise RuntimeError("TAFR vLLM adapter payload is not ready.")
        self.llm_server_manager.load_tafr_lora_adapters(self.tafr_vllm_adapter_payload, adapters=(adapter,))

        rows = []
        total_tokens = 0
        start_time = time.time()
        failures = 0
        for start in range(0, len(sequences), micro_bsz):
            end = min(start + micro_bsz, len(sequences))
            try:
                chunk_rows = self.tafr_llm_client.score_tafr_logprobs(
                    sequences=sequences[start:end],
                    prompt_lens=prompt_lens[start:end],
                    response_lens=response_lens[start:end],
                    adapter=adapter,
                )
            except Exception:
                failures += end - start
                raise
            rows.extend(chunk_rows)
            total_tokens += sum(response_lens[start:end])

        # Use response_mask as shape template if old_log_probs is not yet computed (early call path).
        shape_template = batch.batch.get("old_log_probs", batch.batch["response_mask"])
        output = torch.zeros_like(shape_template, dtype=torch.float32)
        for i, row in enumerate(rows):
            row_tensor = torch.tensor(row, dtype=torch.float32, device=output.device)
            output[i, : row_tensor.numel()] = row_tensor

        if hasattr(self, "tafr_last_logprob_metrics") and self.tafr_last_logprob_metrics is not None:
            self.tafr_last_logprob_metrics[f"tafr_grpo/vllm_{adapter}_score_time"] = time.time() - start_time
            self.tafr_last_logprob_metrics["tafr_grpo/vllm_score_tokens"] = self.tafr_last_logprob_metrics.get(
                "tafr_grpo/vllm_score_tokens", 0.0
            ) + float(total_tokens)
            self.tafr_last_logprob_metrics["tafr_grpo/vllm_score_failures"] = self.tafr_last_logprob_metrics.get(
                "tafr_grpo/vllm_score_failures", 0.0
            ) + float(failures)
        return output

    def _tafr_rows_to_tensor(self, rows: list[list[float]], shape_template: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(shape_template, dtype=torch.float32)
        for i, row in enumerate(rows):
            row_tensor = torch.tensor(row, dtype=torch.float32, device=output.device)
            output[i, : row_tensor.numel()] = row_tensor
        return output

    def _tafr_compute_vllm_log_probs_multi(
        self, batch: DataProto, adapters: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        """Score the rollout sequences under several TAFR adapters in one batched sweep.

        Loads all requested adapters once (each into its own int_id slot) and submits
        anchor+replay requests in a single interleaved gather, so vLLM's continuous
        batcher overlaps them rather than running one full prefill pass per adapter.
        """
        sequences, prompt_lens, response_lens = self._tafr_build_vllm_score_inputs(batch)
        if not getattr(self, "tafr_vllm_adapter_payload", None):
            raise RuntimeError("TAFR vLLM adapter payload is not ready.")

        # Loading the rank-512 all-linear LoRA adapters into vLLM is expensive (~36s,
        # pickle+RPC of the full tensors) and the adapters only change at the EMA refresh.
        # Skip the reload when the payload token is unchanged AND the adapters are still
        # resident; if vLLM evicted them (e.g. across a sleep), the score call raises and
        # we reload once and retry. Best case: ~0s load on the 4/5 steps between refreshes.
        token = getattr(self, "_tafr_payload_token", 0)

        def _load():
            self.llm_server_manager.load_tafr_lora_adapters(self.tafr_vllm_adapter_payload, adapters=adapters)
            self._tafr_loaded_token = token

        _t_load = time.time()
        if getattr(self, "_tafr_loaded_token", None) != token:
            _load()
        load_s = time.time() - _t_load

        def _score():
            return self.tafr_llm_client.score_tafr_logprobs_multi(
                sequences=sequences,
                prompt_lens=prompt_lens,
                response_lens=response_lens,
                adapters=adapters,
            )

        start_time = time.time()
        try:
            rows_by_adapter = _score()
            reload_s = 0.0
        except Exception:
            # Adapters likely evicted (vLLM sleep); reload once and retry.
            _r = time.time()
            _load()
            reload_s = time.time() - _r
            rows_by_adapter = _score()
        print(
            f"TAFR vLLM score split: adapter_load={load_s:.1f}s reload={reload_s:.1f}s "
            f"score={time.time() - start_time - reload_s:.1f}s (seqs={len(sequences)} adapters={len(adapters)})",
            flush=True,
        )

        shape_template = batch.batch.get("old_log_probs", batch.batch["response_mask"])
        outputs = {adapter: self._tafr_rows_to_tensor(rows_by_adapter[adapter], shape_template) for adapter in adapters}
        if hasattr(self, "tafr_last_logprob_metrics") and self.tafr_last_logprob_metrics is not None:
            self.tafr_last_logprob_metrics["tafr_grpo/vllm_multi_score_time"] = time.time() - start_time
            self.tafr_last_logprob_metrics["tafr_grpo/vllm_score_tokens"] = self.tafr_last_logprob_metrics.get(
                "tafr_grpo/vllm_score_tokens", 0.0
            ) + float(sum(response_lens) * len(adapters))
        return outputs

    def _tafr_compute_anchor_log_probs(self, batch: DataProto) -> None:
        target_device = batch.batch.get("old_log_probs", batch.batch["responses"]).device
        if not self.tafr_enabled or self.tafr_config.variant == "replay_only" or float(self.tafr_config.beta) == 0.0:
            return
        if self._tafr_can_score_with_vllm():
            try:
                batch.batch["tafr_anchor_log_probs"] = self._tafr_compute_vllm_log_probs(batch, "anchor").to(
                    target_device
                )
                return
            except Exception as exc:
                # No HF fallback: old_log_probs are vLLM rollout log-probs here, so an
                # HF anchor score would fabricate a nonzero A_reg out of backend skew.
                raise RuntimeError("TAFR vLLM anchor scoring failed; refusing to mix backends.") from exc
        batch_td = batch.to_tensordict()
        tu.assign_non_tensor(batch_td, custom_tafr_grpo=self._tafr_config_dict())
        output = self.actor_rollout_wg.tafr_compute_anchor_log_prob(batch_td)
        anchor_log_probs = tu.get(output, "tafr_anchor_log_probs")
        if anchor_log_probs is None:
            raise RuntimeError("TAFR anchor worker did not return tafr_anchor_log_probs")
        batch.batch["tafr_anchor_log_probs"] = anchor_log_probs.to(target_device).float()
        # pi_old scored through the anchor/failure kernel path (see the worker).
        # Absent on the vLLM path, where old_log_probs are already vLLM rollout scores.
        tafr_old_log_probs = tu.get(output, "tafr_old_log_probs")
        if tafr_old_log_probs is not None:
            batch.batch["tafr_old_log_probs"] = tafr_old_log_probs.to(target_device).float()

    def _tafr_compute_replay_log_probs(self, batch: DataProto) -> None:
        """Score the pi_old rollout samples under frozen pi_replay.

        No generation — same responses already sampled by pi_old, just evaluated
        under pi_replay so the actor loss can compute the replay KL repulsion term:
            - beta * (1 - r_bar_x) * D_KL(pi_replay || pi_theta)
        """
        if not self.tafr_enabled or self.tafr_config.variant == "anchor_only" or float(self.tafr_config.beta) == 0.0:
            return
        if self._tafr_can_score_with_vllm():
            try:
                batch.batch["tafr_replay_log_probs"] = self._tafr_compute_vllm_log_probs(batch, "replay").to(
                    batch.batch["old_log_probs"].device
                )
                return
            except Exception as exc:
                raise RuntimeError("TAFR vLLM replay scoring failed; refusing to mix backends.") from exc
        batch_td = batch.to_tensordict()
        tu.assign_non_tensor(batch_td, custom_tafr_grpo=self._tafr_config_dict())
        output = self.actor_rollout_wg.tafr_compute_replay_log_prob(batch_td)
        replay_log_probs = tu.get(output, "tafr_replay_log_probs")
        if replay_log_probs is None:
            raise RuntimeError("TAFR replay worker did not return tafr_replay_log_probs")
        batch.batch["tafr_replay_log_probs"] = replay_log_probs.to(batch.batch["old_log_probs"].device).float()

    def _tafr_compute_anchor_and_replay_log_probs(self, batch: DataProto) -> None:
        """Fused computation of both anchor and replay log-probs in a single Ray task."""
        if not self.tafr_enabled or float(self.tafr_config.beta) == 0.0:
            return
        need_anchor = self.tafr_config.variant != "replay_only"
        need_replay = self.tafr_config.variant != "anchor_only"
        if not need_anchor and not need_replay:
            return
        if need_anchor and "tafr_anchor_log_probs" in batch.batch:
            return
        if need_replay and "tafr_replay_log_probs" in batch.batch:
            return
        # Determine target device: use old_log_probs if available, else fall back to responses.
        target_device = batch.batch.get("old_log_probs", batch.batch["responses"]).device
        if self._tafr_can_score_with_vllm():
            try:
                adapters = tuple(a for a, needed in (("anchor", need_anchor), ("replay", need_replay)) if needed)
                # Score all needed adapters in a single interleaved prefill sweep
                # (each request carries its own LoRA int_id) instead of one pass each.
                outputs = self._tafr_compute_vllm_log_probs_multi(batch, adapters)
                if need_anchor:
                    batch.batch["tafr_anchor_log_probs"] = outputs["anchor"].to(target_device)
                if need_replay:
                    batch.batch["tafr_replay_log_probs"] = outputs["replay"].to(target_device)
                return
            except Exception as exc:
                raise RuntimeError("TAFR vLLM anchor/replay scoring failed; refusing to mix backends.") from exc
        batch_td = batch.to_tensordict()
        tu.assign_non_tensor(batch_td, custom_tafr_grpo=self._tafr_config_dict())
        output = self.actor_rollout_wg.tafr_compute_anchor_and_replay_log_probs(batch_td)
        if need_anchor:
            anchor_log_probs = tu.get(output, "tafr_anchor_log_probs")
            if anchor_log_probs is None:
                raise RuntimeError("TAFR anchor worker did not return tafr_anchor_log_probs")
            batch.batch["tafr_anchor_log_probs"] = anchor_log_probs.to(target_device).float()
        if need_replay:
            replay_log_probs = tu.get(output, "tafr_replay_log_probs")
            if replay_log_probs is None:
                raise RuntimeError("TAFR replay worker did not return tafr_replay_log_probs")
            batch.batch["tafr_replay_log_probs"] = replay_log_probs.to(target_device).float()

    def _tafr_active_group_mask(self, batch: DataProto) -> torch.Tensor:
        """Mask of rollout rows whose group still contains failures (difficulty > 0).

        When ``score_only_active_groups`` is disabled every row is active, so the
        failure model scores the whole batch (used for ablation).
        """
        if not self.tafr_config.score_only_active_groups:
            return torch.ones(batch.batch["response_mask"].shape[0], dtype=torch.bool)
        group_reward_mean = batch.batch.get("tafr_group_reward_mean", None)
        if group_reward_mean is None:
            return torch.ones(batch.batch["response_mask"].shape[0], dtype=torch.bool)
        return group_reward_mean < 1.0

    def _tafr_compute_failure_log_probs_vllm(self, batch: DataProto, active_mask: torch.Tensor) -> torch.Tensor:
        """Score the active rollout rows under the failure LoRA via vLLM prefill."""
        if not self._tafr_can_score_with_vllm():
            raise RuntimeError("TAFR vLLM scoring requested but the failure adapter is not ready.")
        if not getattr(self, "tafr_vllm_adapter_payload", None):
            raise RuntimeError("TAFR vLLM adapter payload is not ready.")
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            return torch.zeros(
                (0, batch.batch["response_mask"].shape[-1]),
                dtype=torch.float32,
                device=batch.batch["response_mask"].device,
            )
        active_batch = batch.select_idxs(active_idx)
        active_log_probs = self._tafr_compute_vllm_log_probs(active_batch, "failure")
        return active_log_probs.to(batch.batch["response_mask"].device)

    def _tafr_compute_failure_log_probs_hf(self, batch: DataProto, active_mask: torch.Tensor) -> torch.Tensor:
        """Score the active rollout rows under the cloned failure model on the actor worker."""
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if active_idx.numel() == 0:
            return torch.zeros(
                (0, batch.batch["response_mask"].shape[-1]),
                dtype=torch.float32,
                device=batch.batch["response_mask"].device,
            )
        active_batch = batch.select_idxs(active_idx)
        batch_td = active_batch.to_tensordict()
        tu.assign_non_tensor(batch_td, custom_tafr_grpo=self._tafr_config_dict())
        output = self.actor_rollout_wg.tafr_compute_failure_log_prob(batch_td)
        failure_log_probs = tu.get(output, "tafr_failure_log_probs")
        if failure_log_probs is None:
            raise RuntimeError("TAFR failure worker did not return tafr_failure_log_probs")
        return failure_log_probs.to(batch.batch["response_mask"].device).float()

    def _tafr_compute_failure_log_probs(self, batch: DataProto, use_vllm: bool) -> None:
        """Compute response-token log-probs under the failure model for active groups.

        Inactive rows (all-correct groups, or every row before the failure model
        is ready) get ``tafr_failure_log_prob = old_log_prob`` so their failure
        advantage is zero by construction.
        """
        if not self.tafr_enabled or self.tafr_config.loss_version != "failure_token_adv":
            return
        # The old-policy forward may not have run yet when this is called (vLLM
        # scoring happens in the awake-after-gen window), so fall back to the
        # rollout log-probs, which equal old_log_probs in reuse mode.
        shape_template = batch.batch.get(
            "old_log_probs",
            batch.batch.get("rollout_log_probs", batch.batch["response_mask"]),
        )
        if not self.tafr_failure_model_ready:
            batch.batch["tafr_failure_log_prob"] = shape_template.detach().clone().float()
            if hasattr(self, "tafr_last_logprob_metrics") and self.tafr_last_logprob_metrics is not None:
                self.tafr_last_logprob_metrics["tafr_grpo/vllm_score_tokens"] = 0.0
            return
        active_mask = self._tafr_active_group_mask(batch)
        active_log_probs = (
            self._tafr_compute_failure_log_probs_vllm(batch, active_mask)
            if use_vllm
            else self._tafr_compute_failure_log_probs_hf(batch, active_mask)
        )
        failure_log_prob = shape_template.detach().clone().float()
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        failure_log_prob[active_idx] = active_log_probs.to(failure_log_prob.dtype)
        batch.batch["tafr_failure_log_prob"] = failure_log_prob

    def _tafr_sync_inactive_failure_log_probs(self, batch: DataProto) -> None:
        """Overwrite all-correct group rows with old_log_prob (inactive by construction).

        Before the failure model is ready, every row is overwritten so the
        failure signal is exactly zero everywhere (warm start, plan §10).
        """
        if not self.tafr_enabled or self.tafr_config.loss_version != "failure_token_adv":
            return
        if "tafr_failure_log_prob" not in batch.batch or "old_log_probs" not in batch.batch:
            return
        # Must be the same pi_old A_reg uses, or "inactive" rows carry a kernel-path
        # difference instead of an exact zero.
        reg_old = self._tafr_reg_old_log_prob(batch)
        if not self.tafr_failure_model_ready:
            batch.batch["tafr_failure_log_prob"] = reg_old.to(batch.batch["tafr_failure_log_prob"].dtype)
            return
        group_reward_mean = batch.batch.get("tafr_group_reward_mean", None)
        if group_reward_mean is None:
            return
        inactive = group_reward_mean >= 1.0
        batch.batch["tafr_failure_log_prob"][inactive] = reg_old[inactive].to(
            batch.batch["tafr_failure_log_prob"].dtype
        )

    def _tafr_log_rollout_parity_metrics(self, batch: DataProto) -> dict[str, float]:
        """Log the parity between vLLM rollout log-probs and the HF recomputation.

        Used while ``verify_rollout_log_probs=True`` to validate that the exact
        chosen-token log-probabilities captured during generation match the HF
        old-policy recomputation before the separate forward is removed.
        """
        rollout = batch.batch.get("rollout_log_probs", None)
        recomputed = batch.batch.get("old_log_probs", None)
        response_mask = batch.batch.get("response_mask", None)
        if rollout is None or recomputed is None or response_mask is None:
            return {}
        rollout = rollout.float()
        recomputed = recomputed.float()
        mask = response_mask.bool()
        valid = mask.sum().clamp_min(1.0)
        diff = (rollout - recomputed).abs() * mask
        ratio_rollout = torch.exp(rollout - recomputed)
        ratio_diff = (ratio_rollout - 1.0).abs() * mask
        return {
            "tafr_grpo/rollout_logprobs_mean_abs_diff": float((diff.sum() / valid).detach().cpu()),
            "tafr_grpo/rollout_logprobs_max_abs_diff": float(diff.max().detach().cpu()),
            "tafr_grpo/rollout_logprobs_ratio_mean_abs_diff": float((ratio_diff.sum() / valid).detach().cpu()),
            "tafr_grpo/rollout_logprobs_mask_alignment": float(
                (rollout != 0.0).to(rollout.dtype).mul(mask).sum() / valid
            ),
        }

    def _tafr_issue_anchor_replay_async(self, batch: DataProto) -> None:
        """Spawn a background thread to compute anchor+replay log-probs.

        The thread runs the same code path as the synchronous fused call
        (so vLLM and HF fallbacks both work), but lets the main thread
        continue with reward scoring, old_log_prob, ref_log_prob, and
        advantage computation in parallel. The thread writes its result
        directly into ``batch.batch["tafr_anchor_log_probs"]`` and
        ``batch.batch["tafr_replay_log_probs"]``.

        Must be paired with a call to ``_tafr_join_anchor_replay`` before
        the loss step, or the log-probs will not be available. Exceptions
        raised inside the thread are captured and re-raised on join so
        failures surface the same way the synchronous call would.
        """

        if not self.tafr_enabled or float(self.tafr_config.beta) == 0.0:
            return
        # Defensive: an earlier async call is still in flight. Join it
        # first to avoid leaking threads and to surface any pending error.
        if self._tafr_logprob_thread is not None:
            self._tafr_join_anchor_replay(batch)
        self._tafr_logprob_error = None

        def _worker():
            try:
                self._tafr_compute_anchor_and_replay_log_probs(batch)
            except BaseException as exc:  # noqa: BLE001 — re-raised on join
                self._tafr_logprob_error = exc

        self._tafr_logprob_thread = threading.Thread(target=_worker, name="tafr-anchor-replay", daemon=True)
        self._tafr_logprob_thread.start()

    def _tafr_join_anchor_replay(self, batch: DataProto) -> None:
        """Join the background anchor+replay log-prob thread, if any.

        The thread writes directly to ``batch.batch``, so this method
        only waits for completion and re-raises any exception captured
        inside the thread. Safe to call when no thread is in flight.
        """

        thread = self._tafr_logprob_thread
        if thread is None:
            return
        thread.join()
        self._tafr_logprob_thread = None
        error = self._tafr_logprob_error
        self._tafr_logprob_error = None
        if error is not None:
            raise error

    def _tafr_collect_failures(self, batch: DataProto, reward_tensor: torch.Tensor) -> int:
        if not self.tafr_enabled or self.tafr_failure_collector is None:
            return 0
        rewards = reward_tensor.sum(dim=-1).detach().cpu().float().tolist()
        prompts = batch.batch["prompts"].detach().cpu()
        responses = batch.batch["responses"].detach().cpu()
        response_mask = batch.batch["response_mask"].detach().cpu()
        uids = [str(uid) for uid in batch.non_tensor_batch.get("uid", [""] * len(rewards))]

        collected = 0
        for i, reward in enumerate(rewards):
            if float(reward) > 0.0:
                continue
            response_len = int(response_mask[i].sum().item())
            prompt_text = self.tokenizer.decode(prompts[i], skip_special_tokens=True)
            response_text = self.tokenizer.decode(responses[i, :response_len], skip_special_tokens=True)
            collected += int(
                self.tafr_failure_collector.add(
                    prompt=prompt_text,
                    response=response_text,
                    reward=0,
                    metadata={"uid": uids[i], "global_grpo_step": self.global_steps},
                )
            )
        return collected

    def _tafr_run_failure_sft_if_due(self) -> dict[str, float]:
        if not self.tafr_enabled or self.tafr_failure_collector is None:
            return {}
        if not should_run_failure_sft(self.global_steps, self.tafr_config):
            return {"tafr_grpo/did_sft_update_this_step": 0.0}
        if len(self.tafr_failure_collector) == 0:
            # Scheduled interval, but no failures buffered. Track the skip
            # in metrics so the behaviour is visible — the SFT update itself
            # is a no-op (no forward, no backward, no optimizer step).
            return {
                "tafr_grpo/did_sft_update_this_step": 0.0,
                "tafr_grpo/failure_sft_updates": 0.0,
                "tafr_grpo/failure_sft_skipped_empty": 1.0,
            }
        # Use all buffered failures from this interval. Clear only after the
        # worker confirms a real optimizer update so malformed RPC output cannot
        # silently discard the training examples.
        records = self.tafr_failure_collector.to_records()
        output = self.actor_rollout_wg.tafr_failure_sft_update(records, self._tafr_config_dict())
        metrics = self._tafr_rank0_result(output, operation="failure SFT")
        if "tafr_grpo/failure_sft_updates" not in metrics:
            raise RuntimeError("TAFR failure SFT result is missing 'tafr_grpo/failure_sft_updates'.")
        sft_updates = int(metrics["tafr_grpo/failure_sft_updates"])
        if sft_updates <= 0:
            raise RuntimeError("TAFR failure SFT received nonempty records but performed no optimizer updates.")

        self.tafr_failure_collector.clear()
        self.tafr_failure_model_update_count += sft_updates
        # Decouple scorer freshness from the disk-checkpoint schedule: update the
        # failure EMA and re-export/sync the scorer adapter after every successful
        # failure-model update, not only every checkpoint interval.
        self.actor_rollout_wg.tafr_update_failure_ema(self._tafr_config_dict())
        if self.tafr_config.logprob_backend == "vllm":
            self._tafr_sync_vllm_adapters()
        self._tafr_refresh_ready()
        self.tafr_failure_model_updated_since_save = True
        metrics["tafr_adv/failure_model_ready"] = 1.0 if self.tafr_failure_model_ready else 0.0
        metrics["tafr_adv/failure_model_update_count"] = float(self.tafr_failure_model_update_count)
        return {
            "tafr_grpo/did_sft_update_this_step": float(sft_updates > 0),
            **{k: float(v) for k, v in metrics.items()},
        }

    def _tafr_schedule_metrics(self, wrong_collected: int) -> dict[str, float]:
        if not self.tafr_enabled or self.tafr_failure_collector is None:
            return {}
        did_save = should_checkpoint_and_refresh(self.global_steps, self.tafr_config)
        # Mean age, in GRPO steps, of the failures the buffer currently holds. The
        # paper's diagnostics subsection reports it because a learned failure density
        # is only meaningful if it describes behaviour the actor still exhibits -- a
        # buffer whose mean age keeps climbing is modelling a policy that no longer
        # exists, which would explain a replay term that stops helping.
        replay_age = 0.0
        try:
            steps = [
                float(self.global_steps - r.metadata.get("global_grpo_step", self.global_steps))
                for r in getattr(self.tafr_failure_collector, "_examples", [])
                if getattr(r, "metadata", None)
            ]
            replay_age = sum(steps) / len(steps) if steps else 0.0
        except (AttributeError, TypeError):
            pass
        return {
            "tafr_grpo/failure_buffer_mean_age_steps": replay_age,
            "tafr_grpo/global_grpo_step": float(self.global_steps),
            "tafr_grpo/did_checkpoint_save_this_step": float(did_save),
            "tafr_grpo/failure_dataset_size": float(len(self.tafr_failure_collector)),
            "tafr_grpo/number_wrong_responses_collected_this_step": float(wrong_collected),
            "tafr_grpo/ema_gamma": float(self.tafr_config.ema_gamma),
            "tafr_grpo/mix_eta": float(self.tafr_config.mix_eta),
            "tafr_adv/failure_model_ready": 1.0 if self.tafr_failure_model_ready else 0.0,
            "tafr_adv/failure_model_update_count": float(self.tafr_failure_model_update_count),
            **self.tafr_last_logprob_metrics,
        }

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        self._tafr_sync_vllm_adapters()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            if os.environ.get("VERL_VERBOSE_METRICS", "0") == "1":
                pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            self._maybe_save_best_checkpoint(val_metrics)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        # DAPO dynamic sampling: qualified groups carried across generation batches
        # until a full training batch exists, and how many batches that has taken.
        self._dapo_carry = None
        self._dapo_gen_batches = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                # Set by the DAPO dynamic-sampling accumulator when this generation
                # batch did not yield enough qualified groups to update on.
                needs_more_rollouts = False

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)

                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    del combined_gen_batch, combined_gen_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.gpi_ce_enabled:
                        # Append the offline candidates now: before reward scoring, so the
                        # same verifier judges both sources, and before balancing, so the
                        # completed group travels to one rank as a unit.
                        with marked_timer("gpi_build_mixed", timing_raw, color="purple"):
                            if self.gpi_ce_builder is None:
                                self.gpi_ce_builder = gpi_hooks.make_builder(self)
                            batch = gpi_hooks.build_mixed_batch(self, batch)
                            # The offline rows were cloned from an online row and would
                            # otherwise carry ITS agent-loop reward. Score them properly.
                            metrics.update(gpi_hooks.score_offline_rows(self, batch))
                            batch.meta_info["global_token_num"] = torch.sum(
                                batch.batch["attention_mask"], dim=-1
                            ).tolist()

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                        # Buffer failures and compute the group difficulty gate on the RAW
                        # group, BEFORE DAPO filtering. This ordering is required, not
                        # cosmetic: DAPO discards homogeneous groups, and all-wrong groups
                        # are exactly the failures FEPO exists to learn from. Filtering
                        # first would starve the failure buffer of the very signal the
                        # method is built on, and would compute g on a group whose
                        # composition the filter had already changed.
                        tafr_wrong_collected = self._tafr_collect_failures(batch, reward_tensor)
                        self._tafr_annotate_actor_batch(batch, reward_tensor)

                        # DAPO dynamic sampling. Returns None when too few qualified
                        # groups have accumulated, meaning "generate another batch before
                        # updating" -- the actor is untouched in that case, so the groups
                        # carried forward stay on-policy.
                        dapo_result = self._dapo_accumulate(batch, reward_tensor, reward_extra_infos_dict)
                        if dapo_result is None:
                            needs_more_rollouts = True
                        else:
                            batch, reward_tensor, reward_extra_infos_dict, dapo_filter_metrics = dapo_result
                            metrics.update(dapo_filter_metrics)
                            if dapo_filter_metrics:
                                batch.meta_info["global_token_num"] = torch.sum(
                                    batch.batch["attention_mask"], dim=-1
                                ).tolist()

                    if needs_more_rollouts:
                        # DAPO is still short of a full batch of qualified groups. Draw the
                        # next dataloader batch and generate again WITHOUT updating the
                        # actor, so the groups already carried stay on-policy. This
                        # consumes no training step: global_steps is not incremented and
                        # the progress bar does not advance.
                        with marked_timer("stop_profile", timing_raw):
                            self._stop_profiling(curr_step_profile)
                        continue

                    if self.abstract_rl_enabled:
                        # Format bootstrap: graft abstractions into rollouts that emitted
                        # none. After the reward (computed on Y only, so unaffected) and
                        # before old_log_prob, which must cover the grafted tokens. No-op
                        # unless bootstrap_steps > 0, and self-disabling.
                        metrics.update(abstract_hooks.bootstrap_graft(self, batch, timing_raw))

                    # TAFR log-prob scoring, ordered around the vLLM sleep:
                    #   - failure_token_adv + vLLM backend: score the active rows under
                    #     the failure LoRA NOW, while the engine is awake from gen (a
                    #     single prefill pass). Must run before the sleep below.
                    #   - failure_token_adv + HF backend: sleep first to free the
                    #     KV-cache + weight memory, then score on the FSDP actor.
                    #   - k3_legacy: keep the existing anchor/replay scoring path.
                    # Either way old_log_prob / ref / update run with vLLM asleep, so the
                    # actor forwards get the freed memory (avoids OOM at high util).
                    tafr_active = self.tafr_enabled and float(self.tafr_config.beta) > 0.0
                    tafr_new_mode = tafr_active and self.tafr_config.loss_version == "failure_token_adv"
                    score_with_vllm = tafr_active and self._tafr_can_score_with_vllm()
                    if tafr_active and not score_with_vllm and self.tafr_config.logprob_backend == "vllm":
                        # Adapter sync failed / returned no payload. Scoring on HF while
                        # old_log_probs come from the vLLM rollout would fake A_reg.
                        raise RuntimeError(
                            "TAFR logprob_backend='vllm' but the vLLM scorer is unavailable "
                            "(adapters_ready=%s, client=%s); refusing to score on HF against "
                            "vLLM old_log_probs." % (self.tafr_vllm_adapters_ready, self.tafr_llm_client is not None)
                        )
                    if tafr_new_mode and score_with_vllm:
                        with marked_timer("tafr_score_vllm", timing_raw, color="purple"):
                            self._tafr_compute_failure_log_probs(batch, use_vllm=True)
                            self._tafr_compute_anchor_log_probs(batch)

                    slept_after_gen = True
                    self.checkpoint_manager.sleep_replicas()

                    if tafr_new_mode and not score_with_vllm:
                        with marked_timer("tafr_score_hf", timing_raw, color="purple"):
                            self._tafr_compute_failure_log_probs(batch, use_vllm=False)
                            self._tafr_compute_anchor_log_probs(batch)
                    elif not tafr_new_mode and tafr_active and not score_with_vllm:
                        self._tafr_issue_anchor_replay_async(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    # In failure_token_adv mode the old-policy forward is removed:
                    # the exact chosen-token log-probabilities captured during vLLM
                    # rollout become old_log_probs (see plan §5). Before enabling the
                    # reuse by default, run verify_rollout_log_probs to log the parity
                    # against the HF recomputation path.
                    reuse_rollout = (
                        self.tafr_enabled
                        and self.tafr_config.loss_version == "failure_token_adv"
                        and self.tafr_config.reuse_rollout_log_probs
                        and "rollout_log_probs" in batch.batch
                        and not bypass_recomputing_logprobs
                    )
                    if reuse_rollout and batch.batch["rollout_log_probs"].shape != batch.batch["response_mask"].shape:
                        raise ValueError(
                            "TAFR rollout_log_probs must have the same [batch, response_length] shape as response_mask"
                        )
                    verify_rollout = reuse_rollout and self.tafr_config.verify_rollout_log_probs
                    if reuse_rollout and "old_log_probs" not in batch.batch:
                        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                    if reuse_rollout and not verify_rollout:
                        metrics["tafr_grpo/rollout_logprobs_reused"] = 1.0
                    elif bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    elif self.gpi_ce_enabled and self.gpi_ce_config.fuse_old_log_prob:
                        # Skip the old-policy forward entirely. With ppo_epochs=1 and one
                        # optimizer step, theta_old == theta during the update, so the
                        # training forward's own detached scores ARE the old scores and
                        # the GPI-CE loss builds q* from them. Exact, not an approximation.
                        metrics["gpi/fused_old_log_prob"] = 1.0
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    if not (self.gpi_ce_enabled and self.gpi_ce_config.fuse_old_log_prob):
                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if verify_rollout:
                        metrics.update(self._tafr_log_rollout_parity_metrics(batch))
                        metrics["tafr_grpo/rollout_logprobs_reused"] = 0.0
                    # All-correct rows are inactive by construction: their failure
                    # log-probs equal the old log-probs so the failure advantage is 0.
                    self._tafr_sync_inactive_failure_log_probs(batch)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        if self.abstract_rl_enabled:
                            # Parse <abstract>, score it teacher-forced against the
                            # pre-update policy, and replace r_i with U_i before the
                            # group-relative advantage is taken.
                            metrics.update(abstract_hooks.score_and_shape(self, batch, timing_raw))

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        if self.gpi_ce_enabled:
                            # No advantage, no critic: the reward tilts the old finite-group
                            # distribution into q* and the actor is fit to it by cross entropy.
                            gpi_hooks.check_batch_integrity(batch, self.gpi_ce_config.group_size)
                            metrics.update(gpi_hooks.compute_targets(self, batch, reward_tensor))
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
                        # Wait for legacy anchor+replay scoring before constructing A_reg.
                        self._tafr_join_anchor_replay(batch)
                        self._tafr_build_regularization_advantage(batch)
                        # All vllm work is done — sleep now so the backward pass gets the
                        # freed KV cache memory (unless we already slept right after gen).
                        if not slept_after_gen:
                            self.checkpoint_manager.sleep_replicas()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        tafr_save_step = should_checkpoint_and_refresh(self.global_steps, self.tafr_config)
                        if (self.config.trainer.save_freq > 0 or tafr_save_step) and (
                            is_last_step
                            or tafr_save_step
                            or (
                                self.config.trainer.save_freq > 0
                                and self.global_steps % self.config.trainer.save_freq == 0
                            )
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # Run failure-SFT BEFORE the weight sync: at this point vLLM is
                        # still asleep (slept after the GRPO update), so the SFT
                        # forward/backward gets the freed rollout memory instead of OOMing
                        # against an awake engine. Doing it here also means the subsequent
                        # update_weights pushes the POST-SFT actor weights to vLLM, so the
                        # next rollout reflects the SFT update.
                        metrics.update(self._tafr_run_failure_sft_if_due())
                        metrics.update(self._tafr_schedule_metrics(tafr_wrong_collected))

                        # update weights from trainer to rollout (wakes vLLM, syncs the
                        # post-GRPO + post-SFT actor weights)
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        self._maybe_save_best_checkpoint(val_metrics)
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(self._compute_train_comparison_metrics(batch))
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                if self.abstract_rl_enabled:
                    metrics.update(abstract_hooks.efficiency_metrics(self, timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                if self.tafr_enabled:
                    # Actor-loss metrics describe the pre-SFT batch. Report readiness/count
                    # from the final trainer state so the activation step is visible immediately.
                    metrics["tafr_adv/failure_model_ready"] = 1.0 if self.tafr_failure_model_ready else 0.0
                    metrics["tafr_adv/failure_model_update_count"] = float(self.tafr_failure_model_update_count)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    if os.environ.get("VERL_VERBOSE_METRICS", "0") == "1":
                        pprint(f"Final validation metrics: {last_val_metrics}")
                    self._print_metric_summary("final_val", last_val_metrics, self.global_steps)
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
