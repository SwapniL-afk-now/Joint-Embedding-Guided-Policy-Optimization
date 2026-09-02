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
"""
A Ray logger will receive logging info from different processes.
"""

import datetime
import logging
import numbers
import os
import pprint

import torch


def concat_dict_to_str(dict: dict, step):
    if os.environ.get("VERL_CONSOLE_FULL_METRICS", "0") == "1":
        output = [f"step:{step}"]
        for k, v in dict.items():
            if isinstance(v, numbers.Number):
                output.append(f"{k}:{pprint.pformat(v)}")
        return " - ".join(output)

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
        "eval_greedy/math500_pass1",
        "eval_greedy/aime24_pass1",
        "eval_greedy/aime25_pass1",
        "eval_greedy/amc23_pass1",
        "eval_greedy/minerva_pass1",
        "eval_greedy/olympiadbench_pass1",
        "eval_greedy/avg_pass1",
        # GPI-CE: target_mass_offline is the one that says whether the offline
        # candidates earn any of q*'s mass; degenerate_group_fraction is how much of
        # the batch has tied rewards and therefore contributes exactly zero gradient.
        "gpi/loss",
        "gpi/target_mass_offline",
        "gpi/old_mass_offline",
        "gpi/target_reward_improvement",
        "gpi/degenerate_group_fraction",
        "gpi/projection_kl",
        "gpi/target_entropy",
        "gpi/policy_entropy",
        "gpi/reward_offline_mean",
        "gpi/reward_online_mean",
        "gpi/seq_score_offline_mean",
        "gpi/seq_score_online_mean",
        "sdc/loss",
        "sdc/active_token_fraction",
        "sdc/importance_weight_mean",
        "sdc/log_ratio_clipped_fraction",
        "sdc/importance_weight_clip_fraction",
        "sdc/contrast_p10",
        "sdc/contrast_p50",
        "sdc/contrast_p90",
        "sdc/contrast_max",
        "sdc/contrast_sequence_mean",
        "sdc/contrast_center",
        "sdc/contrast_rms",
        "sdc/reliability_gate",
        "sdc/quality_auc_proxy",
        "sdc/zero_contrast_fraction",
        "sdc/sft_pair_response_tokens",
        "sdc/sft_pair_skipped_unmatched",
        "sdc/success_sft_updates",
        "sdc/failure_sft_updates",
        "sdc/actor_total_parameters",
        "sdc/actor_trainable_parameters",
        "sdc/success_total_parameters",
        "sdc/success_trainable_parameters",
        "sdc/success_optimizer_parameters",
        "sdc/failure_total_parameters",
        "sdc/failure_trainable_parameters",
        "sdc/failure_optimizer_parameters",
        "sdc/success_sft_response_tokens",
        "sdc/failure_sft_response_tokens",
        "sdc/success_buffer_age",
        "sdc/failure_buffer_age",
        "sdc/success_model_update_count",
        "sdc/failure_model_update_count",
        "sdc/models_ready",
        "sdc/success_sft_loss",
        "sdc/failure_sft_loss",
        "sdc/success_buffer_size",
        "sdc/failure_buffer_size",
        "sdc/did_sft_update_this_step",
        "sdc/global_policy_step",
        *(k for k in dict if k.startswith("sdc/timing/")),
        "actor/optimization/optimizer_step_calls_this_batch",
        "val/amc23/pass_at_1",
        "val/amc23/pass_at_8",
        "val/amc23/avg_at_k",
        "val/amc23/avg_at_8",
        "perf/time_per_step",
        "perf/step_seconds",
        "perf/responses_per_second",
        "timing_s/step",
        # JEPA-GRPO
        "grpo_pg_loss",
        "grpo_clip_fraction",
        "grad_norm",
        "jepa/lejepa_loss",
        "jepa/align_loss",
        "jepa/sigreg_loss",
        "jepa/n_valid_pairs",
        "jepa/frac_questions_with_both_correct",
        "cot/pass_at_1",
        "cot/avg_reward",
        "code/pass_at_1",
        "code/avg_reward",
    ]
    # Per-phase timings are how you find where a step actually goes; printing the
    # whole family costs nothing and beats guessing from the total.
    preferred = preferred + sorted(k for k in dict if k.startswith("timing_s/") and k not in preferred)
    preferred = preferred + sorted(k for k in dict if k.startswith("abstract_rl/") and k not in preferred)
    output = [f"step:{step}"]
    seen = set()
    for k in preferred:
        if k in dict and isinstance(dict[k], numbers.Number):
            output.append(f"{k}:{float(dict[k]):.4g}")
            seen.add(k)

    val_suffixes = {"pass_at_1", "pass_at_8", "avg_at_k", "avg_at_8"}
    for k in sorted(dict):
        parts = k.split("/")
        if len(parts) == 3 and parts[0] == "val" and parts[2] in val_suffixes and k not in seen:
            v = dict[k]
            if isinstance(v, numbers.Number):
                output.append(f"{k}:{float(v):.4g}")
                seen.add(k)

    if len(output) == 1:
        for k, v in dict.items():
            if isinstance(v, numbers.Number):
                output.append(f"{k}:{float(v):.4g}")
                if len(output) >= 12:
                    break
    return " - ".join(output)


class LocalLogger:
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class DecoratorLoggerBase:
    """
    Base class for all decorators that log messages.

    Args:
        role (str): The role (the name) of the logger.
        logger (logging.Logger): The logger instance to use for logging.
        level (int): The logging level.
        rank (int): The rank of the process.
        log_only_rank_0 (bool): If True, only log for rank 0.
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        print(f"[Rank {rank}] {message}", flush=True)


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information and a timestamp.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    now = datetime.datetime.now()
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"
    if not log_only_rank_0 or rank == 0:
        print(message, flush=True)


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")
