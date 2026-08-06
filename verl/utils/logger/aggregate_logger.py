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
        "fepo/solver_loss",
        "fepo/solver_pg_loss",
        "fepo/failure_sft_loss",
        "solver_grad_norm",
        "failure_grad_norm",
        "train/accuracy",
        "train/failure_rate",
        "train/pass_at_1",
        "train/pass_at_8",
        "train/unique_answer_ratio_at_k",
        "train/exploration_collapse_rate",
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
        # Is A_reg per-token in effect, or only in shape? within >> between means
        # it varies along each rollout instead of just rescaling whole rollouts.
        "tafr_adv/raw_anchor_rms",
        "tafr_adv/raw_failure_rms",
        "tafr_adv/anchor_contribution_rms",
        "tafr_adv/failure_contribution_rms",
        "tafr_adv/failure_over_anchor_ratio",
        "tafr_adv/advantage_abs_p50",
        "tafr_adv/advantage_abs_p90",
        "tafr_adv/advantage_abs_p99",
        "tafr_adv/advantage_abs_max_all",
        "tafr_adv/advantage_within_seq_std",
        "tafr_adv/advantage_between_seq_std",
        "tafr_adv/failure_signal_within_seq_std",
        "tafr_adv/failure_signal_between_seq_std",
        "tafr_grpo/loss_total",
        "tafr_grpo/loss_grpo",
        "tafr_grpo/kl_anchor",
        "tafr_grpo/kl_replay",
        "tafr_grpo/replay_gate_mean",
        "tafr_grpo/mean_group_reward",
        "tafr_grpo/failure_dataset_size",
        "tafr_grpo/did_sft_update_this_step",
        "tafr_grpo/did_checkpoint_save_this_step",
        "tafr_grpo/failure_sft_loss",
        "tafr_grpo/failure_sft_updates",
        "tafr_adv/failure_model_ready",
        "tafr_adv/failure_model_update_count",
        "tafr_adv/active_group_fraction",
        "tafr_adv/scored_token_fraction",
        "tafr_adv/failure_signal_mean",
        "tafr_adv/failure_signal_std",
        "tafr_adv/regularization_loss",
        "tafr_adv/weighted_regularization_loss",
        "tafr_adv/combined_policy_loss",
        "tafr_adv/loss_delta_vs_grpo",
        "tafr_adv/stability_signal_mean",
        "tafr_adv/stability_signal_std",
        "tafr_adv/regularization_advantage_rms",
        "tafr_adv/grpo_advantage_rms",
        "tafr_adv/weighted_regularization_advantage_rms",
        "tafr_adv/regularization_to_grpo_rms_ratio",
        "tafr_adv/normalization_scale",
        "tafr_adv/all_correct/failure_component_abs_max",
        "tafr_adv/failure_loss",
        "tafr_adv/weighted_failure_loss",
        "tafr_adv/all_correct/advantage_abs_max",
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
