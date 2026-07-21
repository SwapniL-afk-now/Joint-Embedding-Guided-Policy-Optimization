# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""JEPA-GRPO Ray trainer.

Extends RayPPOTrainer with a two-view training loop:
  1. CoT rollout → standard Dr.GRPO update (via parent's update_actor)
  2. Code rollout → JEPA alignment update (via jepa_update on the worker)

The Code view uses the same vLLM rollout infrastructure (hybrid engine) as
the CoT view; only the system prompt differs.  JEPA embeddings are computed
on the actor worker itself so gradients are never shipped over Ray.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.fepo.math_parser import compute_math_reward
from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import simple_timer

logger = logging.getLogger(__name__)


class JEPARayPPOTrainer(RayPPOTrainer):
    """Ray PPO trainer augmented with LeJEPA representation alignment.

    The training step becomes:
        1. CoT rollout       (vLLM, standard)
        2. CoT reward        (math string match)
        3. CoT advantages    (Dr.GRPO / GRPO)
        4. GRPO update       (parent._update_actor)
        5. Code rollout      (vLLM, code system prompt)
        6. Code reward       (math string match on extracted answer)
        7. Build JEPA pairs  (prompts correct in both views)
        8. JEPA update       (worker.jepa_update — separate backward)
        9. EMA sync          (happens inside worker.jepa_update)
       10. Weight sync to rollout
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jepa_cfg = JEPARayConfig.from_config(self.config.get("jepa", {}))
        self.jepa_cfg.validate(self.config.actor_rollout_ref.rollout.n)

        # Corrected TCR-JEPA: the CoT cache stores verified-correct SOLUTION TEXTS
        # {dataset_index (int): [y^+ str, ...]}, NOT latent vectors. The target
        # representation z^+ is recomputed ONLINE by the current policy at train time
        # (worker.jepa_update), so there are no stale frozen-reference latents.
        #
        # Loaded whenever JEPA is enabled OR the representation t-SNE is on: the
        # diagnostic plots the teacher views themselves, so the BASELINE arm of a
        # JEPA-vs-baseline comparison (jepa.enable=False) must load them too or its
        # figure would have nothing to compare the student against.
        self.teacher_targets: dict[int, list[str]] | None = None
        if self.jepa_cfg.enable or self._rep_tsne_enabled:
            raw = torch.load(self.jepa_cfg.teacher_cache_path, map_location="cpu")
            self.teacher_targets = {int(k): list(v) for k, v in raw.items() if v}

        # Plateau-latch state for jepa.auto_off_enable (see _maybe_disable_jepa_signal).
        # Per-arm plateau latches: cot/code/self each plateau and latch INDEPENDENTLY,
        # disabling only their own loss arm (and matching-view shaping). `_jepa_signal_off`
        # (global) latches True only once EVERY tracked signal is off, at which point the
        # whole JEPA block is skipped to save the forward.
        self._jepa_signal_off = False
        self._off_arm: dict[str, bool] = {k: False for k in ("cot", "code", "self", "shaping", "global")}
        self._align_best: dict[str, float] = {k: float("-inf") for k in self._off_arm}
        self._align_stall: dict[str, int] = {k: 0 for k in self._off_arm}

        # A SECOND cache of Code-view verified-correct texts (coder-model solutions),
        # same {index: [y^+ str, ...]} format. Encoded online at train time too.
        # CoT-only mode ('jepa-tcr-cot') has no code cache; leave this None so the
        # code arm / reward shaping / self-consistency all self-gate off.
        self.code_teacher_targets: dict[int, list[str]] | None = None
        if (self.jepa_cfg.enable or self._rep_tsne_enabled) and self.jepa_cfg.code_teacher_cache_path:
            raw_code = torch.load(self.jepa_cfg.code_teacher_cache_path, map_location="cpu")
            self.code_teacher_targets = {int(k): list(v) for k, v in raw_code.items() if v}

        # Target-view tokenization cache. _tokenize_target is deterministic in
        # (problem, y^+, view) but was re-applying the chat template + tokenizing
        # the same targets twice per step (shaping + jepa batch build), every
        # step. Unbounded growth is fine: bounded by the teacher caches' size.
        self._tok_target_cache: dict[tuple[str, str, str], torch.Tensor] = {}

    @property
    def _rep_tsne_enabled(self) -> bool:
        return bool(self.config.trainer.get("rep_tsne_enable", False))

    # ------------------------------------------------------ worker setup ----
    def init_workers(self):
        super().init_workers()
        # jepa_init is also required with jepa.enable=False when the representation
        # t-SNE is on: it reads the predictor via score_cot_embeddings, which asserts
        # the worker has a jepa config. The config's own arms stay off, so this
        # initialises the read path without adding any loss term.
        if self.jepa_cfg.enable or self._rep_tsne_enabled:
            import dataclasses

            if self.jepa_cfg.predictor_k > 0:
                self.jepa_cfg.predictor_token_ids = self._resolve_predictor_token_ids(
                    self.jepa_cfg.predictor_k
                )
                # llm-jepa-contrastive and llm-jepa-infoNCE read WRONG anchors through a
                # separate <|bad_predictor_i|> sequence so good/bad reads are distinguishable.
                if self.jepa_cfg.loss_type in ("llm-jepa-contrastive", "llm-jepa-infoNCE"):
                    self.jepa_cfg.bad_predictor_token_ids = self._resolve_predictor_token_ids(
                        self.jepa_cfg.predictor_k, prefix="bad_predictor"
                    )

            cfg_dict = dataclasses.asdict(self.jepa_cfg)
            self.actor_rollout_wg.jepa_init(cfg_dict)

    def _resolve_predictor_token_ids(self, k: int, prefix: str = "predictor") -> list[int]:
        """Add the paper's LITERAL predictor special tokens and return the append
        sequence — exactly the official finetune.py mechanics:

          - k distinct special tokens ``<|{prefix}_1|>`` .. ``<|{prefix}_k|>`` are
            added to the tokenizer via ``add_special_tokens`` (only those not already
            in the vocab, mirroring finetune.py's ``token not in tokenizer.vocab``);
          - they are appended to the Text view in DESCENDING slot order
            ``<|{prefix}_k|>, ..., <|{prefix}_1|>`` (finetune.py's ``while to_add``
            back-append loop), so the last — read — token is always ``<|{prefix}_1|>``.

        ``prefix`` defaults to ``predictor`` (the good/teacher read); llm-jepa-contrastive
        also resolves ``bad_predictor`` for the wrong-anchor <bad_pred> read.

        The one mechanical difference from finetune.py: no ``resize_token_embeddings``.
        Qwen2.5's embedding matrix is padded (config vocab 151936 vs ~151665 tokenizer
        ids), so the new ids land in ALREADY-ALLOCATED spare rows — resizing would
        break LoRA adapter shapes and vLLM weight sync, and finetune.py's resize is a
        no-op whenever the matrix already has the rows. worker.jepa_init asserts the
        ids are in-bounds. The new tokens never appear in rollout prompts, so vLLM's
        own tokenizer copy is unaffected.
        """
        tok = self.tokenizer
        names = [f"<|{prefix}_{i}|>" for i in range(1, k + 1)]
        vocab = tok.get_vocab()
        new_tokens = [t for t in names if t not in vocab]
        if new_tokens:
            tok.add_special_tokens({"additional_special_tokens": new_tokens})
            logger.info("[jepa] added %d literal predictor special tokens: %s",
                        len(new_tokens), new_tokens)
        ids = [int(tok.convert_tokens_to_ids(t)) for t in names]   # slot 1..k
        assert all(i is not None and i >= 0 for i in ids), f"predictor token ids unresolved: {ids}"
        return list(reversed(ids))   # append order: <|predictor_k|> .. <|predictor_1|>

    # ------------------------------------------- code-view tokenisation -----
    def _tokenize_code_prompts(self, batch: DataProto) -> DataProto:
        """Build code-view gen_batch by replacing the system prompt.

        The AgentLoop handles tokenization internally via apply_chat_template;
        we only need to supply raw_prompt (list of messages) in non_tensor_batch.
        """
        code_sys = self.jepa_cfg.code_system_prompt

        raw_problems = [
            info["problem"] if isinstance(info, dict) else str(info)
            for info in batch.non_tensor_batch["extra_info"]
        ]

        messages_list = np.array(
            [
                [
                    {"role": "system", "content": code_sys},
                    {"role": "user", "content": problem},
                ]
                for problem in raw_problems
            ],
            dtype=object,
        )

        # Build a DataProto with a dummy tensor so DataProto has a known batch size.
        bsz = len(raw_problems)
        dummy = torch.zeros(bsz, 1, dtype=torch.uint8)
        code_batch = DataProto.from_single_dict({"dummy_tensor": dummy})
        # AgentLoop looks for raw_prompt in non_tensor_batch
        code_batch.non_tensor_batch["raw_prompt"] = messages_list
        # Copy metadata needed by the reward fn
        code_batch.non_tensor_batch.update({
            k: v for k, v in batch.non_tensor_batch.items()
            if k in {"uid", "data_source", "reward_model"}
        })
        # Tag rows as code-view via extra_info so the reward manager
        # (reward_loop/reward_manager/dapo.py) EXECUTES the program and scores
        # its printed answer instead of string-matching the source. COPY each
        # dict: the originals are shared by reference with the CoT batch.
        code_batch.non_tensor_batch["extra_info"] = np.array(
            [
                {**info, "view": "code"} if isinstance(info, dict) else info
                for info in batch.non_tensor_batch["extra_info"]
            ],
            dtype=object,
        )
        code_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        code_batch.meta_info["global_steps"] = batch.meta_info.get("global_steps", 0)
        return code_batch

    # ------------------------------------------- dual-view validation -------
    def _validate(self, merged: bool = False):
        metrics = super()._validate(merged=merged)
        if not merged:
            self._maybe_log_validation_rep_tsne()
        return metrics

    def _maybe_log_validation_rep_tsne(self) -> None:
        """Log a t-SNE of one response per problem from the exact val suite."""
        cfg = self.config.trainer
        if not bool(cfg.get("rep_tsne_enable", False)):
            return
        # rep_tsne_mode: "train" (default) = the paper-style train-batch figure only
        # (see _maybe_log_train_rep_tsne); "val"/"both" keep this benchmark-faceted
        # plot. Validation problems have NO teacher CoT/Code views — the teacher
        # caches are keyed to train-parquet indices — so this path can only ever
        # compare against an answer-conditioned proxy target, not a teacher view.
        if str(cfg.get("rep_tsne_mode", "train")) not in ("val", "both"):
            return
        freq = max(1, int(cfg.get("rep_tsne_freq", cfg.test_freq)))
        if self.global_steps != 0 and self.global_steps % freq != 0:
            return
        raw = getattr(self, "_last_validation_rep_samples", None)
        if not raw or not raw.get("token_ids"):
            logger.warning("representation t-SNE skipped: no validation rows")
            return

        chosen, seen = [], set()
        for i, uid in enumerate(raw["uids"]):
            if uid not in seen:
                seen.add(uid); chosen.append(i)
        rows = [raw["token_ids"][i] for i in chosen]
        target_rows = []
        eos = [] if self.tokenizer.eos_token_id is None else [int(self.tokenizer.eos_token_id)]
        for i in chosen:
            answer = self.tokenizer(
                raw["ground_truths"][i], add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].cpu()
            target_rows.append(torch.cat([
                raw["prompt_ids"][i].long(), answer.long(), torch.tensor(eos, dtype=torch.long)
            ]))
        all_rows = rows + target_rows
        if len(rows) < 4:
            return
        max_len = max(int(row.numel()) for row in all_rows)
        ids = torch.full((len(all_rows), max_len), int(self.tokenizer.pad_token_id or 0), dtype=torch.long)
        mask = torch.zeros_like(ids)
        lengths = torch.tensor([row.numel() for row in all_rows], dtype=torch.long)
        for i, row in enumerate(all_rows):
            ids[i, :row.numel()] = row.long(); mask[i, :row.numel()] = 1
        td = TensorDict({
            "input_ids": ids, "attention_mask": mask, "lengths": lengths,
            "micro_bs": torch.full((len(all_rows),), int(cfg.get("rep_tsne_micro_batch_size", 4))),
        }, batch_size=[len(all_rows)])
        out = self.actor_rollout_wg.embed_validation_sequences(td)
        if isinstance(out, list):
            out = out[0]
        all_emb = out["embeddings"].detach().cpu().numpy()
        emb, target_emb = np.split(all_emb, 2)

        import json, os
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        sources = np.asarray([raw["data_sources"][i] for i in chosen], dtype=object)
        correct = np.asarray([raw["scores"][i] > 0 for i in chosen], dtype=bool)
        perplexity = min(float(cfg.get("rep_tsne_perplexity", 30)), max(2.0, (len(rows)-1)/3))
        proj = TSNE(n_components=2, init="pca", learning_rate="auto",
                    perplexity=perplexity,
                    random_state=int(cfg.get("rep_tsne_seed", 14142))).fit_transform(emb)
        out_dir = os.path.join(str(cfg.default_local_dir), "representation_tsne")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"step_{int(self.global_steps)}")
        # Paper-style representation diagnostics. Compare generated-answer
        # embeddings with answer-conditioned target embeddings for the same prompts.
        difference = emb - target_emb
        singular_values = np.linalg.svd(
            difference - difference.mean(axis=0, keepdims=True), compute_uv=False
        )
        sv_mass = singular_values / max(float(singular_values.sum()), 1e-12)
        effective_rank = float(np.exp(-(sv_mass * np.log(sv_mass + 1e-12)).sum()))
        cosine = np.sum(emb * target_emb, axis=1) / np.maximum(
            np.linalg.norm(emb, axis=1) * np.linalg.norm(target_emb, axis=1), 1e-12
        )
        split = max(1, len(rows) // 2)
        x_train, y_train = emb[:split], target_emb[:split]
        x_test, y_test = emb[split:], target_emb[split:]
        if len(x_test):
            gram = x_train @ x_train.T
            alpha = np.linalg.solve(gram + 1e-3 * np.eye(len(x_train)), y_train)
            mapped = (x_test @ x_train.T) @ alpha
            linear_residual = float(
                np.linalg.norm(mapped - y_test) / max(float(np.linalg.norm(y_test)), 1e-12)
            )
        else:
            linear_residual = float("nan")

        np.savez(stem+".npz", embeddings=emb, target_embeddings=target_emb,
                 differences=difference, singular_values=singular_values, cosine=cosine,
                 projection=proj, sources=sources, correct=correct,
                 uids=np.asarray([raw["uids"][i] for i in chosen], dtype=object))
        with open(stem+".json", "w") as f:
            json.dump({"step": int(self.global_steps), "count": len(rows),
                       "perplexity": perplexity,
                       "sources": sorted(set(sources.tolist())),
                       "cosine_mean": float(cosine.mean()),
                       "cosine_std": float(cosine.std()),
                       "effective_rank": effective_rank,
                       "linear_residual": linear_residual,
                       "top100_sv_mean": float(singular_values[:100].mean())}, f, indent=2)
        fig, ax = plt.subplots(figsize=(8, 6)); cmap = plt.get_cmap("tab10")
        for j, source in enumerate(sorted(set(sources.tolist()))):
            src = sources == source
            for ok, marker, name in ((True,"o","correct"),(False,"x","incorrect")):
                sel = src & (correct == ok)
                if sel.any():
                    ax.scatter(proj[sel,0], proj[sel,1], color=cmap(j), marker=marker,
                               alpha=.8, label=f"{source} — {name}")
        ax.set_title(f"Validation response representations — step {self.global_steps}")
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); png = stem+".png"; fig.savefig(png, dpi=180); plt.close(fig)

        diag_fig, (sv_ax, cos_ax) = plt.subplots(1, 2, figsize=(11, 4))
        sv_ax.plot(np.arange(1, min(100, len(singular_values)) + 1), singular_values[:100])
        sv_ax.set_yscale("log")
        sv_ax.set_title("Difference singular-value spectrum")
        sv_ax.set_xlabel("Singular-value index"); sv_ax.set_ylabel("Singular value")
        cos_ax.hist(1.0 - cosine, bins=min(25, max(5, len(cosine) // 4)), alpha=.85)
        cos_ax.set_title("Generated-to-target cosine distance")
        cos_ax.set_xlabel("1 - cosine similarity"); cos_ax.set_ylabel("Count")
        diag_fig.suptitle(f"Validation representation diagnostics — step {self.global_steps}")
        diag_fig.tight_layout(); diag_png = stem+"_diagnostics.png"
        diag_fig.savefig(diag_png, dpi=180); plt.close(diag_fig)
        if "wandb" in list(cfg.logger):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "val/representation_tsne": wandb.Image(png),
                        "val/representation_diagnostics": wandb.Image(diag_png),
                        "val/representation_cosine_mean": float(cosine.mean()),
                        "val/representation_cosine_std": float(cosine.std()),
                        "val/representation_effective_rank": effective_rank,
                        "val/representation_linear_residual": linear_residual,
                        "val/representation_top100_sv_mean": float(singular_values[:100].mean()),
                    }, step=self.global_steps)
            except Exception as exc:
                logger.warning("W&B t-SNE logging failed: %s", exc)
        print(f"[representation_tsne] wrote {png} ({len(rows)} benchmark problems)")

    # ------------------------------------ paper-style train representations -
    def _collect_rep_groups(self, batch, reward_tensor, view_tags):
        """Gather the three view groups the LLM-JEPA representation figure needs.

        Returns (p, Z, meta) where ``p`` are the student predictor reads
        ``Pred(Enc([x, y_S, [PRED]xk]))`` for EVERY CoT rollout of the sampled
        prompts (correct and incorrect alike — the incorrect ones are group 4 of
        Figure B), ``Z`` are the encoded teacher views ``Enc([y^T])``, and ``meta``
        carries the per-anchor correctness flag and the gather indices from each
        anchor into its own teacher CoT / teacher Code row.

        Unlike _build_jepa_batch_llm_jepa this deliberately does NOT apply
        jepa_anchor_set or max_anchors_per_prompt: the loss trains on a capped,
        correct-only subset, but the diagnostic must see the full rollout cloud or
        the clusters it is meant to reveal are the ones it filtered away. Teacher
        texts are taken at a FIXED index (0) rather than cycled, so the teacher
        points are the same across steps and the figure is comparable over time.
        """
        cfg = self.config.trainer
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)
        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id

        rows_by_uid: dict = defaultdict(list)
        idx_by_uid: dict = {}
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v != "cot" or int(all_attn_mask[i].sum()) == 0:
                continue
            rows_by_uid[u].append(i)
            if u not in idx_by_uid:
                info = extra_infos[i]
                idx_by_uid[u] = int(info["index"]) if isinstance(info, dict) else None

        max_prompts = int(cfg.get("rep_tsne_max_prompts", 32))
        anchors: list[int] = []
        pair_idx: list[tuple[int, int]] = []
        uniq_pos: dict = {}
        uniq_rows: list[torch.Tensor] = []
        uniq_view: list[str] = []

        def _uniq_target(view: str, problem: str, text: str) -> int:
            key = (view, text)
            pos = uniq_pos.get(key)
            if pos is None:
                pos = len(uniq_rows)
                uniq_pos[key] = pos
                uniq_rows.append(self._tokenize_target(problem, text, view))
                uniq_view.append(view)
            return pos

        n_prompts = 0
        for u in dict.fromkeys(uids):
            if max_prompts > 0 and n_prompts >= max_prompts:
                break
            ds_idx = idx_by_uid.get(u)
            if ds_idx is None:
                continue
            Z_cot = self.teacher_targets.get(ds_idx) if self.teacher_targets else None
            if not Z_cot:
                continue
            Z_code = self.code_teacher_targets.get(ds_idx) if self.code_teacher_targets else None
            sel = rows_by_uid.get(u, [])
            if not sel:
                continue
            info = extra_infos[sel[0]]
            problem = info["problem"] if isinstance(info, dict) else str(info)
            cpos = _uniq_target("cot", problem, Z_cot[0])
            kpos = _uniq_target("code", problem, Z_code[0]) if Z_code else -1
            for idx in sel:
                anchors.append(idx)
                pair_idx.append((cpos, kpos))
            n_prompts += 1

        if len(anchors) < 4 or not uniq_rows:
            return None

        def _strip_eos(t: torch.Tensor) -> torch.Tensor:
            return t[:-1] if (eos_id is not None and t.shape[0] > 1 and int(t[-1]) == eos_id) else t

        def _pack(rows: list[torch.Tensor]):
            lens = [int(t.shape[0]) for t in rows]
            m = max(lens)
            ids = torch.full((len(rows), m), pad_id, dtype=all_input_ids.dtype)
            msk = torch.zeros((len(rows), m), dtype=torch.long)
            for r, t in enumerate(rows):
                ids[r, : t.shape[0]] = t
                msk[r, : t.shape[0]] = 1
            return ids, msk, torch.tensor(lens, dtype=torch.long)

        a_ids, a_mask, a_len = _pack(
            [_strip_eos(all_input_ids[i][all_attn_mask[i].bool()]) for i in anchors]
        )
        t_ids, t_mask, t_len = _pack(uniq_rows)

        cot_out = self.actor_rollout_wg.score_cot_embeddings(
            DataProto.from_single_dict({
                "cot_input_ids": a_ids, "cot_attn_mask": a_mask, "cot_lengths": a_len,
            }).to_tensordict()
        )
        if isinstance(cot_out, list):
            cot_out = cot_out[0]
        tgt_out = self.actor_rollout_wg.embed_targets(
            DataProto.from_single_dict({
                "target_input_ids": t_ids, "target_attn_mask": t_mask, "target_lengths": t_len,
            }).to_tensordict()
        )
        if isinstance(tgt_out, list):
            tgt_out = tgt_out[0]

        meta = {
            "correct": np.asarray([bool(rew[i] > 0) for i in anchors], dtype=bool),
            "cot_idx": np.asarray([c for c, _ in pair_idx], dtype=np.int64),
            "code_idx": np.asarray([k for _, k in pair_idx], dtype=np.int64),
            "uids": np.asarray([str(uids[i]) for i in anchors], dtype=object),
            "target_view": np.asarray(uniq_view, dtype=object),
            "n_prompts": n_prompts,
        }
        return cot_out["cot_emb"].float().numpy(), tgt_out["target_emb"].float().numpy(), meta

    def _maybe_log_train_rep_tsne(self, batch, reward_tensor, view_tags) -> None:
        """Paper-style representation figure (LLM-JEPA arXiv:2509.14252, Fig. 4/6).

        The paper plots the two views -- Enc(Text) and Enc(Code) -- in ONE t-SNE and
        reads off the pair geometry: under NTP fine-tuning the clouds are unstructured,
        under LLM-JEPA they organise so that paired points sit in a consistent
        relationship. Fig. 3 (left) is the quantitative form of the same claim: the top
        singular values of the paired difference collapse by orders of magnitude.

        Adapted to this RL setting, the views are:
          A. student CoT predictor read  p = Pred(Enc([x, y_S, [PRED]xk]))
          B. teacher CoT view            Enc([y^T_cot])
          C. teacher Code view           Enc([y^T_code])
        and the loss aligns A to B and A to C. Two figures are emitted:
          fig A  -- correct student anchors + both teacher views (the paper's plot)
          fig B  -- the same plus INCORRECT student anchors, which the paper has no
                    analogue for; if the objective is working these should fall off
                    the teacher manifold that the correct ones sit on.
        Figure B is fitted separately, because adding points changes a t-SNE embedding
        and a shared fit would make A incomparable across steps.

        Runs on the TRAIN batch, not validation: the teacher caches are keyed by
        train-parquet dataset_index, so validation problems have no teacher views at
        all. This also matches the paper, which visualises its fine-tuning data.
        """
        cfg = self.config.trainer
        if not self._rep_tsne_enabled:
            return
        if str(cfg.get("rep_tsne_mode", "train")) not in ("train", "both"):
            return
        freq = max(1, int(cfg.get("rep_tsne_freq", cfg.test_freq)))
        if self.global_steps % freq != 0:
            return
        if not self.teacher_targets:
            logger.warning("train representation t-SNE skipped: no teacher CoT cache")
            return
        try:
            got = self._collect_rep_groups(batch, reward_tensor, view_tags)
        except Exception as exc:
            logger.warning("train representation t-SNE skipped: %s", exc)
            return
        if got is None:
            logger.warning("train representation t-SNE skipped: too few anchors")
            return
        p, Z, meta = got

        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        correct = meta["correct"]
        cot_idx, code_idx = meta["cot_idx"], meta["code_idx"]
        has_code = code_idx >= 0
        t_cot = Z[cot_idx]
        out_dir = os.path.join(str(cfg.default_local_dir), "representation_tsne")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"train_step_{int(self.global_steps)}")

        # ---- Fig. 3-left analogue: paired-difference spectra, per arm ----
        def _spectrum(diff: np.ndarray) -> tuple[np.ndarray, float]:
            if len(diff) < 2:
                return np.zeros(1), float("nan")
            sv = np.linalg.svd(diff - diff.mean(axis=0, keepdims=True), compute_uv=False)
            mass = sv / max(float(sv.sum()), 1e-12)
            return sv, float(np.exp(-(mass * np.log(mass + 1e-12)).sum()))

        def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.sum(a * b, axis=1) / np.maximum(
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12
            )

        sv_cot, erank_cot = _spectrum(p - t_cot)
        cos_cot = _cos(p, t_cot)
        if has_code.any():
            t_code = Z[np.where(has_code, code_idx, 0)]
            sv_code, erank_code = _spectrum((p - t_code)[has_code])
            cos_code = _cos(p, t_code)
        else:
            t_code = None
            sv_code, erank_code = np.zeros(1), float("nan")
            cos_code = np.zeros(0)

        # ---- the two t-SNE figures ----
        def _fit(points: np.ndarray) -> np.ndarray:
            perp = min(float(cfg.get("rep_tsne_perplexity", 30)), max(2.0, (len(points) - 1) / 3))
            return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perp,
                        random_state=int(cfg.get("rep_tsne_seed", 14142))).fit_transform(points)

        z_cot_rows = np.unique(cot_idx)
        z_code_rows = np.unique(code_idx[has_code]) if has_code.any() else np.zeros(0, dtype=np.int64)

        def _panel(ax, include_wrong: bool, title: str):
            sel = np.ones_like(correct) if include_wrong else correct
            if sel.sum() < 2:
                return
            pts = np.concatenate([p[sel], Z[z_cot_rows], Z[z_code_rows]], axis=0)
            proj = _fit(pts)
            n_s = int(sel.sum())
            s_proj = proj[:n_s]
            cot_proj = proj[n_s: n_s + len(z_cot_rows)]
            code_proj = proj[n_s + len(z_cot_rows):]
            # Pair segments: each student anchor to its own teacher views.
            cot_at = {int(r): k for k, r in enumerate(z_cot_rows)}
            code_at = {int(r): k for k, r in enumerate(z_code_rows)}
            for k, a in enumerate(np.where(sel)[0]):
                tgt = cot_proj[cot_at[int(cot_idx[a])]]
                ax.plot([s_proj[k, 0], tgt[0]], [s_proj[k, 1], tgt[1]],
                        color="0.75", lw=.3, zorder=0)
                if has_code[a] and int(code_idx[a]) in code_at:
                    tgt = code_proj[code_at[int(code_idx[a])]]
                    ax.plot([s_proj[k, 0], tgt[0]], [s_proj[k, 1], tgt[1]],
                            color="0.88", lw=.3, zorder=0)
            c_sel = correct[sel]
            ax.scatter(s_proj[c_sel, 0], s_proj[c_sel, 1], c="tab:blue", marker="o",
                       s=14, alpha=.75, label="student CoT (correct)")
            if include_wrong and (~c_sel).any():
                ax.scatter(s_proj[~c_sel, 0], s_proj[~c_sel, 1], c="tab:red", marker="x",
                           s=18, alpha=.75, label="student CoT (incorrect)")
            ax.scatter(cot_proj[:, 0], cot_proj[:, 1], c="tab:green", marker="^",
                       s=44, edgecolors="k", linewidths=.4, label="teacher CoT")
            if len(code_proj):
                ax.scatter(code_proj[:, 0], code_proj[:, 1], c="tab:orange", marker="s",
                           s=44, edgecolors="k", linewidths=.4, label="teacher Code")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=7, loc="best")

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))
        _panel(ax_a, False, "(a) correct student CoT vs teacher CoT / Code")
        _panel(ax_b, True, "(b) + incorrect student CoT")
        fig.suptitle(
            f"Representations — step {self.global_steps} "
            f"(k={self.jepa_cfg.predictor_k}, jepa={'on' if self.jepa_cfg.enable else 'off'})"
        )
        fig.tight_layout()
        png = stem + ".png"
        fig.savefig(png, dpi=180)
        plt.close(fig)

        # ---- diagnostics: spectra + correct/incorrect cosine split ----
        dfig, (sv_ax, cos_ax) = plt.subplots(1, 2, figsize=(11, 4))
        sv_ax.plot(np.arange(1, min(100, len(sv_cot)) + 1), sv_cot[:100], label="p - Enc(teacher CoT)")
        if len(sv_code) > 1:
            sv_ax.plot(np.arange(1, min(100, len(sv_code)) + 1), sv_code[:100],
                       label="p - Enc(teacher Code)")
        sv_ax.set_yscale("log")
        sv_ax.set_title("Paired-difference singular values (paper Fig. 3 left)")
        sv_ax.set_xlabel("Singular-value index"); sv_ax.set_ylabel("Singular value")
        sv_ax.legend(fontsize=8)
        for flag, name, color in ((True, "correct", "tab:blue"), (False, "incorrect", "tab:red")):
            vals = 1.0 - cos_cot[correct == flag]
            if len(vals):
                cos_ax.hist(vals, bins=min(30, max(5, len(vals) // 4)), alpha=.6,
                            color=color, label=f"{name} (n={len(vals)})")
        cos_ax.set_title("Student-to-teacher-CoT cosine distance")
        cos_ax.set_xlabel("1 - cosine similarity"); cos_ax.set_ylabel("Count")
        cos_ax.legend(fontsize=8)
        dfig.suptitle(f"Representation diagnostics — step {self.global_steps}")
        dfig.tight_layout()
        dpng = stem + "_diagnostics.png"
        dfig.savefig(dpng, dpi=180)
        plt.close(dfig)

        def _m(v):
            return float(v.mean()) if len(v) else float("nan")

        summary = {
            "step": int(self.global_steps),
            "n_anchors": int(len(p)),
            "n_correct": int(correct.sum()),
            "n_prompts": int(meta["n_prompts"]),
            "n_teacher_cot": int(len(z_cot_rows)),
            "n_teacher_code": int(len(z_code_rows)),
            "cos_cot_correct": _m(cos_cot[correct]),
            "cos_cot_wrong": _m(cos_cot[~correct]),
            "cos_code_correct": _m(cos_code[correct]) if len(cos_code) else float("nan"),
            "cos_code_wrong": _m(cos_code[~correct]) if len(cos_code) else float("nan"),
            "effective_rank_cot": erank_cot,
            "effective_rank_code": erank_code,
            "top100_sv_mean_cot": float(sv_cot[:100].mean()),
            "top100_sv_mean_code": float(sv_code[:100].mean()),
        }
        np.savez(stem + ".npz", student=p, targets=Z, correct=correct,
                 cot_idx=cot_idx, code_idx=code_idx, uids=meta["uids"],
                 target_view=meta["target_view"], sv_cot=sv_cot, sv_code=sv_code,
                 cos_cot=cos_cot, cos_code=cos_code)
        with open(stem + ".json", "w") as f:
            json.dump(summary, f, indent=2)

        if "wandb" in list(cfg.logger):
            try:
                import wandb

                if wandb.run is not None:
                    payload = {f"rep/{k}": v for k, v in summary.items() if k != "step"}
                    payload["rep/tsne"] = wandb.Image(png)
                    payload["rep/diagnostics"] = wandb.Image(dpng)
                    wandb.log(payload, step=self.global_steps)
            except Exception as exc:
                logger.warning("W&B train t-SNE logging failed: %s", exc)
        print(f"[representation_tsne] wrote {png} "
              f"({len(p)} anchors / {meta['n_prompts']} prompts, "
              f"cos correct={summary['cos_cot_correct']:.3f} wrong={summary['cos_cot_wrong']:.3f})")

    def _validate_once(self, merged: bool = False, seed: int | None = None):
        """CoT validation (parent) plus an extra Code-view pass.

        Training pass@K pools 4 CoT + 4 Code rollouts per prompt, but the stock
        `_validate_once` is CoT-only, so val pass@K was not measuring the same
        quantity. This override keeps every existing CoT-only key (`val/*`,
        `val-core/*`, `val-aux/*` — best-checkpoint selection reads those, so
        its semantics are unchanged) and ADDS:

          - ``val-code/{source}/*``: grouped metrics over the Code-view samples.
          - ``val-dual/{source}/*``: per-prompt UNION of both views. Rows are
            interleaved cot,code,cot,code,... so with val n=8 per view,
            ``pass_at_8`` reads "4 CoT + 4 Code" (directly comparable to
            train/pass_at_8) and ``pass_at_16`` is the full 8+8 union.

        The two passes iterate `val_dataloader` independently; prompts are
        matched by dataloader order (the val dataloader is sequential). If the
        two passes disagree on prompt count or data_source order, the union
        metrics are skipped with a warning rather than reported misaligned.
        """
        # Deliberately NOT gated on jepa_cfg.enable: enable=False only turns off
        # the shaping/aux-loss arms, while the Code view keeps being rolled out
        # and trained on (n_code > 0), so val must keep measuring it.
        dual = bool(self.jepa_cfg.n_code > 0
                    and getattr(self.jepa_cfg, "dual_view_val", False))
        if merged or not dual:
            return super()._validate_once(merged=merged, seed=seed)

        # ── CoT pass: parent loop, raw per-sample results, then the exact
        # same metric computation the non-merged parent path performs. ──
        cot_raw = super()._validate_once(merged=True, seed=seed)
        if not cot_raw.get("data_sources"):
            return {}
        cot_ds = [str(s) for s in np.concatenate(cot_raw["data_sources"], axis=0)]
        cot_uids = [str(u) for u in cot_raw["sample_uids"]]
        cot_extra = cot_raw["reward_extra_infos_dict"]
        metric_dict = self._val_metrics_update(
            np.array(cot_ds, dtype=object), cot_uids, cot_extra, cot_raw["sample_turns"]
        )

        acc_src = cot_extra.get("acc") or cot_extra.get("score") or cot_extra.get("reward") or []
        cot_acc = [float(a) for a in acc_src]
        cot_pred = [str(p) for p in cot_extra.get("pred", [])]
        if len(cot_acc) != len(cot_uids):
            logger.warning("dual-view val: CoT acc/uid length mismatch; skipping code view")
            return metric_dict

        code = self._validate_code_view(seed=seed)
        if code is None:
            return metric_dict

        metric_dict.update(self._compute_grouped_accuracy_metrics(
            code["ds"], code["uid"],
            {"acc": code["acc"], "score": code["acc"], "pred": code["pred"]},
            prefix="val-code", outputs=code["out"],
        ))

        # ── Union: group each pass's rows by first-seen uid order, pair
        # prompts positionally, interleave rows cot,code,cot,code,... ──
        def _first_seen(uids_list):
            order, rows = [], defaultdict(list)
            for i, u in enumerate(uids_list):
                if u not in rows:
                    order.append(u)
                rows[u].append(i)
            return order, rows

        cot_order, cot_rows = _first_seen(cot_uids)
        code_order, code_rows = _first_seen(code["uid"])
        if len(cot_order) != len(code_order):
            logger.warning(
                "dual-view val: prompt count mismatch (cot=%d code=%d); skipping val-dual",
                len(cot_order), len(code_order),
            )
            return metric_dict

        have_pred = len(cot_pred) == len(cot_uids)
        dual_ds, dual_uid, dual_acc, dual_pred = [], [], [], []
        from itertools import zip_longest
        for u_c, u_k in zip(cot_order, code_order):
            ci, ki = cot_rows[u_c], code_rows[u_k]
            if cot_ds[ci[0]] != code["ds"][ki[0]]:
                logger.warning(
                    "dual-view val: data_source order mismatch (%s vs %s); skipping val-dual",
                    cot_ds[ci[0]], code["ds"][ki[0]],
                )
                return metric_dict
            for a, b in zip_longest(ci, ki):
                if a is not None:
                    dual_uid.append(u_c)
                    dual_ds.append(cot_ds[a])
                    dual_acc.append(cot_acc[a])
                    if have_pred:
                        dual_pred.append(cot_pred[a])
                if b is not None:
                    dual_uid.append(u_c)
                    dual_ds.append(code["ds"][b])
                    dual_acc.append(code["acc"][b])
                    if have_pred:
                        dual_pred.append(code["pred"][b])

        metric_dict.update(self._compute_grouped_accuracy_metrics(
            dual_ds, dual_uid,
            {"acc": dual_acc, "score": dual_acc, "pred": dual_pred},
            prefix="val-dual",
        ))
        return metric_dict

    def _validate_code_view(self, seed: int | None = None) -> dict | None:
        """Generate + score one Code-view pass over the whole val set.

        Mirrors the generation/reward mechanics of the parent `_validate_once`
        loop (same val_kwargs sampling, same padding, same reward extraction),
        with the prompt rebuilt through `_tokenize_code_prompts` so the system
        prompt matches the training-time code view.
        """
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from verl.trainer.ppo.reward import extract_reward

        # Same back-to-back vLLM RPC drain guard as fit()'s cot->code handoff.
        import time as _time
        _time.sleep(float(os.environ.get("JEPA_VLLM_DRAIN_S", "5")))

        n_val = self.config.actor_rollout_ref.rollout.val_kwargs.n
        uids, dss, accs, preds, outs = [], [], [], [], []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )
            test_batch = test_batch.repeat(repeat_times=n_val, interleave=True)

            code_gen_batch = self._tokenize_code_prompts(test_batch)
            # Replace train-time meta (temperature) with the parent's val meta
            # so the rollout uses val_kwargs sampling, exactly like the CoT pass.
            code_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            if seed is not None:
                code_gen_batch.meta_info["seed"] = int(seed)

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            padded, pad_size = pad_dataproto_to_divisor(code_gen_batch, size_divisor)
            out_padded = self.async_rollout_manager.generate_sequences(padded)
            out = unpad_dataproto(out_padded, pad_size=pad_size)
            # Same union-hygiene as fit(): drop per-call keys that would
            # collide or end up length-mismatched.
            out.meta_info.pop("timing", None)
            out.non_tensor_batch.pop("raw_prompt", None)
            out.non_tensor_batch.pop("uid", None)

            test_batch = test_batch.union(out)
            test_batch.meta_info["validate"] = True
            reward_tensor, reward_extra = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
            n_rows = len(scores)

            acc_vals = reward_extra.get("acc")
            if acc_vals is not None and len(acc_vals) == n_rows:
                acc_list = [float(a) for a in acc_vals]
            else:
                acc_list = [float(s) for s in scores]
            pred_vals = reward_extra.get("pred")
            if pred_vals is not None and len(pred_vals) == n_rows:
                pred_list = [str(p) if p is not None else "<unparsed>" for p in pred_vals]
            else:
                pred_list = ["<unparsed>"] * n_rows

            uids.extend(str(u) for u in test_batch.non_tensor_batch["uid"])
            dss.extend(
                str(s) for s in test_batch.non_tensor_batch.get("data_source", ["unknown"] * n_rows)
            )
            accs.extend(acc_list)
            preds.extend(pred_list)
            outs.extend(self.tokenizer.batch_decode(test_batch.batch["responses"], skip_special_tokens=True))

        if not uids:
            return None
        return {"uid": uids, "ds": dss, "acc": accs, "pred": preds, "out": outs}

    # --------------------------------------- JEPA batch construction --------
    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract the last printed number/expression from a Python code response."""
        import re
        # Look for boxed answer first (model may still produce it)
        m = re.search(r"\\boxed\{([^}]+)\}", response)
        if m:
            return m.group(1).strip()
        # Otherwise take the last line that looks numeric
        for line in reversed(response.strip().splitlines()):
            line = line.strip()
            if line and re.match(r"^-?\d", line):
                return line
        return response.strip()

    def _compute_train_comparison_metrics(self, batch: DataProto) -> dict[str, float | int]:
        """Compute train pass@K from raw verifier correctness, grouped across views.

        JEPA/TCR reward shaping can make a sequence's shaped reward positive even
        when the math/code verifier marked it wrong. Training pass@K should answer
        "did any of the 4 CoT + 4 Code samples solve the prompt?", so it must use
        raw verifier `acc` and the shared prompt `uid`, not shaped token rewards.
        """
        raw_acc = batch.non_tensor_batch.get("acc", None)
        if raw_acc is None or len(raw_acc) != len(batch):
            return super()._compute_train_comparison_metrics(batch)

        acc_values = [float(v) for v in raw_acc]
        uids = [str(uid) for uid in batch.non_tensor_batch.get("uid", [])]
        data_sources = [str(src) for src in batch.non_tensor_batch.get("data_source", ["unknown"] * len(acc_values))]
        preds = [str(pred if pred is not None else "<unparsed>") for pred in batch.non_tensor_batch.get("pred", [])]
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        reward_extra = {"acc": acc_values, "score": acc_values, "pred": preds}

        grouped = self._compute_grouped_accuracy_metrics(
            data_sources, uids, reward_extra, prefix="train-dataset", outputs=outputs
        )
        correct_count = sum(float(v > 0.0) for v in acc_values)
        total = len(acc_values)
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

    @staticmethod
    def _group_rows_by_uid(
        uids: np.ndarray, view_tags: np.ndarray, rew: torch.Tensor
    ) -> tuple[dict, dict, list]:
        """Group row indices by (uid, view), preserving first-seen prompt order.

        Replaces the old `flat_idx = p_idx * rollout_n + g` stride arithmetic,
        which assumed two SEPARATE, fixed-stride rollout_n-sized batches. Now
        that cot+code rows live in one combined batch (built via
        `DataProto.concat`, not positional interleaving — see fit()), the
        only thing that ties a prompt's rows together is a shared `uid`.

        Returns (cot_by_uid, code_by_uid, valid_uids) where valid_uids is the
        ordered list of uids with >=1 correct cot row AND >=1 correct code row.
        """
        cot_by_uid: dict = defaultdict(list)
        code_by_uid: dict = defaultdict(list)
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v == "cot":
                cot_by_uid[u].append(i)
            elif v == "code":
                code_by_uid[u].append(i)

        valid_uids = []
        for u in dict.fromkeys(uids):  # dedup, preserves first-seen order
            cot_idxs = cot_by_uid.get(u, [])
            code_idxs = code_by_uid.get(u, [])
            if cot_idxs and code_idxs and (rew[cot_idxs] > 0).any() and (rew[code_idxs] > 0).any():
                valid_uids.append(u)
        return cot_by_uid, code_by_uid, valid_uids

    def _template_suffix_ids(self) -> list:
        """Token ids of the chat template's constant end-of-message suffix.

        Computed once by diffing the rendering of a marker assistant message
        against the marker's own tokens (Qwen: ["<|im_end|>", "\n"]). Used to
        replicate the official run.sh ``last_token=-3`` read for Qwen — the
        embedding is taken at the last CONTENT token, not the constant suffix.
        """
        cached = getattr(self, "_template_suffix_cache", None)
        if cached is not None:
            return cached
        marker = "XQZV"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": marker}], tokenize=False, add_generation_prompt=False
        )
        tail = rendered.split(marker, 1)[1] if marker in rendered else ""
        sfx = self.tokenizer(tail, add_special_tokens=False)["input_ids"] if tail else []
        self._template_suffix_cache = list(sfx)
        return self._template_suffix_cache

    def _tokenize_target(self, problem_text: str, resp_text: str, view: str) -> torch.Tensor:
        """Tokenize the verified-correct teacher target view for online z^+ recompute.

        loss_type='llm-jepa' (paper-exact view separation): the official finetune.py
        builds the target view from ``messages[2:3]`` — the ASSISTANT MESSAGE ALONE,
        chat-templated separately. The views share NO prompt: Enc(Code) never sees
        the question, and no system prompt is included. We mirror that exactly:
        the target is the teacher response rendered as a lone assistant message
        (whatever the model's chat template emits for it, as in the reference).
        ``problem_text`` is deliberately ignored (and excluded from the cache key,
        which also lets identical texts dedup across prompts).

        Legacy tcr modes keep the prior [x, y^+] framing (code system prompt + user
        message + response) — the policy encodes these ids (last-real-token,
        normalized) at train time; there are no cached latents.
        """
        llm_jepa = self.jepa_cfg.loss_type == "llm-jepa"
        cache_key = ("" if llm_jepa else problem_text, resp_text, view)
        hit = self._tok_target_cache.get(cache_key)
        if hit is not None:
            return hit
        if llm_jepa:
            # finetune.py: assistant_messages = messages[2:3]; apply_chat_template(
            #     assistant_messages, tokenize=False, add_generation_prompt=False)
            atext = self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": resp_text}],
                tokenize=False, add_generation_prompt=False,
            )
            max_len = int(self.jepa_cfg.target_max_length)
            if max_len <= 0:
                max_len = int(self.config.data.max_prompt_length + self.config.data.max_response_length)
            ids = self.tokenizer(atext, return_tensors="pt",
                                 truncation=True, max_length=max_len)["input_ids"][0]
            # Official read index: run.sh sets last_token=-3 for Qwen, i.e. the
            # embedding is read at the LAST CONTENT TOKEN, skipping the constant
            # "<|im_end|>\n" template suffix. Reading the constant suffix token
            # instead makes every target's read nearly identical (anisotropic
            # last-token states) and inflates cosine similarity from step 0.
            # The model is causal, so DROPPING the suffix tokens and reading the
            # (new) last token is bit-identical to their index-offset read.
            # Conditional on an exact suffix match so truncated rows are untouched.
            sfx = self._template_suffix_ids()
            if len(sfx) and ids.shape[0] > len(sfx) and ids[-len(sfx):].tolist() == sfx:
                ids = ids[: ids.shape[0] - len(sfx)]
            self._tok_target_cache[cache_key] = ids
            return ids
        msgs = []
        if view == "code" and self.jepa_cfg.code_system_prompt:
            msgs.append({"role": "system", "content": self.jepa_cfg.code_system_prompt})
        msgs.append({"role": "user", "content": problem_text})
        # Use the same chat-template controls as the student rollout.  In
        # particular, Qwen3 targets must be rendered with enable_thinking=False
        # when the student is trained to produce short non-thinking responses.
        template_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        ptext = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        # Teacher trajectories are a separate, no-grad target view.  Do not cap
        # them to the student's rollout budget: target_max_length lets the target
        # encoder consume the full long-thinking trace in one causal forward.
        # A zero value preserves the former prompt+response behavior.
        max_len = int(self.jepa_cfg.target_max_length)
        if max_len <= 0:
            max_len = int(self.config.data.max_prompt_length + self.config.data.max_response_length)
        ids = self.tokenizer(ptext + resp_text, return_tensors="pt",
                             truncation=True, max_length=max_len)["input_ids"][0]
        self._tok_target_cache[cache_key] = ids
        return ids

    def _build_jepa_batch_tcr_dual(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build the jepa-tcr-dual batch: SEPARATE CoT and Code student anchor
        blocks, each with its OWN precomputed teacher-correct target cache, plus a
        per-CoT-row self-consistency partner pointing at the paired Code row.

        Layout:
          - cot block (A_c rows): correct CoT rollouts of prompts with a stored CoT
            verified-correct text; a parallel `target_input_ids` row holds the
            tokenized [x, y^+] view, encoded online by the policy in the worker.
          - code block (A_k rows): correct Code rollouts of prompts with a stored
            Code verified-correct text; same parallel target row.
          - `self_partner` (A_c,) long: for each CoT row, the row index INTO THE CODE
            BLOCK of its paired (same-prompt, same-slot) code rollout, or -1. The
            worker reads that code row's BOUNDARY embedding as the stop-grad
            self-consistency target z_self.

        Group ids are shared per prompt across both views so the stratified,
        prompt-averaged aggregation lines up. A prompt missing one view's cache
        still contributes the other view's anchors (with no self-consistency pair).
        Returns None if fewer than min_valid_pairs CoT anchors exist.
        """
        assert self.teacher_targets is not None, "CoT teacher target cache not loaded"
        cot_only = self.jepa_cfg.loss_type == "jepa-tcr-cot"
        assert cot_only or self.code_teacher_targets is not None, "Code teacher target cache not loaded"
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)

        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0

        rows_by_uid: dict = defaultdict(lambda: {"cot": [], "code": []})
        idx_by_uid: dict = {}
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v not in ("cot", "code"):
                continue
            rows_by_uid[u][v].append(i)
            if u not in idx_by_uid:
                info = extra_infos[i]
                idx_by_uid[u] = int(info["index"]) if isinstance(info, dict) else None

        anchor_set = self.jepa_cfg.jepa_anchor_set

        def _select(idxs):
            if anchor_set == "correct":
                sel = [i for i in idxs if rew[i] > 0]
            elif anchor_set == "wrong":
                sel = [i for i in idxs if rew[i] <= 0]
            else:  # "all"
                sel = list(idxs)
            return [i for i in sel if int(all_attn_mask[i].sum()) > 0]

        def _match_row(k, n_u):
            return int(torch.randint(n_u, (1,)).item()) if self.jepa_cfg.tcr_match == "random" else k % n_u

        def _problem(i):
            info = extra_infos[i]
            return info["problem"] if isinstance(info, dict) else str(info)

        # Single combined anchor block (CoT rows FIRST, then Code rows) so the
        # DataProto keeps one uniform batch dim even when the two views have
        # different anchor counts. `is_code` recovers the split; `self_partner`
        # (only set on CoT rows) indexes into the CODE sub-block (0..A_k-1).
        cot_ids_list, cot_lengths, cot_tgt_ids, cot_gid, cot_ic = [], [], [], [], []
        code_ids_list, code_lengths, code_tgt_ids, code_gid, code_ic = [], [], [], [], []
        self_pairs: list = []   # (cot_row_pos, code_row_pos) — positions within each sub-block
        n_correct_list = []
        group_counter = 0
        for u in dict.fromkeys(uids):
            ds_idx = idx_by_uid.get(u)
            Z_cot = self.teacher_targets.get(ds_idx) if ds_idx is not None else None
            Z_code = (self.code_teacher_targets.get(ds_idx)
                      if (ds_idx is not None and self.code_teacher_targets is not None) else None)
            cot_sel = _select(rows_by_uid[u]["cot"]) if Z_cot else []
            code_sel = _select(rows_by_uid[u]["code"]) if Z_code else []
            # n_code == 0: no code rollouts exist. Reuse the CoT anchors as the
            # "code" block paired with the CODE teacher cache, turning align_code
            # into a cot->code-teacher pull. These synthesized rows get no
            # self-consistency partner (cos_self self-gates to 0).
            synthesized_code = not code_sel and self.jepa_cfg.n_code == 0 and bool(Z_code)
            if synthesized_code:
                code_sel = list(cot_sel)
            if not cot_sel and not code_sel:
                continue
            gid = group_counter
            group_counter += 1
            n_correct_list.append(sum(1 for i in cot_sel + code_sel if rew[i] > 0))
            n_slots = max(len(cot_sel), len(code_sel))
            for k in range(n_slots):
                cot_pos = code_pos = None
                if k < len(cot_sel):
                    idx = cot_sel[k]
                    cot_pos = len(cot_ids_list)
                    cot_ids_list.append(all_input_ids[idx][all_attn_mask[idx].bool()])
                    cot_lengths.append(int(all_attn_mask[idx].sum()))
                    cot_tgt_ids.append(self._tokenize_target(_problem(idx), Z_cot[_match_row(k, len(Z_cot))], "cot"))
                    cot_gid.append(gid)
                    cot_ic.append(bool(rew[idx] > 0))
                if k < len(code_sel):
                    idx = code_sel[k]
                    code_pos = len(code_ids_list)
                    code_ids_list.append(all_input_ids[idx][all_attn_mask[idx].bool()])
                    code_lengths.append(int(all_attn_mask[idx].sum()))
                    code_tgt_ids.append(self._tokenize_target(_problem(idx), Z_code[_match_row(k, len(Z_code))], "code"))
                    code_gid.append(gid)
                    code_ic.append(bool(rew[idx] > 0))
                if cot_pos is not None and code_pos is not None and not synthesized_code:
                    self_pairs.append((cot_pos, code_pos))

        A_c = len(cot_ids_list)
        A_k = len(code_ids_list)
        # CoT-only mode has no code arm (A_k==0 by construction); dual requires both.
        if A_c < self.jepa_cfg.min_valid_pairs:
            return None
        if A_k == 0 and self.jepa_cfg.loss_type != "jepa-tcr-cot":
            return None

        # Concatenate the two sub-blocks (CoT first) into one padded block.
        all_ids_list = cot_ids_list + code_ids_list
        all_lengths = cot_lengths + code_lengths
        max_len = max(s.shape[0] for s in all_ids_list)
        anchor_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id) for t in all_ids_list
        ])
        anchor_mask = torch.stack([
            torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, max_len - length))
            for length in all_lengths
        ])
        # Parallel verified-correct TARGET block [x, y^+], CoT-first then Code, aligned
        # 1:1 with the anchor block. Encoded online by the policy (worker.jepa_update)
        # into z^+ — no cached latents.
        tgt_ids_all = cot_tgt_ids + code_tgt_ids
        tgt_lengths = [int(t.shape[0]) for t in tgt_ids_all]
        tgt_max = max(tgt_lengths)
        target_input_ids = torch.stack([
            torch.nn.functional.pad(t, (0, tgt_max - t.shape[0]), value=pad_id) for t in tgt_ids_all
        ])
        target_attn_mask = torch.stack([
            torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, tgt_max - length))
            for length in tgt_lengths
        ])
        is_code = torch.tensor([False] * A_c + [True] * A_k, dtype=torch.bool)
        group_id = torch.tensor(cot_gid + code_gid, dtype=torch.long)
        is_correct = torch.tensor(cot_ic + code_ic, dtype=torch.bool)
        # self_partner aligned to the FULL block (length A_c + A_k); set on CoT rows only.
        self_partner = torch.full((A_c + A_k,), -1, dtype=torch.long)
        for c_pos, k_pos in self_pairs:
            self_partner[c_pos] = k_pos   # k_pos is the position within the CODE sub-block

        jepa_batch = DataProto.from_single_dict({
            "anchor_input_ids": anchor_ids,
            "anchor_attn_mask": anchor_mask,
            "anchor_lengths": torch.tensor(all_lengths, dtype=torch.long),
            "target_input_ids": target_input_ids,
            "target_attn_mask": target_attn_mask,
            "target_lengths": torch.tensor(tgt_lengths, dtype=torch.long),
            "is_code": is_code,
            "anchor_group_id": group_id,
            "anchor_is_correct": is_correct,
            "self_partner": self_partner,
        })
        jepa_batch.meta_info["n_correct_cot_mean"] = float(np.mean(n_correct_list)) if n_correct_list else 0.0
        jepa_batch.meta_info["n_anchors"] = A_c
        jepa_batch.meta_info["n_anchors_code"] = A_k
        jepa_batch.meta_info["n_self_pairs"] = len(self_pairs)
        return jepa_batch

    def _build_jepa_batch_llm_jepa(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build the paper-exact LLM-JEPA batch (loss_type='llm-jepa').

        Anchors are the student's CoT rollouts (the only view generated in this
        mode). Each anchor is matched to ONE pregenerated big-teacher CoT text and
        ONE big-teacher Code text (cycled k % n_u, same as the other modes). The
        teacher target rows are DEDUPLICATED here at the trainer: each distinct
        (ds_idx, view, text) is tokenized and shipped exactly once, and anchors
        carry gather indices (tgt_cot_idx / tgt_code_idx) into that unique block.
        The worker encodes the unique block WITH gradient (no stop-grad — paper
        fidelity) inside the same GradCache pool as the anchors.

        A prompt without a cached teacher-CoT text contributes no anchors (the CoT
        arm is mandatory); a missing Code text only masks that anchor's code arm
        (tgt_code_idx = -1). Returns None if fewer than min_valid_pairs anchors.
        """
        assert self.teacher_targets is not None, "CoT teacher target cache not loaded"
        assert self.code_teacher_targets is not None, "Code teacher target cache not loaded"
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)
        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0

        rows_by_uid: dict = defaultdict(list)
        idx_by_uid: dict = {}
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v != "cot":
                continue
            rows_by_uid[u].append(i)
            if u not in idx_by_uid:
                info = extra_infos[i]
                idx_by_uid[u] = int(info["index"]) if isinstance(info, dict) else None

        anchor_set = self.jepa_cfg.jepa_anchor_set

        def _select(idxs):
            if anchor_set == "correct":
                sel = [i for i in idxs if rew[i] > 0]
            elif anchor_set == "wrong":
                sel = [i for i in idxs if rew[i] <= 0]
            else:  # "all"
                sel = list(idxs)
            return [i for i in sel if int(all_attn_mask[i].sum()) > 0]

        def _match_row(k, n_u):
            return int(torch.randint(n_u, (1,)).item()) if self.jepa_cfg.tcr_match == "random" else k % n_u

        def _problem(i):
            info = extra_infos[i]
            return info["problem"] if isinstance(info, dict) else str(info)

        anchors: list[int] = []
        pair_idx: list[tuple[int, int]] = []      # (uniq_cot_pos, uniq_code_pos or -1)
        # Paper-exact view separation: the target view is the teacher response ALONE
        # (no prompt — see _tokenize_target), so dedup keys on (view, text) only:
        # identical teacher texts collapse even across different prompts.
        uniq_pos: dict = {}                       # (view, text) -> pos in uniq block
        uniq_rows: list[torch.Tensor] = []        # tokenized unique target rows

        def _uniq_target(ds_idx: int, view: str, problem: str, text: str) -> int:
            key = (view, text)
            pos = uniq_pos.get(key)
            if pos is None:
                pos = len(uniq_rows)
                uniq_pos[key] = pos
                uniq_rows.append(self._tokenize_target(problem, text, view))
            return pos

        n_correct_list = []
        for u in dict.fromkeys(uids):
            ds_idx = idx_by_uid.get(u)
            if ds_idx is None:
                continue
            Z_cot = self.teacher_targets.get(ds_idx)
            if not Z_cot:
                continue  # CoT teacher view is mandatory for an anchor
            Z_code = self.code_teacher_targets.get(ds_idx)
            sel = _select(rows_by_uid.get(u, []))
            # Paper-parity + speed: the reference trains ONE (Text, Code) pair per
            # example per step; cap the anchors encoded per prompt accordingly.
            cap = int(self.jepa_cfg.max_anchors_per_prompt)
            if cap > 0:
                sel = sel[:cap]
            if not sel:
                continue
            n_correct_list.append(sum(1 for i in sel if rew[i] > 0))
            for k, idx in enumerate(sel):
                problem = _problem(idx)
                cpos = _uniq_target(ds_idx, "cot", problem, Z_cot[_match_row(k, len(Z_cot))])
                kpos = (_uniq_target(ds_idx, "code", problem, Z_code[_match_row(k, len(Z_code))])
                        if Z_code else -1)
                anchors.append(idx)
                pair_idx.append((cpos, kpos))

        A = len(anchors)
        if A < self.jepa_cfg.min_valid_pairs:
            return None

        # DataProto/TensorDict require ONE uniform leading batch dim, but the anchor
        # count A and unique-target count U generally differ. Pad both blocks to
        # R = max(A, U) with zero-LENGTH rows (lengths==0 marks padding; real rows
        # come first). The worker slices real rows back out before the joint
        # forward, so padding rows are never tokenized through the model.
        U = len(uniq_rows)
        R = max(A, U)

        # Anchor block (student CoT rollouts, real tokens only, right-padded).
        # Official [PRED] placement: finetune.py appends the predictor tokens INSIDE
        # the message content, so the template's <|im_end|> comes AFTER them and is
        # causally invisible to the pred read. Our rollout rows end with the eos
        # (<|im_end|>); strip it so the worker-appended [PRED] tokens directly follow
        # the last content token — the exact causal layout of the reference.
        eos_id = self.tokenizer.eos_token_id

        def _strip_eos(t: torch.Tensor) -> torch.Tensor:
            return t[:-1] if (eos_id is not None and t.shape[0] > 1 and int(t[-1]) == eos_id) else t

        ids_list = [_strip_eos(all_input_ids[i][all_attn_mask[i].bool()]) for i in anchors]
        lengths = [int(t.shape[0]) for t in ids_list]
        max_len = max(lengths)
        anchor_ids = torch.full((R, max_len), pad_id, dtype=all_input_ids.dtype)
        anchor_mask = torch.zeros((R, max_len), dtype=torch.long)
        for r, t in enumerate(ids_list):
            anchor_ids[r, : t.shape[0]] = t
            anchor_mask[r, : t.shape[0]] = 1

        # Unique teacher-target block.
        t_lengths = [int(t.shape[0]) for t in uniq_rows]
        t_max = max(t_lengths)
        target_ids = torch.full((R, t_max), pad_id, dtype=all_input_ids.dtype)
        target_mask = torch.zeros((R, t_max), dtype=torch.long)
        for r, t in enumerate(uniq_rows):
            target_ids[r, : t.shape[0]] = t
            target_mask[r, : t.shape[0]] = 1

        jepa_batch = DataProto.from_single_dict({
            "anchor_input_ids": anchor_ids,
            "anchor_attn_mask": anchor_mask,
            "anchor_lengths": torch.tensor(lengths + [0] * (R - A), dtype=torch.long),
            "target_input_ids": target_ids,
            "target_attn_mask": target_mask,
            "target_lengths": torch.tensor(t_lengths + [0] * (R - U), dtype=torch.long),
            "tgt_cot_idx": torch.tensor(
                [c for c, _ in pair_idx] + [-1] * (R - A), dtype=torch.long),
            "tgt_code_idx": torch.tensor(
                [k for _, k in pair_idx] + [-1] * (R - A), dtype=torch.long),
        })
        jepa_batch.meta_info["n_correct_cot_mean"] = float(np.mean(n_correct_list)) if n_correct_list else 0.0
        jepa_batch.meta_info["n_anchors"] = A
        jepa_batch.meta_info["n_anchors_code"] = int(sum(1 for _, k in pair_idx if k >= 0))
        jepa_batch.meta_info["n_unique_targets"] = len(uniq_rows)
        jepa_batch.meta_info["n_self_pairs"] = 0
        return jepa_batch

    def _build_jepa_batch_llm_jepa_contrastive(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build the llm-jepa-contrastive batch (loss_type='llm-jepa-contrastive').

        Three arms, self-gating per prompt (see core_algos.llm_jepa_contrastive_loss):
          A  correct CoT anchors -> pregenerated teacher CoT + Code (arm A, as llm-jepa),
             read via <|predictor_i|>.
          B  a prompt's wrong CoT rollouts: the FIRST becomes a plain-read target, the rest
             become anchors read via <|bad_predictor_i|> and pulled toward it (no stop-grad).
          C  for prompts with both, every (good anchor, bad anchor) pair contributes a
             (1+cos)/2 contrastive push between their reads.

        Ships ONE anchor block (good rows first, then bad rows; `anchor_is_bad` splits) and
        ONE target block (unique teacher rows first, then per-prompt wrong-target rows;
        `target_is_wrong` splits), plus per-anchor gather indices, per-anchor/pair prompt
        `group_id`s (for the per-prompt collapse), and the contrast pair index lists. Both
        blocks are padded to R = max(N_anchor, U_target) with zero-length rows.
        Returns None if fewer than min_valid_pairs good anchors.
        """
        assert self.teacher_targets is not None, "CoT teacher target cache not loaded"
        assert self.code_teacher_targets is not None, "Code teacher target cache not loaded"
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)
        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id
        cap = int(self.jepa_cfg.max_anchors_per_prompt)

        rows_by_uid: dict = defaultdict(list)
        idx_by_uid: dict = {}
        prob_by_uid: dict = {}
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v != "cot":
                continue
            rows_by_uid[u].append(i)
            if u not in idx_by_uid:
                info = extra_infos[i]
                idx_by_uid[u] = int(info["index"]) if isinstance(info, dict) else None
                prob_by_uid[u] = info["problem"] if isinstance(info, dict) else str(info)

        def _strip_eos(t: torch.Tensor) -> torch.Tensor:
            return t[:-1] if (eos_id is not None and t.shape[0] > 1 and int(t[-1]) == eos_id) else t

        def _rollout_ids(i: int) -> torch.Tensor:
            return _strip_eos(all_input_ids[i][all_attn_mask[i].bool()])

        def _match_row(k, n_u):
            return int(torch.randint(n_u, (1,)).item()) if self.jepa_cfg.tcr_match == "random" else k % n_u

        # Unique teacher-target rows (view, text) dedup — identical to _build_jepa_batch_llm_jepa.
        uniq_pos: dict = {}
        uniq_rows: list[torch.Tensor] = []

        def _uniq_target(view: str, problem: str, text: str) -> int:
            key = (view, text)
            pos = uniq_pos.get(key)
            if pos is None:
                pos = len(uniq_rows)
                uniq_pos[key] = pos
                uniq_rows.append(self._tokenize_target(problem, text, view))
            return pos

        good_rows: list[torch.Tensor] = []      # student CoT correct rollouts (anchors)
        good_gather: list[tuple[int, int]] = []  # (teacher cot pos, teacher code pos or -1)
        good_group: list[int] = []
        bad_rows: list[torch.Tensor] = []        # student CoT wrong rollouts (anchors)
        bad_gather: list[int] = []               # index into wrong-target block
        bad_group: list[int] = []
        wrong_rows: list[torch.Tensor] = []      # per-prompt chosen wrong target (plain read)
        contrast_pairs: list[tuple[int, int, int]] = []  # (good pos, bad pos, group)

        pid = 0
        for u in dict.fromkeys(uids):
            rows = [i for i in rows_by_uid.get(u, []) if int(all_attn_mask[i].sum()) > 0]
            correct = [i for i in rows if rew[i] > 0]
            wrong = [i for i in rows if rew[i] <= 0]
            ds_idx = idx_by_uid.get(u)
            problem = prob_by_uid.get(u, "")

            good_local: list[int] = []
            # Arm A: correct anchors -> teacher CoT (+ Code) — needs the CoT cache.
            Z_cot = self.teacher_targets.get(ds_idx) if ds_idx is not None else None
            if Z_cot and correct:
                Z_code = self.code_teacher_targets.get(ds_idx)
                sel = correct[:cap] if cap > 0 else correct
                for k, idx in enumerate(sel):
                    cpos = _uniq_target("cot", problem, Z_cot[_match_row(k, len(Z_cot))])
                    kpos = (_uniq_target("code", problem, Z_code[_match_row(k, len(Z_code))])
                            if Z_code else -1)
                    good_local.append(len(good_rows))
                    good_rows.append(_rollout_ids(idx))
                    good_gather.append((cpos, kpos))
                    good_group.append(pid)

            bad_local: list[int] = []
            # Arm B: >=2 wrong rollouts — first is the plain-read target, rest are anchors.
            if len(wrong) >= 2:
                wpos = len(wrong_rows)
                wrong_rows.append(_rollout_ids(wrong[0]))
                rest = wrong[1:]
                rest = rest[:cap] if cap > 0 else rest
                for idx in rest:
                    bad_local.append(len(bad_rows))
                    bad_rows.append(_rollout_ids(idx))
                    bad_gather.append(wpos)
                    bad_group.append(pid)

            # Arm C: contrast every good x bad read for prompts with both.
            for gp in good_local:
                for bp in bad_local:
                    contrast_pairs.append((gp, bp, pid))

            if good_local or bad_local:
                pid += 1

        Ng, Nb = len(good_rows), len(bad_rows)
        if Ng < self.jepa_cfg.min_valid_pairs:
            return None
        Ut, Uw = len(uniq_rows), len(wrong_rows)
        N, U = Ng + Nb, Ut + Uw
        R = max(N, U)
        Nc = len(contrast_pairs)

        def _pack(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
            lengths = [int(t.shape[0]) for t in rows]
            width = max(lengths) if rows else 1
            ids = torch.full((R, width), pad_id, dtype=all_input_ids.dtype)
            mask = torch.zeros((R, width), dtype=torch.long)
            for r, t in enumerate(rows):
                ids[r, : t.shape[0]] = t
                mask[r, : t.shape[0]] = 1
            return ids, mask, lengths

        anchor_ids, anchor_mask, anchor_len = _pack(good_rows + bad_rows)
        target_ids, target_mask, target_len = _pack(uniq_rows + wrong_rows)

        def _padL(vals: list[int], n_real: int) -> torch.Tensor:
            return torch.tensor(vals + [-1] * (R - n_real), dtype=torch.long)

        jepa_batch = DataProto.from_single_dict({
            "anchor_input_ids": anchor_ids,
            "anchor_attn_mask": anchor_mask,
            "anchor_lengths": torch.tensor(anchor_len + [0] * (R - N), dtype=torch.long),
            "anchor_is_bad": torch.tensor([0] * Ng + [1] * Nb + [-1] * (R - N), dtype=torch.long),
            "anchor_group_id": _padL(good_group + bad_group, N),
            "target_input_ids": target_ids,
            "target_attn_mask": target_mask,
            "target_lengths": torch.tensor(target_len + [0] * (R - U), dtype=torch.long),
            "target_is_wrong": torch.tensor([0] * Ut + [1] * Uw + [-1] * (R - U), dtype=torch.long),
            # good-anchor gather (aligned to the first Ng anchor rows; -1 on bad/pad rows)
            "tgt_cot_idx": _padL([c for c, _ in good_gather] + [-1] * Nb, N),
            "tgt_code_idx": _padL([k for _, k in good_gather] + [-1] * Nb, N),
            # bad-anchor gather into the wrong-target block (-1 on good/pad rows)
            "bad_tgt_idx": _padL([-1] * Ng + bad_gather, N),
            # contrast pair index lists (into the good / bad blocks) + group id
            "contrast_good_idx": _padL([g for g, _, _ in contrast_pairs], Nc),
            "contrast_bad_idx": _padL([b for _, b, _ in contrast_pairs], Nc),
            "contrast_group_id": _padL([g for _, _, g in contrast_pairs], Nc),
        })
        jepa_batch.meta_info["n_anchors"] = Ng
        jepa_batch.meta_info["n_bad_anchors"] = Nb
        jepa_batch.meta_info["n_contrast_pairs"] = Nc
        jepa_batch.meta_info["n_unique_targets"] = Ut
        jepa_batch.meta_info["n_wrong_targets"] = Uw
        return jepa_batch

    def _build_jepa_batch_llm_jepa_infonce(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build the llm-jepa-infoNCE batch (loss_type='llm-jepa-infoNCE').

        Data-selection rule (not loss weighting): only MIXED prompts — those with at
        least one correct AND one wrong CoT rollout — contribute. Each mixed prompt
        ships min(n_correct, n_wrong) disjoint (y+, y-) pairs (uniform random
        matching), all sharing one uniformly sampled verified teacher CoT solution
        t+_cot and, when the code teacher cache has one, one uniformly sampled
        teacher Code solution t+_code (second anchored view in the loss). All-good
        and all-wrong prompts ship nothing (they still feed GRPO).

        Ships ONE anchor block of 2M rows — the M positive rollouts first (read via
        <|predictor_i|>), then their M negatives (read via <|bad_predictor_i|>) — and
        ONE target block
        of deduplicated teacher rows (CoT and Code mixed) with per-pair gather
        indices (tgt_cot_idx, tgt_code_idx; -1 = no code view). The worker encodes
        targets with the EMA encoder under no_grad (z+ = sg(f_EMA)).
        Returns None if fewer than min_valid_pairs mixed prompts.
        """
        assert self.teacher_targets is not None, "CoT teacher target cache not loaded"
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)
        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id

        rows_by_uid: dict = defaultdict(list)
        idx_by_uid: dict = {}
        prob_by_uid: dict = {}
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v != "cot":
                continue
            rows_by_uid[u].append(i)
            if u not in idx_by_uid:
                info = extra_infos[i]
                idx_by_uid[u] = int(info["index"]) if isinstance(info, dict) else None
                prob_by_uid[u] = info["problem"] if isinstance(info, dict) else str(info)

        def _strip_eos(t: torch.Tensor) -> torch.Tensor:
            return t[:-1] if (eos_id is not None and t.shape[0] > 1 and int(t[-1]) == eos_id) else t

        def _rollout_ids(i: int) -> torch.Tensor:
            return _strip_eos(all_input_ids[i][all_attn_mask[i].bool()])

        def _pick(idxs: list) -> int:
            return idxs[int(torch.randint(len(idxs), (1,)).item())]

        # Unique teacher-target rows, (view, text) dedup — as in the sibling builders.
        uniq_pos: dict = {}
        uniq_rows: list[torch.Tensor] = []
        pos_rows: list[torch.Tensor] = []   # y+ per mixed prompt
        neg_rows: list[torch.Tensor] = []   # y- per mixed prompt
        tgt_idx: list[int] = []             # teacher CoT row per pair
        tgt_code: list[int] = []            # teacher Code row per pair (-1 = none)
        n_mixed_prompts = 0

        def _uniq(view: str, problem: str, text: str) -> int:
            key = (view, text)
            pos = uniq_pos.get(key)
            if pos is None:
                pos = len(uniq_rows)
                uniq_pos[key] = pos
                uniq_rows.append(self._tokenize_target(problem, text, view))
            return pos

        for u in dict.fromkeys(uids):
            rows = [i for i in rows_by_uid.get(u, []) if int(all_attn_mask[i].sum()) > 0]
            correct = [i for i in rows if rew[i] > 0]
            wrong = [i for i in rows if rew[i] <= 0]
            if not correct or not wrong:
                continue  # mixed prompts only
            ds_idx = idx_by_uid.get(u)
            Z_cot = self.teacher_targets.get(ds_idx) if ds_idx is not None else None
            if not Z_cot:
                continue
            problem = prob_by_uid.get(u, "")
            pos = _uniq("cot", problem, Z_cot[_pick(range(len(Z_cot)))])
            code_targets = getattr(self, "code_teacher_targets", None)
            Z_code = code_targets.get(ds_idx) if code_targets else None
            cpos = _uniq("code", problem, Z_code[_pick(range(len(Z_code)))]) if Z_code else -1
            k = min(len(correct), len(wrong))
            cperm = [correct[j] for j in torch.randperm(len(correct)).tolist()[:k]]
            wperm = [wrong[j] for j in torch.randperm(len(wrong)).tolist()[:k]]
            for ci, wi in zip(cperm, wperm):
                pos_rows.append(_rollout_ids(ci))
                neg_rows.append(_rollout_ids(wi))
                tgt_idx.append(pos)
                tgt_code.append(cpos)
            n_mixed_prompts += 1

        M = len(pos_rows)
        if M < self.jepa_cfg.min_valid_pairs:
            return None
        U = len(uniq_rows)
        N = 2 * M
        R = max(N, U)

        def _pack(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
            lengths = [int(t.shape[0]) for t in rows]
            width = max(lengths) if rows else 1
            ids = torch.full((R, width), pad_id, dtype=all_input_ids.dtype)
            mask = torch.zeros((R, width), dtype=torch.long)
            for r, t in enumerate(rows):
                ids[r, : t.shape[0]] = t
                mask[r, : t.shape[0]] = 1
            return ids, mask, lengths

        anchor_ids, anchor_mask, anchor_len = _pack(pos_rows + neg_rows)
        target_ids, target_mask, target_len = _pack(uniq_rows)

        jepa_batch = DataProto.from_single_dict({
            "anchor_input_ids": anchor_ids,
            "anchor_attn_mask": anchor_mask,
            "anchor_lengths": torch.tensor(anchor_len + [0] * (R - N), dtype=torch.long),
            "target_input_ids": target_ids,
            "target_attn_mask": target_mask,
            "target_lengths": torch.tensor(target_len + [0] * (R - U), dtype=torch.long),
            "tgt_cot_idx": torch.tensor(tgt_idx + [-1] * (R - M), dtype=torch.long),
            "tgt_code_idx": torch.tensor(tgt_code + [-1] * (R - M), dtype=torch.long),
        })
        jepa_batch.meta_info["n_anchors"] = M
        jepa_batch.meta_info["n_unique_targets"] = U
        jepa_batch.meta_info["n_mixed_prompts"] = n_mixed_prompts
        jepa_batch.meta_info["pairs_per_prompt"] = M / max(n_mixed_prompts, 1)
        return jepa_batch

    # ------------------------------------------- TCR reward shaping (idea #2) -
    @staticmethod
    def _stratified_shaping(
        s: torch.Tensor,
        group_ids: list,
        is_correct: torch.Tensor,
        beta: float,
        sigma_floor: float,
    ) -> tuple[torch.Tensor, int]:
        """Within-(group, is_correct)-stratum standardized shaping term β·ŝ.

        For each (group_id, correctness) stratum with >=2 members, standardize the
        scores `s` (mean 0, std clamped at `sigma_floor`) and scale by β. Singleton
        or degenerate strata contribute 0. Returns (shaped (n,), n_strata_shaped).

        Pure function of its tensors (no model/state) so it is unit-testable. Key
        invariants it guarantees: (a) a constant added to all of one stratum's
        scores leaves ŝ unchanged (global-shift invariance); (b) shaping is computed
        independently per correctness stratum, so it never moves mass across the
        correct/wrong boundary.
        """
        n = s.shape[0]
        shaped = torch.zeros(n, dtype=torch.float32)
        strata: dict = defaultdict(list)
        for j in range(n):
            strata[(group_ids[j], bool(is_correct[j]))].append(j)
        n_shaped = 0
        for members in strata.values():
            if len(members) < 2:
                continue
            idx = torch.tensor(members)
            sv = s[idx].float()
            shat = (sv - sv.mean()) / max(float(sv.std(unbiased=False)), sigma_floor)
            for m, val in zip(members, shat):
                shaped[m] = beta * float(val)
            n_shaped += 1
        return shaped, n_shaped

    def _compute_tcr_reward_shaping(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> tuple[torch.Tensor, dict]:
        """jepa-tcr-dual teacher-alignment reward shaping (folded into token_level_rewards).

        Scores every CoT *and* Code rollout's [PRED] latent against its question's
        verified-correct targets (recomputed online by the current policy from the
        stored y^+ text, predictor_k=0), standardizes the score WITHIN its
        (uid, view, is_correct) reward stratum, and returns a per-row additive
        reward term β·ŝ_i. Both views are shaped symmetrically with the dual-view
        JEPA loss arm.

        Within-stratum centering makes the term (a) invariant to a global latent
        shift (defeating the cos_correct≈cos_wrong shortcut) and (b) unable to flip
        a correct-vs-wrong ordering (ground truth always dominates). Rows whose
        prompt has no cached target, or singleton/degenerate strata, get 0.

        Returns:
            (shape_per_row, metrics): shape_per_row is a (B,) CPU float tensor
            aligned 1:1 with `batch` rows (0 for non-cot / unshaped rows).
        """
        assert self.teacher_targets is not None, "teacher target cache not loaded"
        B = len(batch)
        shape_per_row = torch.zeros(B, dtype=torch.float32)
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)
        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0

        # Per-view target cache: CODE rows are scored against the CODE teacher cache and
        # COT rows against the COT cache (each rollout aligned to its own view's teacher).
        code_cache = self.code_teacher_targets

        def _cache_for(view: str):
            return code_cache if (view == "code" and code_cache is not None) else self.teacher_targets

        # Collect all CoT *and* Code rows whose prompt has cached targets. Both views
        # are scored against the teacher and shaped, mirroring the dual-view JEPA arm
        # (a code rollout that lands near the teacher should earn the same advantage
        # bonus a CoT one does). View is tracked so each view is standardized within
        # its own (uid, view, is_correct) stratum below.
        scored_rows: list[int] = []
        scored_views: list[str] = []
        for i, v in enumerate(view_tags):
            if v not in ("cot", "code") or int(all_attn_mask[i].sum()) == 0:
                continue
            info = extra_infos[i]
            ds_idx = int(info["index"]) if isinstance(info, dict) else None
            if ds_idx is None or _cache_for(v).get(ds_idx) is None:
                continue
            scored_rows.append(i)
            scored_views.append(v)
        if len(scored_rows) < self.jepa_cfg.min_valid_pairs:
            return shape_per_row, {"shaping/frac_groups_shaped": 0.0, "shaping/n_rows_scored": 0.0}

        # Forward-only [PRED] embeddings for the scored rows (worker RPC, no backward).
        ids_list = [all_input_ids[i][all_attn_mask[i].bool()] for i in scored_rows]
        lengths = [int(all_attn_mask[i].sum()) for i in scored_rows]
        max_len = max(s.shape[0] for s in ids_list)
        padded_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id) for t in ids_list
        ])
        padded_mask = torch.stack([
            torch.nn.functional.pad(torch.ones(L, dtype=torch.long), (0, max_len - L)) for L in lengths
        ])
        score_td = DataProto.from_single_dict({
            "cot_input_ids": padded_ids,
            "cot_attn_mask": padded_mask,
            "cot_lengths": torch.tensor(lengths, dtype=torch.long),
        }).to_tensordict()
        out = self.actor_rollout_wg.score_cot_embeddings(score_td)
        if isinstance(out, list):
            out = out[0] if out else None
        emb = out["cot_emb"].float()  # (n_rows, d), L2-normalized

        # Recompute this batch's verified-correct target embeddings z^+ ONLINE (current
        # policy, predictor_k=0), replacing the old cached frozen-ref latents. Each unique
        # (ds_idx, view) target-text set is tokenized and embedded once via the worker.
        def _pf(i):
            info = extra_infos[i]
            return info["problem"] if isinstance(info, dict) else str(info)

        flat_target_ids: list = []
        slots: dict = {}   # (ds_idx, view) -> (start, end) into flat_target_ids
        for j, i in enumerate(scored_rows):
            key = (int(extra_infos[i]["index"]), scored_views[j])
            if key in slots:
                continue
            start = len(flat_target_ids)
            for t in _cache_for(key[1])[key[0]]:
                flat_target_ids.append(self._tokenize_target(_pf(i), t, key[1]))
            slots[key] = (start, len(flat_target_ids))

        t_lengths = [int(t.shape[0]) for t in flat_target_ids]
        t_max = max(t_lengths)
        t_ids = torch.stack([
            torch.nn.functional.pad(t, (0, t_max - t.shape[0]), value=pad_id) for t in flat_target_ids
        ])
        t_mask = torch.stack([
            torch.nn.functional.pad(torch.ones(L, dtype=torch.long), (0, t_max - L)) for L in t_lengths
        ])
        tgt_td = DataProto.from_single_dict({
            "target_input_ids": t_ids,
            "target_attn_mask": t_mask,
            "target_lengths": torch.tensor(t_lengths, dtype=torch.long),
        }).to_tensordict()
        tout = self.actor_rollout_wg.embed_targets(tgt_td)
        if isinstance(tout, list):
            tout = tout[0] if tout else None
        Z_all = tout["target_emb"].float()  # (M, d), L2-normalized

        # s_i = max_k <p_i, z^+_k> over the question's online targets.
        s = torch.empty(len(scored_rows), dtype=torch.float32)
        is_correct = torch.empty(len(scored_rows), dtype=torch.bool)
        gids = []
        for j, i in enumerate(scored_rows):
            a, b = slots[(int(extra_infos[i]["index"]), scored_views[j])]
            Z = Z_all[a:b].to(emb.dtype)
            s[j] = (emb[j].unsqueeze(0) * Z).sum(dim=-1).max()
            is_correct[j] = bool(rew[i] > 0)
            # Stratum key includes the view so CoT and Code are standardized against
            # their own kind (their similarity-to-teacher scales differ).
            gids.append((uids[i], scored_views[j]))

        # Standardize within each (uid, view, is_correct) stratum, then scale by β.
        beta = float(self.jepa_cfg.tcr_reward_beta)
        sigma_floor = float(self.jepa_cfg.tcr_reward_sigma_floor)
        shaped, n_shaped_groups = self._stratified_shaping(
            s, gids, is_correct, beta=beta, sigma_floor=sigma_floor,
        )
        for j in range(len(scored_rows)):
            shape_per_row[scored_rows[j]] = float(shaped[j])

        # Monitors (do not affect the optimized objective).
        s_np = s.numpy()
        corr = 0.0
        if is_correct.any() and (~is_correct).any():
            corr = float(np.corrcoef(s_np, is_correct.numpy().astype(np.float32))[0, 1])
        metrics = {
            "shaping/corr_s_correct": corr,
            "shaping/s_mean_correct": float(s[is_correct].mean()) if is_correct.any() else 0.0,
            "shaping/s_mean_wrong": float(s[~is_correct].mean()) if (~is_correct).any() else 0.0,
            "shaping/adv_std": float(shape_per_row[shape_per_row != 0].std(unbiased=False)) if (shape_per_row != 0).any() else 0.0,
            "shaping/n_rows_scored": float(len(scored_rows)),
            "shaping/n_rows_scored_cot": float(scored_views.count("cot")),
            "shaping/n_rows_scored_code": float(scored_views.count("code")),
            "shaping/n_strata_shaped": float(n_shaped_groups),
            "shaping/frac_rows_shaped": float((shape_per_row != 0).sum()) / max(1, len(scored_rows)),
        }
        return shape_per_row, metrics

    # --------------------------------------- auto-disable on alignment plateau -
    def _tracked_signals(self) -> list[tuple[str, str]]:
        """(arm_name, metric_key) pairs to track for the plateau latch (higher=better).

        jepa-tcr-dual tracks its three arms (cot/code/self) independently. An explicit
        `auto_off_metric` override collapses to one global signal (legacy).

        Arms whose view is not rolled out are excluded: with n_code=0 the code and
        self arms never produce a measurement, so tracking them would make the
        global all-arms-off latch unreachable (JEPA could never fully disable).
        """
        if self.jepa_cfg.auto_off_metric:
            return [("global", self.jepa_cfg.auto_off_metric)]
        signals = []
        if self.jepa_cfg.n_cot > 0:
            signals.append(("cot", "jepa/cos_cot"))
        # llm-jepa has a code ARM without code ROLLOUTS: the student CoT predicts the
        # pregenerated teacher Code view, so cos_code is a real signal even at n_code=0.
        llm_jepa_code_arm = (self.jepa_cfg.loss_type == "llm-jepa"
                             and self.code_teacher_targets is not None)
        if self.jepa_cfg.n_code > 0 or llm_jepa_code_arm:
            signals.append(("code", "jepa/cos_code"))
        if self.jepa_cfg.n_cot > 0 and self.jepa_cfg.n_code > 0:
            signals.append(("self", "jepa/cos_self"))
        return signals

    def _maybe_disable_jepa_signal(self, metrics: dict) -> None:
        """Latch each tracked JEPA signal OFF independently once it plateaus.

        For each tracked (arm, metric): after warmup, a step that fails to beat that
        arm's running best by `auto_off_min_delta` increments its own stall counter;
        once it reaches `auto_off_patience`, only THAT arm latches off for the rest of
        training (its loss term + matching-view shaping go to zero). The global
        `_jepa_signal_off` latches True only once EVERY tracked arm is off, skipping the
        whole JEPA block. Steps where a metric is absent/zero (no anchors of that view)
        are ignored so they neither advance nor reset that arm's counter.
        """
        cfg = self.jepa_cfg
        if not cfg.enable or not cfg.auto_off_enable:
            return
        # llm-jepa-contrastive must never be auto-disabled: this plateau latch watches
        # only the cot/code ALIGNMENT cosines and is blind to the contrastive + SIGReg
        # arms, so it wrongly euthanizes JEPA the moment alignment plateaus (~step 87 in
        # the smoke run) even though the contrastive/anti-collapse signal is still active.
        # llm-jepa-infoNCE likewise: cos_cot plateauing says nothing about the
        # separation half of the softmax competition.
        if cfg.loss_type in ("llm-jepa-contrastive", "llm-jepa-infoNCE"):
            metrics["jepa/signal_off"] = 0.0
            return
        metrics["jepa/signal_off"] = float(self._jepa_signal_off)
        if self._jepa_signal_off:
            return
        signals = self._tracked_signals()
        warm = self.global_steps >= cfg.auto_off_warmup_steps
        for arm, key in signals:
            metrics[f"jepa/off_{arm}"] = float(self._off_arm[arm])
            if not warm or self._off_arm[arm]:
                continue
            val = metrics.get(key)
            if val is None:
                continue
            val = float(val)
            if val == 0.0:   # no anchors of this view this step -> not a real measurement
                continue
            # Gate: plateau counter does not start until this arm has crossed the
            # minimum cosine threshold (auto_off_min_cos). Keeps the latch from
            # firing during the early ramp before alignment is established.
            if cfg.auto_off_min_cos > 0.0 and val < cfg.auto_off_min_cos:
                metrics[f"jepa/align_best_{arm}"] = self._align_best[arm]
                metrics[f"jepa/align_stall_{arm}"] = float(self._align_stall[arm])
                continue
            if val > self._align_best[arm] + cfg.auto_off_min_delta:
                self._align_best[arm] = val
                self._align_stall[arm] = 0
            else:
                self._align_stall[arm] += 1
                if self._align_stall[arm] >= cfg.auto_off_patience:
                    self._off_arm[arm] = True
                    logger.info(
                        "[jepa] arm '%s' plateaued on %s (val=%.4f best=%.4f, stalled %d>=%d steps) "
                        "-> disabling this arm (loss term + matching-view shaping)",
                        arm, key, val, self._align_best[arm], self._align_stall[arm], cfg.auto_off_patience,
                    )
            metrics[f"jepa/off_{arm}"] = float(self._off_arm[arm])
            metrics[f"jepa/align_best_{arm}"] = self._align_best[arm]
            metrics[f"jepa/align_stall_{arm}"] = float(self._align_stall[arm])
        # Fully off only when every tracked arm has individually latched.
        if signals and all(self._off_arm[arm] for arm, _ in signals):
            if not self._jepa_signal_off:
                logger.info("[jepa] all tracked arms disabled -> JEPA signal fully off for the rest of training")
            self._jepa_signal_off = True
        metrics["jepa/signal_off"] = float(self._jepa_signal_off)

    # --------------------------------------- latch state checkpointing -------
    # The plateau latches live on the trainer object; without persisting them a
    # resume silently re-enables JEPA until it re-plateaus (warmup + patience).
    _LATCH_FILE = "jepa_latch.json"

    def _save_checkpoint(self):
        super()._save_checkpoint()
        step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        state = {
            "jepa_signal_off": self._jepa_signal_off,
            "off_arm": self._off_arm,
            # -inf is not valid JSON; store None for "no best yet"
            "align_best": {k: (None if v == float("-inf") else v) for k, v in self._align_best.items()},
            "align_stall": self._align_stall,
        }
        try:
            with open(os.path.join(step_folder, self._LATCH_FILE), "w") as f:
                json.dump(state, f, indent=1)
        except OSError as e:
            logger.warning("[jepa] could not save latch state: %s", e)

    def _load_checkpoint(self):
        ret = super()._load_checkpoint()
        if self.global_steps > 0:
            latch_path = os.path.join(
                self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", self._LATCH_FILE
            )
            if os.path.exists(latch_path):
                with open(latch_path) as f:
                    state = json.load(f)
                self._jepa_signal_off = bool(state["jepa_signal_off"])
                self._off_arm.update({k: bool(v) for k, v in state["off_arm"].items()})
                self._align_best.update(
                    {k: (float("-inf") if v is None else float(v)) for k, v in state["align_best"].items()}
                )
                self._align_stall.update({k: int(v) for k, v in state["align_stall"].items()})
                logger.info(
                    "[jepa] restored latch state from %s (signal_off=%s, off_arm=%s)",
                    latch_path, self._jepa_signal_off, self._off_arm,
                )
        return ret

    # ------------------------------------------------ memory diagnostics -----
    @staticmethod
    def _log_gpu_mem(tag: str) -> None:
        """Log device-level GPU memory (used/total MiB) at a phase boundary.

        Uses `nvidia-smi` rather than torch so the driver process does NOT create
        a CUDA context (which would itself consume GPU memory on this memory-tight
        colocated setup). Device-level used memory captures BOTH the vLLM EngineCore
        and the FSDP-actor worker processes. Best-effort: never raises into the loop.

        Opt-in via JEPA_LOG_GPU_MEM=1: five nvidia-smi subprocesses per step cost
        ~0.5-1 s of wall clock, only worth paying when debugging memory.
        """
        if os.environ.get("JEPA_LOG_GPU_MEM", "0") != "1":
            return
        try:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            logger.info("[jepa-gpu-mem] %s: %s MiB (used,total per GPU)", tag, out.replace("\n", " | "))
        except Exception as e:  # noqa: BLE001 - diagnostics must never break training
            logger.warning("[jepa-gpu-mem] %s: snapshot failed (%s)", tag, e)

    # ---------------------------------------------------- training loop -----
    def fit(self):
        """JEPA-GRPO training loop.

        Compared to RayPPOTrainer.fit(), after each GRPO actor update we:
          1. Generate Code-view rollouts for the same prompts
          2. Score correctness with the math reward function
          3. Build JEPA pairs and run jepa_update on the worker
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        from verl.trainer.ppo.ray_trainer import compute_advantage, compute_response_mask
        from verl.trainer.ppo.reward import extract_reward
        from verl.utils.metric import reduce_metrics

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)
            self._maybe_save_best_checkpoint(val_metrics)
            if self.config.trainer.get("val_only", False):
                return

        from tqdm import tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps,
                            desc="JEPA-GRPO Training")

        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict[str, Any] = {}
                timing_raw: dict[str, float] = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # ── Step 1: unified CoT + Code rollout ──────────────────
                # Both views now contribute to the GRPO update below (the old
                # code generated a SEPARATE rollout_n-sized Code batch later,
                # whose reward only ever fed JEPA pairing — half the rollout
                # compute never reached the policy gradient). GRPO's grouping
                # is uid-based (compute_grpo_outcome_advantage groups by
                # data.non_tensor_batch["uid"], not by position), so the cot
                # and code sub-batches just need matching uids per prompt —
                # no positional interleaving is required.
                with simple_timer("cot_gen", timing_raw):
                    rollout_n = self.config.actor_rollout_ref.rollout.n
                    n_cot = self.jepa_cfg.n_cot
                    n_code = self.jepa_cfg.n_code

                    sub_batches = []

                    if n_cot > 0:
                        gen_batch_cot = self._get_gen_batch(batch)
                        gen_batch_cot.meta_info["global_steps"] = self.global_steps
                        gen_batch_cot_rep = gen_batch_cot.repeat(repeat_times=n_cot, interleave=True)
                        cot_gen_output = self.async_rollout_manager.generate_sequences(gen_batch_cot_rep)
                        # Each generate_sequences call attaches its own per-call "timing"
                        # diagnostics dict to meta_info. Pop+merge into timing_raw (the
                        # pattern used elsewhere in verl, e.g. ray_trainer.py's
                        # `timing_raw.update(combined_gen_output.meta_info["timing"])`)
                        # instead of letting it ride along on the DataProto — otherwise it
                        # collides later: cot's and code's "timing" dicts are different
                        # objects, and both union() and DataProto.concat() assert equality
                        # on overlapping non-"metrics" meta_info keys.
                        if "timing" in cot_gen_output.meta_info:
                            timing_raw.update(
                                {f"cot_gen/{k}": v for k, v in cot_gen_output.meta_info.pop("timing").items()}
                            )
                        # AgentLoop echoes raw_prompt back into its output. cot_sub is unioned
                        # onto `batch.repeat(...)` (not onto a gen_batch that itself carries
                        # raw_prompt — unlike the old code), so if only ONE of cot/code drops
                        # this key, DataProto.concat ends up with a non_tensor_batch entry whose
                        # length matches just one sub instead of the full combined batch size.
                        # Drop it symmetrically from both; nothing downstream reads it.
                        cot_gen_output.non_tensor_batch.pop("raw_prompt", None)

                        cot_sub = batch.repeat(repeat_times=n_cot, interleave=True)
                        # .repeat() passes meta_info by reference (verl/protocol.py), so
                        # cot_sub.meta_info IS batch.meta_info here — decouple with a shallow
                        # copy before union() mutates it, so this sub's union() can't leak
                        # into the shared `batch` object (and from there into code_sub below).
                        cot_sub.meta_info = dict(cot_sub.meta_info)
                        cot_sub = cot_sub.union(cot_gen_output)
                        cot_sub.non_tensor_batch["view"] = np.array(["cot"] * len(cot_sub), dtype=object)
                        sub_batches.append(cot_sub)

                    if n_cot > 0 and n_code > 0:
                        # Issuing a second vLLM engine RPC immediately after generate_sequences()
                        # returns reproducibly segfaults vLLM's executor (cuMemcpy) at the start
                        # of the first real training step — reproduced 3x, including once where
                        # the "second RPC" was a checkpoint_manager.sleep_replicas() call (NOT
                        # another generation), so this isn't specific to generation-vs-generation;
                        # it's specific to back-to-back vLLM RPCs with no real wall-clock gap.
                        # generate_sequences() is wrapped in asyncio.run (verl/utils/ray_utils.py),
                        # which should block until fully complete, but empirically something in
                        # vLLM's async engine (e.g. background request/KV-cache bookkeeping) is
                        # still settling when the next RPC submits new CUDA work. The old code
                        # never hit this because substantial real CPU/FSDP work (reward,
                        # advantage, old-logprob) always sat between its two generate_sequences
                        # calls — never back-to-back. A plain delay is a blunt instrument, but
                        # it's the minimal, lowest-risk way to give vLLM's async state time to
                        # drain without restructuring the data flow (see git history for two
                        # real bugs introduced by data-flow changes in this same unification).
                        import time as _time
                        _time.sleep(float(os.environ.get("JEPA_VLLM_DRAIN_S", "5")))

                    if n_code > 0:
                        code_gen_batch = self._tokenize_code_prompts(batch)  # batch is still un-repeated here
                        code_gen_batch.meta_info["global_steps"] = self.global_steps
                        code_gen_batch_rep = code_gen_batch.repeat(repeat_times=n_code, interleave=True)
                        code_gen_output = self.async_rollout_manager.generate_sequences(code_gen_batch_rep)
                        # See cot_gen_output comment above — drop symmetrically from both views.
                        code_gen_output.non_tensor_batch.pop("raw_prompt", None)
                        # See cot_gen_output comment above — pop+merge "timing" instead of
                        # letting it collide with cot's (different) "timing" dict at union/concat.
                        if "timing" in code_gen_output.meta_info:
                            timing_raw.update(
                                {f"code_gen/{k}": v for k, v in code_gen_output.meta_info.pop("timing").items()}
                            )

                        code_sub = batch.repeat(repeat_times=n_code, interleave=True)
                        code_sub.meta_info = dict(code_sub.meta_info)  # see cot_sub comment above
                        code_sub = code_sub.union(code_gen_output)
                        code_sub.non_tensor_batch["view"] = np.array(["code"] * len(code_sub), dtype=object)
                        sub_batches.append(code_sub)

                    batch = DataProto.concat(sub_batches)
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                # ── Step 2: rewards (full cot+code batch) ──
                # Advantage is computed later (Step 2.7), AFTER TCR shaping is folded into
                # token_level_rewards, so the shaping flows through the same per-group
                # standardization as the raw reward.
                with simple_timer("cot_reward", timing_raw):
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        batch = batch.union(self._compute_reward_colocate(batch))
                    reward_tensor, reward_extra_infos = extract_reward(batch)
                    if reward_extra_infos:
                        for key, values in reward_extra_infos.items():
                            if len(values) == len(batch):
                                batch.non_tensor_batch[key] = np.array(values, dtype=object)
                    batch.batch["token_level_scores"] = reward_tensor
                    if not self.config.algorithm.use_kl_in_reward:
                        # clone(): Step 2.5's TCR shaping adds beta*s_hat to
                        # token_level_rewards IN-PLACE; without the copy that
                        # write aliases token_level_scores AND reward_tensor,
                        # contaminating cot/code pass_at_1, avg_reward and the
                        # JEPA "correct"-anchor selection (rew > 0) downstream.
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"].clone()

                # ── Step 2.2: DAPO dynamic sampling (group filter) ──────
                # The dataloader yields gen_batch_size prompts; keep only up to
                # train_batch_size "mixed" uid-groups (>=1 correct AND >=1 wrong
                # across the group's 4 CoT + 4 Code rows) — same
                # _filter_dapo_groups the parent fit applies. Without this the
                # step trained on the full unfiltered gen batch: ~70% zero-
                # advantage all-wrong groups AND >train_batch_size groups,
                # silently violating fuse_old_log_prob's single-optimizer-step
                # assumption. Raw pre-filter rates are logged as train_raw/* so
                # true model progress stays visible next to the (by-construction
                # higher) post-filter train/accuracy.
                with simple_timer("dapo_filter", timing_raw):
                    raw_acc_col = batch.non_tensor_batch.get("acc", None)
                    if raw_acc_col is not None and len(raw_acc_col) == len(batch):
                        raw_correct = np.array([float(a) > 0.0 if a is not None else False for a in raw_acc_col])
                    else:
                        raw_correct = (reward_tensor.sum(dim=-1) > 0).cpu().numpy()
                    raw_uids = batch.non_tensor_batch["uid"]
                    uid_any_correct: dict = {}
                    for u, c in zip(raw_uids, raw_correct):
                        uid_any_correct[u] = uid_any_correct.get(u, False) or bool(c)
                    metrics["train_raw/accuracy"] = float(raw_correct.mean()) if len(raw_correct) else 0.0
                    metrics["train_raw/pass_at_8"] = (
                        float(sum(uid_any_correct.values()) / len(uid_any_correct)) if uid_any_correct else 0.0
                    )
                    metrics["train_raw/response_count"] = float(len(batch))

                    reward_extra_infos_dict = {
                        k: list(batch.non_tensor_batch[k]) for k in ("acc",) if k in batch.non_tensor_batch
                    }
                    batch, reward_tensor, reward_extra_infos_dict, dapo_filter_metrics = self._filter_dapo_groups(
                        batch, reward_tensor, reward_extra_infos_dict
                    )
                    metrics.update(dapo_filter_metrics)

                # Sleep rollout replicas BEFORE the TCR-shaping forward (Step 2.5) and
                # old_log_prob so vLLM releases its KV reservation before the actor runs the
                # forward-only embedding pass / recomputes log-probs (the ~7 GiB lm_head logits
                # block). Otherwise that block is allocated while vLLM still holds its full KV
                # pool -> OOM near the card ceiling ("Tried to allocate 6.90 GiB ... 92.14 GiB
                # in use"). Called exactly once per step; vLLM is woken again only at weight_sync_2.
                #
                # DRAIN GUARD: issuing a vLLM RPC (sleep_replicas IS one) immediately after
                # generate_sequences() reproducibly segfaults vLLM's executor when there is no
                # real wall-clock gap (see the same-class issue + _time.sleep(5) guard in the
                # Step-1 cot/code generation block). Only the reward block (~15 ms) now sits
                # between generation and the sleep, so we reinstate the documented short drain
                # before the sleep RPC.
                self._log_gpu_mem("after_gen_before_sleep")
                import time as _time
                _time.sleep(float(os.environ.get("JEPA_VLLM_DRAIN_S", "5")))
                with simple_timer("sleep_replicas_1", timing_raw):
                    self.checkpoint_manager.sleep_replicas()
                self._log_gpu_mem("after_sleep")

                # ── Step 2.5: TCR reward shaping (idea #2) ───────────────
                # Fold teacher-alignment β·ŝ into token_level_rewards BEFORE compute_advantage,
                # so the signal is standardized per group together with the raw reward — rather
                # than riding post-hoc on the already-normalized advantage (which also double-
                # counted under DAPO's norm_adv_by_std_in_grpo). vLLM is asleep and the actor
                # FSDP is warm here, so the forward-only embedding pass is memory-safe.
                # beta == 0 disables shaping mathematically; skip the whole
                # forward-only embedding pass it would otherwise still pay for.
                if self.jepa_cfg.enable and not self._jepa_signal_off and float(self.jepa_cfg.tcr_reward_beta) != 0.0:
                    with simple_timer("tcr_reward_shaping", timing_raw):
                        view_tags_s = batch.non_tensor_batch["view"]
                        shape_per_row, shaping_metrics = self._compute_tcr_reward_shaping(
                            batch=batch, reward_tensor=reward_tensor, view_tags=view_tags_s,
                        )
                        # Per-view plateau latch: zero shaping for a view whose arm is off.
                        if self._off_arm["cot"]:
                            shape_per_row[torch.from_numpy(view_tags_s == "cot")] = 0.0
                        if self._off_arm["code"]:
                            shape_per_row[torch.from_numpy(view_tags_s == "code")] = 0.0
                        # Add β·ŝ at each row's LAST valid response token (matching the sparse
                        # token_level_scores layout), so GRPO's per-sequence reward sum gains
                        # exactly β·ŝ — not β·ŝ × response_len (which spreading across the mask
                        # would cause).
                        tlr = batch.batch["token_level_rewards"]
                        rmask = batch.batch["response_mask"]
                        B_s, T_s = tlr.shape
                        col_idx = torch.arange(T_s, device=tlr.device).unsqueeze(0).expand(B_s, T_s)
                        masked_idx = torch.where(rmask.bool(), col_idx, torch.full_like(col_idx, -1))
                        last_idx = masked_idx.max(dim=1).values  # (B,), -1 if no response token
                        has_resp = last_idx >= 0
                        shape_vec = shape_per_row.to(device=tlr.device, dtype=tlr.dtype)
                        sel = has_resp.nonzero(as_tuple=True)[0]
                        tlr[sel, last_idx[sel]] += shape_vec[sel]
                        metrics.update(shaping_metrics)

                # ── Step 2.7: advantages (over the SHAPED token_level_rewards) ──
                with simple_timer("compute_adv", timing_raw):
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=rollout_n,
                        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        config=self.config.algorithm,
                    )

                # ── Step 3.5: Build JEPA pairs (moved ahead of the actor update so
                # eligibility for a fused single optimizer step is known upfront —
                # pair-building only needs `batch`/`reward_tensor`/`view_tags`, all
                # already available here; see _build_jepa_batch_tcr_dual docstring).
                view_tags = batch.non_tensor_batch["view"]
                rew_scalar_all = reward_tensor.sum(dim=-1)
                cot_mask_rows = (view_tags == "cot")
                code_mask_rows = (view_tags == "code")

                jepa_batch = None
                jepa_active = self.jepa_cfg.enable and not self._jepa_signal_off
                if jepa_active:
                    metrics["jepa/n_cot"] = float(n_cot)
                    metrics["jepa/n_code"] = float(n_code)
                    with simple_timer("jepa_build_batch", timing_raw):
                        if self.jepa_cfg.loss_type == "llm-jepa":
                            jepa_batch = self._build_jepa_batch_llm_jepa(
                                batch=batch,
                                reward_tensor=reward_tensor,
                                view_tags=view_tags,
                            )
                        elif self.jepa_cfg.loss_type == "llm-jepa-contrastive":
                            jepa_batch = self._build_jepa_batch_llm_jepa_contrastive(
                                batch=batch,
                                reward_tensor=reward_tensor,
                                view_tags=view_tags,
                            )
                        elif self.jepa_cfg.loss_type == "llm-jepa-infoNCE":
                            jepa_batch = self._build_jepa_batch_llm_jepa_infonce(
                                batch=batch,
                                reward_tensor=reward_tensor,
                                view_tags=view_tags,
                            )
                        else:
                            jepa_batch = self._build_jepa_batch_tcr_dual(
                                batch=batch,
                                reward_tensor=reward_tensor,
                                view_tags=view_tags,
                            )

                # Paper-style representation figure. Placed here, BEFORE the actor
                # update, so the embeddings come from the same weights that produced
                # these rollouts; running it after Step 4/5 would mix a policy that
                # has already moved with rollouts from the policy that preceded it.
                with simple_timer("rep_tsne", timing_raw):
                    self._maybe_log_train_rep_tsne(batch, reward_tensor, view_tags)

                # ── Step 3: Compute old log-probs & (optional) ref ──────
                # Actor/FSDP-only (compute_log_prob); does NOT call vLLM. All rollout outputs
                # it reads are already materialized in `batch` (DataProto.concat in Step 1),
                # so sleeping vLLM first is safe.
                #
                # Fused path: when the update is a single on-policy optimizer step
                # (ppo_epochs==1 and ppo_mini_batch_size >= train_batch_size), the
                # old log-probs equal the log-probs the update forward recomputes on
                # the same weights, so the PPO ratio is identically 1.0. In that case
                # we skip this whole extra forward over all rollout tokens and let the
                # loss default old_log_prob = log_prob.detach() (see losses.py).
                actor_cfg = self.config.actor_rollout_ref.actor
                _fuse = bool(actor_cfg.get("fuse_old_log_prob", False))
                _single_opt_step = (
                    int(actor_cfg.ppo_epochs) == 1
                    and int(actor_cfg.ppo_mini_batch_size) >= int(self.config.data.train_batch_size)
                )
                fuse_old_log_prob = _fuse and _single_opt_step
                if _fuse and not _single_opt_step:
                    print(
                        "[fuse_old_log_prob] disabled: requires ppo_epochs==1 and "
                        f"ppo_mini_batch_size ({actor_cfg.ppo_mini_batch_size}) >= "
                        f"train_batch_size ({self.config.data.train_batch_size}); "
                        "falling back to a separate old_log_prob forward."
                    )
                self._log_gpu_mem("before_old_log_prob")
                with simple_timer("old_log_prob", timing_raw):
                    if not fuse_old_log_prob:
                        old_log_prob, _old_log_prob_mfu = self._compute_old_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    if self.use_reference_policy:
                        ref_log_prob = self._compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                self._log_gpu_mem("after_old_log_prob")

                # ── Step 4: GRPO actor update ────────────────────────────
                # The actor takes its OWN complete optimizer step here (policy grad
                # applied + FSDP grad finalize). The JEPA update (Step 5) then takes a
                # SECOND, separate step for the auxiliary grad. Two sequential steps —
                # NOT a fused single step: the earlier "defer the policy step and fuse
                # it into jepa_update" scheme silently DROPPED the policy gradient,
                # because FSDP stashes a deferred (un-stepped) grad in its sharded
                # internal buffer and never materialises it onto .grad for jepa_update
                # to add to. θ → θ−lr·g_policy → θ−lr·α·g_JEPA; at this lr the two-step
                # update is indistinguishable from the summed-loss single step.
                with simple_timer("update_actor", timing_raw):
                    actor_output = self._update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_metrics)

                # ── Step 5: JEPA (if enabled) ────────────────────────────
                # The code-framed rollouts already live in `batch` (tagged
                # view=="code" in Step 1) and already received reward in
                # Step 2 — no separate rollout/reward pass needed here
                # anymore, only pair-building for the auxiliary loss.
                # jepa-tcr-dual keeps BOTH the Step-2.5 reward shaping (beta) AND this
                # differentiable dual loss arm (align_cot + align_code + self-consistency
                # + SIGReg).
                if jepa_active:
                    if jepa_batch is not None:
                        # JEPA update on worker (embedding extract + backward + EMA sync).
                        # Also applies the deferred policy gradient from Step 4 as part of
                        # the same, single optimizer step (see worker.jepa_update).
                        with simple_timer("jepa_update", timing_raw):
                            jepa_td = jepa_batch.to_tensordict()
                            # Broadcast to match jepa_td's batch dim (TensorDict requires
                            # assigned tensors to share the leading batch_size shape).
                            jepa_td["global_step"] = torch.full(
                                (jepa_td.batch_size[0],), float(self.global_steps)
                            )
                            # Per-arm plateau latches -> worker zeroes the disabled align
                            # term's gradient (preds still feed SIGReg). dual mode only.
                            if self.jepa_cfg.loss_type in ("llm-jepa", "jepa-tcr-dual", "jepa-tcr-cot"):
                                _bs = jepa_td.batch_size[0]
                                _cot_only = self.jepa_cfg.loss_type == "jepa-tcr-cot"
                                jepa_td["align_cot_on"] = torch.full((_bs,), float(not self._off_arm["cot"]))
                                # CoT-only mode has no code / self-consistency arms.
                                # (llm-jepa DOES have a code arm — the teacher Code view.)
                                jepa_td["align_code_on"] = torch.full((_bs,), 0.0 if _cot_only else float(not self._off_arm["code"]))
                                jepa_td["self_on"] = torch.full((_bs,), 0.0 if _cot_only else float(not self._off_arm["self"]))
                            jepa_output = self.actor_rollout_wg.jepa_update(jepa_td)
                            # ONE_TO_ALL dispatch returns a list; take rank-0 output
                            if isinstance(jepa_output, list):
                                jepa_output = jepa_output[0] if jepa_output else None
                            if jepa_output is not None:
                                for k, v in jepa_output.items():
                                    if isinstance(v, torch.Tensor):
                                        metrics[k] = float(v.item())
                            if "n_correct_cot_mean" in jepa_batch.meta_info:
                                metrics["jepa/n_correct_cot_mean"] = jepa_batch.meta_info["n_correct_cot_mean"]
                            if "n_mixed_prompts" in jepa_batch.meta_info:
                                metrics["jepa/n_mixed_prompts"] = float(jepa_batch.meta_info["n_mixed_prompts"])
                                metrics["jepa/pairs_per_prompt"] = float(jepa_batch.meta_info["pairs_per_prompt"])
                    else:
                        metrics["jepa/skipped"] = 1.0
                        metrics["jepa/n_valid_pairs"] = 0.0

                    # NOTE: do NOT sleep the rollout replicas here. vLLM was already put
                    # to sleep at sleep_replicas_1 (above, before update_actor) and nothing
                    # wakes it again until weight_sync_2 at the end of the step. The
                    # pre-unification loop had a weight_sync_1 (wake) + a separate JEPA-only
                    # code generation between the two sleeps, so this second sleep acted on an
                    # AWAKE engine; unification removed that wake+generation but left this
                    # sleep behind. Sleeping an already-slept engine reproducibly segfaults
                    # vLLM's executor in cuMemcpy at the first training step (EngineCore dies
                    # -> "collective_rpc sleep ... cancelled" -> EngineDeadError). jepa_update
                    # is a worker-side FSDP RPC and does not touch the rollout engine.

                # Track code/cot accuracy (sliced from the single combined reward_tensor)
                code_rew_scalar = rew_scalar_all[torch.from_numpy(code_mask_rows)]
                metrics["code/pass_at_1"] = float((code_rew_scalar > 0).float().mean()) if len(code_rew_scalar) else 0.0
                metrics["code/avg_reward"] = float(code_rew_scalar.mean()) if len(code_rew_scalar) else 0.0

                # ── Step 6: Weight sync to rollout (wakes vLLM) ─────────
                self._log_gpu_mem("before_weight_sync_2")
                with simple_timer("weight_sync_2", timing_raw):
                    self.checkpoint_manager.update_weights(self.global_steps)
                self._log_gpu_mem("after_weight_sync_2")

                # ── CoT-based train metrics (rich grouped stats) ─────────
                cot_rew_scalar = rew_scalar_all[torch.from_numpy(cot_mask_rows)]
                metrics["cot/pass_at_1"] = float((cot_rew_scalar > 0).float().mean()) if len(cot_rew_scalar) else 0.0
                metrics["cot/avg_reward"] = float(cot_rew_scalar.mean()) if len(cot_rew_scalar) else 0.0
                # NOTE: compute_data_metrics/_compute_train_comparison_metrics below now span
                # the FULL combined (cot+code) batch, not cot-only as before this change — this
                # is intended (both modalities now receive gradient), but means train/accuracy,
                # response_length/* etc. will show a discontinuity at the cutover step in wandb.
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(self._compute_train_comparison_metrics(batch))
                metrics["train/global_step"] = self.global_steps

                # Auto-disable the JEPA aux signal (shaping/differentiable) once the
                # teacher-alignment metric this step has plateaued. Evaluated AFTER the
                # step's shaping/jepa metrics are in `metrics`; latches for all later steps.
                self._maybe_disable_jepa_signal(metrics)

                # ── Validation ───────────────────────────────────────────
                is_last_step = self.global_steps >= self.total_training_steps
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with simple_timer("validate", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                    self._maybe_save_best_checkpoint(val_metrics)

                # ── Checkpoint ───────────────────────────────────────────
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with simple_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                metrics.update({f"timing_s/{k}": v for k, v in timing_raw.items()})
                # Sum only top-level segments. The per-call generate_sequences
                # diagnostics merged in above as "cot_gen/*" / "code_gen/*" are
                # sub-timings ALREADY contained in the top-level "cot_gen" /
                # "code_gen" timers — including them double-counts and inflated
                # step_total to ~minutes (3625s observed). Drop any key with "/".
                metrics["timing_s/step_total"] = sum(
                    v for k, v in timing_raw.items() if "/" not in k
                )

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                # The full metrics dict (~150 keys) on the tqdm postfix floods the
                # console and slows the terminal. Only show it when explicitly opted in
                # via VERL_CONSOLE_FULL_METRICS=1; otherwise a tiny curated subset.
                if os.environ.get("VERL_CONSOLE_FULL_METRICS", "0") == "1":
                    progress_bar.set_postfix(metrics)
                else:
                    _short = {
                        k: round(metrics[k], 4)
                        for k in ("actor/loss", "actor/grad_norm", "jepa/tcr_loss",
                                  "jepa/grad_norm", "train/accuracy", "timing_s/step_total")
                        if k in metrics
                    }
                    progress_bar.set_postfix(_short)

                if is_last_step:
                    return

                self.global_steps += 1
