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
"""JEPA-GRPO Ray worker.

Extends ActorRolloutRefWorker with:
  - EMA target encoder for the Code view (stored as plain param dict, not PEFT adapter)
  - Hook-based embedding extraction on the final norm (avoids output_hidden_states=True overhead)
  - jepa_update() RPC: runs a JEPA-only forward+backward+optimizer step, using either
    the LeJEPA loss (default) or the LLM-JEPA paper's prediction loss (jepa.loss_type)
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional

import torch
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine_workers import ActorRolloutRefWorker

from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.experimental.jepa_grpo.core_algos import (
    hgrpo_loss,
    llm_jepa_contrastive_loss,
    llm_jepa_geometry_loss,
    llm_jepa_infonce_loss,
    llm_jepa_paper_loss,
    llm_jepa_tcr_dual_loss,
    llm_jepa_triplet_loss,
)


def _assert_policy_grad_resident(actor_grad_norm: float) -> None:
    """Fail loudly if the deferred policy gradient never reached .grad.

    Fused mode relies on the actor's backward leaving its gradient resident (the actor
    skipped its optimizer step). A prior attempt at this fusion reported FSDP stashing the
    deferred grad in a sharded internal buffer instead, which silently dropped the policy
    update — a zero norm here is exactly that failure, so raise rather than train on a
    JEPA-only gradient.
    """
    if not (actor_grad_norm > 0.0):
        raise RuntimeError(
            "jepa.fuse_optimizer_step: the deferred policy gradient is not resident on "
            f".grad (norm={actor_grad_norm}). Fusing would apply the JEPA gradient alone "
            "and silently drop this step's policy update. Set jepa.fuse_optimizer_step="
            "False to use the two-step path, or materialize the FSDP grad before stepping."
        )


class JEPAActorRolloutRefWorker(ActorRolloutRefWorker):
    """Extends ActorRolloutRefWorker with JEPA embedding extraction and EMA target encoder.

    The standard GRPO loss (update_actor) is unchanged.  A new ``jepa_update``
    RPC runs a *separate* backward+optimizer-step for the LeJEPA loss, keeping
    the implementation self-contained without touching the FSDP engine internals.

    EMA is stored as a plain ``dict[str, Tensor]`` keyed on LoRA adapter param
    names so vLLM never sees unknown weight keys.
    """

    # ------------------------------------------------------------------ init --
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        # Initialised lazily on first jepa_init call so config is available
        self.ema_weights: Optional[dict[str, torch.Tensor]] = None
        self.jepa_cfg: Optional[JEPARayConfig] = None
        # JEPA_UPDATE_NO_GC=1: run the GRPO actor update WITHOUT gradient
        # checkpointing (skips the full recompute pass, ~25-35% faster update)
        # while keeping GC enabled for every other phase — the JEPA GradCache
        # backward genuinely needs it. Toggled around train_mini_batch only.
        import os as _os
        if _os.environ.get("JEPA_UPDATE_NO_GC", "0") == "1" and hasattr(self, "actor"):
            _orig_tmb = self.actor.train_mini_batch

            def _tmb_no_gc(data, _orig=_orig_tmb):
                m = self.actor.engine.module  # FSDP/PEFT forward getattr to the HF model
                m.gradient_checkpointing_disable()
                try:
                    return _orig(data)
                finally:
                    m.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )

            self.actor.train_mini_batch = _tmb_no_gc
            print("[JEPA_UPDATE_NO_GC] gradient checkpointing disabled for update_actor only")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def jepa_init(self, jepa_cfg_dict: dict) -> None:
        """Initialise EMA weights from current LoRA adapter parameters."""
        self.jepa_cfg = JEPARayConfig(**jepa_cfg_dict)
        # Literal predictor tokens (paper §3.1): the trainer added
        # <|predictor_1|>..<|predictor_k|> to the TOKENIZER only. Their new ids must
        # land inside the model's preallocated (padded) embedding rows, since we do
        # not resize_token_embeddings (a resize would break LoRA shapes and vLLM
        # weight sync). Qwen2.5 has ~270 spare rows, so any sane k fits.
        if self.jepa_cfg.predictor_k > 0:
            ids = list(self.jepa_cfg.predictor_token_ids)
            assert len(ids) == self.jepa_cfg.predictor_k, (
                f"predictor_k={self.jepa_cfg.predictor_k} but "
                f"{len(ids)} predictor_token_ids were resolved"
            )
            module = self.actor.engine.module
            inner = getattr(module, "_fsdp_wrapped_module", module)
            emb_rows = inner.get_input_embeddings().weight.shape[0]
            assert max(ids) < emb_rows, (
                f"predictor token id {max(ids)} >= embedding rows {emb_rows}: this "
                "model has no spare padded vocab rows; resize_token_embeddings would "
                "be required (unsupported under LoRA + vLLM weight sync)"
            )
        self.ema_weights = {
            n: p.data.clone().detach()
            for n, p in self._adapter_named_params()
        }

        # Trainable predictor-token embeddings. The literal predictor ids land in
        # Qwen's spare PADDED embedding rows, which are all ~identical (bf16 noise
        # apart) AND frozen under LoRA — so <predictor_i>/<bad_predictor_i> carry no
        # learnable "good vs bad" (or slot) signal on their own. Give each predictor
        # id its OWN small trainable delta, ADDED to the frozen base embedding by a
        # forward hook on embed_tokens wherever that id appears. The predictor ids
        # never occur in rollout/policy inputs, so the hook is a no-op there and vLLM
        # is untouched; only the JEPA anchor forwards (which append the ids) train it.
        self._predictor_hook_handle = None
        self.predictor_delta = None
        self._pred_id_to_row = {}
        # Hook is a true no-op unless explicitly armed (only during the JEPA anchor
        # forward), so the hot policy/rollout forward pays no per-forward .any() sync.
        self._predictor_hook_active = False
        if self.jepa_cfg.predictor_k > 0:
            pred_ids = list(self.jepa_cfg.predictor_token_ids)
            if self.jepa_cfg.loss_type in ("llm-jepa-contrastive", "llm-jepa-infoNCE"):
                pred_ids = pred_ids + list(self.jepa_cfg.bad_predictor_token_ids)
            pred_ids = list(dict.fromkeys(int(i) for i in pred_ids))  # unique, ordered
            self._pred_id_to_row = {tid: r for r, tid in enumerate(pred_ids)}
            emb_w = self._embed_tokens().weight
            hidden = emb_w.shape[1]
            dev = emb_w.device
            # Small distinct init (seeded identically on every rank) so good/bad and
            # the k slots start differentiated; then broadcast to be exactly in sync.
            # Deterministic CPU seed → identical init on every rank (no broadcast
            # needed); DP grad sync happens per-step in jepa_update.
            g = torch.Generator(device="cpu").manual_seed(1234)
            # Init per-row norm ~0.3 (= 0.0066·√2048) — small relative to a natural
            # token embedding, distinct per predictor id, so training starts from a
            # sane in-distribution point (no cap enforced afterward; trained like any
            # other embedding row).
            init = (torch.randn(len(pred_ids), hidden, generator=g) * 0.0066).to(dev, torch.float32)
            self.predictor_delta = torch.nn.Parameter(init)
            # No bespoke optimizer/LR/clip/norm-cap: the predictor tokens are trained
            # by the SAME optimizer, at the SAME learning rate (and schedule), as
            # every other trainable parameter — just another param group on the
            # actor's real optimizer. It isn't FSDP-sharded (plain replicated param),
            # so jepa_update still manually all-reduces its grad across DP ranks
            # before the shared engine.optimizer_step() call.
            self.actor.engine.optimizer.add_param_group({"params": [self.predictor_delta]})

            embed_mod = self._embed_tokens()

            def _predictor_embed_hook(_mod, _inp, _out, _self=self):
                # Add the trainable delta at every predictor-id position. Armed only
                # during the JEPA anchor forward (see jepa_update), so the policy /
                # rollout forward returns immediately with no .any() sync and no
                # graph dependency on predictor_delta.
                if not _self._predictor_hook_active or _self.predictor_delta is None:
                    return _out
                input_ids = _inp[0]
                delta = None
                for tid, row in _self._pred_id_to_row.items():
                    mask = input_ids == tid
                    if bool(mask.any()):
                        contrib = mask.unsqueeze(-1).to(_out.dtype) * _self.predictor_delta[row].to(_out.dtype)
                        delta = contrib if delta is None else delta + contrib
                return _out if delta is None else _out + delta

            self._predictor_hook_handle = embed_mod.register_forward_hook(_predictor_embed_hook)
            print(f"[jepa] trainable predictor embeddings: {len(pred_ids)} tokens "
                  f"(ids {pred_ids}) | trained via actor optimizer (same lr/schedule "
                  f"as every other trainable parameter, no special clip/cap)",
                  flush=True)

    # --------------------------------------------------------------- helpers --
    def _adapter_named_params(self):
        """Yield (name, param) for LoRA adapter parameters only."""
        module = self.actor.engine.module
        # FSDP wraps the module; unwrap to access peft internals
        inner = getattr(module, "_fsdp_wrapped_module", module)
        for n, p in inner.named_parameters():
            if "lora_" in n:
                yield n, p

    def _last_layer(self) -> torch.nn.Module:
        """Return the last transformer decoder layer of the policy model."""
        module = self.actor.engine.module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        # Handles Qwen2/LLaMA-style naming: model.model.layers or model.layers
        try:
            return inner.model.model.layers[-1]
        except AttributeError:
            return inner.model.layers[-1]

    def _final_norm(self) -> torch.nn.Module:
        """Return the final layer norm of the base transformer.

        Mirrors the FSDP-unwrap logic in _last_layer() but targets the norm
        applied after all decoder layers (the one that produces last_hidden_state).
        """
        module = self.actor.engine.module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        try:
            return inner.model.model.norm  # Qwen2-style double nesting
        except AttributeError:
            return inner.model.norm        # LLaMA-style

    def _decoder_layers(self) -> list:
        """Return the list of decoder layer modules (Qwen2DecoderLayer etc.)."""
        module = self.actor.engine.module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        for attr in ("model.model.layers", "model.layers"):
            obj = inner
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                pass
        return []

    def _embed_tokens(self) -> torch.nn.Module:
        """Return the input token-embedding module (nn.Embedding). Mirrors the
        FSDP-unwrap logic in _final_norm()."""
        module = self.actor.engine.module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        try:
            return inner.model.model.embed_tokens   # Qwen2-style double nesting
        except AttributeError:
            return inner.model.embed_tokens         # LLaMA-style

    @contextlib.contextmanager
    def _no_gc_ctx(self):
        """Temporarily disable gradient checkpointing on each decoder layer.

        Patches the per-layer ``gradient_checkpointing`` boolean directly instead
        of calling ``model.gradient_checkpointing_disable()``.  The HF API also
        removes ``enable_input_require_grads`` hooks which breaks LoRA backward;
        this surgical patch avoids that side effect.

        With GC off, the forward stores all intermediate activations and backward
        runs without recompute — eliminating the cudaErrorIllegalAddress that
        FSDP1's GC-recompute path triggers on single-GPU NO_SHARD configurations
        when backward is computed from an intermediate hook capture rather than
        the model's final output tensor.

        Note: each element of ``self.layers`` is an FSDP wrapper around the
        actual Qwen2DecoderLayer.  We must unwrap via ``_fsdp_wrapped_module``
        to reach the ``GradientCheckpointingLayer`` that owns the flag.
        """
        layers = self._decoder_layers()
        gc_states = {}
        for fsdp_layer in layers:
            # Unwrap FSDP to reach the GradientCheckpointingLayer
            layer = getattr(fsdp_layer, "_fsdp_wrapped_module", fsdp_layer)
            if hasattr(layer, "gradient_checkpointing"):
                gc_states[id(layer)] = layer.gradient_checkpointing
                layer.gradient_checkpointing = False
        try:
            yield
        finally:
            for fsdp_layer in layers:
                layer = getattr(fsdp_layer, "_fsdp_wrapped_module", fsdp_layer)
                if id(layer) in gc_states:
                    layer.gradient_checkpointing = gc_states[id(layer)]

    @contextlib.contextmanager
    def _ema_ctx(self):
        """Context manager: temporarily overwrite live LoRA params with EMA values.

        Uses in-place copy (not data-pointer swap) so that FSDP1's FlatParameter
        view pointers remain intact.  Swapping p.data to a different tensor would
        redirect the parameter away from FSDP1's managed flat storage.
        """
        saved = {}
        for n, p in self._adapter_named_params():
            saved[n] = p.data.clone()                                   # save live values
            p.data.copy_(self.ema_weights[n].to(p.device, dtype=p.dtype))  # overwrite in-place
        try:
            yield
        finally:
            for n, p in self._adapter_named_params():
                p.data.copy_(saved[n])                                  # restore live values in-place

    @torch.no_grad()
    def _sync_ema(self) -> None:
        """Exponential moving average update: ema ← decay*ema + (1-decay)*live."""
        decay = self.jepa_cfg.ema_decay
        for n, p in self._adapter_named_params():
            self.ema_weights[n].mul_(decay).add_(p.data.detach(), alpha=1.0 - decay)

    # ------------------------------------------------ embedding extraction ---
    def _pack_left_padded(
        self,
        input_ids: torch.Tensor,   # (B, L) left-padded
        seq_lengths: torch.Tensor, # (B,) real lengths
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove left-padding and pack into verl's (1, total_nnz) format.

        verl's monkey-patched flash attention (use_remove_padding=True) expects
        packed input with no attention_mask — it infers cu_seqlens from
        position_ids.  Passing a padded [B, L] tensor + attention_mask instead
        triggers HuggingFace's _upad_input → torch.nonzero, which causes a
        cudaErrorIllegalAddress on the FSDP-managed CUDA context.

        Returns:
            packed_ids:  (1, total_nnz) token ids, padding removed
            packed_pos:  (1, total_nnz) position ids, 0-indexed per sequence
            last_idx:    (B,) index in the packed dim of each sequence's last token
        """
        B, L = input_ids.shape
        chunks, pos_chunks, last_indices = [], [], []
        offset = 0
        for b in range(B):
            rlen = int(seq_lengths[b].item())
            pad = L - rlen
            chunks.append(input_ids[b, pad:])                              # real tokens only
            pos_chunks.append(torch.arange(rlen, device=device, dtype=torch.long))
            offset += rlen
            last_indices.append(offset - 1)

        packed_ids = torch.cat(chunks).unsqueeze(0)           # (1, total_nnz)
        packed_pos = torch.cat(pos_chunks).unsqueeze(0)       # (1, total_nnz)
        last_idx   = torch.tensor(last_indices, device=device) # (B,)
        return packed_ids, packed_pos, last_idx

    @staticmethod
    def _pad_concat_batches(
        groups: "list[tuple[torch.Tensor, torch.Tensor]]",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate N (N_i, L_i) batches along the batch dim into one (sum(N_i), L').

        Used to merge the CoT, Code, and (for the triplet mode) wrong-Code
        views into a single batch for a joint live forward (see jepa_update).
        Every group's L dimension is zero-padded to the longest one; this is
        safe because `_extract_embeddings` only ever reads real tokens via
        `attention_mask` (extra zero-padded columns with mask=0 are simply
        ignored).
        """
        L = max(ids.shape[1] for ids, _ in groups)
        padded_ids, padded_mask = [], []
        for ids, mask in groups:
            pad = (0, L - ids.shape[1])
            padded_ids.append(torch.nn.functional.pad(ids, pad) if pad[1] else ids)
            padded_mask.append(torch.nn.functional.pad(mask, pad) if pad[1] else mask)
        return torch.cat(padded_ids, dim=0), torch.cat(padded_mask, dim=0)

    def _pad_concat_batch(
        self,
        ids_a: torch.Tensor, mask_a: torch.Tensor,
        ids_b: torch.Tensor, mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-group convenience wrapper around `_pad_concat_batches`."""
        return self._pad_concat_batches([(ids_a, mask_a), (ids_b, mask_b)])

    def _extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        use_ema: bool,
        requires_grad: bool,
        predictor_k: "int | list[int]" = 0,
        predictor_token_ids: "list[int] | None" = None,
        also_boundary: bool = False,
    ) -> torch.Tensor:
        """Run the policy model on the full batch and return last-token embeddings.

        All N sequences are processed in a SINGLE forward call so that FSDP1
        sees exactly one forward per backward — its expected usage pattern.
        Running N separate forwards before one backward confuses FSDP1's hook
        state machine and causes CUBLAS_STATUS_INTERNAL_ERROR.

        Sequences are re-packed into an (N, max_rlen) tensor with real tokens
        at the start (right-padded with zeros).  A monotonically-increasing
        position_ids shared across rows ensures _is_packed_sequence() returns
        False, routing flash attention through flash_attn_func (not varlen).
        A proper 2D attention_mask prevents padding positions from polluting
        the representations of shorter sequences.

        Args:
            input_ids: (N, L) token ids, may be left or right padded
            attention_mask: (N, L) 1 for real tokens, 0 for padding
            seq_lengths: (N,) actual sequence lengths (= attention_mask.sum per row)
            use_ema: if True, temporarily swap in EMA weights
            requires_grad: if False, wrap forward in torch.no_grad()
            predictor_k: number of LLM-JEPA tied-weight predictor tokens
                (arXiv:2509.14252 §3.1) to append after each row's real tokens.
                Either a single int broadcast to every row, or a per-row
                list/sequence of ints (used to mix predictor and non-predictor
                rows in one joint forward — e.g. CoT rows get k>0, Code rows
                get k=0). k=0 is a no-op — Pred(x) = x, identical to prior
                behavior. When k>0 for a row, the literal predictor special
                tokens are appended and that row's returned embedding is read
                from the LAST predictor token (= <|predictor_1|>) instead of
                the last real token, reusing the model's own weights (no new
                parameters) as the "tied-weight predictor".
            predictor_token_ids: full predictor append sequence
                [id(<|predictor_K|>), ..., id(<|predictor_1|>)] — the official
                finetune.py order. Required when any row has predictor_k > 0.
            also_boundary: if True, ALSO return the per-row BOUNDARY embedding —
                the last REAL token (index rlen-1), i.e. the read taken BEFORE any
                appended [PRED] tokens. This is a free second read of the same
                packed forward (used for the jepa-tcr-dual self-consistency target
                z_self = Enc(Code_S) boundary). No-op cost: the hidden states are
                already materialized. Changes the return arity (see Returns).

        Returns:
            also_boundary=False: ((N, d) [PRED]/last-token embeddings, logits_anchor)
            also_boundary=True:  ((N, d) [PRED] embeddings, (N, d) boundary
                                  embeddings, logits_anchor)
            All embeddings are unit-normalised float32.
        """
        device = next(self.actor.engine.module.parameters()).device
        N = input_ids.shape[0]
        rlens = seq_lengths.long().tolist()  # one host sync instead of N .item() calls
        if isinstance(predictor_k, int):
            predictor_ks = [predictor_k] * N
        else:
            predictor_ks = [int(k) for k in predictor_k]
            assert len(predictor_ks) == N, "predictor_k list must match batch size"
        base_max_rlen = max(rlens) if rlens else 1
        max_rlen = base_max_rlen + (max(predictor_ks) if predictor_ks else 0)

        # Build (N, max_rlen) batch: real tokens contiguous at start, zeros for padding.
        # Works for both left-padded and right-padded inputs via the attention_mask.
        # Where predictor_ks[b] > 0, the LITERAL predictor special tokens follow that
        # row's real tokens (still causally attending to them) before trailing
        # padding. `predictor_token_ids` is the full append sequence
        # [<|predictor_K|>, ..., <|predictor_1|>] (official finetune.py order); a
        # row with pk <= K takes the LAST pk entries so the read token is always
        # <|predictor_1|>.
        # predictor_token_ids may be ONE shared append sequence (list[int], the common
        # case) OR a PER-ROW list of sequences (list[list[int]]) so a single joint forward
        # can mix different predictor vocabularies — e.g. llm-jepa-contrastive appends
        # <|predictor_i|> on good rows and <|bad_predictor_i|> on wrong rows. Row b always
        # takes the LAST pk entries of its sequence (read token = slot 1).
        pred_seq = None
        per_row_seqs = None
        if any(pk > 0 for pk in predictor_ks):
            assert predictor_token_ids, "rows with predictor_k > 0 need predictor_token_ids"
            if isinstance(predictor_token_ids[0], (list, tuple)):
                assert len(predictor_token_ids) == N, "per-row predictor_token_ids must match batch size"
                per_row_seqs = [
                    torch.tensor(list(s), dtype=input_ids.dtype, device=device) if s else None
                    for s in predictor_token_ids
                ]
                for b, pk in enumerate(predictor_ks):
                    assert pk <= 0 or (per_row_seqs[b] is not None and pk <= per_row_seqs[b].shape[0]), (
                        f"row {b}: predictor_k {pk} exceeds its resolved predictor tokens")
            else:
                assert max(predictor_ks) <= len(predictor_token_ids), (
                    f"predictor_k {max(predictor_ks)} exceeds the {len(predictor_token_ids)} "
                    "resolved predictor tokens"
                )
                pred_seq = torch.tensor(list(predictor_token_ids), dtype=input_ids.dtype, device=device)
        packed_ids = torch.zeros(N, max_rlen, dtype=input_ids.dtype, device=device)
        packed_attn = torch.zeros(N, max_rlen, dtype=torch.long, device=device)
        for b, rlen in enumerate(rlens):
            pk = predictor_ks[b]
            if rlen > 0:
                real_toks = input_ids[b][attention_mask[b].bool()].to(device)  # (rlen,)
                packed_ids[b, :rlen] = real_toks
                packed_attn[b, :rlen] = 1
                if pk > 0:
                    seq = per_row_seqs[b] if per_row_seqs is not None else pred_seq
                    packed_ids[b, rlen:rlen + pk] = seq[-pk:]
                    packed_attn[b, rlen:rlen + pk] = 1

        # Monotonically increasing position_ids for ALL positions (including padding).
        # _is_packed_sequence() returns False for monotonic positions → regular
        # flash_attn_func, not the varlen path.
        packed_pos = torch.arange(max_rlen, device=device, dtype=torch.long).unsqueeze(0).expand(N, -1)

        norm = self._final_norm()
        captured: dict[str, torch.Tensor] = {}

        def _hook(module, inp, out, _cap=captured):
            _cap["h"] = out[0] if isinstance(out, tuple) else out

        ema_ctx = self._ema_ctx() if use_ema else contextlib.nullcontext()
        grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()

        handle = norm.register_forward_hook(_hook)
        try:
            with ema_ctx, grad_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.actor.engine.module(
                    input_ids=packed_ids,
                    attention_mask=packed_attn,
                    position_ids=packed_pos,
                    use_cache=False,
                    return_dict=True,  # required by the fused-kernels patched forward
                    # Only the anchor scalar below needs *a* logits tensor, not the
                    # full (N, max_rlen, vocab) one HF computes by default (logits_to_keep=0).
                    # With long packed batches this materializes tens of GB just to be
                    # multiplied by zero — logits_to_keep=1 keeps only the last position's
                    # logits, cutting that allocation by a factor of max_rlen.
                    logits_to_keep=1,
                )
        finally:
            handle.remove()

        last_h = captured["h"]   # (N, max_rlen, d) — kept in model dtype; only the
        # gathered read positions are cast to float32 below (the old full-tensor
        # .float() materialized an extra (N, max_rlen, d) fp32 copy per forward).

        # Keep a zero-valued scalar connected to the model's returned logits.
        # This forces backward to flow through the model's actual output tensor,
        # which is required for FSDP1's root-module post-backward hook to fire.
        # Without it, backward enters from the intermediate captured["h"] and
        # skips the root FSDP output, leaving its state machine in an
        # inconsistent state that causes cudaErrorIllegalAddress.
        # The multiplier is exactly 0.0 so this contributes zero gradient.
        # With model.use_fused_kernels=True the patched forward returns
        # CausalLMOutputForPPO (log_probs/entropy, no logits tensor); anchor on
        # log_probs in that case — any root-forward output tensor works.
        _anchor_src = getattr(outputs, "logits", None)
        if _anchor_src is not None:
            logits_anchor = _anchor_src[:, 0, 0].sum() * 0.0
        else:
            # shape varies by backend ((N, T) torch vs flat (N*T,) triton) — stay agnostic
            logits_anchor = outputs.log_probs.reshape(-1)[0].sum() * 0.0

        # Vectorized gather of the per-row read positions (replaces the per-row
        # Python loop). With predictor_ks[b]=0 the read is rlen-1 (last real
        # token); with predictor_ks[b]>0 it is that row's last predictor token
        # = Pred(Enc(Text)). Rows with rlen==0 get a zero embedding, as before.
        rlen_t = torch.tensor(rlens, device=device, dtype=torch.long)
        pk_t = torch.tensor(predictor_ks, device=device, dtype=torch.long)
        rows = torch.arange(N, device=device)
        empty = (rlen_t == 0).unsqueeze(-1)
        read_idx = (rlen_t + pk_t - 1).clamp(min=0)
        emb = torch.nn.functional.normalize(last_h[rows, read_idx].float(), dim=-1)
        emb = torch.where(empty, torch.zeros_like(emb), emb)

        if also_boundary:
            # Boundary = last REAL token (pre-[PRED]); equals `emb` when k=0.
            bnd_idx = (rlen_t - 1).clamp(min=0)
            bnd = torch.nn.functional.normalize(last_h[rows, bnd_idx].float(), dim=-1)
            bnd = torch.where(empty, torch.zeros_like(bnd), bnd)
            return emb, bnd, logits_anchor
        return emb, logits_anchor  # (N, d), scalar

    def _embed_chunked_no_grad(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        use_ema: bool,
        micro_bs: int,
        predictor_k: "int | list[int]" = 0,
        predictor_token_ids: "list[int] | None" = None,
    ) -> torch.Tensor:
        """No-grad embedding extraction, chunked to bound forward activation memory.

        No backward is needed here (used for the EMA Code target encoder in
        "lejepa" mode, and for the jepa-tcr-reward forward-only scoring pass), so
        this is just a plain loop + concat — no GradCache machinery required.

        `predictor_k`/`predictor_token_ids` mirror `_extract_embeddings`: pass
        cfg.predictor_k to read the Pred(Enc(text)) [PRED]-token embedding instead
        of the last real token. A per-row list is sliced per chunk.
        """
        N = ids.shape[0]
        per_row_k = isinstance(predictor_k, (list, tuple))

        if N <= micro_bs:
            emb, _ = self._extract_embeddings(
                ids, mask, lengths, use_ema=use_ema, requires_grad=False,
                predictor_k=(list(predictor_k) if per_row_k else predictor_k),
                predictor_token_ids=predictor_token_ids,
            )
            return emb

        # Sort rows by length before chunking so each chunk is padded to ITS OWN
        # max length instead of (in expectation) the global max — with skewed
        # response lengths this cuts the padded FLOPs of every chunk forward
        # substantially. Results are un-permuted before returning, so this is
        # exact w.r.t. the unsorted path.
        order = torch.argsort(lengths.long(), descending=True)
        order_list = order.tolist()
        ids_s, mask_s, len_s = ids[order], mask[order], lengths[order]
        k_s = [predictor_k[i] for i in order_list] if per_row_k else predictor_k

        def _k(start, end):
            return list(k_s[start:end]) if per_row_k else k_s

        chunks = []
        for start in range(0, N, micro_bs):
            end = min(start + micro_bs, N)
            emb, _ = self._extract_embeddings(
                ids_s[start:end], mask_s[start:end], len_s[start:end], use_ema=use_ema, requires_grad=False,
                predictor_k=_k(start, end), predictor_token_ids=predictor_token_ids,
            )
            chunks.append(emb)
        out_sorted = torch.cat(chunks, dim=0)
        inv = torch.empty(N, dtype=torch.long, device=out_sorted.device)
        inv[order.to(out_sorted.device)] = torch.arange(N, device=out_sorted.device)
        return out_sorted[inv]

    def _embed_sharded_no_grad(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        micro_bs: int,
        predictor_k: "int | list[int]" = 0,
        predictor_token_ids: "list[int] | None" = None,
    ) -> torch.Tensor:
        """Data-parallel wrapper around _embed_chunked_no_grad.

        The TCR-shaping RPCs are ONE_TO_ALL, so without this every rank redundantly
        embeds the FULL batch and only rank 0's copy is kept. Here each rank embeds
        a contiguous 1/world_size row shard and the (small, N x d fp32) results are
        all-gathered, so wall-clock scales ~1/world_size. Per-row outputs are
        independent, so the result is bitwise-identical to the unsharded path.
        """
        N = ids.shape[0]
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        per_row_k = isinstance(predictor_k, (list, tuple))

        def _run(i_lo: int, i_hi: int) -> torch.Tensor:
            if i_hi <= i_lo:
                d = self.actor.engine.module.config.hidden_size
                return torch.zeros(0, d, dtype=torch.float32)
            k = list(predictor_k[i_lo:i_hi]) if per_row_k else predictor_k
            return self._embed_chunked_no_grad(
                ids[i_lo:i_hi], mask[i_lo:i_hi], lengths[i_lo:i_hi],
                use_ema=False, micro_bs=micro_bs,
                predictor_k=k, predictor_token_ids=predictor_token_ids,
            ).float().cpu()

        if world == 1:
            return _run(0, N)

        rank = torch.distributed.get_rank()
        bounds = [round(N * r / world) for r in range(world + 1)]
        local = _run(bounds[rank], bounds[rank + 1])
        gathered: list = [None] * world
        torch.distributed.all_gather_object(gathered, local)
        return torch.cat([g for g in gathered], dim=0)

    # ------------------------------------------------ TCR reward scoring ---
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def embed_validation_sequences(self, data: TensorDict) -> TensorDict:
        """Read-only boundary embeddings, identical for Dr.GRPO and JEPA."""
        micro_bs = int(data["micro_bs"].reshape(-1)[0].item())
        with self.actor.engine.eval_mode():
            emb = self._embed_sharded_no_grad(
                data["input_ids"], data["attention_mask"], data["lengths"],
                micro_bs=max(1, micro_bs), predictor_k=0,
            )
        aggressive_empty_cache(force_sync=True)
        return TensorDict({"embeddings": emb.float().cpu()}, batch_size=[])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def score_cot_embeddings(self, data: TensorDict) -> TensorDict:
        """Forward-only [PRED]-token embeddings for jepa-tcr-reward shaping.

        Unlike jepa_update this runs NO backward, NO EMA, and NO optimizer step:
        it just returns p_i = Pred(Enc([x, y_S,i, [PRED]])) for every passed CoT
        rollout row, L2-normalized, so the trainer can score them against the
        cached teacher-correct targets and fold the result into the advantage.

        Args:
            data: TensorDict with cot_input_ids (N, L), cot_attn_mask (N, L),
                cot_lengths (N,).
        Returns:
            TensorDict with key "cot_emb" (N, d) float32 unit embeddings.
        """
        assert self.jepa_cfg is not None, "Call jepa_init() before score_cot_embeddings()"
        engine = self.actor.engine
        cfg = self.jepa_cfg
        micro_bs = max(1, cfg.embed_micro_batch_size)
        N = data["cot_input_ids"].shape[0]
        with engine.eval_mode():
            emb = self._embed_sharded_no_grad(
                data["cot_input_ids"], data["cot_attn_mask"], data["cot_lengths"],
                micro_bs=micro_bs,
                predictor_k=[cfg.predictor_k] * N,
                predictor_token_ids=cfg.predictor_token_ids,
            )
        # No empty_cache here: embed_targets() always follows in the same shaping
        # pass and does the (expensive, force-synced) flush once for both RPCs.
        return TensorDict({"cot_emb": emb.float().cpu()}, batch_size=[])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def embed_targets(self, data: TensorDict) -> TensorDict:
        """Forward-only LAST-REAL-TOKEN embeddings of verified-correct target views.

        Encodes [x, y^+] with the CURRENT policy (predictor_k=0, no EMA, no backward),
        L2-normalized — the online z^+ used by jepa-tcr-dual reward shaping, replacing
        the old cached frozen-reference latents. Matches the predictor_k=0 read the
        alignment loss uses for its targets.
        """
        assert self.jepa_cfg is not None, "Call jepa_init() before embed_targets()"
        micro_bs = max(1, self.jepa_cfg.embed_micro_batch_size)
        with self.actor.engine.eval_mode():
            emb = self._embed_sharded_no_grad(
                data["target_input_ids"], data["target_attn_mask"], data["target_lengths"],
                micro_bs=micro_bs, predictor_k=0,
            )
        aggressive_empty_cache(force_sync=True)
        return TensorDict({"target_emb": emb.float().cpu()}, batch_size=[])

    def _embed_chunked_with_backward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        predictor_k: "list[int]",
        loss_fn: "Callable[..., tuple[torch.Tensor, dict]]",
        micro_bs: int,
        alpha: float,
        also_boundary: bool = False,
        predictor_token_ids: "list | None" = None,
    ) -> dict:
        """Embed the joint (N, L) batch, compute a loss over it, and run backward
        — splitting the forward+backward into micro-batches of size `micro_bs`
        rows when N > micro_bs, so peak activation memory is bounded by one
        micro-batch instead of the full joint batch (the cause of the
        "Tried to allocate ... GiB" OOM during `scaled_loss.backward()` at full
        batch size: 64 prompts x up to 3 views/prompt x up to ~3072+predictor_k
        tokens each, all packed into a single forward, easily exceeds the
        94.97 GiB card once activations for backward are retained).

        Implements GradCache (Gao et al., "Scaling Deep Contrastive Learning
        Batch Size under Memory Limited Setup"): the JEPA/SIGReg/triplet losses
        are global statistics over the *whole* embedding pool, so they can't be
        computed independently per chunk — but the loss's gradient w.r.t. each
        row's embedding CAN be computed cheaply once the full (N, d) embedding
        tensor is assembled (d is tiny vs. activation memory). Three passes:
          1. Forward each chunk (no_grad — only the values are needed) to get
             its embeddings, cloned into a fresh leaf tensor per chunk.
          2. Concatenate the detached per-chunk leaves into the full (N, d)
             tensor, run the actual loss function on it (cheap — no transformer
             involved), and call .backward(). This populates `.grad` on each
             detached per-chunk leaf with exactly the gradient the full-batch
             loss would have produced for that chunk's rows.
          3. Re-forward each chunk (grad enabled, same inputs) and backward
             using that cached `.grad` as the upstream gradient — this is the
             ONLY pass that touches model parameters, and it only ever holds
             one chunk's activations at a time. The per-chunk `logits_anchor`
             is included in this same backward call (see `_extract_embeddings`)
             so FSDP1's root-module post-backward hook still fires correctly
             for every chunk.

        Trades 2x forward compute (each chunk's transformer forward runs twice:
        once to cache embeddings, once for the real backward) for activation
        memory bounded by `micro_bs` rows instead of N rows — exact, not an
        approximation, since the loss is computed once on the full pool.

        When ``also_boundary`` is True, the per-row BOUNDARY embedding (last real
        token, pre-[PRED]) is captured DETACHED from the SAME pass-1 forward (no
        extra forward) and ``loss_fn`` is called as ``loss_fn(joint_emb, boundary)``
        instead of ``loss_fn(joint_emb)``. Used by jepa-tcr-dual so the stop-grad
        self-consistency target z_self rides for free on the encode that already
        produces the [PRED] reads.

        Returns the metrics dict from `loss_fn`; backward is a side effect, the
        caller must still call `engine.optimizer_step()` afterward.
        """
        N = ids.shape[0]
        cfg = self.jepa_cfg
        # One shared append sequence (list[int]) or a per-row list of sequences
        # (list[list[int]]); the per-row form must ride the length-sort permutation
        # in lockstep with pk_s so each chunk still gets its own rows' tokens.
        ptids = cfg.predictor_token_ids if predictor_token_ids is None else predictor_token_ids
        per_row_pt = bool(ptids) and isinstance(ptids[0], (list, tuple))

        def _run_loss(emb, boundary):
            return loss_fn(emb, boundary) if also_boundary else loss_fn(emb)

        if N <= micro_bs:
            if also_boundary:
                joint_emb, boundary, logits_anchor = self._extract_embeddings(
                    ids, mask, lengths, use_ema=False, requires_grad=True,
                    predictor_k=predictor_k, predictor_token_ids=ptids,
                    also_boundary=True,
                )
                boundary = boundary.detach()
            else:
                joint_emb, logits_anchor = self._extract_embeddings(
                    ids, mask, lengths, use_ema=False, requires_grad=True,
                    predictor_k=predictor_k, predictor_token_ids=ptids,
                )
                boundary = None
            loss, metrics = _run_loss(joint_emb, boundary)
            (alpha * loss + logits_anchor).backward()
            return metrics

        # Sort rows by length so each chunk is padded to its own max length,
        # not (in expectation) the global max — same exact-result trick as
        # `_embed_chunked_no_grad`. The loss in pass 2 is computed on the pool
        # RESTORED to the caller's row order (loss_fn's masks/group ids index
        # the original layout); the un-permute is differentiable, so pass 2's
        # backward still lands the right per-row gradient on each sorted leaf.
        order = torch.argsort(lengths.long(), descending=True)
        order_list = order.tolist()
        ids_s, mask_s, len_s = ids[order], mask[order], lengths[order]
        pk_s = [predictor_k[i] for i in order_list]
        pt_s = [ptids[i] for i in order_list] if per_row_pt else None

        def _pt(start, end):
            return pt_s[start:end] if per_row_pt else ptids

        bounds = [(s, min(s + micro_bs, N)) for s in range(0, N, micro_bs)]

        # Pass 1: cache per-chunk embeddings under no_grad. Only the VALUES are
        # needed here (pass 2 differentiates w.r.t. fresh leaf tensors; pass 3
        # is the only pass that touches model parameters), so building a graph
        # or storing activations in this pass is pure waste — no_grad halves
        # pass-1 memory and lets larger micro-batches through unchanged.
        # The boundary read (stop-grad by design) is captured here too.
        cached = []
        boundary_chunks = [] if also_boundary else None
        with torch.no_grad():
            for start, end in bounds:
                if also_boundary:
                    chunk_emb, chunk_bnd, _ = self._extract_embeddings(
                        ids_s[start:end], mask_s[start:end], len_s[start:end], use_ema=False, requires_grad=False,
                        predictor_k=pk_s[start:end], predictor_token_ids=_pt(start, end),
                        also_boundary=True,
                    )
                    boundary_chunks.append(chunk_bnd)
                else:
                    chunk_emb, _ = self._extract_embeddings(
                        ids_s[start:end], mask_s[start:end], len_s[start:end], use_ema=False, requires_grad=False,
                        predictor_k=pk_s[start:end], predictor_token_ids=_pt(start, end),
                    )
                cached.append(chunk_emb.clone().requires_grad_(True))

        # Pass 2: compute the real loss on the full assembled pool (restored to
        # original row order), backward into the cached per-chunk leaves only
        # (cheap — no transformer graph).
        dev = cached[0].device
        inv = torch.empty(N, dtype=torch.long, device=dev)
        inv[order.to(dev)] = torch.arange(N, device=dev)
        joint_emb_cached = torch.cat(cached, dim=0)[inv]
        boundary_all = torch.cat(boundary_chunks, dim=0)[inv] if also_boundary else None
        loss, metrics = _run_loss(joint_emb_cached, boundary_all)
        (alpha * loss).backward()

        # Pass 3: re-forward each (sorted) chunk live and backprop the cached
        # gradient through it into the model parameters, one chunk's
        # activations at a time.
        for (start, end), cached_chunk in zip(bounds, cached):
            chunk_emb_live, chunk_logits_anchor = self._extract_embeddings(
                ids_s[start:end], mask_s[start:end], len_s[start:end], use_ema=False, requires_grad=True,
                predictor_k=pk_s[start:end], predictor_token_ids=_pt(start, end),
            )
            torch.autograd.backward(
                [chunk_emb_live, chunk_logits_anchor],
                [cached_chunk.grad, torch.ones((), device=chunk_logits_anchor.device, dtype=chunk_logits_anchor.dtype)],
            )

        return metrics

    # --------------------------------------------------------- JEPA update ---
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def jepa_update(self, data: TensorDict) -> TensorDict:
        """Run a JEPA-only forward+backward+optimizer step.

        Objective: ``jepa-tcr-dual`` (the only supported loss_type) — ONE live
        encoder for all views (no EMA/target network on the loss path), predictor
        tokens on the CoT and Code anchor rows so each view's p = Pred(Enc(student
        response)). Each view's pred is pulled toward its OWN cached teacher-correct
        target (align_cot / align_code), plus a self-consistency pull of pred_CoT
        toward the stop-grad Code_S boundary read, with SIGReg over [pred_cot,
        pred_code] for anti-collapse (see core_algos.llm_jepa_tcr_dual_loss).

        Args:
            data: TensorDict with keys:
                cot_input_ids  (N, L_cot)   — CoT prompt token ids (padded)
                cot_attn_mask  (N, L_cot)   — CoT attention mask
                cot_lengths    (N,)          — actual CoT sequence lengths
                code_input_ids (N, L_code)  — Code prompt+response token ids
                code_attn_mask (N, L_code)  — Code attention mask
                code_lengths   (N,)         — actual Code sequence lengths
                  where N = number of valid (cot_correct AND code_correct) pairs

        Returns:
            TensorDict with JEPA training metrics.
        """
        assert self.jepa_cfg is not None, "Call jepa_init() before jepa_update()"
        assert self.ema_weights is not None, "EMA not initialised"
        assert self.jepa_cfg.loss_type in (
            "llm-jepa", "llm-jepa-contrastive", "llm-jepa-infoNCE", "llm-jepa-geometry",
            "llm-jepa-triplet", "h-grpo", "jepa-tcr-dual", "jepa-tcr-cot"), (
            f"worker.jepa_update only supports loss_type 'llm-jepa', 'llm-jepa-contrastive', "
            f"'llm-jepa-infoNCE', 'llm-jepa-geometry', 'llm-jepa-triplet', 'h-grpo', "
            f"'jepa-tcr-dual' or 'jepa-tcr-cot', got {self.jepa_cfg.loss_type!r}"
        )
        if self.jepa_cfg.predictor_k > 0:
            assert len(self.jepa_cfg.predictor_token_ids) == self.jepa_cfg.predictor_k, (
                "predictor_k > 0 requires the resolved predictor_token_ids sequence; "
                "JEPARayPPOTrainer.init_workers() should have set this before jepa_init()"
            )

        # llm-jepa pads the anchor block to max(A, U) rows with lengths==0 markers
        # (see _build_jepa_batch_llm_jepa); count REAL anchors, not padded rows.
        if "anchor_lengths" in data:
            n_pairs = int((data["anchor_lengths"] > 0).sum().item())
        elif "anchor_input_ids" in data:
            n_pairs = data["anchor_input_ids"].shape[0]
        else:
            n_pairs = data["cot_input_ids"].shape[0]
        engine = self.actor.engine
        cfg = self.jepa_cfg
        # Fused mode (jepa.fuse_optimizer_step): the actor DEFERRED its optimizer step,
        # so the policy gradient is resident on .grad and unapplied — this call owns the
        # single combined step. Two-step mode: the actor already stepped.
        _fuse = data.get("fuse_optimizer_step", None)
        fuse = bool(float(_fuse[0].item())) if _fuse is not None else False
        if n_pairs < self.jepa_cfg.min_valid_pairs:
            # Two-step design: the actor already took its own optimizer step (policy
            # grad applied) before this call. With too few JEPA pairs there is no
            # auxiliary grad to add, so there is simply nothing to do here — do NOT
            # step again (that would re-apply the leftover, already-stepped grad).
            # FUSED design: the opposite — the policy grad has NOT been applied yet, so
            # returning here would silently drop this iteration's entire policy update.
            # Step it alone (no JEPA contribution to add).
            if fuse:
                with engine.train_mode():
                    actor_gn = engine.grad_norm_only()
                    _assert_policy_grad_resident(actor_gn)
                    engine.optimizer_step()
                    engine.lr_scheduler_step()
                return TensorDict(
                    {"jepa/skipped": torch.tensor(1.0),
                     "jepa/fused": torch.tensor(1.0),
                     "actor/grad_norm": torch.tensor(float(actor_gn)),
                     "jepa/grad_norm_total": torch.tensor(float(actor_gn)),
                     "jepa/n_valid_pairs": torch.tensor(float(n_pairs))},
                    batch_size=[],
                )
            return TensorDict(
                {"jepa/skipped": torch.tensor(1.0),
                 "jepa/n_valid_pairs": torch.tensor(float(n_pairs))},
                batch_size=[],
            )

        # Bounds peak activation memory during the joint forward+backward to
        # roughly `micro_bs` rows instead of the full joint batch (which can be
        # up to 3 views x train_batch_size rows in triplet mode) — see
        # `_embed_chunked_with_backward`'s docstring for the OOM this fixes.
        micro_bs = max(1, cfg.embed_micro_batch_size)

        # Linear alpha warmup: ramps the EFFECTIVE alpha from 0 -> cfg.alpha
        # over cfg.alpha_warmup_steps JEPA-update calls (disabled when 0, the
        # default — full alpha from the first call, matching prior behavior).
        # `global_step` is set by ray_trainer.py on the input TensorDict;
        # falls back to "no warmup" (full alpha) if absent for any reason.
        # The ramp is RELATIVE to the first update this process sees, so a run
        # resumed from step N still warms up over the next warmup_steps instead
        # of jumping straight to full alpha (min(step, N)/warmup would be 1).
        if cfg.alpha_warmup_steps > 0:
            default_step = torch.tensor([float(cfg.alpha_warmup_steps)])
            step = float(data.get("global_step", default_step)[0].item())
            if getattr(self, "_alpha_ramp_start", None) is None:
                self._alpha_ramp_start = step
            alpha = cfg.alpha * min(1.0, (step - self._alpha_ramp_start) / cfg.alpha_warmup_steps)
        else:
            alpha = cfg.alpha

        actor_gn = 0.0
        with engine.train_mode():
            # Two-step design: the actor (ray_trainer Step 4) already ran its own
            # complete optimizer step for the policy loss, which leaves its (applied)
            # gradient resident on .grad. Zero it here so the JEPA backward below and
            # the JEPA optimizer_step apply ONLY the auxiliary gradient — otherwise
            # the leftover policy grad would be double-applied.
            # predictor_delta is a param group on this same optimizer, so this
            # zeros its grad too — no separate zero_grad needed.
            #
            # FUSED design: the resident grad is the DEFERRED, un-applied policy grad.
            # Keep it — the JEPA backward below accumulates onto it and the single step
            # at the end applies the sum. Measure it first: this is the only point where
            # the policy contribution is separable, so it is what keeps actor/grad_norm
            # exact and comparable with the two-step runs.
            if fuse:
                actor_gn = engine.grad_norm_only()
                _assert_policy_grad_resident(actor_gn)
            else:
                engine.optimizer_zero_grad()
            # Arm the predictor-embedding hook for the JEPA forward only.
            self._predictor_hook_active = self.predictor_delta is not None

            # ONE live encoder for all views, predictor tokens on the CoT rows
            # only so p^c = Pred(Enc(correct CoT)). The cot/code/wrong blocks of
            # the input TensorDict are each padded to a common row count with
            # `*_lengths == 0` marking padding rows (see ray_trainer builders) —
            # filter those out before the joint forward so only real rows are
            # encoded. The joint forward + loss differ per loss_type below.
            if cfg.loss_type == "llm-jepa":
                # Paper-exact LLM-JEPA (arXiv:2509.14252 / official finetune.py),
                # adapted so the student's generated CoT predicts the PREGENERATED
                # big-teacher CoT and Code views. Fidelity notes:
                #   - NO stop-gradient: the reference concatenates Text/Code views
                #     into ONE differentiable forward. We replicate that by placing
                #     the anchor rows AND the (deduplicated) teacher-target rows in
                #     one joint GradCache block — pass 2's loss backward reaches the
                #     leaves of BOTH branches, and pass 3 pushes gradient into the
                #     model through anchors and targets alike.
                #   - Pred = tied-weight predictor via cfg.predictor_k appended
                #     [PRED] tokens on the ANCHOR rows only (targets read the plain
                #     last real token, i.e. Enc without Pred — as in the paper).
                #   - d(.,.) = cosine distance; flat mean; no SIGReg.
                # The trainer ships UNIQUE target rows plus per-anchor gather
                # indices (tgt_cot_idx / tgt_code_idx), so each distinct teacher
                # text is tokenized, padded, and encoded exactly once.
                # Both blocks were padded to a common row count R with zero-length
                # rows (real rows first) to satisfy the uniform TensorDict batch
                # dim; slice the real rows back out so padding is never forwarded.
                anchor_lengths_all = data["anchor_lengths"].long()
                target_lengths_all = data["target_lengths"].long()
                N = int((anchor_lengths_all > 0).sum().item())
                U = int((target_lengths_all > 0).sum().item())
                anchor_ids = data["anchor_input_ids"][:N]
                anchor_mask = data["anchor_attn_mask"][:N]
                anchor_lengths = anchor_lengths_all[:N]
                target_ids = data["target_input_ids"][:U]
                target_mask = data["target_attn_mask"][:U]
                target_lengths = target_lengths_all[:U]
                tgt_cot_idx = data["tgt_cot_idx"].long()[:N]
                tgt_code_idx = data["tgt_code_idx"].long()[:N]

                def _arm_on(key):
                    v = data.get(key, None)
                    return True if v is None else bool(v.reshape(-1)[0].item())
                align_cot_on = _arm_on("align_cot_on")
                align_code_on = _arm_on("align_code_on")

                joint_ids, joint_mask = self._pad_concat_batch(
                    anchor_ids, anchor_mask, target_ids, target_mask
                )
                joint_lengths = torch.cat([anchor_lengths, target_lengths])
                # [PRED]×k on anchors; targets are plain Enc reads (k=0).
                joint_predictor_k = [cfg.predictor_k] * N + [0] * U

                def _loss_fn(joint_emb, _N=N, _cot_idx=tgt_cot_idx, _code_idx=tgt_code_idx,
                             _cot_on=align_cot_on, _code_on=align_code_on):
                    dev = joint_emb.device
                    pred = joint_emb[:_N]
                    tgt = joint_emb[_N:]
                    z_cot = tgt[_cot_idx.to(dev)]
                    has_code = (_code_idx >= 0).to(dev)
                    z_code = tgt[_code_idx.clamp(min=0).to(dev)]
                    return llm_jepa_paper_loss(
                        pred=pred, target_cot=z_cot, target_code=z_code,
                        has_code=has_code,
                        align_cot_on=_cot_on, align_code_on=_code_on,
                    )

                jepa_metrics = self._embed_chunked_with_backward(
                    joint_ids, joint_mask, joint_lengths, joint_predictor_k,
                    _loss_fn, micro_bs, alpha,
                )
            elif cfg.loss_type == "llm-jepa-contrastive":
                # Three-arm loss over student CoT rollouts (see
                # core_algos.llm_jepa_contrastive_loss and
                # ray_trainer._build_jepa_batch_llm_jepa_contrastive). One joint block:
                # good anchors (<|predictor_i|>) | bad anchors (<|bad_predictor_i|>) |
                # unique teacher targets | per-prompt wrong targets — real rows first,
                # padding (lengths==0) sliced off.
                anchor_lengths_all = data["anchor_lengths"].long()
                target_lengths_all = data["target_lengths"].long()
                N = int((anchor_lengths_all > 0).sum().item())
                U = int((target_lengths_all > 0).sum().item())
                is_bad = data["anchor_is_bad"].long()[:N]
                is_wrong = data["target_is_wrong"].long()[:U]
                Ng = int((is_bad == 0).sum().item())
                Nb = N - Ng
                Ut = int((is_wrong == 0).sum().item())

                anchor_ids = data["anchor_input_ids"][:N]
                anchor_mask = data["anchor_attn_mask"][:N]
                anchor_lengths = anchor_lengths_all[:N]
                target_ids = data["target_input_ids"][:U]
                target_mask = data["target_attn_mask"][:U]
                target_lengths = target_lengths_all[:U]

                # Gather indices: first Ng anchor rows are good, next Nb are bad.
                tgt_cot_idx = data["tgt_cot_idx"].long()[:Ng]
                tgt_code_idx = data["tgt_code_idx"].long()[:Ng]
                bad_tgt_idx = data["bad_tgt_idx"].long()[Ng:N]
                good_group_id = data["anchor_group_id"].long()[:Ng]
                bad_group_id = data["anchor_group_id"].long()[Ng:N]
                cg_all = data["contrast_good_idx"].long()
                Nc = int((cg_all >= 0).sum().item())
                contrast_good_idx = cg_all[:Nc]
                contrast_bad_idx = data["contrast_bad_idx"].long()[:Nc]
                contrast_group_id = data["contrast_group_id"].long()[:Nc]

                joint_ids, joint_mask = self._pad_concat_batch(
                    anchor_ids, anchor_mask, target_ids, target_mask
                )
                joint_lengths = torch.cat([anchor_lengths, target_lengths])
                joint_predictor_k = [cfg.predictor_k] * N + [0] * U
                good_seq = list(cfg.predictor_token_ids)
                bad_seq = list(cfg.bad_predictor_token_ids)
                # Per-row append vocab: good anchors -> good tokens, bad -> bad tokens,
                # targets -> none (k=0). Rides the GradCache length-sort permutation.
                joint_pt = [good_seq] * Ng + [bad_seq] * Nb + [[]] * U

                def _loss_fn(joint_emb, _Ng=Ng, _Nb=Nb, _Ut=Ut, _Nc=Nc,
                             _cot=tgt_cot_idx, _code=tgt_code_idx, _bt=bad_tgt_idx,
                             _gg=good_group_id, _bg=bad_group_id,
                             _cgi=contrast_good_idx, _cbi=contrast_bad_idx, _cg=contrast_group_id):
                    dev = joint_emb.device
                    good_pred = joint_emb[:_Ng]
                    bad_pred = joint_emb[_Ng:_Ng + _Nb]
                    tgt = joint_emb[_Ng + _Nb:]        # U target rows: teacher [:Ut], wrong [Ut:]
                    teacher = tgt[:_Ut]
                    wrong = tgt[_Ut:]
                    z_cot = teacher[_cot.to(dev)]
                    has_code = (_code >= 0).to(dev)
                    z_code = teacher[_code.clamp(min=0).to(dev)]
                    bad_target = wrong[_bt.to(dev)] if _Nb > 0 else bad_pred
                    c_good = good_pred[_cgi.to(dev)] if _Nc > 0 else good_pred[:0]
                    c_bad = bad_pred[_cbi.to(dev)] if _Nc > 0 else bad_pred[:0]
                    return llm_jepa_contrastive_loss(
                        good_pred=good_pred, target_cot=z_cot, target_code=z_code,
                        has_code=has_code, good_group_id=_gg.to(dev),
                        bad_pred=bad_pred, bad_target=bad_target, bad_group_id=_bg.to(dev),
                        contrast_good=c_good, contrast_bad=c_bad, contrast_group_id=_cg.to(dev),
                        # Anti-collapse: SIGReg over ALL unique reads in this joint forward
                        # (good_pred + bad_pred + teacher + wrong) — joint_emb itself.
                        sig_pool=joint_emb, sigreg_lambda=cfg.triplet_sigreg_lambda,
                        sig_M=cfg.n_projections, sig_t_min=cfg.t_min, sig_t_max=cfg.t_max,
                        sig_s=cfg.epps_pulley_s,
                    )

                jepa_metrics = self._embed_chunked_with_backward(
                    joint_ids, joint_mask, joint_lengths, joint_predictor_k,
                    _loss_fn, micro_bs, alpha, predictor_token_ids=joint_pt,
                )
            elif cfg.loss_type == "h-grpo":
                # One flat student block (M rollout rows, ALL of them -- not correct/wrong
                # pairs) plus a deduplicated teacher block with one anchor per GROUP.
                # Every row gets the SAME read operator: predictor_k is forced to 0 by
                # config validation, because appending a label-dependent [PRED]/[BAD_PRED]
                # (as the infoNCE/triplet branch below does) would let the group softmax
                # separate rollouts by their appended token instead of their content.
                anchor_lengths_all = data["anchor_lengths"].long()
                target_lengths_all = data["target_lengths"].long()
                M = int((anchor_lengths_all > 0).sum().item())
                U = int((target_lengths_all > 0).sum().item())
                tgt_group_idx = data["tgt_cot_idx"].long()
                tgt_group_idx = tgt_group_idx[tgt_group_idx >= 0]   # (P,) one per group
                student_label = data["student_label"].float()[:M]
                student_gid = data["student_group_id"].long()[:M]
                assert int(student_gid.max().item()) + 1 == tgt_group_idx.shape[0], (
                    f"h-grpo: {int(student_gid.max().item()) + 1} groups in student_group_id "
                    f"but {tgt_group_idx.shape[0]} teacher anchors"
                )

                joint_ids, joint_mask = self._pad_concat_batch(
                    data["anchor_input_ids"][:M], data["anchor_attn_mask"][:M],
                    data["target_input_ids"][:U], data["target_attn_mask"][:U],
                )
                joint_lengths = torch.cat([anchor_lengths_all[:M], target_lengths_all[:U]])
                joint_predictor_k = [0] * (M + U)

                def _loss_fn(joint_emb, _M=M, _tidx=tgt_group_idx,
                             _lab=student_label, _gid=student_gid):
                    dev = joint_emb.device
                    return hgrpo_loss(
                        student=joint_emb[:_M],
                        label=_lab.to(dev),
                        teacher=joint_emb[_M:][_tidx.to(dev)],
                        group_id=_gid.to(dev),
                    )

                jepa_metrics = self._embed_chunked_with_backward(
                    joint_ids, joint_mask, joint_lengths, joint_predictor_k,
                    _loss_fn, micro_bs, alpha, predictor_token_ids=None,
                )
            elif cfg.loss_type in ("llm-jepa-infoNCE", "llm-jepa-geometry", "llm-jepa-triplet"):
                # Single discriminative InfoNCE over mixed prompts (see
                # core_algos.llm_jepa_infonce_loss and
                # ray_trainer._build_jepa_batch_llm_jepa_infonce). Anchor block =
                # M positive rollouts (read via <|predictor_i|>) then their M
                # negatives (read via <|bad_predictor_i|>). CoT and Code teacher
                # rows share the joint forward for efficiency, but the geometry loss
                # detaches both teacher views: they are fixed anchors for attraction
                # and repulsion. SIGReg receives only the correct/wrong student rows,
                # so anti-collapse gradients cannot leak into teacher targets. The
                # InfoNCE sibling keeps its existing differentiable-target behavior.
                anchor_lengths_all = data["anchor_lengths"].long()
                target_lengths_all = data["target_lengths"].long()
                N = int((anchor_lengths_all > 0).sum().item())
                U = int((target_lengths_all > 0).sum().item())
                M = N // 2
                assert N == 2 * M, f"infoNCE anchor block must be 2M rows, got {N}"
                anchor_ids = data["anchor_input_ids"][:N]
                anchor_mask = data["anchor_attn_mask"][:N]
                anchor_lengths = anchor_lengths_all[:N]
                target_ids = data["target_input_ids"][:U]
                target_mask = data["target_attn_mask"][:U]
                target_lengths = target_lengths_all[:U]
                tgt_idx = data["tgt_cot_idx"].long()[:M]
                tgt_code_idx = data["tgt_code_idx"].long()[:M]
                has_code = tgt_code_idx >= 0
                pair_group_id = data["pair_group_id"].long()[:M]

                joint_ids, joint_mask = self._pad_concat_batch(
                    anchor_ids, anchor_mask, target_ids, target_mask
                )
                joint_lengths = torch.cat([anchor_lengths, target_lengths])
                if cfg.predictor_k > 0:
                    good_seq = list(cfg.predictor_token_ids)
                    bad_seq = list(cfg.bad_predictor_token_ids)
                    # [PRED] on good anchors, [BAD_PRED] on wrong anchors; targets are
                    # plain Enc reads (k=0).
                    joint_predictor_k = [cfg.predictor_k] * N + [0] * U
                    joint_pt = [good_seq] * M + [bad_seq] * M + [[]] * U
                else:
                    # Identity predictor Pred(x)=x: no appended token anywhere, every read
                    # is the last CONTENT token. p+/p- differ only by rollout correctness.
                    joint_predictor_k = [0] * (N + U)
                    joint_pt = None  # _embed_chunked_with_backward falls back to [] -> last-token

                def _loss_fn(joint_emb, _N=N, _M=M, _tidx=tgt_idx, _cidx=tgt_code_idx,
                             _hc=has_code, _gid=pair_group_id):
                    dev = joint_emb.device
                    tgt = joint_emb[_N:]
                    teacher_z = tgt[_tidx.to(dev)]
                    teacher_zc = tgt[_cidx.clamp(min=0).to(dev)]
                    if cfg.loss_type == "llm-jepa-geometry":
                        return llm_jepa_geometry_loss(
                            correct=joint_emb[:_M],
                            wrong=joint_emb[_M:_N],
                            teacher=teacher_z,
                            teacher_code=teacher_zc,
                            has_code=_hc,
                            group_id=_gid,
                            tau=cfg.geometry_tau,
                            margin=cfg.geometry_margin,
                            align_weight=cfg.geometry_align_weight,
                            wrong_teacher_weight=cfg.geometry_wrong_teacher_weight,
                            wrong_correct_weight=cfg.geometry_wrong_correct_weight,
                            sig_pool=joint_emb[:_N],
                            sigreg_lambda=cfg.triplet_sigreg_lambda,
                            sig_M=cfg.n_projections,
                            sig_t_min=cfg.t_min,
                            sig_t_max=cfg.t_max,
                            sig_s=cfg.epps_pulley_s,
                        )
                    if cfg.loss_type == "llm-jepa-triplet":
                        # Teacher-anchored triplet: paper align (correct->teacher,
                        # both views) + hinge (wrong >= margin farther than correct).
                        # triplet_detach_teacher controls whether the anchor is frozen.
                        return llm_jepa_triplet_loss(
                            correct=joint_emb[:_M],
                            wrong=joint_emb[_M:_N],
                            teacher=teacher_z,
                            teacher_code=teacher_zc,
                            has_code=_hc,
                            group_id=_gid,
                            tau=cfg.triplet_tau,
                            margin=cfg.triplet_margin,
                            repel_weight=cfg.triplet_repel_weight,
                            include_cot=cfg.triplet_include_cot,
                            detach_teacher=cfg.triplet_detach_teacher,
                            sig_pool=joint_emb[:_N],
                            sigreg_lambda=cfg.triplet_sigreg_lambda,
                            sig_M=cfg.n_projections,
                            sig_t_min=cfg.t_min,
                            sig_t_max=cfg.t_max,
                            sig_s=cfg.epps_pulley_s,
                        )
                    return llm_jepa_infonce_loss(
                        pred_pos=joint_emb[:_M],
                        pred_neg=joint_emb[_M:_N],
                        teacher=teacher_z,
                        teacher_code=teacher_zc,
                        has_code=_hc,
                        tau=cfg.infonce_tau,
                    )

                jepa_metrics = self._embed_chunked_with_backward(
                    joint_ids, joint_mask, joint_lengths, joint_predictor_k,
                    _loss_fn, micro_bs, alpha, predictor_token_ids=joint_pt,
                )
            elif cfg.loss_type in ("jepa-tcr-dual", "jepa-tcr-cot"):
                # Dual-target self-consistent TCR ('jepa-tcr-cot' is the same path with
                # an empty code arm: is_code all-False, self_mask all-False, so the loss
                # degrades to align_cot + SIGReg(pred_cot)).
                # One combined anchor block (CoT rows
                # first, then Code rows; `is_code` recovers the split), each row with
                # [PRED]. Each view's pred is pulled toward its OWN verified-correct
                # target z^+, recomputed ONLINE by the current policy from the stored
                # y^+ text (stop-grad, no EMA); each CoT pred is additionally pulled
                # toward the stop-grad Code_S BOUNDARY read of its paired code rollout
                # (self-consistency). SIGReg over both views' preds.
                anchor_ids = data["anchor_input_ids"]
                anchor_mask = data["anchor_attn_mask"]
                anchor_lengths = data["anchor_lengths"]
                is_code = data["is_code"].bool()
                cot_sel = ~is_code
                N = anchor_ids.shape[0]

                # Corrected TCR-JEPA: recompute the verified-correct targets z^+ ONLINE
                # with the CURRENT policy (no EMA), under stop-gradient, instead of
                # reading stale latents cached by a frozen reference encoder. The target
                # block [x, y^+] is encoded last-real-token + L2-normalized (predictor_k=0,
                # matching the prediction branch's pooling), so targets always live in the
                # policy's present representation space. Rows are CoT-first then Code,
                # aligned 1:1 with the anchor block (see ray_trainer target builder). The
                # no-grad forward before the grad anchor forward mirrors the lejepa EMA
                # path and keeps FSDP1 to one backward per step.
                target_ids = data["target_input_ids"]
                target_mask = data["target_attn_mask"]
                target_lengths = data["target_lengths"]
                with torch.no_grad():
                    # Dedup identical target rows before encoding: the batch builder
                    # emits one target row PER ANCHOR, and anchors of the same prompt
                    # frequently map to the same y^+ text (k % n_u matching), so the
                    # unique row count is often a small fraction of the block. Encode
                    # each unique row once and scatter back — exact, order-preserving.
                    tgt_lens = target_lengths.long().tolist()
                    keys = [
                        target_ids[b, : tgt_lens[b]].cpu().numpy().tobytes()
                        for b in range(target_ids.shape[0])
                    ]
                    first_row: dict[bytes, int] = {}
                    uniq_rows: list[int] = []
                    inverse: list[int] = []
                    for b, k in enumerate(keys):
                        if k not in first_row:
                            first_row[k] = len(uniq_rows)
                            uniq_rows.append(b)
                        inverse.append(first_row[k])
                    uniq_idx = torch.tensor(uniq_rows, dtype=torch.long)
                    uniq_emb = self._embed_chunked_no_grad(
                        target_ids[uniq_idx], target_mask[uniq_idx], target_lengths[uniq_idx],
                        use_ema=False, micro_bs=micro_bs, predictor_k=0,
                    )
                    teacher_target = uniq_emb[torch.tensor(inverse, dtype=torch.long, device=uniq_emb.device)]

                # z_self (the Code_S boundary read, stop-grad) rides on the SAME forward
                # that produces the [PRED] reads: `_embed_chunked_with_backward(..,
                # also_boundary=True)` hands the loss_fn the per-row boundary embeddings
                # (last real token, pre-[PRED]) for free — no separate forward. The
                # loss_fn gathers each CoT row's partner (self_partner indexes the code
                # sub-block) from the boundary of the corresponding ABSOLUTE code row.
                joint_predictor_k = [cfg.predictor_k] * N
                group_id = data.get("anchor_group_id", None)
                is_correct = data.get("anchor_is_correct", None)
                self_partner = data.get("self_partner", None)
                # Per-arm plateau latches (broadcast scalars from the trainer). When an
                # arm is off, its align term contributes 0 grad but preds stay in SIGReg.
                def _arm_on(key):
                    v = data.get(key, None)
                    return True if v is None else bool(v.reshape(-1)[0].item())
                align_cot_on = _arm_on("align_cot_on")
                align_code_on = _arm_on("align_code_on")
                self_on = _arm_on("self_on")
                # Absolute row indices of the code sub-block (code rows are last, in order).
                code_abs = is_code.nonzero(as_tuple=True)[0]
                cot_rows = cot_sel.nonzero(as_tuple=True)[0]
                n_cot = int(cot_rows.numel())
                n_code = int(code_abs.numel())

                def _loss_fn(joint_emb, boundary, _cot=cot_sel, _code=is_code,
                             _tt=teacher_target, _gid=group_id, _ic=is_correct,
                             _sp=self_partner, _code_abs=code_abs, _cot_rows=cot_rows,
                             _n_cot=n_cot, _n_code=n_code,
                             _cot_on=align_cot_on, _code_on=align_code_on, _self_on=self_on):
                    dev, dt = joint_emb.device, joint_emb.dtype
                    tt = _tt.to(device=dev, dtype=dt)
                    gid = _gid.to(dev) if _gid is not None else None
                    ic = _ic.to(dev) if _ic is not None else None
                    # Build z_self from the boundary reads of the SAME forward. Boundary
                    # is detached; partner index (into the code sub-block) -> absolute row.
                    d = joint_emb.shape[-1]
                    self_target = boundary.new_zeros((_n_cot, d))
                    self_mask = torch.zeros(_n_cot, dtype=torch.bool, device=dev)
                    if _sp is not None:
                        sp_cot = _sp.to(device=dev, dtype=torch.long)[_cot_rows.to(dev)]
                        has = (sp_cot >= 0) & (sp_cot < _n_code)
                        if has.any():
                            rows = torch.arange(_n_cot, device=dev)[has]
                            abs_idx = _code_abs.to(dev)[sp_cot[has]]
                            self_target[rows] = boundary[abs_idx]
                            self_mask[rows] = True
                    return llm_jepa_tcr_dual_loss(
                        pred_cot=joint_emb[_cot],
                        teacher_target_cot=tt[_cot],
                        pred_code=joint_emb[_code],
                        teacher_target_code=tt[_code],
                        self_target=self_target,
                        self_mask=self_mask,
                        cot_group_id=gid[_cot] if gid is not None else None,
                        cot_is_correct=ic[_cot] if ic is not None else None,
                        code_group_id=gid[_code] if gid is not None else None,
                        code_is_correct=ic[_code] if ic is not None else None,
                        self_consist_w=cfg.self_consist_w,
                        align_cot_on=_cot_on,
                        align_code_on=_code_on,
                        self_on=_self_on,
                        lambda_=cfg.triplet_sigreg_lambda,
                        M=cfg.n_projections,
                        t_min=cfg.t_min, t_max=cfg.t_max, s=cfg.epps_pulley_s,
                    )

                jepa_metrics = self._embed_chunked_with_backward(
                    anchor_ids, anchor_mask, anchor_lengths, joint_predictor_k, _loss_fn,
                    micro_bs, alpha, also_boundary=True,
                )

            # TWO-STEP: apply ONLY the JEPA gradient (grad was zeroed at the top of this
            # train_mode; the actor already stepped the policy grad in ray_trainer
            # Step 4). jepa/grad_norm is the α·JEPA gradient norm; the policy step's
            # norm is logged as actor/grad_norm by the actor update.
            # FUSED: .grad now holds g_policy + α·g_JEPA. One clip (the actor's own
            # optim clip_grad — jepa.max_grad_norm and the ratio-matched clip below are
            # both bypassed) and one optimizer step over the sum.
            self._predictor_hook_active = False  # disarm before any non-JEPA forward
            # Ratio-matched clip: cap the JEPA update at ratio * the live actor
            # grad-norm (passed in from ray_trainer Step 4) so the auxiliary step
            # auto-balances against the policy step whether the policy runs
            # suppressed (~0.02) or healthy (~0.29), instead of being pinned at a
            # fixed ceiling. JEPA_CLIP_RATIO=0 (default) keeps the old fixed clip.
            # ponytail: ratio=1 matches magnitudes; tune the env if JEPA should be
            # deliberately sub/super the policy step.
            import os as _os
            _agn = data.get("actor_grad_norm", None)
            _agn = float(_agn[0].item()) if _agn is not None else 0.0
            if fuse:
                # Single clip over the summed gradient = the actor's own optim
                # clip_grad. A JEPA-specific ceiling here would be a second trust
                # region on a loss that no longer has its own step.
                clip_used = None
            else:
                clip_used = self.jepa_cfg.max_grad_norm
                _ratio = float(_os.environ.get("JEPA_CLIP_RATIO", "0"))
                if _ratio > 0.0 and _agn > 0.0:
                    clip_used = min(clip_used, _ratio * _agn)
            # predictor_delta is a plain (non-FSDP) replicated param, not covered by
            # FSDP's own gradient-sync hooks — average its grad across DP ranks
            # manually (no-op when DP size is 1) BEFORE the shared optimizer step
            # below, which now updates it together with every other trainable
            # parameter (same optimizer, same lr, no special clip/cap).
            pred_gn = 0.0    # grad norm (post DP-avg), for logging only
            _prev_delta = None
            if self.predictor_delta is not None and self.predictor_delta.grad is not None:
                dp_size = engine.get_data_parallel_size()
                if dp_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        self.predictor_delta.grad, op=torch.distributed.ReduceOp.SUM,
                        group=engine.get_data_parallel_group())
                    self.predictor_delta.grad.div_(dp_size)
                pred_gn = float(self.predictor_delta.grad.detach().norm().cpu())
                _prev_delta = self.predictor_delta.data.clone()  # for ||Δθ|| applied

            grad_norm = engine.optimizer_step(clip_grad_override=clip_used)
            if fuse:
                # The actor deferred its step, so engine_workers skipped the LR
                # scheduler too (it is gated on apply_step). Advance it here or the LR
                # freezes — invisible under the constant schedule this config uses, a
                # real bug under warmup/cosine.
                engine.lr_scheduler_step()

            pred_pn = 0.0     # delta param norm after the shared step
            pred_max_row = 0.0  # largest per-row delta norm
            pred_step = 0.0   # ||Δθ|| actually applied to the deltas this step
            if _prev_delta is not None:
                pred_pn = float(self.predictor_delta.data.norm().cpu())
                pred_max_row = float(self.predictor_delta.data.norm(dim=-1).max().cpu())
                pred_step = float((self.predictor_delta.data - _prev_delta).norm().cpu())
            self._sync_ema()
            aggressive_empty_cache(force_sync=True)
            _total_loss_value = jepa_metrics.get("jepa/llm_jepa_loss", 0.0)
            _gn = float(grad_norm) if grad_norm is not None else 0.0
            out = {
                "jepa/total_loss": torch.tensor(float(_total_loss_value)),
                "jepa/n_valid_pairs": torch.tensor(float(n_pairs)),
                "jepa/skipped": torch.tensor(0.0),
                "jepa/actor_grad_norm_seen": torch.tensor(float(_agn)),
                "jepa/pred_embed_grad_norm": torch.tensor(float(pred_gn)),
                "jepa/pred_embed_norm": torch.tensor(float(pred_pn)),
                "jepa/pred_embed_max_row_norm": torch.tensor(float(pred_max_row)),
                "jepa/pred_embed_step_norm": torch.tensor(float(pred_step)),
            }
            if fuse:
                # grad_norm here is the PRE-CLIP norm of the SUM, not a JEPA-only norm —
                # publish it under a distinct key so the jepa/grad_norm series keeps one
                # meaning (α·JEPA alone) across runs. actor/grad_norm is the exact
                # policy-only norm measured before the JEPA backward, same definition the
                # actor's own step reports in two-step mode.
                out["jepa/fused"] = torch.tensor(1.0)
                out["jepa/grad_norm_total"] = torch.tensor(_gn)
                out["actor/grad_norm"] = torch.tensor(float(actor_gn))
            else:
                out["jepa/fused"] = torch.tensor(0.0)
                out["jepa/grad_norm"] = torch.tensor(_gn)
                out["jepa/clip_used"] = torch.tensor(float(clip_used))
            out.update({k: torch.tensor(float(v)) for k, v in jepa_metrics.items()})
            return TensorDict(out, batch_size=[])
