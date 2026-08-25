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
"""CPU tests for loss_type='llm-jepa-cluster' (correct-cluster LLM-JEPA)."""

import pytest
import torch
import torch.nn.functional as F

from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.experimental.jepa_grpo.core_algos import llm_jepa_cluster_loss


def _cfg(**kw):
    base = {"enable": True, "loss_type": "llm-jepa-cluster", "predictor_k": 0, "n_code": 0}
    base.update(kw)
    return JEPARayConfig.from_config(base)


# ---------------------------------------------------------------- config ----

def test_cluster_needs_no_teacher_cache():
    """The only loss_type with no teacher target, so it must bypass the shared check."""
    _cfg().validate(rollout_n=8)


@pytest.mark.parametrize("kw,match", [
    ({"predictor_k": 2}, "predictor_k"),
    ({"n_code": 4}, "n_code"),
    ({"min_correct_for_cluster": 1}, "min_correct_for_cluster"),
])
def test_cluster_rejects_bad_config(kw, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**kw).validate(rollout_n=8)


def test_ordering_flags_default_off():
    """Defaults must reproduce the existing behaviour: fused off, JEPA after policy."""
    c = JEPARayConfig()
    assert c.update_before_policy is False
    assert c.fuse_optimizer_step is False


# ------------------------------------------------------------------ loss ----

def test_matches_naive_pairwise_definition():
    """Closed form == the literal mean over ordered pairs, per prompt then over prompts."""
    torch.manual_seed(3)
    reps = torch.randn(9, 32)
    gid = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2])
    loss, _ = llm_jepa_cluster_loss(reps, gid)

    z = F.normalize(reps.float(), dim=-1)
    per_prompt = []
    for p in gid.unique().tolist():
        rows = (gid == p).nonzero().flatten().tolist()
        pairs = [1.0 - float(z[i] @ z[j]) for i in rows for j in rows if i != j]
        per_prompt.append(sum(pairs) / len(pairs))
    assert float(loss) == pytest.approx(sum(per_prompt) / len(per_prompt), abs=1e-5)


def test_eligibility_gate_excludes_singletons():
    """A prompt with one correct response has no pair and must not contribute."""
    reps = torch.randn(5, 16)
    gid = torch.tensor([0, 0, 1, 2, 2])          # prompt 1 is a singleton
    _, m = llm_jepa_cluster_loss(reps, gid)
    assert m["jepa/n_eligible_prompts"] == 2


def test_per_prompt_averaging_not_flat_over_pairs():
    """A 4-correct group (12 pairs) must not outweigh a 2-correct group (2 pairs)."""
    tight = F.normalize(torch.tensor([[1.0, 0.01], [1.0, -0.01]]), dim=-1)
    spread = F.normalize(torch.randn(4, 2), dim=-1)
    reps = torch.cat([tight, spread])
    gid = torch.tensor([0, 0, 1, 1, 1, 1])
    loss, _ = llm_jepa_cluster_loss(reps, gid)

    d_tight = 1.0 - float(F.normalize(tight, dim=-1)[0] @ F.normalize(tight, dim=-1)[1])
    zs = F.normalize(spread, dim=-1)
    pr = [1.0 - float(zs[i] @ zs[j]) for i in range(4) for j in range(4) if i != j]
    d_spread = sum(pr) / len(pr)
    # equal weight per prompt, NOT weighted 2:12 by pair count
    assert float(loss) == pytest.approx((d_tight + d_spread) / 2, abs=1e-5)


def test_no_stop_gradient_on_either_side():
    """Both members of every pair must receive gradient (paper has no stop-grad)."""
    reps = torch.randn(6, 16, requires_grad=True)
    gid = torch.tensor([0, 0, 0, 1, 1, 2])
    llm_jepa_cluster_loss(reps, gid)[0].backward()
    per_row = reps.grad.abs().sum(dim=-1)
    assert (per_row[:5] > 0).all()
    assert float(per_row[5]) == 0.0          # ineligible singleton


def test_groups_are_isolated():
    """No cross-prompt term: perturbing one prompt cannot change another's distance."""
    torch.manual_seed(7)
    reps = torch.randn(6, 16)
    gid = torch.tensor([0, 0, 0, 1, 1, 1])
    a = llm_jepa_cluster_loss(reps[:3], gid[:3])[0]
    reps2 = reps.clone()
    reps2[3:] = torch.randn(3, 16)
    b = llm_jepa_cluster_loss(reps2[:3], gid[:3])[0]
    assert float(a) == pytest.approx(float(b), abs=1e-7)


def test_bf16_reads_do_not_destroy_the_signal():
    """Reads arrive bf16 from the FSDP forward; loss lives near 1.0 where bf16 steps
    by 2^-8. Both the accumulation and the returned scalar must stay fp32."""
    torch.manual_seed(11)
    reps = torch.randn(8, 1536).bfloat16()
    gid = torch.zeros(8, dtype=torch.long)
    exact = float(llm_jepa_cluster_loss(reps.double(), gid)[0])
    got, _ = llm_jepa_cluster_loss(reps, gid)
    assert got.dtype == torch.float32
    assert float(got) == pytest.approx(exact, abs=1e-5)


def test_collapse_is_observable():
    """Attraction-only has no anti-collapse term by design, so the metrics must be
    able to reveal collapse: identical reads everywhere drive both cosines to 1."""
    v = F.normalize(torch.randn(1, 16), dim=-1).repeat(6, 1)
    gid = torch.tensor([0, 0, 0, 1, 1, 1])
    loss, m = llm_jepa_cluster_loss(v, gid)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    assert m["jepa/cos_correct_correct"] == pytest.approx(1.0, abs=1e-4)
    assert m["jepa/cos_all_pairs"] == pytest.approx(1.0, abs=1e-4)

    spread = F.normalize(torch.randn(6, 16), dim=-1)
    _, ms = llm_jepa_cluster_loss(spread, gid)
    assert ms["jepa/cos_all_pairs"] < m["jepa/cos_all_pairs"]


def test_empty_and_all_ineligible_keep_backward_valid():
    reps = torch.randn(3, 16, requires_grad=True)
    loss, m = llm_jepa_cluster_loss(reps, torch.tensor([0, 1, 2]))   # all singletons
    assert float(loss) == 0.0 and m["jepa/n_eligible_prompts"] == 0
    loss.backward()                                                  # must not raise
    assert reps.grad is not None

    l0, m0 = llm_jepa_cluster_loss(torch.zeros(0, 16), torch.zeros(0, dtype=torch.long))
    assert float(l0) == 0.0 and m0["jepa/pool_size"] == 0
