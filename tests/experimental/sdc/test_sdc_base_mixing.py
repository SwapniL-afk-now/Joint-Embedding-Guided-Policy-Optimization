from copy import deepcopy

import torch
from torch import nn

from verl.experimental.sdc.config import SDCConfig
from verl.experimental.sdc.sdc_state import (
    build_sdc_checkpoint_state,
    clone_frozen_state,
    mix_module_with_base,
    mix_state_with_base,
)


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    else:
        assert left == right


def test_exact_gamma_interpolation():
    destination = {"weight": torch.tensor([5.0, 7.0])}
    base = {"weight": torch.tensor([1.0, 3.0])}
    mix_state_with_base(destination, base, gamma=0.9)
    torch.testing.assert_close(destination["weight"], torch.tensor([4.6, 6.6]))


def test_gamma_one_is_sft_and_gamma_zero_is_frozen_base():
    base = {"weight": torch.tensor([1.0, 3.0])}
    sft = {"weight": torch.tensor([5.0, 7.0])}

    gamma_one = {"weight": sft["weight"].clone()}
    mix_state_with_base(gamma_one, base, gamma=1.0)
    torch.testing.assert_close(gamma_one["weight"], sft["weight"], rtol=0, atol=0)

    gamma_zero = {"weight": sft["weight"].clone()}
    mix_state_with_base(gamma_zero, base, gamma=0.0)
    torch.testing.assert_close(gamma_zero["weight"], base["weight"], rtol=0, atol=0)


def test_success_and_failure_use_one_immutable_base_across_refreshes():
    base = clone_frozen_state({"weight": torch.tensor([1.0, 3.0])})
    base_before = deepcopy(base)
    success = {"weight": torch.tensor([5.0, 7.0])}
    failure = {"weight": torch.tensor([-2.0, 9.0])}
    mix_state_with_base(success, base, gamma=0.9)
    mix_state_with_base(failure, base, gamma=0.9)

    expected_success = 0.9 * torch.tensor([5.0, 7.0]) + 0.1 * base["weight"]
    expected_failure = 0.9 * torch.tensor([-2.0, 9.0]) + 0.1 * base["weight"]
    torch.testing.assert_close(success["weight"], expected_success)
    torch.testing.assert_close(failure["weight"], expected_failure)
    _assert_nested_equal(base, base_before)

    second_success = {"weight": torch.tensor([12.0, -1.0])}
    second_failure = {"weight": torch.tensor([4.0, 8.0])}
    mix_state_with_base(second_success, base, gamma=0.9)
    mix_state_with_base(second_failure, base, gamma=0.9)
    _assert_nested_equal(base, base_before)


def test_actor_updates_do_not_mutate_frozen_base():
    actor = {"weight": torch.tensor([1.0, 3.0])}
    base = clone_frozen_state(actor)
    actor["weight"].add_(10.0)
    torch.testing.assert_close(base["weight"], torch.tensor([1.0, 3.0]), rtol=0, atol=0)


def test_optimizer_state_survives_base_mixing():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_before = deepcopy(optimizer.state_dict())
    base = clone_frozen_state(model.state_dict())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3.0)
    mix_module_with_base(model, base, gamma=0.9)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)


def test_checkpoint_resume_preserves_both_branches_base_gamma_and_counters(tmp_path):
    actor = nn.Linear(2, 1)
    base = clone_frozen_state(actor.state_dict())
    success = nn.Linear(2, 1)
    failure = nn.Linear(2, 1)
    success_optimizer = torch.optim.AdamW(success.parameters(), lr=0.1)
    failure_optimizer = torch.optim.AdamW(failure.parameters(), lr=0.1)

    for model, optimizer in ((success, success_optimizer), (failure, failure_optimizer)):
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    checkpoint = build_sdc_checkpoint_state(
        global_step=7,
        success_model_update_count=4,
        failure_model_update_count=4,
        models_ready=True,
        base_state=base,
        base_mix_gamma=0.9,
        success_buffer_state={"n": 2},
        failure_buffer_state={"n": 3},
    )
    path = tmp_path / "sdc.pt"
    torch.save(
        {
            "state": checkpoint,
            "success": success.state_dict(),
            "failure": failure.state_dict(),
            "success_optimizer": success_optimizer.state_dict(),
            "failure_optimizer": failure_optimizer.state_dict(),
        },
        path,
    )
    restored = torch.load(path, map_location="cpu", weights_only=False)
    restored_success = nn.Linear(2, 1)
    restored_failure = nn.Linear(2, 1)
    restored_success_optimizer = torch.optim.AdamW(restored_success.parameters(), lr=0.1)
    restored_failure_optimizer = torch.optim.AdamW(restored_failure.parameters(), lr=0.1)
    restored_success.load_state_dict(restored["success"])
    restored_failure.load_state_dict(restored["failure"])
    restored_success_optimizer.load_state_dict(restored["success_optimizer"])
    restored_failure_optimizer.load_state_dict(restored["failure_optimizer"])

    _assert_nested_equal(restored_success.state_dict(), success.state_dict())
    _assert_nested_equal(restored_failure.state_dict(), failure.state_dict())
    _assert_nested_equal(restored_success_optimizer.state_dict(), success_optimizer.state_dict())
    _assert_nested_equal(restored_failure_optimizer.state_dict(), failure_optimizer.state_dict())
    _assert_nested_equal(restored["state"]["sdc_base_state_cpu"], base)
    assert restored["state"]["base_mix_gamma"] == 0.9
    assert restored["state"]["sdc_success_model_update_count"] == 4
    assert restored["state"]["sdc_failure_model_update_count"] == 4
    assert restored["state"]["sdc_models_ready"] is True

    with torch.no_grad():
        restored_success.weight.add_(2.0)
    mix_module_with_base(restored_success, restored["state"]["sdc_base_state_cpu"], 0.9)
    expected = 0.9 * (success.weight.detach() + 2.0) + 0.1 * base["weight"]
    torch.testing.assert_close(restored_success.weight, expected)


def test_unmatched_side_does_not_mix_either_branch():
    base = {"weight": torch.tensor([1.0])}
    success = {"weight": torch.tensor([5.0])}
    failure = {"weight": torch.tensor([7.0])}
    before_success, before_failure = success["weight"].clone(), failure["weight"].clone()
    success_records, failure_records = [], [{"response": "only failure"}]
    if success_records and failure_records:
        mix_state_with_base(success, base, 0.9)
        mix_state_with_base(failure, base, 0.9)
    torch.testing.assert_close(success["weight"], before_success, rtol=0, atol=0)
    torch.testing.assert_close(failure["weight"], before_failure, rtol=0, atol=0)


def test_base_mix_gamma_validation_is_explicit():
    try:
        SDCConfig(base_mix_gamma=1.01)
    except ValueError as exc:
        assert "base_mix_gamma" in str(exc)
    else:
        raise AssertionError("out-of-range base_mix_gamma must be rejected")
