# Fused single-optimizer-step pieces that are testable without FSDP/GPU:
#   - the guard that turns "deferred policy grad went missing" into a hard error
#   - grad_norm_only(): reads the norm of whatever is resident on .grad, unchanged
# The FSDP grad-materialization behaviour itself needs a live engine; the guard is
# what makes that failure loud at runtime instead of silent.
import torch
import torch.nn as nn

from verl.experimental.jepa_grpo.worker import _assert_policy_grad_resident
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def test_guard_rejects_missing_policy_grad():
    _assert_policy_grad_resident(0.05)  # resident -> fine
    for missing in (0.0, -1.0):
        try:
            _assert_policy_grad_resident(missing)
        except RuntimeError as exc:
            assert "fuse_optimizer_step" in str(exc)
        else:
            raise AssertionError(f"expected RuntimeError for grad norm {missing}")


def _norm_of(module):
    return torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in module.parameters())).item()


def test_grad_norm_only_is_exact_and_non_destructive():
    torch.manual_seed(0)
    module = nn.Linear(8, 4)
    module(torch.randn(5, 8)).sum().backward()

    expected = _norm_of(module)
    before = [p.grad.detach().clone() for p in module.parameters()]

    engine = FSDPEngine.__new__(FSDPEngine)  # bypass __init__: only .module is used
    engine.module = module
    measured = FSDPEngine.grad_norm_only(engine)

    assert abs(measured - expected) < 1e-5, (measured, expected)
    for p, b in zip(module.parameters(), before):
        assert torch.equal(p.grad, b), "grad_norm_only must not modify gradients"


def test_grad_norm_only_sees_accumulated_sum():
    # Fused mode relies on this: the JEPA backward adds onto the resident policy grad,
    # so a second measurement must reflect the SUM, not the last backward alone.
    torch.manual_seed(0)
    module = nn.Linear(8, 4)
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.module = module

    module(torch.randn(5, 8)).sum().backward()
    first = FSDPEngine.grad_norm_only(engine)
    module(torch.randn(5, 8)).sum().backward()  # accumulates, no zero_grad
    total = FSDPEngine.grad_norm_only(engine)

    assert first > 0 and total > 0
    assert abs(total - _norm_of(module)) < 1e-5
    assert abs(total - first) > 1e-6, "second backward must change the resident grad"


if __name__ == "__main__":
    test_guard_rejects_missing_policy_grad()
    test_grad_norm_only_is_exact_and_non_destructive()
    test_grad_norm_only_sees_accumulated_sum()
    print("ok")


def test_train_mode_exit_preserves_a_deferred_gradient():
    """The bug that killed the first h-grpo launch.

    EngineTrainModeCtx.__exit__ used to call optimizer_zero_grad() unconditionally, so a
    caller that ran train_batch(apply_step=False) found .grad WIPED by the time it went to
    fuse a second backward onto it -- surfacing as _assert_policy_grad_resident(norm=0.0).
    keep_grad=True must suppress that zeroing; keep_grad=False (every other caller) must
    not change.
    """
    import types

    from verl.workers.engine.fsdp.transformer_impl import EngineTrainModeCtx, FSDPEngine

    def _run(keep_grad):
        zeroed = []
        # __exit__ asserts isinstance(engine, FSDPEngine), so build a real instance
        # without running __init__ and populate only what the exit path touches.
        engine = FSDPEngine.__new__(FSDPEngine)
        engine._is_offload_param = False
        engine._is_offload_optimizer = False
        engine.optimizer = types.SimpleNamespace(zero_grad=lambda: zeroed.append(True))
        ctx = EngineTrainModeCtx(engine, disable_auto_offload=True, keep_grad=keep_grad)
        ctx.prev_sp_group = None          # normally set by __enter__
        ctx.__exit__(None, None, None)
        return bool(zeroed)

    assert _run(keep_grad=True) is False, "deferred gradient must survive the context exit"
    assert _run(keep_grad=False) is True, "default behavior must be unchanged"
