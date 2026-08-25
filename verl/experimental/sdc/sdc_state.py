"""Small, dependency-light helpers for SDC reference-model state."""

from __future__ import annotations

from typing import Mapping

import torch


def clone_frozen_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return an immutable-by-convention, detached CPU snapshot."""

    return {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in state.items()
        if torch.is_tensor(value)
    }


def pin_base_state_once(base_state: dict) -> bool:
    """Lazily pin CPU base tensors so mixing can issue async H2D copies.

    Pinning happens at first mix rather than snapshot creation: snapshots are
    also embedded into checkpoint payloads where pageable host memory is fine,
    and this keeps one-time pinning costs off cold paths. Dropping the
    snapshot later releases the pinned pages through ordinary garbage
    collection (PyTorch has no separate unpin operation).

    Returns True when the caller may use ``non_blocking=True`` copies.
    """

    if not torch.cuda.is_available():
        return False
    for name, value in list(base_state.items()):
        if torch.is_tensor(value) and value.device.type == "cpu" and not value.is_pinned():
            try:
                base_state[name] = value.pin_memory()
            except RuntimeError:
                # Pinned-allocator exhaustion or no accessible CUDA context:
                # fall back to synchronous transfers for everything after.
                return False
    return True


@torch.no_grad()
def mix_state_with_base(
    destination_state: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
    gamma: float,
) -> None:
    """Apply symmetric frozen-base interpolation in place.

    Floating tensors use ``gamma * destination + (1 - gamma) * base``.  Other
    tensors are copied from the common base, avoiding arithmetic on integer
    buffers while keeping both branches deterministic. Base tensors are pinned
    once (lazily) and copied to the destination device with non-blocking H2D
    transfers; the fp32 accumulation below runs on the same stream, so dtype,
    order of operations, and numerics are identical to synchronous copies.
    """

    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"SDC base_mix_gamma must be in [0, 1], got {gamma}.")

    non_blocking = isinstance(base_state, dict) and pin_base_state_once(base_state)
    for name, destination in destination_state.items():
        base = base_state.get(name)
        if base is None or not torch.is_tensor(destination):
            continue
        base_on_destination = base.to(device=destination.device, non_blocking=non_blocking)
        if torch.is_floating_point(destination):
            # Use a temporary higher-precision accumulator without changing
            # the destination parameter/buffer dtype.
            mixed = (
                destination.detach().to(dtype=torch.float32) * gamma
                + base_on_destination.to(dtype=torch.float32) * (1.0 - gamma)
            )
            destination.copy_(mixed.to(dtype=destination.dtype))
        else:
            destination.copy_(base_on_destination.to(dtype=destination.dtype))


def mix_module_with_base(module: torch.nn.Module, base_state: Mapping[str, torch.Tensor], gamma: float) -> None:
    """Mix every matching module state tensor against one frozen base."""

    with torch.no_grad():
        mix_state_with_base(module.state_dict(), base_state, gamma)


def build_sdc_checkpoint_state(
    *,
    global_step: int,
    success_model_update_count: int,
    failure_model_update_count: int,
    models_ready: bool,
    base_state: Mapping[str, torch.Tensor] | None = None,
    base_adapter_state: Mapping[str, torch.Tensor] | None = None,
    adapter_only: bool = False,
    base_mix_gamma: float,
    success_buffer_state=None,
    failure_buffer_state=None,
    base_saved: bool = False,
    base_global_step: int | None = None,
) -> dict:
    """Build the replicated SDC sidecar payload used by FSDP/FSDP2.

    Slim full-parameter checkpoints pass ``base_state=None`` together with
    ``base_saved=True`` and ``base_global_step`` referencing the earlier save
    that embedded ``sdc_base_state_cpu``. Old payloads without those keys load
    exactly as before.
    """

    return {
        "global_step": int(global_step),
        "sdc_success_model_update_count": int(success_model_update_count),
        "sdc_failure_model_update_count": int(failure_model_update_count),
        "sdc_models_ready": bool(models_ready),
        "base_mix_gamma": float(base_mix_gamma),
        "sdc_base_state_cpu": clone_frozen_state(base_state) if base_state is not None else None,
        "sdc_base_adapter_state_cpu": (
            clone_frozen_state(base_adapter_state) if base_adapter_state is not None else None
        ),
        "sdc_adapter_only": bool(adapter_only),
        "success_buffer_state": success_buffer_state,
        "failure_buffer_state": failure_buffer_state,
        "base_saved": bool(base_saved),
        "base_global_step": None if base_global_step is None else int(base_global_step),
    }
