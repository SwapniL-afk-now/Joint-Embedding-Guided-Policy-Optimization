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


@torch.no_grad()
def mix_state_with_base(
    destination_state: Mapping[str, torch.Tensor],
    base_state: Mapping[str, torch.Tensor],
    gamma: float,
) -> None:
    """Apply symmetric frozen-base interpolation in place.

    Floating tensors use ``gamma * destination + (1 - gamma) * base``.  Other
    tensors are copied from the common base, avoiding arithmetic on integer
    buffers while keeping both branches deterministic.
    """

    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"SDC base_mix_gamma must be in [0, 1], got {gamma}.")

    for name, destination in destination_state.items():
        base = base_state.get(name)
        if base is None or not torch.is_tensor(destination):
            continue
        base_on_destination = base.to(device=destination.device)
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
) -> dict:
    """Build the replicated SDC sidecar payload used by FSDP/FSDP2."""

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
    }
