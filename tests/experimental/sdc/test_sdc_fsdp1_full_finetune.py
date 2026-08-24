from __future__ import annotations

import datetime
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from verl.experimental.sdc.sft_trainer import configure_sft_trainable_parameters
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _TinyFullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(3, 4)
        self.second = nn.Linear(4, 2)
        self.register_buffer("full_finetune_buffer", torch.zeros(3, dtype=torch.int64))


def _full_finetune_sync_worker(rank: int, init_method: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model = _TinyFullModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        FSDPEngine._sdc_ensure_optimizer_state(optimizer)

        with torch.no_grad():
            for parameter_index, parameter in enumerate(model.parameters()):
                parameter.fill_(float(parameter_index + 1) if rank == 0 else 0.0)
            model.full_finetune_buffer.fill_(9 if rank == 0 else 0)
            for parameter_index, parameter in enumerate(optimizer.param_groups[0]["params"]):
                state = optimizer.state[parameter]
                state["step"].fill_(7 if rank == 0 else 0)
                state["exp_avg"].fill_(float(10 + parameter_index) if rank == 0 else 0.0)
                state["exp_avg_sq"].fill_(float(20 + parameter_index) if rank == 0 else 0.0)

        engine = object.__new__(FSDPEngine)
        engine._is_lora = False
        # Force tensors, including individual parameters, to span many chunks.
        engine._sdc_sync_chunk_bytes = 16
        engine._sdc_sync_model_and_optimizer(model, optimizer)

        for parameter_index, parameter in enumerate(model.parameters()):
            torch.testing.assert_close(
                parameter,
                torch.full_like(parameter, float(parameter_index + 1)),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            model.full_finetune_buffer,
            torch.full_like(model.full_finetune_buffer, 9),
            rtol=0,
            atol=0,
        )
        for parameter_index, parameter in enumerate(optimizer.param_groups[0]["params"]):
            state = optimizer.state[parameter]
            assert state["step"].device.type == "cpu"
            assert state["step"].item() == 7
            torch.testing.assert_close(
                state["exp_avg"],
                torch.full_like(state["exp_avg"], float(10 + parameter_index)),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                state["exp_avg_sq"],
                torch.full_like(state["exp_avg_sq"], float(20 + parameter_index)),
                rtol=0,
                atol=0,
            )
    finally:
        dist.destroy_process_group()


def test_full_finetune_rank_zero_state_syncs_across_two_ranks():
    with tempfile.TemporaryDirectory() as directory:
        init_method = f"file://{os.path.join(directory, 'process_group')}"
        mp.spawn(_full_finetune_sync_worker, args=(init_method,), nprocs=2, join=True)


def test_full_finetune_checkpoint_load_fails_on_missing_state():
    engine = object.__new__(FSDPEngine)
    engine._is_lora = False
    model = nn.Linear(2, 1)
    incomplete = {"weight": model.weight.detach().clone()}

    try:
        engine._sdc_load_model_state(model, incomplete)
    except RuntimeError as exc:
        assert "Missing key" in str(exc)
    else:
        raise AssertionError("Full-finetuning SDC restore must reject incomplete model state.")


def test_full_finetune_adamw_step_stays_on_cpu_after_state_move():
    model = _TinyFullModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    FSDPEngine._sdc_ensure_optimizer_state(optimizer)

    FSDPEngine._sdc_move_optimizer_state(optimizer, torch.device("cpu"))

    for parameter in optimizer.param_groups[0]["params"]:
        state = optimizer.state[parameter]
        assert state["step"].device.type == "cpu"
        assert state["exp_avg"].device == parameter.device
        assert state["exp_avg_sq"].device == parameter.device


def test_full_finetune_optimizer_updates_every_model_parameter():
    model = _TinyFullModel()
    trainable = configure_sft_trainable_parameters(model, lora_enabled=False)
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()

    parameters = list(model.parameters())
    assert len(trainable) == len(parameters)
    assert all(parameter.requires_grad for parameter in parameters)
    assert all(parameter in optimizer.state for parameter in parameters)
    assert all(not torch.equal(old, parameter) for old, parameter in zip(before, parameters, strict=True))
