"""CPU-only checks for examples/sdc_grid/*.

These tests never launch training and never touch a GPU: they run each launcher
under `--dry-run` (preflight only) or the SDC_GRID_DUMP_OVERRIDES / _DIVERGENCES
escape hatches in common.sh, which build the hydra override list and exit before
main_ppo is ever imported. Composition against the real ppo_trainer config schema
is done in-process via hydra.compose, not by launching the trainer.

The GXPO_GOLDEN_FAIRNESS_KEYS snapshot below is the resolved command line of the
live GXPO reference run (tmux `qwen25-1p5b-throughput0902`, launched from
qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh -> common.sh, pid 3922267), captured
2026-09-02. It is intentionally a frozen snapshot, not a live re-parse of those
scripts: several values (e.g. ppo_max_token_len_per_gpu=49152) come from an env
override at launch time and are not visible in the scripts' own defaults. If the
SDC grid's inherited values ever need to change, update them in both
examples/sdc_grid/common.sh and this snapshot together, deliberately.
"""

import functools
import os
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRID_DIR = os.path.join(REPO_ROOT, "examples", "sdc_grid")
CONFIG_DIR = os.path.join(REPO_ROOT, "verl", "trainer", "config")

# Asset locations follow the same env vars common.sh honours, so the suite runs
# on any box where the checkout lives outside /workspace. Only when the RESOLVED
# paths are missing do we skip.
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/Qwen2.5-Math-1.5B-Instruct")
DATA_ROOT = os.environ.get("GXPO_DATA_ROOT", "/workspace/data")
VAL_CACHE_ROOT = os.environ.get("SDC_VAL_CACHE_ROOT", os.path.join(DATA_ROOT, "sdc_validation_normalized"))
# FA3 is unavailable on some GPUs (e.g. Blackwell sm_120); common.sh preflights
# whatever ACTOR_ATTENTION_IMPL asks for, so the golden snapshot follows it.
ACTOR_ATTENTION_IMPL = os.environ.get("ACTOR_ATTENTION_IMPL", "flash_attention_3")
_REQUIRED_PATHS = [
    os.path.join(MODEL_PATH, "config.json"),
    os.path.join(DATA_ROOT, "dapo_math", "train.parquet"),
    os.path.join(DATA_ROOT, "lighteval-math", "train.parquet"),
    os.path.join(VAL_CACHE_ROOT, "math500.parquet"),
]
_MISSING_PATHS = [p for p in _REQUIRED_PATHS if not os.path.exists(p)]
pytestmark = pytest.mark.skipif(
    bool(_MISSING_PATHS),
    reason=(
        "sdc_grid launcher preflight needs the model/dataset checkout; missing "
        f"{_MISSING_PATHS} (set MODEL_PATH / GXPO_DATA_ROOT / SDC_VAL_CACHE_ROOT)"
    ),
)

BASES = ("drgrpo", "dapo", "avspo")
ALPHAS = ("1.0", "0.9", "0.7", "0.5")
# Zero-advantage ("all-wrong group") repair arms: sdc_tr at alpha=0.5 plus an
# explicit policy_loss.sdc_tr_degenerate_coef.
ZEROADV_ALPHA = "0.5"
ZEROADV_COEF = "0.5"

# (rel_path, base, mode, alpha, degenerate_coef); degenerate_coef is None when the
# arm must leave the knob at its 0.0 dataclass default.
ARMS = []
for _base in BASES:
    ARMS.append((f"base/{_base}.sh", _base, "base", None, None))
    ARMS.append((f"sdc/{_base}.sh", _base, "sdc", None, None))
    for _alpha in ALPHAS:
        ARMS.append((f"sdc_tr/{_base}_a{_alpha}.sh", _base, "sdc_tr", _alpha, None))
    ARMS.append(
        (
            f"sdc_tr_zeroadv/{_base}_a{ZEROADV_ALPHA}_dg{ZEROADV_COEF}.sh",
            _base,
            "sdc_tr",
            ZEROADV_ALPHA,
            ZEROADV_COEF,
        )
    )

ARM_IDS = [a[0] for a in ARMS]

BASE_ADV_ESTIMATOR = {"drgrpo": "grpo", "dapo": "grpo", "avspo": "avspo"}
BASE_NORM_ADV = {"drgrpo": False, "dapo": True, "avspo": True}
BASE_CLIP = {"drgrpo": (0.2, 0.2), "dapo": (0.2, 0.28), "avspo": (0.2, 0.2)}

# See module docstring for provenance.
GXPO_GOLDEN_FAIRNESS_KEYS = {
    "data.train_batch_size": "256",
    "data.val_batch_size": "128",
    "data.max_prompt_length": "1024",
    "data.max_response_length": "3072",
    "data.filter_overlong_prompts": "True",
    "data.truncation": "error",
    "data.seed": "3407",
    "actor_rollout_ref.model.use_remove_padding": "True",
    "actor_rollout_ref.model.enable_gradient_checkpointing": "True",
    "actor_rollout_ref.model.use_liger": "True",
    "+actor_rollout_ref.model.override_config.attn_implementation": ACTOR_ATTENTION_IMPL,
    "actor_rollout_ref.actor.optim.lr": "1e-6",
    "actor_rollout_ref.actor.optim.fused": "True",
    "actor_rollout_ref.actor.use_torch_compile": "True",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "64",
    "actor_rollout_ref.actor.use_dynamic_bsz": "True",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "49152",
    "actor_rollout_ref.actor.clip_ratio": "0.2",
    "actor_rollout_ref.actor.clip_ratio_c": "3.0",
    "actor_rollout_ref.actor.grad_clip": "1.0",
    "actor_rollout_ref.actor.entropy_coeff": "0",
    "actor_rollout_ref.actor.use_kl_loss": "True",
    "actor_rollout_ref.actor.kl_loss_coef": "0.01",
    "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
    "actor_rollout_ref.actor.fsdp_config.fsdp_size": "4",
    "actor_rollout_ref.actor.fsdp_config.param_offload": "False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "False",
    "actor_rollout_ref.actor.data_loader_seed": "3407",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": "1",
    "actor_rollout_ref.rollout.n": "8",
    "actor_rollout_ref.rollout.temperature": "1.0",
    "actor_rollout_ref.rollout.top_p": "1.0",
    "actor_rollout_ref.rollout.enforce_eager": "False",
    "actor_rollout_ref.rollout.enable_chunked_prefill": "True",
    "actor_rollout_ref.rollout.val_kwargs.n": "1",
    "actor_rollout_ref.rollout.val_kwargs.do_sample": "False",
    "actor_rollout_ref.rollout.val_kwargs.temperature": "0",
    "actor_rollout_ref.rollout.val_kwargs.top_p": "1.0",
    "actor_rollout_ref.ref.fsdp_config.param_offload": "True",
    "trainer.critic_warmup": "0",
    "trainer.n_gpus_per_node": "4",
    "trainer.nnodes": "1",
    "trainer.save_freq": "25",
    "trainer.test_freq": "5",
    "trainer.val_before_train": "True",
    "trainer.total_training_steps": "400",
    "trainer.total_epochs": "100",
}

EXPECTED_DIVERGENCE_KEYS = {
    "data.system_prompt",
    "trainer.max_steps",
    "trainer.keep_last_ckpts",
    "trainer.keep_all_ckpts",
    "trainer.keep_last_validations",
    "rollout.attention_backend",
    "model.attn_implementation",
    "actor.optim.name",
    "rollout.free_cache_engine",
    "rollout.gpu_memory_utilization",
    "rollout.max_num_seqs",
    "rollout.max_num_batched_tokens",
    "rollout.log_prob_micro_batch_size_per_gpu",
    "rollout.mode",
    "trainer.resume_mode",
}


def _run_entrypoint(rel_path, extra_env=None, dry_run=False):
    script = os.path.join(GRID_DIR, rel_path)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    if extra_env:
        env.update(extra_env)
    args = ["bash", script]
    if dry_run:
        args.append("--dry-run")
    return subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180)


@functools.lru_cache(maxsize=None)
def _dump_overrides(rel_path):
    # Each call spawns two Python subprocesses (verl-import guard + FA3/Liger
    # dependency check) inside common.sh, ~5s apiece. Several tests need the same
    # arm's overrides, so memoize per rel_path rather than re-running preflight
    # for every assertion.
    proc = _run_entrypoint(rel_path, extra_env={"SDC_GRID_DUMP_OVERRIDES": "1"})
    assert proc.returncode == 0, (
        f"{rel_path} failed to dump overrides\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return tuple(line for line in proc.stdout.splitlines() if line.strip())


def _overrides_to_dict(overrides):
    result = {}
    for ov in overrides:
        key, _, value = ov.partition("=")
        result[key] = value
    return result


def _values_equal(a, b):
    a, b = a.strip(), b.strip()
    if a.lower() == b.lower():
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


@pytest.mark.parametrize("rel_path,base,mode,alpha,degenerate_coef", ARMS, ids=ARM_IDS)
def test_entrypoint_dry_run_succeeds(rel_path, base, mode, alpha, degenerate_coef):
    proc = _run_entrypoint(rel_path, dry_run=True)
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "preflight OK" in proc.stdout
    assert "resolved launch configuration" in proc.stdout


@pytest.mark.parametrize("rel_path,base,mode,alpha,degenerate_coef", ARMS, ids=ARM_IDS)
def test_arm_inherits_gxpo_fairness_keys(rel_path, base, mode, alpha, degenerate_coef):
    overrides = _overrides_to_dict(_dump_overrides(rel_path))

    missing = [k for k in GXPO_GOLDEN_FAIRNESS_KEYS if k not in overrides]
    assert not missing, f"{rel_path} is missing GXPO-inherited keys: {missing}"

    mismatched = {
        k: (GXPO_GOLDEN_FAIRNESS_KEYS[k], overrides[k])
        for k in GXPO_GOLDEN_FAIRNESS_KEYS
        if not _values_equal(GXPO_GOLDEN_FAIRNESS_KEYS[k], overrides[k])
    }
    assert not mismatched, f"{rel_path} diverges from the GXPO reference on: {mismatched}"


def test_divergences_are_exhaustive():
    proc = _run_entrypoint("base/drgrpo.sh", extra_env={"SDC_GRID_DUMP_DIVERGENCES": "1"})
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, "SDC_GRID_DIVERGENCES dumped nothing"

    keys = set()
    for line in lines:
        parts = line.split("|")
        assert len(parts) == 4, f"malformed divergence row (expected 4 '|'-separated fields): {line!r}"
        keys.add(parts[0])

    missing = EXPECTED_DIVERGENCE_KEYS - keys
    extra = keys - EXPECTED_DIVERGENCE_KEYS
    assert not missing, f"documented divergences disappeared from common.sh: {missing}"
    assert not extra, f"undocumented divergences appeared in common.sh: {extra}"


@pytest.mark.parametrize("rel_path,base,mode,alpha,degenerate_coef", ARMS, ids=ARM_IDS)
def test_arm_hydra_composition(rel_path, base, mode, alpha, degenerate_coef):
    hydra = pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    overrides = _dump_overrides(rel_path)

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)

    expected_loss_mode = "sdc_tr" if mode == "sdc_tr" else "vanilla"
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == expected_loss_mode

    assert cfg.algorithm.adv_estimator == BASE_ADV_ESTIMATOR[base]
    assert bool(cfg.algorithm.norm_adv_by_std_in_grpo) == BASE_NORM_ADV[base]

    expected_low, expected_high = BASE_CLIP[base]
    assert float(cfg.actor_rollout_ref.actor.clip_ratio_low) == expected_low
    assert float(cfg.actor_rollout_ref.actor.clip_ratio_high) == expected_high

    expected_sdc_enable = mode != "base"
    assert bool(cfg.custom_sdc.enable) == expected_sdc_enable

    if mode == "sdc_tr":
        assert float(cfg.actor_rollout_ref.actor.policy_loss.sdc_tr_alpha) == float(alpha)
        assert cfg.actor_rollout_ref.actor.policy_loss.sdc_tr_tilt_clip is not None
        # The zero-advantage repair is opt-in: only the sdc_tr_zeroadv arms turn
        # it on, every other sdc_tr arm must stay at the 0.0 default.
        expected_coef = float(degenerate_coef) if degenerate_coef is not None else 0.0
        assert float(cfg.actor_rollout_ref.actor.policy_loss.sdc_tr_degenerate_coef) == expected_coef
    else:
        # base/sdc arms never touch the sdc_tr-only knobs; they should stay at
        # the dataclass default (1.0 / bit-for-bit vanilla) rather than being
        # silently set to something else.
        assert float(cfg.actor_rollout_ref.actor.policy_loss.sdc_tr_alpha) == 1.0
        assert float(cfg.actor_rollout_ref.actor.policy_loss.sdc_tr_degenerate_coef) == 0.0


@pytest.mark.parametrize("base", BASES)
def test_sdc_tr_alpha_one_differs_from_sdc_sibling_only_in_loss_mode(base):
    sdc_overrides = _overrides_to_dict(_dump_overrides(f"sdc/{base}.sh"))
    tr_overrides = _overrides_to_dict(_dump_overrides(f"sdc_tr/{base}_a1.0.sh"))

    # SDC-TR arms intentionally receive distinct run identities so their
    # checkpoints and metrics cannot overwrite the additive SDC siblings.
    allowed_changed = {
        "actor_rollout_ref.actor.policy_loss.loss_mode",
        "trainer.experiment_name",
        "trainer.default_local_dir",
    }
    allowed_tr_only = {
        "actor_rollout_ref.actor.policy_loss.sdc_tr_alpha",
        "actor_rollout_ref.actor.policy_loss.sdc_tr_tilt_clip",
        # Opt-in zero-advantage repair knob; the a1.0 arms leave it at its 0.0
        # default, but tolerate it being emitted explicitly.
        "actor_rollout_ref.actor.policy_loss.sdc_tr_degenerate_coef",
    }

    unexpected = {}
    for key in set(sdc_overrides) | set(tr_overrides):
        a, b = sdc_overrides.get(key), tr_overrides.get(key)
        if a == b:
            continue
        if key in allowed_changed or key in allowed_tr_only:
            continue
        unexpected[key] = (a, b)

    assert not unexpected, (
        f"sdc/{base}.sh and sdc_tr/{base}_a1.0.sh diverge beyond the sdc_tr wrapper: {unexpected}"
    )
    assert sdc_overrides["actor_rollout_ref.actor.policy_loss.loss_mode"] == "vanilla"
    assert tr_overrides["actor_rollout_ref.actor.policy_loss.loss_mode"] == "sdc_tr"


@pytest.mark.parametrize("base", BASES)
def test_zeroadv_arm_differs_from_sdc_tr_sibling_only_in_degenerate_coef(base):
    """sdc_tr_zeroadv/<base>_a0.5_dg0.5.sh must be the a0.5 arm plus one knob."""
    tr_overrides = _overrides_to_dict(_dump_overrides(f"sdc_tr/{base}_a{ZEROADV_ALPHA}.sh"))
    zeroadv_overrides = _overrides_to_dict(
        _dump_overrides(f"sdc_tr_zeroadv/{base}_a{ZEROADV_ALPHA}_dg{ZEROADV_COEF}.sh")
    )

    coef_key = "actor_rollout_ref.actor.policy_loss.sdc_tr_degenerate_coef"
    # Run identity keys legitimately differ (the arm has its own experiment name
    # and checkpoint dir).
    allowed_changed = {coef_key, "trainer.experiment_name", "trainer.default_local_dir"}

    unexpected = {}
    for key in set(tr_overrides) | set(zeroadv_overrides):
        a, b = tr_overrides.get(key), zeroadv_overrides.get(key)
        if a == b or key in allowed_changed:
            continue
        unexpected[key] = (a, b)

    assert not unexpected, (
        f"sdc_tr_zeroadv/{base} diverges from its sdc_tr a{ZEROADV_ALPHA} sibling beyond "
        f"the repair knob: {unexpected}"
    )
    assert float(zeroadv_overrides[coef_key]) == float(ZEROADV_COEF)
    # The plain sdc_tr arms must not silently start emitting the knob.
    assert coef_key not in tr_overrides or float(tr_overrides[coef_key]) == 0.0
    assert "_dg" in zeroadv_overrides["trainer.experiment_name"]
