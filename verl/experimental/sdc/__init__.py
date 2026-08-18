"""Algorithm-agnostic Success/Failure Distribution Contrast (SDC)."""

from verl.experimental.sdc.config import (
    SDCConfig,
    sdc_enabled,
    should_checkpoint,
    should_run_sft,
    validate_sdc_config,
)
from verl.experimental.sdc.data_collector import (
    FailureDataCollector,
    OutcomeDataCollector,
    OutcomeExample,
    SuccessDataCollector,
    extract_sdc_outcome,
    validate_sdc_outcome,
)
from verl.experimental.sdc.sdc_loss import compute_sdc_loss, contrast_quality_metrics

__all__ = [
    "SDCConfig",
    "sdc_enabled",
    "should_checkpoint",
    "should_run_sft",
    "validate_sdc_config",
    "FailureDataCollector",
    "SuccessDataCollector",
    "OutcomeDataCollector",
    "OutcomeExample",
    "extract_sdc_outcome",
    "validate_sdc_outcome",
    "compute_sdc_loss",
    "contrast_quality_metrics",
]
