"""Deprecated compatibility imports for :mod:`verl.experimental.sdc`."""

import torch

from verl.experimental.sdc.data_collector import *


def binary_outcome_scores(outcome_tensor):
    return validate_sdc_outcome(outcome_tensor).to(dtype=torch.float32)
