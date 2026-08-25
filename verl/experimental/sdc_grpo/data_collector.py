"""Deprecated compatibility imports for :mod:`verl.experimental.sdc`."""

from verl.experimental.sdc.data_collector import *  # noqa: F401,F403
from verl.experimental.sdc.data_collector import binary_outcome_scores

__all__ = [name for name in globals() if not name.startswith("_")]
