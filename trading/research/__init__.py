# -*- coding: utf-8 -*-
"""Evolve signal-edge research package.

Phase 0 shared harness lives in ``signal_edge_harness``. This is the first
installment of an ongoing program — add new signal OOS tests through that
module rather than bespoke purge/DSR pipelines.
"""

from trading.research.signal_edge_harness import (  # noqa: F401
    DISCLOSURE,
    TargetSpec,
    TrialSpec,
    run_signal_edge_oos,
)

__all__ = [
    "DISCLOSURE",
    "TargetSpec",
    "TrialSpec",
    "run_signal_edge_oos",
]
