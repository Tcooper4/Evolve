"""
Optimization Utilities

Utility functions and classes for the optimization module.
"""

try:
    from .consolidator import OptimizerConsolidator, run_optimizer_consolidation
except ImportError:
    OptimizerConsolidator = None
    run_optimizer_consolidation = None

__all__ = ["OptimizerConsolidator", "run_optimizer_consolidation"]
