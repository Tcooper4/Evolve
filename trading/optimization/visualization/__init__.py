"""
Optimization Visualization Module

Visualization tools for optimization results.
"""

try:
    from ..optimization_visualizer import OptimizationVisualizer
except ImportError:
    OptimizationVisualizer = None

__all__ = [
    'OptimizationVisualizer'
] 