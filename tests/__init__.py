"""Test suite for the trading system."""

# Core test modules
__all__ = [
    # Unit tests
    "test_agents",
    "test_analysis",
    "test_audit_return_statements",
    "test_basic_functionality",
    "test_basic_imports",
    "test_benchmark",
    "test_check_system",
    "test_complete_optimization_consolidation",
    "test_comprehensive_audit",
    "test_comprehensive_codebase_review",
    "test_comprehensive_return_fix",
    "test_conftest",
    "test_core_functionality",
    "test_demo_unified_interface",
    "test_edge_cases",
    "test_ensemble_voting",
    "test_fixes",
    "test_fix_optimization_imports",
    "test_focused_return_fix",
    "test_full_pipeline",
    "test_imports",
    "test_institutional_upgrade",
    "test_model_selection_strategy_signals",
    "test_performance",
    "test_position_sizer",
    "test_post_upgrade_audit",
    "test_post_upgrade_return_audit",
    "test_production_readiness",
    "test_prompt_template_formatter",
    "test_quick_comprehensive_audit",
    "test_quick_fix",
    "test_real_world_scenario",
    "test_remove_duplicates",
    "test_return_statements",
    "test_rl",
    "test_router",
    "test_simple_audit",
    "test_strategy_combinations",
    "test_system_check",
    "test_system_status",
    "test_system_upgrade",
    "test_targeted_audit",
    "test_task_dashboard",
    "test_task_integration",
    # Integration tests
    "test_integration",
    # Strategy tests
    "test_strategies",
    # Agent tests
    "test_agent_registry",
    "test_agents",
    # Backtesting tests
    "test_backtesting",
    # Forecasting tests
    "test_forecasting",
    # NLP tests
    "test_nlp",
    # Optimization tests
    "test_optimization",
    # Risk tests
    "test_risk",
    # Unit tests
    "unit",
]

# Test discovery patterns
test_patterns = [
    "test_*.py",
    "*_test.py",
    "test_*_*.py",
]

# Test categories
test_categories = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions",
    "system": "System-wide tests",
    "performance": "Performance and load tests",
    "security": "Security and validation tests",
    "regression": "Regression tests",
}

# Test configuration
test_config = {
    "default_timeout": 30,
    "max_retries": 3,
    "parallel_execution": True,
    "coverage_threshold": 80,
    "slow_test_threshold": 5.0,
}

# Import all test modules for discovery
try:
    from . import (  # Core functionality tests; System tests; Audit tests; Import tests; Performance tests; Agent tests; Strategy tests; Model tests; Optimization tests; Return statement tests; Edge case tests; Production tests; Fix tests; Upgrade tests; Other tests
        test_agent_registry,
        test_agents,
        test_audit_return_statements,
        test_basic_functionality,
        test_basic_imports,
        test_benchmark,
        test_check_system,
        test_complete_optimization_consolidation,
        test_comprehensive_audit,
        test_comprehensive_return_fix,
        test_core_functionality,
        test_demo_unified_interface,
        test_edge_cases,
        test_ensemble_voting,
        test_fix_optimization_imports,
        test_fixes,
        test_focused_return_fix,
        test_full_pipeline,
        test_imports,
        test_institutional_upgrade,
        test_model_selection_strategy_signals,
        test_optimization,
        test_performance,
        test_position_sizer,
        test_post_upgrade_audit,
        test_post_upgrade_return_audit,
        test_production_readiness,
        test_prompt_template_formatter,
        test_quick_comprehensive_audit,
        test_quick_fix,
        test_real_world_scenario,
        test_remove_duplicates,
        test_return_statements,
        test_rl,
        test_router,
        test_simple_audit,
        test_strategies,
        test_strategy_combinations,
        test_system_check,
        test_system_status,
        test_system_upgrade,
        test_targeted_audit,
        test_task_dashboard,
        test_task_integration,
    )
except ImportError as e:
    # Log import errors but don't fail
    import logging

    logging.warning(f"Some test modules could not be imported: {e}")

# Test discovery function


def discover_tests(pattern=None, category=None):
    """Discover test modules based on pattern and category."""
    import glob
    import os

    if pattern is None:
        pattern = "test_*.py"

    test_files = []
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        for file in files:
            if glob.fnmatch.fnmatch(file, pattern):
                test_files.append(os.path.join(root, file))

    if category:
        # Filter by category if specified
        test_files = [f for f in test_files if category in f]

    return test_files


# Test runner configuration


def get_test_config():
    """Get test configuration."""
    return test_config.copy()


def set_test_config(**kwargs):
    """Update test configuration."""
    test_config.update(kwargs)
