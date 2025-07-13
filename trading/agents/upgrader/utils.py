"""
Utility functions for the Upgrader Agent.

This module contains helper functions for model and pipeline component detection,
drift detection, and status checking.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger("UpgraderUtils")


def check_model_status(model_id: str, config: Dict) -> Tuple[bool, str]:
    """
    Check if a model needs upgrading based on various criteria.

    Args:
        model_id: The ID of the model to check
        config: The model's configuration dictionary

    Returns:
        Tuple[bool, str]: (needs_upgrade, reason)
            - needs_upgrade: True if the model needs upgrading
            - reason: Explanation for why the model needs upgrading
    """
    try:
        # Check if model exists
        if not os.path.exists(f"models/{model_id}"):
            return True, "Model missing"

        # Check model version
        current_version = _get_model_version(model_id)
        latest_version = config.get("version", "1.0.0")

        if current_version != latest_version:
            return True, f"Version mismatch: {current_version} -> {latest_version}"

        # Check for model drift
        if detect_drift(model_id):
            return True, "Model drift detected"

        return False, ""

    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        return False, f"Error: {str(e)}"


def check_component_status(component: str) -> Tuple[bool, str]:
    """
    Check if a pipeline component needs upgrading.

    Args:
        component: Path to the component to check

    Returns:
        Tuple[bool, str]: (needs_upgrade, reason)
            - needs_upgrade: True if the component needs upgrading
            - reason: Explanation for why the component needs upgrading
    """
    try:
        # Check if component exists
        if not os.path.exists(component):
            return True, "Component missing"

        # Check for deprecated logic
        if check_deprecated_logic(component):
            return True, "Deprecated logic detected"

        # Check for missing parameters
        if check_missing_parameters(component):
            return True, "Missing parameters detected"

        return False, ""

    except Exception as e:
        logger.error(f"Error checking component status: {str(e)}")
        return False, f"Error: {str(e)}"


def detect_drift(model_id: str) -> bool:
    """
    Detect if a model has drifted from its expected behavior.

    Args:
        model_id: The ID of the model to check

    Returns:
        bool: True if drift is detected, False otherwise
    """
    try:
        # Load model performance history
        performance_file = f"logs/model_performance_{model_id}.json"
        if not os.path.exists(performance_file):
            logger.warning(f"No performance data found for model {model_id}")
            return False

        with open(performance_file, "r") as f:
            performance_data = json.load(f)

        # Get recent performance metrics
        recent_metrics = performance_data.get("recent_metrics", [])
        if len(recent_metrics) < 10:
            logger.warning(f"Insufficient recent metrics for drift detection: {len(recent_metrics)}")
            return False

        # Calculate performance degradation
        recent_mae = np.mean([m.get("mae", 0) for m in recent_metrics[-10:]])
        historical_mae = np.mean([m.get("mae", 0) for m in recent_metrics[:-10]])

        if historical_mae > 0:
            degradation_ratio = recent_mae / historical_mae
            has_drifted = degradation_ratio > 1.2  # 20% degradation threshold

            logger.info(
                f"Drift detection for model {model_id}: degradation_ratio={degradation_ratio:.3f}, drifted={has_drifted}"
            )
            return has_drifted

        return False

    except Exception as e:
        logger.error(f"Error detecting drift: {str(e)}")
        return False


def check_deprecated_logic(component: str) -> bool:
    """
    Check if a component contains deprecated logic.

    Args:
        component: Path to the component to check

    Returns:
        bool: True if deprecated logic is found, False otherwise
    """
    try:
        if not os.path.exists(component):
            logger.warning(f"Component not found: {component}")
            return False

        # Read the component file
        with open(component, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for deprecated patterns
        deprecated_patterns = [
            "DEPRECATED",
            "deprecated",
            "legacy",
            "old_",
            "TODO.*deprecated",
            "FIXME.*deprecated",
            "print\\(",  # print statements (should use logger)
            "except:",  # bare except blocks
            "import \\*",  # wildcard imports
        ]

        deprecated_count = 0
        for pattern in deprecated_patterns:
            matches = re.findall(pattern, content)
            deprecated_count += len(matches)

        has_deprecated = deprecated_count > 0

        if has_deprecated:
            logger.warning(f"Deprecated logic found in {component}: {deprecated_count} patterns")

        return has_deprecated

    except Exception as e:
        logger.error(f"Error checking deprecated logic: {str(e)}")
        return False


def check_missing_parameters(component: str) -> bool:
    """
    Check if a component is missing required parameters.

    Args:
        component: Path to the component to check

    Returns:
        bool: True if missing parameters are found, False otherwise
    """
    try:
        if not os.path.exists(component):
            logger.warning(f"Component not found: {component}")
            return False

        # Read the component file
        with open(component, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for common missing parameter patterns
        missing_patterns = [
            "TODO.*parameter",
            "FIXME.*parameter",
            "raise NotImplementedError",
            "pass  # TODO",
            "return None  # TODO",
            "placeholder",
            "stub",
        ]

        missing_count = 0
        for pattern in missing_patterns:
            matches = re.findall(pattern, content)
            missing_count += len(matches)

        has_missing = missing_count > 0

        if has_missing:
            logger.warning(f"Missing parameters found in {component}: {missing_count} patterns")

        return has_missing

    except Exception as e:
        logger.error(f"Error checking parameters: {str(e)}")
        return False


def _get_model_version(model_id: str) -> str:
    """
    Get the current version of a model.

    Args:
        model_id: The ID of the model

    Returns:
        str: The model's version
    """
    try:
        version_file = Path(f"models/{model_id}/version.txt")
        if version_file.exists():
            return version_file.read_text().strip()
        return "1.0.0"

    except Exception as e:
        logger.error(f"Error getting model version: {str(e)}")
        return "1.0.0"


def get_pipeline_components() -> List[str]:
    """
    Get a list of all pipeline components that need to be checked.

    Returns:
        List[str]: List of component paths
    """
    try:
        components = []
        pipeline_dir = Path("trading/pipeline")

        if pipeline_dir.exists():
            for file in pipeline_dir.rglob("*.py"):
                components.append(str(file))

        return components

    except Exception as e:
        logger.error(f"Error getting pipeline components: {str(e)}")
        return []


def validate_upgrade_result(result: Dict) -> bool:
    """
    Validate the result of an upgrade operation.

    Args:
        result: Dictionary containing upgrade result information

    Returns:
        bool: True if the upgrade was successful, False otherwise
    """
    try:
        required_fields = ["success", "model_id", "timestamp"]
        return all(field in result for field in required_fields) and result["success"]

    except Exception as e:
        logger.error(f"Error validating upgrade result: {str(e)}")
        return False
