"""Development tool to initialize and verify logging files.

This module provides functionality to initialize and verify the existence of required
logging files for the trading system. It supports both text and JSONL formats,
and can be configured via an external JSON configuration file.

Example:
    >>> result = init_log_files(verbose=True)
    >>> print(result)
    {'app.log': 'created', 'audit.log': 'exists', 'metrics.jsonl': 'created'}
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_log_config() -> list:
    """Load log file configuration from JSON file.

    Returns:
        list: Configuration for required log files

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(__file__).parent / "config" / "log_files.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise


def init_log_files(
    log_dir: str = "logs", verbose: bool = True
) -> Dict[str, Literal["created", "exists"]]:
    """Initialize or verify log files.

    Args:
        log_dir: Directory containing log files
        verbose: Whether to log creation/verification messages

    Returns:
        Dict mapping filenames to their status ("created" or "exists")

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Get current timestamp
    timestamp = datetime.utcnow().isoformat()

    # Load configuration
    log_config = load_log_config()

    # Track results
    results: Dict[str, Literal["created", "exists"]] = {}

    # Initialize each required file
    for config in log_config:
        filename = config["filename"]
        file_path = log_path / filename

        # Create file if it doesn't exist
        if not file_path.exists():
            with open(file_path, "w") as file:
                if config["format"] == "jsonl":
                    # JSON Lines format
                    content = config["template"]
                    if isinstance(content, dict):
                        content = {
                            k: v.format(timestamp=timestamp) for k, v in content.items()
                        }
                    json.dump(content, file)
                    file.write("\n")
                else:
                    # Text format
                    file.write(config["template"].format(timestamp=timestamp))

            results[filename] = "created"
            if verbose:
                logger.info(f"Created log file: {filename}")
        else:
            results[filename] = "exists"
            if verbose:
                logger.info(f"Verified log file exists: {filename}")

    return results


def main() -> None:
    """Main entry point."""
    try:
        # Initialize log files
        results = init_log_files(verbose=True)

        # Log summary
        logger.info("\nLog file initialization summary:")
        for filename, status in results.items():
            logger.info(f"- {filename}: {status}")

        logger.info(
            "\nNote: These files are for development only and should not be committed to git."
        )
    except Exception as e:
        logger.error(f"Failed to initialize log files: {e}")
        raise


if __name__ == "__main__":
    main()
