#!/usr/bin/env python3
"""
Report Service Launcher

Launches the automated report generation service.
"""

import logging
import os
import sys
from pathlib import Path

from utils.launch_utils import setup_logging

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging():
    """Set up logging for the service."""
    return setup_logging(service_name="report_service")


def main():
    """Main launcher function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Get configuration from environment variables
    config = {
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "redis_db": int(os.getenv("REDIS_DB", "0")),
        "service_name": os.getenv("REPORT_SERVICE_NAME", "report_service"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "notion_token": os.getenv("NOTION_TOKEN"),
        "slack_webhook": os.getenv("SLACK_WEBHOOK"),
        "output_dir": os.getenv("REPORT_OUTPUT_DIR", "reports"),
    }

    # Email configuration
    email_config = {}
    if os.getenv("EMAIL_SMTP_SERVER"):
        email_config = {
            "smtp_server": os.getenv("EMAIL_SMTP_SERVER"),
            "smtp_port": int(os.getenv("EMAIL_SMTP_PORT", "587")),
            "username": os.getenv("EMAIL_USERNAME"),
            "password": os.getenv("EMAIL_PASSWORD"),
            "from_email": os.getenv("EMAIL_FROM"),
            "to_email": os.getenv("EMAIL_TO"),
        }

    logger.info("Starting Report Service...")
    logger.info(f"Configuration: {config}")

    try:
        # Initialize and start service
        service = ReportService(
            redis_host=config["redis_host"],
            redis_port=config["redis_port"],
            redis_db=config["redis_db"],
            service_name=config["service_name"],
            openai_api_key=config["openai_api_key"],
            notion_token=config["notion_token"],
            slack_webhook=config["slack_webhook"],
            email_config=email_config,
            output_dir=config["output_dir"],
        )

        # Start the service
        service.start()

    except KeyboardInterrupt:
        logger.info("Shutting down Report Service...")
        service.stop()
    except Exception as e:
        logger.error(f"Error starting Report Service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
