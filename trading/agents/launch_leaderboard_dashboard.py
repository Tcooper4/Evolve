#!/usr/bin/env python3
"""
Launch Agent Leaderboard Dashboard

This script launches the agent leaderboard dashboard for monitoring agent performance.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required_packages = ["streamlit", "pandas", "plotly"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install them with: pip install " + " ".join(missing_packages))
        return False

    return True


def check_dashboard_file() -> bool:
    """Check if dashboard file exists."""
    dashboard_path = Path("trading/agents/leaderboard_dashboard.py")

    if not dashboard_path.exists():
        logger.error(f"❌ Dashboard file not found: {dashboard_path}")
        return False

    return True


def launch_dashboard():
    """Launch the agent leaderboard dashboard."""
    try:
        import subprocess
        import time
        import webbrowser

        # Configuration
        host = "localhost"
        port = 8502
        dashboard_path = "trading/agents/leaderboard_dashboard.py"

        logger.info(f"🚀 Launching Agent Leaderboard Dashboard...")
        logger.info(f"📍 URL: http://{host}:{port}")
        logger.info(f"📁 Dashboard: {dashboard_path}")
        logger.info("=" * 60)

        # Launch Streamlit
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(port),
            "--server.address",
            host,
        ]

        process = subprocess.Popen(cmd)

        # Wait a moment for the server to start
        time.sleep(3)

        # Open browser
        webbrowser.open(f"http://{host}:{port}")

        # Wait for process to complete
        process.wait()

    except KeyboardInterrupt:
        logger.info("\n👋 Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to launch dashboard: {e}")


def main():
    """Main function."""
    logger.info("🔧 Checking dependencies...")

    if not check_dependencies():
        return

    if not check_dashboard_file():
        return

    logger.info("✅ All dependencies are installed")

    # Launch dashboard
    launch_dashboard()


if __name__ == "__main__":
    main()
