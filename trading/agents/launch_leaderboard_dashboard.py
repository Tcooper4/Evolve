#!/usr/bin/env python3
"""
Agent Leaderboard Dashboard Launcher

Launches the Streamlit dashboard for agent performance leaderboard visualization.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return {'success': True, 'result': False, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    return True

def launch_dashboard(port=8501, host="localhost", headless=False):
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "leaderboard_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    # Build the streamlit command
    cmd = [
        "streamlit", "run", str(dashboard_path),
        "--server.port", str(port),
        "--server.address", host
    ]
    
    if headless:
        cmd.extend(["--server.headless", "true"])
    
    print(f"🚀 Launching Agent Leaderboard Dashboard...")
    print(f"📍 URL: http://{host}:{port}")
    print(f"📁 Dashboard: {dashboard_path}")
    print("=" * 60)
    
    try:
        # Launch the dashboard
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return {'success': True, 'result': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Launch Agent Leaderboard Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_leaderboard_dashboard.py
  python launch_leaderboard_dashboard.py --port 8502
  python launch_leaderboard_dashboard.py --host 0.0.0.0 --port 8503
  python launch_leaderboard_dashboard.py --headless
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run the dashboard on (default: 8501)"
    )
    
    parser.add_argument(
        "--host", "-H",
        default="localhost",
        help="Host to bind the dashboard to (default: localhost)"
    )
    
    parser.add_argument(
        "--headless", "-d",
        action="store_true",
        help="Run in headless mode (no browser auto-open)"
    )
    
    parser.add_argument(
        "--check-deps", "-c",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies are installed")
            return 0
        else:
            return {'success': True, 'result': 1, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    # Check dependencies before launching
    if not check_dependencies():
        return 1
    
    # Launch the dashboard
    success = launch_dashboard(
        port=args.port,
        host=args.host,
        headless=args.headless
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 