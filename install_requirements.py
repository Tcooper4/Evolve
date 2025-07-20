"""
Requirements Installation Script

This script installs all required packages for the Evolve trading system
in logical batches to handle dependency conflicts gracefully.
"""

import subprocess
import sys
import os


def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success: {cmd}")
            return True
        else:
            print(f"✗ Failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception: {cmd}")
        print(f"Error: {e}")
        return False


def install_packages(packages, batch_name):
    """Install a batch of packages."""
    print(f"\n--- Installing {batch_name} ---")
    cmd = f"pip install {' '.join(packages)}"
    return run_command(cmd)


def main():
    # Define packages in logical batches
    batches = {
        "Core HTTP/API": [
            "aiohttp==3.12.14",
            "aiosmtplib==4.0.1",
            "alpaca-py==0.20.0",
            "alpha_vantage==3.0.0",
            "requests>=2.25.0"
        ],
        "Data Science Core": [
            "numpy==1.25.2",
            "pandas==2.0.3",
            "scipy==1.11.3",
            "scikit-learn==1.3.2"
        ],
        "Web Framework": [
            "Flask==3.1.1",
            "flask-cors==6.0.1",
            "flask-limiter==3.12",
            "flask-socketio==5.5.1",
            "fastapi==0.116.1",
            "uvicorn==0.35.0"
        ],
        "Data Visualization": [
            "matplotlib==3.10.3",
            "plotly==6.2.0",
            "seaborn==0.13.2",
            "dash==3.1.1",
            "dash-bootstrap-components==2.0.3"
        ],
        "Machine Learning": [
            "torch==2.1.2",
            "torch-geometric==2.6.1",
            "lightgbm==4.6.0",
            "xgboost==3.0.2",
            "catboost==1.2.8",
            "transformers==4.53.2"
        ],
        "Utilities": [
            "python-dotenv==1.1.1",
            "PyYAML==6.0.2",
            "pytz==2024.1",
            "python-dateutil==2.9.0.post0",
            "filelock==3.18.0",
            "schedule==1.2.2"
        ],
        "Optional/Advanced": [
            "streamlit==1.46.1",
            "mlflow==3.1.1",
            "optuna==4.4.0",
            "ray==2.47.1",
            "prophet==1.1.7"
        ]
    }

    print("Starting batch installation of requirements...")

    # Install each batch
    for batch_name, packages in batches.items():
        success = install_packages(packages, batch_name)
        if not success:
            print(f"Warning: {batch_name} installation failed, continuing...")

    print("\n--- Installation Complete ---")
    print("Note: Some packages may have failed due to dependency conflicts.")
    print("You can try installing individual packages manually if needed.")


if __name__ == "__main__":
    main() 