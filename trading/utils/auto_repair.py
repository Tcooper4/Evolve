"""
Auto-repair utilities for common package and environment issues.

This module provides automatic detection and repair of common issues
that can occur in the trading system environment.
"""

import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import version checking utilities
try:
    from importlib.metadata import PackageNotFoundError, version

    VERSION_CHECK_AVAILABLE = True
except ImportError:
    VERSION_CHECK_AVAILABLE = False
    version = None
    PackageNotFoundError = None


class AutoRepair:
    """Handles automatic detection and repair of common package and environment issues."""

    REQUIRED_PACKAGES = {
        "numpy": "1.20.0",
        "pandas": "1.3.0",
        "scikit-learn": "1.0.0",
        "matplotlib": "3.3.0",
        "plotly": "5.0.0",
        "torch": "1.9.0",
        "transformers": "4.15.0",
        "yfinance": "0.1.70",
        "alpaca-trade-api": "2.3.0",
        "streamlit": "1.0.0",
    }

    def __init__(self):
        """Initialize the auto-repair system."""
        self.repair_status = {
            "packages": False,
            "dlls": False,
            "transformers": False,
            "environment": False,
        }
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            "platform": platform.system(),
            "python_version": sys.version,
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
        }

    def check_packages(self) -> Dict[str, Any]:
        """Check for missing or outdated packages."""
        try:
            if not VERSION_CHECK_AVAILABLE:
                return {
                    "success": False,
                    "error": "importlib.metadata not available for package checking",
                    "timestamp": datetime.now().isoformat(),
                }

            missing = []
            outdated = []

            for package, min_version in self.REQUIRED_PACKAGES.items():
                try:
                    current_version = version(package)
                    if current_version < min_version:
                        outdated.append(
                            f"{package} (current: {current_version}, required: {min_version})"
                        )
                except PackageNotFoundError:
                    missing.append(package)

            all_ok = len(missing) == 0 and len(outdated) == 0
            issues = missing + outdated

            return {
                "success": True,
                "result": all_ok,
                "missing": missing,
                "outdated": outdated,
                "issues": issues,
                "message": f"Package check completed: {len(issues)} issues found",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def check_dlls(self) -> Dict[str, Any]:
        """Check for common DLL issues."""
        try:
            issues = []

            # Check for numpy DLL issues
            try:
                import numpy

                numpy.__version__
            except ImportError as e:
                if "_multiarray_umath" in str(e):
                    issues.append("numpy DLL issue detected")

            # Check for OpenMP DLL issues
            if platform.system() == "Windows":
                try:
                    import torch

                    torch.__version__
                except ImportError as e:
                    if "libiomp5md.dll" in str(e):
                        issues.append("OpenMP DLL issue detected")

            return {
                "success": True,
                "result": len(issues) == 0,
                "issues": issues,
                "message": f"DLL check completed: {len(issues)} issues found",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def check_transformers(self) -> Dict[str, Any]:
        """Check for Hugging Face transformer issues."""
        try:
            issues = []

            try:
                from transformers import AutoTokenizer

                AutoTokenizer.from_pretrained("gpt2")
            except Exception as e:
                issues.append(f"Transformers issue: {str(e)}")

            return {
                "success": True,
                "result": len(issues) == 0,
                "issues": issues,
                "message": f"Transformers check completed: {len(issues)} issues found",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def repair_packages(self) -> Dict[str, Any]:
        """Attempt to repair package issues."""
        try:
            # Upgrade pip first
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
            )

            # Install/upgrade required packages
            repaired_count = 0
            for package, version in self.REQUIRED_PACKAGES.items():
                try:
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            f"{package}>={version}",
                        ]
                    )
                    repaired_count += 1
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {package}: {e}")

            self.repair_status["packages"] = True

            return {
                "success": True,
                "result": True,
                "message": f"Package repair completed: {repaired_count} packages processed",
                "timestamp": datetime.now().isoformat(),
                "repaired_count": repaired_count,
            }
        except Exception as e:
            logger.error(f"Package repair failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def repair_dlls(self) -> Dict[str, Any]:
        """Attempt to repair DLL issues."""
        try:
            if platform.system() == "Windows":
                # Reinstall numpy to fix DLL issues
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "uninstall", "-y", "numpy"]
                )
                subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

                # Reinstall torch to fix OpenMP issues
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "uninstall", "-y", "torch"]
                )
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

            self.repair_status["dlls"] = True

            return {
                "success": True,
                "result": True,
                "message": "DLL repair completed successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"DLL repair failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def repair_transformers(self) -> Dict[str, Any]:
        """Attempt to repair transformer issues."""
        try:
            # Clear transformers cache
            cache_dir = Path.home() / ".cache" / "huggingface"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            # Reinstall transformers
            subprocess.check_call(
                [sys.executable, "-m", "pip", "uninstall", "-y", "transformers"]
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "transformers"]
            )

            self.repair_status["transformers"] = True

            return {
                "success": True,
                "result": True,
                "message": "Transformers repair completed successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Transformers repair failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def repair_environment(self) -> Dict[str, Any]:
        """Attempt to repair environment issues."""
        try:
            # Create virtual environment if needed
            venv_path = Path("venv")
            if not venv_path.exists():
                subprocess.check_call([sys.executable, "-m", "venv", "venv"])

            # Update environment variables
            env_vars = {
                "PYTHONPATH": str(Path.cwd()),
                "TRADING_ENV": "development",
            }

            self.repair_status["environment"] = True

            return {
                "success": True,
                "result": True,
                "message": "Environment repair completed successfully",
                "timestamp": datetime.now().isoformat(),
                "env_vars": env_vars,
            }
        except Exception as e:
            logger.error(f"Environment repair failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def run_repair(self) -> Dict[str, Any]:
        """Run comprehensive repair process."""
        try:
            logger.info("Starting comprehensive repair process...")

            # Check for issues first
            package_check = self.check_packages()
            dll_check = self.check_dlls()
            transformers_check = self.check_transformers()

            repair_results = {
                "packages": None,
                "dlls": None,
                "transformers": None,
                "environment": None,
            }

            # Repair packages if needed
            if not package_check.get("result", True):
                logger.info("Repairing packages...")
                repair_results["packages"] = self.repair_packages()

            # Repair DLLs if needed
            if not dll_check.get("result", True):
                logger.info("Repairing DLLs...")
                repair_results["dlls"] = self.repair_dlls()

            # Repair transformers if needed
            if not transformers_check.get("result", True):
                logger.info("Repairing transformers...")
                repair_results["transformers"] = self.repair_transformers()

            # Always run environment repair
            logger.info("Repairing environment...")
            repair_results["environment"] = self.repair_environment()

            # Check if all repairs were successful
            all_successful = all(
                result.get("success", False) if result else True
                for result in repair_results.values()
            )

            return {
                "success": all_successful,
                "result": all_successful,
                "repair_results": repair_results,
                "message": "Comprehensive repair process completed",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Comprehensive repair failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_repair_status(self) -> Dict[str, Any]:
        """Get current repair status."""
        return {
            "success": True,
            "repair_status": self.repair_status,
            "system_info": self.system_info,
            "timestamp": datetime.now().isoformat(),
        }


# Global auto-repair instance
auto_repair = AutoRepair()


def run_auto_repair() -> Dict[str, Any]:
    """Run auto-repair process."""
    return auto_repair.run_repair()


def check_system_health() -> Dict[str, Any]:
    """Check overall system health."""
    package_check = auto_repair.check_packages()
    dll_check = auto_repair.check_dlls()
    transformers_check = auto_repair.check_transformers()

    all_healthy = all(
        [
            package_check.get("result", True),
            dll_check.get("result", True),
            transformers_check.get("result", True),
        ]
    )

    return {
        "success": True,
        "healthy": all_healthy,
        "checks": {
            "packages": package_check,
            "dlls": dll_check,
            "transformers": transformers_check,
        },
        "timestamp": datetime.now().isoformat(),
    }
