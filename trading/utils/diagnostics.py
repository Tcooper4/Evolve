"""System diagnostics and health checks for the trading platform."""

import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError as e:
    print("⚠️ psutil not available. Disabling system resource monitoring.")
    print(f"   Missing: {e}")
    psutil = None
    PSUTIL_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print("⚠️ PyTorch not available. Disabling PyTorch diagnostics.")
    print(f"   Missing: {e}")
    torch = None
    TORCH_AVAILABLE = False

from trading.utils.auto_repair import auto_repair
from trading.utils.error_logger import error_logger


class SystemDiagnostics:
    """System diagnostics and health checks."""

    def __init__(self):
        """Initialize the diagnostics system."""
        self.logger = logging.getLogger(__name__)
        self.health_status = {}
        self.last_check = None

    def check_data_loading(self) -> Tuple[bool, List[str]]:
        """Check data loading capabilities."""
        issues = []

        try:
            # Test numpy
            arr = np.random.rand(1000, 1000)
            np.save("test.npy", arr)
            loaded = np.load("test.npy")
            os.remove("test.npy")
            if not np.array_equal(arr, loaded):
                issues.append("NumPy data loading issue")
        except Exception as e:
            issues.append(f"NumPy error: {str(e)}")

        try:
            # Test pandas
            df = pd.DataFrame(np.random.rand(100, 10))
            df.to_csv("test.csv")
            loaded = pd.read_csv("test.csv")
            os.remove("test.csv")
            if not df.equals(loaded):
                issues.append("Pandas data loading issue")
        except Exception as e:
            issues.append(f"Pandas error: {str(e)}")

        return {
            "success": True,
            "result": len(issues) == 0,
            "issues": issues,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def check_forecasting_models(self) -> Tuple[bool, List[str]]:
        """Check forecasting model availability."""
        issues = []

        if not TORCH_AVAILABLE:
            issues.append("PyTorch not available for model diagnostics")
            return {
                "success": True,
                "result": False,
                "issues": issues,
                "message": "PyTorch not available",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # Check PyTorch
            if not torch.cuda.is_available():
                issues.append("CUDA not available for PyTorch")

            # Test model loading
            model = torch.nn.Linear(10, 1)
            x = torch.randn(100, 10)
            y = model(x)
            if y.shape != (100, 1):
                issues.append("PyTorch model shape mismatch")
        except Exception as e:
            issues.append(f"PyTorch error: {str(e)}")

        return {
            "success": True,
            "result": len(issues) == 0,
            "issues": issues,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def check_agent_communication(self) -> Tuple[bool, List[str]]:
        """Check agent communication channels."""
        issues = []

        # Check required directories
        required_dirs = [
            "trading/agents",
            "trading/memory",
            "trading/models",
            "trading/data",
        ]

        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                issues.append(f"Missing directory: {dir_path}")

        # Check memory access
        try:
            memory_dir = Path("trading/memory")
            if not memory_dir.exists():
                memory_dir.mkdir(parents=True)

            # Test file creation
            test_file = memory_dir / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            issues.append(f"Memory access error: {str(e)}")

        return {
            "success": True,
            "result": len(issues) == 0,
            "issues": issues,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def check_system_resources(self) -> Tuple[bool, List[str]]:
        """Check system resource availability."""
        issues = []

        if not PSUTIL_AVAILABLE:
            issues.append("psutil not available for system resource monitoring")
            return {
                "success": True,
                "result": False,
                "issues": issues,
                "message": "psutil not available",
                "timestamp": datetime.now().isoformat(),
            }

        # Check CPU
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 90:
            issues.append(f"High CPU usage: {cpu_percent}%")

        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            issues.append(f"High memory usage: {memory.percent}%")

        # Check disk space
        disk = psutil.disk_usage("/")
        if disk.percent > 90:
            issues.append(f"Low disk space: {disk.percent}% used")

        return {
            "success": True,
            "result": len(issues) == 0,
            "issues": issues,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def run_health_check(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "checks": {},
            "issues": [],
        }

        # Run all checks
        checks = {
            "data_loading": self.check_data_loading,
            "forecasting_models": self.check_forecasting_models,
            "agent_communication": self.check_agent_communication,
            "system_resources": self.check_system_resources,
        }

        for name, check_func in checks.items():
            is_healthy, issues = check_func()
            results["checks"][name] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "issues": issues,
            }
            if issues:
                results["issues"].extend(issues)

        # Update overall status
        if results["issues"]:
            results["status"] = "unhealthy"

        # Save results
        self.health_status = results
        self.last_check = datetime.now()

        # Log issues
        if results["issues"]:
            error_logger.log_error("Health check failed", context=results)

        return results

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.last_check or (datetime.now() - self.last_check).seconds > 300:
            return self.run_health_check()
        return self.health_status

    def save_report(self, filepath: str = "health_report.json") -> None:
        """Save health check report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "health_status": self.health_status,
            "system_info": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "memory": f"{psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB",
            },
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)


# Create singleton instance
diagnostics = SystemDiagnostics()
