"""Auto-repair system for handling common package and environment issues."""

import sys
import os
import subprocess
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pkg_resources
import platform
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoRepair:
    """Handles automatic detection and repair of common package and environment issues."""
    
    REQUIRED_PACKAGES = {
        'numpy': '1.24.0',
        'pandas': '2.0.0',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'streamlit': '1.24.0',
        'plotly': '5.15.0',
        'scikit-learn': '1.3.0',
        'yfinance': '0.2.28',
        'openai': '1.0.0',
        'huggingface-hub': '0.16.0'
    }
    
    def __init__(self):
        """Initialize the auto-repair system."""
        self.repair_log = []
        self.system_info = self._get_system_info()
        self.repair_status = {
            'packages': False,
            'dlls': False,
            'transformers': False,
            'environment': False
        }
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for diagnostics."""
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine()
        }
    
    def check_packages(self) -> Tuple[bool, List[str]]:
        """Check for missing or outdated packages."""
        missing = []
        outdated = []
        
        for package, min_version in self.REQUIRED_PACKAGES.items():
            try:
                pkg = pkg_resources.working_set.by_key[package]
                if pkg.version < min_version:
                    outdated.append(f"{package} (current: {pkg.version}, required: {min_version})")
            except KeyError:
                missing.append(package)
        
        return len(missing) == 0 and len(outdated) == 0, missing + outdated
    
    def check_dlls(self) -> Tuple[bool, List[str]]:
        """Check for common DLL issues."""
        issues = []
        
        # Check for numpy DLL issues
        try:
            import numpy
            numpy.__version__
        except ImportError as e:
            if "_multiarray_umath" in str(e):
                issues.append("numpy DLL issue detected")
        
        # Check for OpenMP DLL issues
        if platform.system() == 'Windows':
            try:
                import torch
                torch.__version__
            except ImportError as e:
                if "libiomp5md.dll" in str(e):
                    issues.append("OpenMP DLL issue detected")
        
        return len(issues) == 0, issues
    
    def check_transformers(self) -> Tuple[bool, List[str]]:
        """Check for Hugging Face transformer issues."""
        issues = []
        
        try:
            from transformers import AutoTokenizer, AutoModel
            AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            issues.append(f"Transformers issue: {str(e)}")
        
        return len(issues) == 0, issues
    
    def repair_packages(self) -> bool:
        """Attempt to repair package issues."""
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install/upgrade required packages
            for package, version in self.REQUIRED_PACKAGES.items():
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    f"{package}>={version}"
                ])
            
            self.repair_status['packages'] = True
            return True
        except Exception as e:
            logger.error(f"Package repair failed: {str(e)}")
            return False
    
    def repair_dlls(self) -> bool:
        """Attempt to repair DLL issues."""
        try:
            if platform.system() == 'Windows':
                # Reinstall numpy to fix DLL issues
                subprocess.check_call([
                    sys.executable, "-m", "pip", "uninstall", "-y", "numpy"
                ])
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "numpy"
                ])
                
                # Reinstall torch to fix OpenMP issues
                subprocess.check_call([
                    sys.executable, "-m", "pip", "uninstall", "-y", "torch"
                ])
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "torch"
                ])
            
            self.repair_status['dlls'] = True
            return True
        except Exception as e:
            logger.error(f"DLL repair failed: {str(e)}")
            return False
    
    def repair_transformers(self) -> bool:
        """Attempt to repair transformer issues."""
        try:
            # Clear transformers cache
            cache_dir = Path.home() / ".cache" / "huggingface"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            
            # Reinstall transformers
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", "-y", "transformers"
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "transformers"
            ])
            
            self.repair_status['transformers'] = True
            return True
        except Exception as e:
            logger.error(f"Transformers repair failed: {str(e)}")
            return False
    
    def repair_environment(self) -> bool:
        """Attempt to repair the Python environment."""
        try:
            # Create virtual environment if needed
            venv_path = Path("venv")
            if not venv_path.exists():
                subprocess.check_call([
                    sys.executable, "-m", "venv", "venv"
                ])
            
            # Update environment variables
            os.environ["PYTHONPATH"] = str(Path.cwd())
            
            self.repair_status['environment'] = True
            return True
        except Exception as e:
            logger.error(f"Environment repair failed: {str(e)}")
            return False
    
    def run_repair(self) -> Dict[str, Any]:
        """Run all repair checks and fixes."""
        results = {
            'status': 'success',
            'issues_found': [],
            'issues_fixed': [],
            'system_info': self.system_info
        }
        
        # Check and repair packages
        packages_ok, package_issues = self.check_packages()
        if not packages_ok:
            results['issues_found'].extend(package_issues)
            if self.repair_packages():
                results['issues_fixed'].extend(package_issues)
        
        # Check and repair DLLs
        dlls_ok, dll_issues = self.check_dlls()
        if not dlls_ok:
            results['issues_found'].extend(dll_issues)
            if self.repair_dlls():
                results['issues_fixed'].extend(dll_issues)
        
        # Check and repair transformers
        transformers_ok, transformer_issues = self.check_transformers()
        if not transformers_ok:
            results['issues_found'].extend(transformer_issues)
            if self.repair_transformers():
                results['issues_fixed'].extend(transformer_issues)
        
        # Check and repair environment
        if not all(self.repair_status.values()):
            if self.repair_environment():
                results['issues_fixed'].append("Environment configuration")
        
        # Update final status
        if results['issues_found'] and not results['issues_fixed']:
            results['status'] = 'failed'
        elif results['issues_found'] and results['issues_fixed']:
            results['status'] = 'partial'
        
        return results
    
    def get_repair_status(self) -> Dict[str, Any]:
        """Get current repair status."""
        return {
            'status': self.repair_status,
            'system_info': self.system_info,
            'log': self.repair_log
        }

# Create singleton instance
auto_repair = AutoRepair() 