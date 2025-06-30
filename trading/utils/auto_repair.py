"""Auto-repair system for handling common package and environment issues."""

import sys
import os
import subprocess
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from importlib.metadata import distributions, version, PackageNotFoundError
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
        
        return {
            'success': True,
            'message': 'AutoRepair system initialized successfully',
            'timestamp': datetime.now().isoformat()
        }

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for diagnostics."""
        system_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine()
        }
        
        return {
            'success': True,
            'result': system_info,
            'message': 'System information collected',
            'timestamp': datetime.now().isoformat()
        }
    
    def check_packages(self) -> Dict[str, Any]:
        """Check for missing or outdated packages."""
        try:
            missing = []
            outdated = []
            
            for package, min_version in self.REQUIRED_PACKAGES.items():
                try:
                    current_version = version(package)
                    if current_version < min_version:
                        outdated.append(f"{package} (current: {current_version}, required: {min_version})")
                except PackageNotFoundError:
                    missing.append(package)
            
            all_ok = len(missing) == 0 and len(outdated) == 0
            issues = missing + outdated
            
            return {
                'success': True,
                'result': all_ok,
                'missing': missing,
                'outdated': outdated,
                'issues': issues,
                'message': f'Package check completed: {len(issues)} issues found',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
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
            if platform.system() == 'Windows':
                try:
                    import torch
                    torch.__version__
                except ImportError as e:
                    if "libiomp5md.dll" in str(e):
                        issues.append("OpenMP DLL issue detected")
            
            return {
                'success': True,
                'result': len(issues) == 0,
                'issues': issues,
                'message': f'DLL check completed: {len(issues)} issues found',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_transformers(self) -> Dict[str, Any]:
        """Check for Hugging Face transformer issues."""
        try:
            issues = []
            
            try:
                from transformers import AutoTokenizer, AutoModel
                AutoTokenizer.from_pretrained("gpt2")
            except Exception as e:
                issues.append(f"Transformers issue: {str(e)}")
            
            return {
                'success': True,
                'result': len(issues) == 0,
                'issues': issues,
                'message': f'Transformers check completed: {len(issues)} issues found',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def repair_packages(self) -> Dict[str, Any]:
        """Attempt to repair package issues."""
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install/upgrade required packages
            repaired_count = 0
            for package, version in self.REQUIRED_PACKAGES.items():
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        f"{package}>={version}"
                    ])
                    repaired_count += 1
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {package}: {e}")
            
            self.repair_status['packages'] = True
            
            return {
                'success': True,
                'result': True,
                'message': f'Package repair completed: {repaired_count} packages processed',
                'timestamp': datetime.now().isoformat(),
                'repaired_count': repaired_count
            }
        except Exception as e:
            logger.error(f"Package repair failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def repair_dlls(self) -> Dict[str, Any]:
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
            
            return {
                'success': True,
                'result': True,
                'message': 'DLL repair completed successfully',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"DLL repair failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def repair_transformers(self) -> Dict[str, Any]:
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
            
            return {
                'success': True,
                'result': True,
                'message': 'Transformers repair completed successfully',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Transformers repair failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def repair_environment(self) -> Dict[str, Any]:
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
            
            return {
                'success': True,
                'result': True,
                'message': 'Environment repair completed successfully',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Environment repair failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_repair(self) -> Dict[str, Any]:
        """Run all repair checks and fixes."""
        try:
            results = {
                'status': 'success',
                'issues_found': [],
                'issues_fixed': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Check packages
            package_check = self.check_packages()
            if not package_check['result']:
                results['issues_found'].extend(package_check['issues'])
                package_repair = self.repair_packages()
                if package_repair['success']:
                    results['issues_fixed'].extend(package_check['issues'])
            
            # Check DLLs
            dll_check = self.check_dlls()
            if not dll_check['result']:
                results['issues_found'].extend(dll_check['issues'])
                dll_repair = self.repair_dlls()
                if dll_repair['success']:
                    results['issues_fixed'].extend(dll_check['issues'])
            
            # Check transformers
            transformer_check = self.check_transformers()
            if not transformer_check['result']:
                results['issues_found'].extend(transformer_check['issues'])
                transformer_repair = self.repair_transformers()
                if transformer_repair['success']:
                    results['issues_fixed'].extend(transformer_check['issues'])
            
            # Repair environment
            env_repair = self.repair_environment()
            if not env_repair['success']:
                results['status'] = 'partial'
            
            return {
                'success': True,
                'result': results,
                'message': f'Auto-repair completed: {len(results["issues_fixed"])} issues fixed',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_repair_status(self) -> Dict[str, Any]:
        """Get current repair status."""
        return {
            'success': True,
            'result': self.repair_status,
            'message': 'Repair status retrieved',
            'timestamp': datetime.now().isoformat()
        }

# Create singleton instance
auto_repair = AutoRepair() 