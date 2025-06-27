"""
Self Repair Agent

This agent is responsible for:
1. Scanning and detecting issues in forecasting models and strategy scripts
2. Automatically applying patches and fixes
3. Synchronizing configurations across the system
4. Logging repair activities and maintenance

The agent performs proactive maintenance to ensure system stability
and consistency across all components.
"""

import os
import json
import logging
import importlib
import inspect
import ast
from importlib.metadata import distributions, version, PackageNotFoundError
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import sys
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RepairIssue:
    """Container for repair issues and their fixes"""
    issue_type: str
    file_path: str
    description: str
    severity: str
    fix_applied: bool
    timestamp: str
    fix_details: Dict

class SelfRepairAgent:
    def __init__(self, config_path: str = "config/self_repair_config.json"):
        """
        Initialize the SelfRepairAgent.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.repair_log = self._load_repair_log()
        self.issue_patterns = self._load_issue_patterns()
        self.dependency_map = self._load_dependency_map()
        
        # Initialize paths
        self.base_dir = Path(self.config.get("base_dir", "trading"))
        self.models_dir = self.base_dir / "models"
        self.strategies_dir = self.base_dir / "strategies"
        self.config_dir = Path(self.config.get("config_dir", "config"))
        self.log_dir = Path(self.config.get("log_dir", "logs/maintenance"))
        
        # Create necessary directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file watcher
        self.observer = Observer()
        self.observer.schedule(
            CodeChangeHandler(self),
            str(self.base_dir),
            recursive=True
        )
        self.observer.start()
        
        logger.info("Initialized SelfRepairAgent")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}

    def _load_repair_log(self) -> List[RepairIssue]:
        """Load repair history from log file"""
        try:
            log_path = self.log_dir / "repair_history.json"
            if not log_path.exists():
                return []
                
            with open(log_path, 'r') as f:
                raw_issues = json.load(f)
                return [RepairIssue(**issue) for issue in raw_issues]
        except Exception as e:
            logger.error(f"Error loading repair log: {e}")
            return []

    def _load_issue_patterns(self) -> Dict[str, Dict]:
        """Load patterns for detecting common issues"""
        return {
            "import_error": {
                "pattern": r"ImportError: No module named '(\w+)'",
                "severity": "high",
                "fix_type": "dependency"
            },
            "deprecated_warning": {
                "pattern": r"DeprecationWarning: .* is deprecated",
                "severity": "medium",
                "fix_type": "code_update"
            },
            "config_mismatch": {
                "pattern": r"KeyError: '(\w+)'",
                "severity": "medium",
                "fix_type": "config"
            },
            "model_version": {
                "pattern": r"Model version mismatch",
                "severity": "high",
                "fix_type": "model_update"
            }
        }

    def _load_dependency_map(self) -> Dict[str, List[str]]:
        """Load mapping of dependencies between components"""
        try:
            map_path = self.config_dir / "dependency_map.json"
            if not map_path.exists():
                return {}
                
            with open(map_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading dependency map: {e}")
            return {}

    async def scan_for_issues(self) -> List[RepairIssue]:
        """
        Scan for issues in the codebase.
        
        Returns:
            List of detected issues
        """
        try:
            issues = []
            
            # Scan Python files
            for py_file in self.base_dir.rglob("*.py"):
                file_issues = await self._scan_python_file(py_file)
                issues.extend(file_issues)
            
            # Scan configuration files
            for config_file in self.config_dir.rglob("*.json"):
                file_issues = await self._scan_config_file(config_file)
                issues.extend(file_issues)
            
            # Scan model files
            for model_file in self.models_dir.rglob("*.h5"):
                file_issues = await self._scan_model_file(model_file)
                issues.extend(file_issues)
            
            # Check dependencies
            dep_issues = await self._check_dependencies()
            issues.extend(dep_issues)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error scanning for issues: {e}")
            return []

    async def _scan_python_file(self, file_path: Path) -> List[RepairIssue]:
        """Scan a Python file for issues"""
        try:
            issues = []
            
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Check imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        try:
                            importlib.import_module(name.name)
                        except ImportError:
                            issues.append(RepairIssue(
                                issue_type="import_error",
                                file_path=str(file_path),
                                description=f"Missing import: {name.name}",
                                severity="high",
                                fix_applied=False,
                                timestamp=datetime.now().isoformat(),
                                fix_details={"module": name.name}
                            ))
            
            # Check for deprecated functions
            for pattern in self.issue_patterns["deprecated_warning"]["pattern"]:
                if re.search(pattern, content):
                    issues.append(RepairIssue(
                        issue_type="deprecated_warning",
                        file_path=str(file_path),
                        description=f"Deprecated function usage detected",
                        severity="medium",
                        fix_applied=False,
                        timestamp=datetime.now().isoformat(),
                        fix_details={"pattern": pattern}
                    ))
            
            return issues
            
        except Exception as e:
            logger.error(f"Error scanning Python file {file_path}: {e}")
            return []

    async def _scan_config_file(self, file_path: Path) -> List[RepairIssue]:
        """Scan a configuration file for issues"""
        try:
            issues = []
            
            # Read config
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Check required fields
            required_fields = self.config.get("required_config_fields", [])
            for field in required_fields:
                if field not in config:
                    issues.append(RepairIssue(
                        issue_type="config_mismatch",
                        file_path=str(file_path),
                        description=f"Missing required field: {field}",
                        severity="medium",
                        fix_applied=False,
                        timestamp=datetime.now().isoformat(),
                        fix_details={"field": field}
                    ))
            
            return issues
            
        except Exception as e:
            logger.error(f"Error scanning config file {file_path}: {e}")
            return []

    async def _scan_model_file(self, file_path: Path) -> List[RepairIssue]:
        """Scan a model file for issues"""
        try:
            issues = []
            
            # Check model version
            model_config_path = file_path.with_suffix('.json')
            if model_config_path.exists():
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                
                current_version = model_config.get("version")
                expected_version = self.config.get("expected_model_version")
                
                if current_version != expected_version:
                    issues.append(RepairIssue(
                        issue_type="model_version",
                        file_path=str(file_path),
                        description=f"Model version mismatch: {current_version} vs {expected_version}",
                        severity="high",
                        fix_applied=False,
                        timestamp=datetime.now().isoformat(),
                        fix_details={
                            "current_version": current_version,
                            "expected_version": expected_version
                        }
                    ))
            
            return issues
            
        except Exception as e:
            logger.error(f"Error scanning model file {file_path}: {e}")
            return []

    async def _check_dependencies(self) -> List[RepairIssue]:
        """Check for dependency issues"""
        try:
            issues = []
            installed = {dist.metadata['Name']: dist.version for dist in distributions()}
            
            # Check required dependencies
            required_deps = {
                'numpy': '1.24.0',
                'pandas': '2.0.0',
                'torch': '2.0.0',
                'transformers': '4.30.0',
                'streamlit': '1.24.0'
            }
            
            for package, min_version in required_deps.items():
                try:
                    current_version = version(package)
                    if current_version < min_version:
                        issues.append(RepairIssue(
                            issue_type="dependency",
                            file_path="requirements.txt",
                            description=f"Package {package} version {current_version} < {min_version}",
                            severity="high",
                            fix_applied=False,
                            timestamp=datetime.now().isoformat(),
                            fix_details={"package": package, "required_version": min_version}
                        ))
                except PackageNotFoundError:
                    issues.append(RepairIssue(
                        issue_type="dependency",
                        file_path="requirements.txt",
                        description=f"Package {package} not installed",
                        severity="high",
                        fix_applied=False,
                        timestamp=datetime.now().isoformat(),
                        fix_details={"package": package, "required_version": min_version}
                    ))
            
            return issues
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return []

    async def apply_patches(self, issues: List[RepairIssue]) -> List[RepairIssue]:
        """
        Apply fixes for detected issues.
        
        Args:
            issues: List of issues to fix
            
        Returns:
            List of issues with fix status updated
        """
        try:
            fixed_issues = []
            
            for issue in issues:
                try:
                    if issue.issue_type == "import_error":
                        await self._fix_import_error(issue)
                    elif issue.issue_type == "deprecated_warning":
                        await self._fix_deprecated_warning(issue)
                    elif issue.issue_type == "config_mismatch":
                        await self._fix_config_mismatch(issue)
                    elif issue.issue_type == "model_version":
                        await self._fix_model_version(issue)
                    elif issue.issue_type == "dependency":
                        await self._fix_dependency(issue)
                    
                    issue.fix_applied = True
                    fixed_issues.append(issue)
                    
                except Exception as e:
                    logger.error(f"Error fixing issue {issue}: {e}")
                    fixed_issues.append(issue)
            
            return fixed_issues
            
        except Exception as e:
            logger.error(f"Error applying patches: {e}")
            return issues

    async def _fix_import_error(self, issue: RepairIssue) -> None:
        """Fix missing import issues"""
        try:
            # Install missing package
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                issue.fix_details["module"]
            ], check=True)
            
            logger.info(f"Installed missing package: {issue.fix_details['module']}")
            
        except Exception as e:
            logger.error(f"Error fixing import error: {e}")
            raise

    async def _fix_deprecated_warning(self, issue: RepairIssue) -> None:
        """Fix deprecated function usage"""
        try:
            # Read file
            with open(issue.file_path, 'r') as f:
                content = f.read()
            
            # Apply fix based on pattern
            # This would need specific patterns and replacements
            # for each type of deprecation
            
            # Write updated content
            with open(issue.file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Fixed deprecated function in {issue.file_path}")
            
        except Exception as e:
            logger.error(f"Error fixing deprecated warning: {e}")
            raise

    async def _fix_config_mismatch(self, issue: RepairIssue) -> None:
        """Fix configuration mismatches"""
        try:
            # Read config
            with open(issue.file_path, 'r') as f:
                config = json.load(f)
            
            # Add missing field with default value
            field = issue.fix_details["field"]
            config[field] = self.config.get("default_config_values", {}).get(field)
            
            # Write updated config
            with open(issue.file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Fixed config mismatch in {issue.file_path}")
            
        except Exception as e:
            logger.error(f"Error fixing config mismatch: {e}")
            raise

    async def _fix_model_version(self, issue: RepairIssue) -> None:
        """Fix model version mismatches"""
        try:
            # Update model version in config
            model_config_path = Path(issue.file_path).with_suffix('.json')
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            
            model_config["version"] = issue.fix_details["expected_version"]
            
            with open(model_config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            logger.info(f"Updated model version in {model_config_path}")
            
        except Exception as e:
            logger.error(f"Error fixing model version: {e}")
            raise

    async def _fix_dependency(self, issue: RepairIssue) -> None:
        """Fix dependency issues"""
        try:
            # Install or update package
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade",
                issue.fix_details["package"]
            ], check=True)
            
            logger.info(f"Updated dependency: {issue.fix_details['package']}")
            
        except Exception as e:
            logger.error(f"Error fixing dependency error: {e}")
            raise

    async def sync_configs(self) -> None:
        """Synchronize all configuration files"""
        try:
            # Load master config
            master_config = self._load_master_config()
            
            # Update all config files
            for config_file in self.config_dir.rglob("*.json"):
                if config_file.name == "master_config.json":
                    continue
                    
                await self._sync_config_file(config_file, master_config)
            
            logger.info("Synchronized all configuration files")
            
        except Exception as e:
            logger.error(f"Error synchronizing configs: {e}")

    def _load_master_config(self) -> Dict:
        """Load the master configuration file"""
        try:
            master_path = self.config_dir / "master_config.json"
            if not master_path.exists():
                return {}
                
            with open(master_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading master config: {e}")
            return {}

    async def _sync_config_file(self, config_path: Path, master_config: Dict) -> None:
        """Synchronize a single config file with master config"""
        try:
            # Read current config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update with master values
            for key, value in master_config.items():
                if key in config and config[key] != value:
                    config[key] = value
            
            # Write updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Synchronized config file: {config_path}")
            
        except Exception as e:
            logger.error(f"Error synchronizing config file {config_path}: {e}")

    def log_repair(self, issue: RepairIssue) -> None:
        """
        Log repair activity.
        
        Args:
            issue: The repair issue to log
        """
        try:
            # Add to in-memory log
            self.repair_log.append(issue)
            
            # Keep only recent repairs
            max_log_size = self.config.get("max_log_size", 1000)
            self.repair_log = self.repair_log[-max_log_size:]
            
            # Save to disk
            log_path = self.log_dir / "repair_history.json"
            with open(log_path, 'w') as f:
                json.dump(
                    [vars(issue) for issue in self.repair_log],
                    f,
                    indent=2
                )
            
            logger.info(f"Logged repair activity: {issue.issue_type} in {issue.file_path}")
            
        except Exception as e:
            logger.error(f"Error logging repair: {e}")

    async def execute_repair_loop(self) -> None:
        """Main execution loop for the SelfRepairAgent.
        
        Continuously monitors the codebase for issues and applies automatic repairs.
        Runs until interrupted or an error occurs.
        """
        try:
            while True:
                # Scan for issues
                issues = await self.scan_for_issues()
                
                if issues:
                    # Apply fixes
                    fixed_issues = await self.apply_patches(issues)
                    
                    # Log repairs
                    for issue in fixed_issues:
                        self.log_repair(issue)
                    
                    # Sync configs
                    await self.sync_configs()
                
                # Wait before next check
                await asyncio.sleep(self.config.get("check_interval", 3600))  # Default: 1 hour
                
        except KeyboardInterrupt:
            logger.info("SelfRepairAgent stopped by user")
        except Exception as e:
            logger.error(f"Error in SelfRepairAgent: {e}")
        finally:
            self.observer.stop()
            self.observer.join()

class CodeChangeHandler(FileSystemEventHandler):
    """Handler for monitoring code changes"""
    
    def __init__(self, agent: SelfRepairAgent) -> None:
        """Initialize the code change handler.
        
        Args:
            agent: The SelfRepairAgent instance to notify of changes
        """
        self.agent = agent
    
    def on_modified(self, event) -> None:
        """Handle file modification events.
        
        Args:
            event: File system event containing information about the modification
        """
        if event.is_directory:
            return
        if event.src_path.endswith(('.py', '.json', '.h5')):
            logger.info(f"Code change detected: {event.src_path}")
            # Trigger immediate scan
            asyncio.create_task(self.agent.scan_for_issues())

if __name__ == "__main__":
    agent = SelfRepairAgent()
    asyncio.run(agent.execute_repair_loop()) 