"""Test repair agent for maintaining test coverage and quality."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pytest
import coverage

from trading.base_agent import BaseMetaAgent
from trading.tests import conftest

class TestRepairAgent(BaseMetaAgent):
    """Agent for maintaining and improving test coverage."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the test repair agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("test_repair", config)
        self.coverage = coverage.Coverage()def run(self) -> Dict[str, Any]:
        """Run test analysis and repair.
        
        Returns:
            Dict containing test analysis results and suggested improvements
        """
        results = {
            "coverage": self._analyze_coverage(),
            "missing_tests": self._find_missing_tests(),
            "suggested_improvements": []
        }
        
        # Generate improvements
        improvements = self._generate_improvements(results)
        results["suggested_improvements"] = improvements
        
        # Log results
        self.log_action("Test analysis completed", results)
        
        return results
    
    def _analyze_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage.
        
        Returns:
            Dict containing coverage analysis results
        """
        self.coverage.start()
        pytest.main(["-xvs", "tests/"])
        self.coverage.stop()
        self.coverage.save()
        
        return {
            "total_coverage": self.coverage.report(),
            "missing_lines": self.coverage.get_missing()
        }
    
    def _find_missing_tests(self) -> List[Dict[str, Any]]:
        """Find modules and functions missing tests.
        
        Returns:
            List of missing test cases
        """
        missing = []
        
        # Analyze each module
        for module in self._get_modules():
            if not self._has_tests(module):
                missing.append({
                    "module": module,
                    "type": "module",
                    "priority": "high"
                })
        
        return missing
    
    def _generate_improvements(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test improvements.
        
        Args:
            results: Results from test analysis
            
        Returns:
            List of suggested improvements
        """
        improvements = []
        
        # Check coverage
        if results["coverage"]["total_coverage"] < 0.8:
            improvements.append({
                "type": "coverage",
                "description": "Low test coverage",
                "suggestion": "Add more test cases"
            })
        
        # Check missing tests
        for missing in results["missing_tests"]:
            improvements.append({
                "type": "missing_test",
                "target": missing["module"],
                "description": f"Missing tests for {missing['module']}",
                "suggestion": f"Create test file: tests/test_{missing['module']}.py"
            })
        
        return improvements
    
    def _get_modules(self) -> List[str]:
        """Get list of modules to test.
        
        Returns:
            List of module names
        """
        modules = []
        for path in Path("trading").rglob("*.py"):
            if path.stem != "__init__":
                modules.append(path.stem)
        return modules
    
    def _has_tests(self, module: str) -> bool:
        """Check if a module has tests.
        
        Args:
            module: Module name
            
        Returns:
            True if module has tests
        """
        test_file = Path(f"tests/test_{module}.py")
        return test_file.exists()