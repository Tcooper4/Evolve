#!/usr/bin/env python3
"""
Targeted Return Statement Audit for Evolve Project
Scans only project files, excluding virtual environments and external libraries.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReturnStatementAuditor:
    def __init__(self):
        self.violations = []
        self.total_files = 0
        self.total_functions = 0

        # Directories to exclude
        self.exclude_dirs = {
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            "node_modules",
            "site-packages",
            "dist",
            "build",
            ".pytest_cache",
            "htmlcov",
            "coverage",
            ".mypy_cache",
            ".tox",
        }

        # File patterns to exclude
        self.exclude_patterns = {
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
            "*.dll",
            "*.exe",
            "*.egg",
            "*.whl",
            "*.tar.gz",
            "*.zip",
        }

    def should_scan_file(self, filepath: Path) -> bool:
        """Check if file should be scanned."""
        # Skip excluded directories
        for part in filepath.parts:
            if part in self.exclude_dirs:
                return False

        # Skip excluded file patterns
        if any(filepath.match(pattern) for pattern in self.exclude_patterns):
            return False

        # Only scan Python files in our project
        if not filepath.suffix == ".py":
            return False

        # Skip files outside our main project directories
        project_dirs = {
            "trading",
            "core",
            "utils",
            "services",
            "strategies",
            "models",
            "data",
            "execution",
            "evaluation",
            "memory",
            "risk",
            "logs",
            "ui",
            "config",
            "pages",
            "scripts",
        }

        # Check if file is in a project directory
        for part in filepath.parts:
            if part in project_dirs:
                return True

        # Also scan root level Python files
        if len(filepath.parts) <= 2:  # Root level files
            return True

        return False

    def find_python_files(self, root_dir: str = ".") -> List[Path]:
        """Find all Python files to audit."""
        root_path = Path(root_dir)
        python_files = []

        for filepath in root_path.rglob("*.py"):
            if self.should_scan_file(filepath):
                python_files.append(filepath)

        return python_files

    def audit_function(self, node: ast.FunctionDef, filepath: str) -> List[Dict[str, Any]]:
        """Audit a single function for return statement violations."""
        violations = []

        # Skip abstract methods and properties
        if any(
            decorator.id == "abstractmethod" for decorator in node.decorator_list if isinstance(decorator, ast.Name)
        ):
            return violations

        if any(
            decorator.attr == "property" for decorator in node.decorator_list if isinstance(decorator, ast.Attribute)
        ):
            return violations

        # Check if function has any return statements
        has_return = False
        has_structured_return = False

        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                has_return = True
                # Check if it's a structured return (dict with success key)
                if isinstance(child.value, ast.Dict) and any(
                    isinstance(k, ast.Constant) and k.value == "success" for k in child.value.keys
                ):
                    has_structured_return = True
                elif (
                    isinstance(child.value, ast.Call)
                    and isinstance(child.value.func, ast.Name)
                    and child.value.func.id in ["dict", "Dict"]
                ):
                    has_structured_return = True

        # Check for violations
        if not has_return:
            violations.append(
                {
                    "type": "missing_return",
                    "function": node.name,
                    "line": node.lineno,
                    "message": "Function has no return statement",
                }
            )
        elif has_return and not has_structured_return:
            violations.append(
                {
                    "type": "unstructured_return",
                    "function": node.name,
                    "line": node.lineno,
                    "message": "Function has return statement but not structured return",
                }
            )

        return violations

    def audit_file(self, filepath: Path) -> Dict[str, Any]:
        """Audit a single file for return statement violations."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            violations = []
            function_count = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
                    node_violations = self.audit_function(node, str(filepath))
                    violations.extend(node_violations)

            return {
                "filepath": str(filepath),
                "violations": violations,
                "function_count": function_count,
                "violation_count": len(violations),
            }

        except Exception as e:
            logger.error(f"Error auditing {filepath}: {e}")
            return {
                "filepath": str(filepath),
                "violations": [],
                "function_count": 0,
                "violation_count": 0,
                "error": str(e),
            }

    def audit_project(self, root_dir: str = ".") -> Dict[str, Any]:
        """Audit the entire project."""
        logger.info("Starting targeted return statement audit...")

        python_files = self.find_python_files(root_dir)
        logger.info(f"Found {len(python_files)} Python files to audit")

        results = {
            "total_files": len(python_files),
            "total_functions": 0,
            "total_violations": 0,
            "files_with_violations": 0,
            "violations_by_type": {},
            "file_results": [],
        }

        for filepath in python_files:
            file_result = self.audit_file(filepath)
            results["file_results"].append(file_result)
            results["total_functions"] += file_result["function_count"]
            results["total_violations"] += file_result["violation_count"]

            if file_result["violation_count"] > 0:
                results["files_with_violations"] += 1

                # Count violations by type
                for violation in file_result["violations"]:
                    violation_type = violation["type"]
                    results["violations_by_type"][violation_type] = (
                        results["violations_by_type"].get(violation_type, 0) + 1
                    )

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print audit summary."""
        print("\n" + "=" * 60)
        print("RETURN STATEMENT AUDIT SUMMARY")
        print("=" * 60)
        print(f"Total files scanned: {results['total_files']}")
        print(f"Total functions found: {results['total_functions']}")
        print(f"Files with violations: {results['files_with_violations']}")
        print(f"Total violations: {results['total_violations']}")
        print()

        if results["violations_by_type"]:
            print("Violations by type:")
            for violation_type, count in results["violations_by_type"].items():
                print(f"  {violation_type}: {count}")
            print()

        if results["total_violations"] > 0:
            print("Files with violations:")
            for file_result in results["file_results"]:
                if file_result["violation_count"] > 0:
                    print(f"  {file_result['filepath']}: {file_result['violation_count']} violations")
                    for violation in file_result["violations"]:
                        print(f"    - {violation['function']} (line {violation['line']}): {violation['message']}")
        else:
            print("✅ No return statement violations found!")

        print("=" * 60)


def main():
    auditor = ReturnStatementAuditor()
    results = auditor.audit_project()
    auditor.print_summary(results)

    # Save detailed results to file
    with open("return_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Audit complete. Results saved to return_audit_results.json")


if __name__ == "__main__":
    main()
