#!/usr/bin/env python3
"""
Comprehensive audit script to verify all functions have proper return statements.
This script checks the entire Evolve codebase for agentic modularity compliance.
"""

import ast
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReturnStatementAuditor:
    """Auditor for checking return statement compliance."""

    def __init__(self, exclude_dirs: Set[str] = None):
        self.exclude_dirs = exclude_dirs or {
            ".venv",
            "venv",
            "env",
            "archive",
            "legacy",
            "test_coverage",
            "__pycache__",
            ".git",
            "htmlcov",
            "site-packages",
            "dist",
            "build",
            "node_modules",
        }
        self.issues = []
        self.passing_functions = []
        self.exempt_functions = set()
        self.return_value_usage = {}  # Track return value usage
        self.unused_returns = []  # Track unused return values
        self.pipeline_functions = set()  # Track pipeline functions

    def audit_codebase(self, root_dir: str = ".") -> Dict[str, Any]:
        """Audit the entire codebase for return statement compliance."""
        logger.info("Starting comprehensive return statement audit...")

        python_files = self._find_python_files(root_dir)
        logger.info(f"Found {len(python_files)} Python files to audit")

        # First pass: identify all functions and their return values
        for file_path in python_files:
            self._audit_file(file_path)

        # Second pass: analyze return value usage
        for file_path in python_files:
            self._analyze_return_usage(file_path)

        # Third pass: identify pipeline functions and check their return usage
        self._analyze_pipeline_return_usage(python_files)

        return self._generate_report()

    def _find_python_files(self, root_dir: str) -> List[str]:
        """Find all Python files in the codebase."""
        python_files = []
        for root, dirs, files in os.walk(root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        return python_files

    def _audit_file(self, file_path: str):
        """Audit a single Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            self._analyze_ast(file_path, tree, content)

        except Exception as e:
            logger.error(f"Error auditing {file_path}: {e}")
            self.issues.append({"file": file_path, "type": "parse_error", "error": str(e)})

    def _analyze_ast(self, file_path: str, tree: ast.AST, content: str):
        """Analyze AST for function definitions and return statements."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._check_function(file_path, node, content)

    def _check_function(self, file_path: str, func_node: ast.FunctionDef, content: str):
        """Check if a function has proper return statements."""
        func_name = func_node.name
        func_line = func_node.lineno

        # Skip __init__ methods (exempt from return requirement)
        if func_name == "__init__":
            self.exempt_functions.add(f"{file_path}:{func_name}")
            return

        # Check if function has any return statements
        has_return = self._has_return_statement(func_node)

        # Check if function has only logging/print statements
        has_only_logging = self._has_only_logging_statements(func_node, content)

        # Check if function has side effects (file operations, network calls, etc.)
        has_side_effects = self._has_side_effects(func_node)

        # Determine if function needs a return statement
        needs_return = self._function_needs_return(func_node, has_return, has_only_logging, has_side_effects)

        # Track return values for usage analysis
        if has_return:
            return_values = self._extract_return_values(func_node)
            self.return_value_usage[f"{file_path}:{func_name}"] = {
                "line": func_line,
                "return_values": return_values,
                "is_pipeline": self._is_pipeline_function(func_node, file_path),
                "usage_count": 0,
            }

            # Mark as pipeline function if applicable
            if self._is_pipeline_function(func_node, file_path):
                self.pipeline_functions.add(f"{file_path}:{func_name}")

        if needs_return:
            self.issues.append(
                {
                    "file": file_path,
                    "line": func_line,
                    "function": func_name,
                    "type": "missing_return",
                    "reason": "Function has side effects or logging but no return statement",
                }
            )
        else:
            self.passing_functions.append(f"{file_path}:{func_name}")

    def _extract_return_values(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract return values from function."""
        return_values = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return_info = {
                    "line": node.lineno,
                    "value": ast.unparse(node.value) if node.value else None,
                    "type": type(node.value).__name__ if node.value else None,
                }
                return_values.append(return_info)

        return return_values

    def _is_pipeline_function(self, func_node: ast.FunctionDef, file_path: str) -> bool:
        """Check if function is part of a pipeline."""
        pipeline_indicators = [
            "pipeline",
            "process",
            "transform",
            "filter",
            "map",
            "reduce",
            "execute",
            "run",
            "step",
            "stage",
            "phase",
            "workflow",
        ]

        func_name = func_node.name.lower()
        file_path_lower = file_path.lower()

        # Check function name
        if any(indicator in func_name for indicator in pipeline_indicators):
            return True

        # Check file path
        if any(indicator in file_path_lower for indicator in ["pipeline", "workflow", "process"]):
            return True

        # Check function body for pipeline patterns
        func_str = ast.unparse(func_node)
        pipeline_patterns = [
            r"\.execute\(",
            r"\.run\(",
            r"\.process\(",
            r"\.transform\(",
            r"pipeline\.",
            r"workflow\.",
            r"step\.",
            r"stage\.",
        ]

        return any(re.search(pattern, func_str) for pattern in pipeline_patterns)

    def _analyze_return_usage(self, file_path: str):
        """Analyze how return values are used in a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            self._find_function_calls(file_path, tree, content)

        except Exception as e:
            logger.error(f"Error analyzing return usage in {file_path}: {e}")

    def _find_function_calls(self, file_path: str, tree: ast.AST, content: str):
        """Find function calls and track return value usage."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._analyze_function_call(file_path, node, content)

    def _analyze_function_call(self, file_path: str, call_node: ast.Call, content: str):
        """Analyze a function call for return value usage."""
        try:
            # Get function name
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                func_name = call_node.func.attr
            else:
                return

            # Check if this is a call to a tracked function
            for tracked_func, info in self.return_value_usage.items():
                if tracked_func.endswith(f":{func_name}"):
                    # Check if return value is used
                    parent = self._get_parent_node(call_node)
                    is_used = self._is_return_value_used(call_node, parent)

                    if not is_used:
                        self.unused_returns.append(
                            {
                                "file": file_path,
                                "line": call_node.lineno,
                                "function_call": func_name,
                                "tracked_function": tracked_func,
                                "return_values": info["return_values"],
                                "is_pipeline": info["is_pipeline"],
                            }
                        )
                    else:
                        info["usage_count"] += 1

        except Exception as e:
            logger.error(f"Error analyzing function call: {e}")

    def _get_parent_node(self, node: ast.AST) -> ast.AST:
        """Get parent node (simplified implementation)."""
        # This is a simplified implementation
        # In a full implementation, you'd need to track parent nodes during AST traversal
        return node

    def _is_return_value_used(self, call_node: ast.Call, parent: ast.AST) -> bool:
        """Check if return value from function call is used."""
        # Check if call is part of an assignment
        if isinstance(parent, ast.Assign):
            return True

        # Check if call is part of a return statement
        if isinstance(parent, ast.Return):
            return True

        # Check if call is part of a conditional
        if isinstance(parent, (ast.If, ast.While, ast.For)):
            return True

        # Check if call is part of another function call
        if isinstance(parent, ast.Call):
            return True

        # Check if call is part of a list/dict comprehension
        if isinstance(parent, (ast.ListComp, ast.DictComp, ast.SetComp)):
            return True

        # Check if call is part of a yield statement
        if isinstance(parent, ast.Yield):
            return True

        # Check if call is part of an expression statement (unused)
        if isinstance(parent, ast.Expr):
            return False

        return False

    def _analyze_pipeline_return_usage(self, python_files: List[str]):
        """Analyze return value usage specifically in pipeline functions."""
        pipeline_issues = []

        for func_key, info in self.return_value_usage.items():
            if info["is_pipeline"]:
                # Pipeline functions should have their return values used
                if info["usage_count"] == 0:
                    pipeline_issues.append(
                        {
                            "function": func_key,
                            "issue": "Pipeline function return value not used",
                            "return_values": info["return_values"],
                            "recommendation": "Consider using return values for pipeline continuation or remove unnecessary returns",
                        }
                    )

                # Check if pipeline function returns meaningful data
                meaningful_returns = self._check_meaningful_pipeline_returns(info["return_values"])
                if not meaningful_returns:
                    pipeline_issues.append(
                        {
                            "function": func_key,
                            "issue": "Pipeline function returns non-meaningful data",
                            "return_values": info["return_values"],
                            "recommendation": "Return meaningful data for pipeline continuation",
                        }
                    )

        self.issues.extend(pipeline_issues)

    def _check_meaningful_pipeline_returns(self, return_values: List[Dict[str, Any]]) -> bool:
        """Check if pipeline function returns meaningful data."""
        for return_info in return_values:
            value = return_info.get("value", "")

            # Check for meaningful return patterns
            meaningful_patterns = [
                r"result",
                r"data",
                r"output",
                r"processed",
                r"transformed",
                r"status",
                r"success",
                r"error",
                r"info",
                r"metadata",
            ]

            if any(re.search(pattern, value, re.IGNORECASE) for pattern in meaningful_patterns):
                return True

            # Check for data structures
            if any(keyword in value.lower() for keyword in ["dict", "list", "dataframe", "series", "array"]):
                return True

        return False

    def _has_return_statement(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has any return statements."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return True
        return False

    def _has_only_logging_statements(self, func_node: ast.FunctionDef, content: str) -> bool:
        """Check if function only contains logging/print statements."""
        lines = content.split("\n")
        func_start = func_node.lineno - 1
        func_end = func_node.end_lineno

        func_lines = lines[func_start:func_end]
        func_content = "\n".join(func_lines)

        # Check for logging patterns
        logging_patterns = [r"logger\.", r"print\(", r"st\.", r"logging\.", r"console\.", r"print\s*\("]

        has_logging = any(re.search(pattern, func_content) for pattern in logging_patterns)

        # Check if function has other meaningful operations
        has_other_ops = self._has_meaningful_operations(func_node)

        return has_logging and not has_other_ops

    def _has_meaningful_operations(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has meaningful operations beyond logging."""
        meaningful_nodes = [
            ast.Assign,
            ast.AugAssign,
            ast.Call,
            ast.If,
            ast.For,
            ast.While,
            ast.Try,
            ast.With,
            ast.Raise,
            ast.Return,
            ast.Yield,
        ]

        for node in ast.walk(func_node):
            if any(isinstance(node, node_type) for node_type in meaningful_nodes):
                # Skip logging calls
                if isinstance(node, ast.Call):
                    if self._is_logging_call(node):
                        continue
                return True

        return False

    def _is_logging_call(self, call_node: ast.Call) -> bool:
        """Check if a call node is a logging call."""
        if isinstance(call_node.func, ast.Attribute):
            attr_name = call_node.func.attr
            if attr_name in ["info", "warning", "error", "debug", "critical"]:
                return True
        elif isinstance(call_node.func, ast.Name):
            if call_node.func.id == "print":
                return True
        return False

    def _has_side_effects(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has side effects."""
        side_effect_patterns = [
            "open(",
            "write(",
            "read(",
            "save(",
            "load(",
            "requests.",
            "urllib.",
            "http.",
            "subprocess.",
            "os.system(",
            "redis.",
            "database.",
            "sqlite.",
            "st.",
            "streamlit.",
        ]

        func_str = ast.unparse(func_node)
        return any(pattern in func_str for pattern in side_effect_patterns)

    def _function_needs_return(
        self, func_node: ast.FunctionDef, has_return: bool, has_only_logging: bool, has_side_effects: bool
    ) -> bool:
        """Determine if a function needs a return statement."""
        # If it already has a return, it's fine
        if has_return:
            return False

        # If it has side effects, it should return a status
        if has_side_effects:
            return True

        # If it only has logging, it should return a status
        if has_only_logging:
            return True

        # Check if function name suggests it should return something
        func_name = func_node.name.lower()
        return_indicators = ["get", "fetch", "load", "read", "calculate", "compute", "process", "validate", "check"]

        return any(indicator in func_name for indicator in return_indicators)

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        total_functions = len(self.passing_functions) + len(self.issues)
        pipeline_functions_count = len(self.pipeline_functions)
        unused_returns_count = len(self.unused_returns)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_functions_audited": total_functions,
                "passing_functions": len(self.passing_functions),
                "functions_with_issues": len(self.issues),
                "pipeline_functions": pipeline_functions_count,
                "unused_return_values": unused_returns_count,
                "coverage_percentage": (len(self.passing_functions) / total_functions * 100)
                if total_functions > 0
                else 0,
            },
            "issues": self.issues,
            "unused_returns": self.unused_returns,
            "pipeline_analysis": {
                "pipeline_functions": list(self.pipeline_functions),
                "pipeline_issues": [issue for issue in self.issues if "pipeline" in str(issue)],
                "recommendations": self._generate_pipeline_recommendations(),
            },
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_pipeline_recommendations(self) -> List[str]:
        """Generate recommendations for pipeline functions."""
        recommendations = []

        # Check for unused pipeline returns
        unused_pipeline_returns = [ur for ur in self.unused_returns if ur["is_pipeline"]]
        if unused_pipeline_returns:
            recommendations.append(f"Found {len(unused_pipeline_returns)} pipeline functions with unused return values")
            recommendations.append(
                "Consider using return values for pipeline continuation or remove unnecessary returns"
            )

        # Check for pipeline functions without meaningful returns
        pipeline_functions_without_meaningful_returns = [
            func
            for func, info in self.return_value_usage.items()
            if info["is_pipeline"] and not self._check_meaningful_pipeline_returns(info["return_values"])
        ]

        if pipeline_functions_without_meaningful_returns:
            recommendations.append(
                f"Found {len(pipeline_functions_without_meaningful_returns)} pipeline functions without meaningful returns"
            )
            recommendations.append("Pipeline functions should return meaningful data for downstream processing")

        return recommendations

    def _generate_recommendations(self) -> List[str]:
        """Generate general recommendations."""
        recommendations = []

        if self.issues:
            recommendations.append(f"Found {len(self.issues)} functions with return statement issues")
            recommendations.append("Review and fix functions that need return statements")

        if self.unused_returns:
            recommendations.append(f"Found {len(self.unused_returns)} unused return values")
            recommendations.append("Consider removing unused return statements or using return values")

        recommendations.append("Ensure all pipeline functions return meaningful data")
        recommendations.append("Use return values for error handling and status reporting")

        return recommendations


def main():
    """Main function to run the audit."""
    auditor = ReturnStatementAuditor()
    report = auditor.audit_codebase()

    print("=== Return Statement Audit Report ===")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Total functions audited: {report['summary']['total_functions_audited']}")
    print(f"Passing functions: {report['summary']['passing_functions']}")
    print(f"Functions with issues: {report['summary']['functions_with_issues']}")
    print(f"Pipeline functions: {report['summary']['pipeline_functions']}")
    print(f"Unused return values: {report['summary']['unused_return_values']}")
    print(f"Coverage: {report['summary']['coverage_percentage']:.1f}%")

    if report["issues"]:
        print("\n=== Issues Found ===")
        for issue in report["issues"]:
            print(f"- {issue['file']}:{issue['line']} - {issue['function']} - {issue['type']}")

    if report["unused_returns"]:
        print("\n=== Unused Return Values ===")
        for unused in report["unused_returns"]:
            print(
                f"- {unused['file']}:{unused['line']} - {unused['function_call']} (from {unused['tracked_function']})"
            )

    if report["pipeline_analysis"]["recommendations"]:
        print("\n=== Pipeline Recommendations ===")
        for rec in report["pipeline_analysis"]["recommendations"]:
            print(f"- {rec}")

    print("\n=== General Recommendations ===")
    for rec in report["recommendations"]:
        print(f"- {rec}")

    return report


if __name__ == "__main__":
    main()
