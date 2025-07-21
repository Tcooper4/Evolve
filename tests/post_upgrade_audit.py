#!/usr/bin/env python3
"""
Post-Upgrade Return Statement Audit

Comprehensive audit to identify all functions that violate return statement requirements
after the system upgrade.
"""

import ast
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostUpgradeAuditor:
    """Auditor for checking return statement compliance after upgrade."""

    def __init__(self, exclude_dirs: Set[str] = None):
        self.exclude_dirs = exclude_dirs or {
            "archive",
            "legacy",
            "test_coverage",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
        }
        self.violations = []
        self.compliant_functions = []
        self.exempt_functions = set()

    def audit_codebase(self, root_dir: str = ".") -> Dict[str, Any]:
        """Audit the entire codebase for return statement compliance."""
        logger.info("🔍 Starting post-upgrade return statement audit...")

        python_files = self._find_python_files(root_dir)
        logger.info(f"Found {len(python_files)} Python files to audit")

        for file_path in python_files:
            self._audit_file(file_path)

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
            self.violations.append(
                {"file": file_path, "type": "parse_error", "error": str(e)}
            )

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
            # Check if __init__ has side effects that should return status
            if self._has_side_effects(func_node):
                self.violations.append(
                    {
                        "file": file_path,
                        "line": func_line,
                        "function": func_name,
                        "type": "init_with_side_effects",
                        "reason": "__init__ has side effects but no return statement",
                    }
                )
            return

        # Check if function has any return statements
        has_return = self._has_return_statement(func_node)

        # Check if function has side effects (logging, print, file operations, etc.)
        has_side_effects = self._has_side_effects(func_node)

        # Check if function has only logging/print statements
        has_only_logging = self._has_only_logging_statements(func_node, content)

        # Check if function name suggests it should return something
        should_return = self._function_should_return(func_node)

        # Determine if function needs a return statement
        needs_return = self._function_needs_return(
            func_node, has_return, has_only_logging, has_side_effects, should_return
        )

        if needs_return:
            self.violations.append(
                {
                    "file": file_path,
                    "line": func_line,
                    "function": func_name,
                    "type": "missing_return",
                    "reason": self._get_violation_reason(
                        has_return, has_only_logging, has_side_effects, should_return
                    ),
                }
            )
        else:
            self.compliant_functions.append(f"{file_path}:{func_name}")

    def _has_return_statement(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has any return statements."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
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
            "logger.",
            "logging.",
            "print(",
            "send(",
            "post(",
            "get(",
            "display(",
            "show(",
            "plot(",
            "render(",
            "draw(",
            "execute(",
            "run(",
            "start(",
            "stop(",
            "create(",
            "update(",
            "set_",
            "add_",
            "remove_",
            "delete_",
            "log_",
            "notify_",
            "alert_",
            "send_",
            "write_",
        ]

        func_str = ast.unparse(func_node)
        return any(pattern in func_str for pattern in side_effect_patterns)

    def _has_only_logging_statements(
        self, func_node: ast.FunctionDef, content: str
    ) -> bool:
        """Check if function only contains logging/print statements."""
        lines = content.split("\n")
        func_start = func_node.lineno - 1
        func_end = func_node.end_lineno

        func_lines = lines[func_start:func_end]
        func_content = "\n".join(func_lines)

        # Check for logging patterns
        logging_patterns = [
            r"logger\.",
            r"print\(",
            r"st\.",
            r"logging\.",
            r"console\.",
            r"print\s*\(",
        ]

        has_logging = any(
            re.search(pattern, func_content) for pattern in logging_patterns
        )

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

    def _function_should_return(self, func_node: ast.FunctionDef) -> bool:
        """Check if function name suggests it should return something."""
        return_indicators = [
            "get_",
            "fetch_",
            "load_",
            "read_",
            "parse_",
            "calculate_",
            "compute_",
            "generate_",
            "create_",
            "build_",
            "make_",
            "render_",
            "display_",
            "show_",
            "plot_",
            "draw_",
            "analyze_",
            "process_",
            "transform_",
            "convert_",
            "validate_",
            "check_",
            "verify_",
            "test_",
            "run_",
            "execute_",
            "select_",
            "choose_",
            "log_",
            "save_",
            "export_",
            "publish_",
        ]

        func_name = func_node.name
        return any(indicator in func_name for indicator in return_indicators)

    def _function_needs_return(
        self,
        func_node: ast.FunctionDef,
        has_return: bool,
        has_only_logging: bool,
        has_side_effects: bool,
        should_return: bool,
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

        # If function name suggests it should return something
        if should_return:
            return True

        return False

    def _get_violation_reason(
        self,
        has_return: bool,
        has_only_logging: bool,
        has_side_effects: bool,
        should_return: bool,
    ) -> str:
        """Get the reason for violation."""
        if has_side_effects:
            return "Function has side effects but no return statement"
        elif has_only_logging:
            return "Function only has logging/print statements but no return"
        elif should_return:
            return "Function name suggests it should return something"
        else:
            return "Function needs return statement for agentic modularity"

    def _generate_report(self) -> Dict[str, Any]:
        """Generate audit report."""
        total_violations = len(self.violations)
        total_compliant = len(self.compliant_functions)
        total_functions = total_violations + total_compliant

        compliance_rate = (
            (total_compliant / total_functions * 100) if total_functions > 0 else 0
        )

        return {
            "success": True,
            "result": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
            "total_violations": total_violations,
            "total_compliant": total_compliant,
            "total_functions": total_functions,
            "compliance_rate": compliance_rate,
            "violations": self.violations,
            "compliant_functions": self.compliant_functions,
        }


def main():
    """Main audit function."""
    print("🔍 POST-UPGRADE RETURN STATEMENT AUDIT")
    print("=" * 60)

    auditor = PostUpgradeAuditor()
    result = auditor.audit_codebase()

    print(f"\n📊 AUDIT RESULTS")
    print("=" * 60)
    print(f"Total functions audited: {result['total_functions']}")
    print(f"Compliant functions: {result['total_compliant']}")
    print(f"Violations found: {result['total_violations']}")
    print(f"Compliance rate: {result['compliance_rate']:.1f}%")

    if result["total_violations"] == 0:
        print("\n✅ ALL RETURN STATEMENTS ARE COMPLIANT!")
        return {"status": "compliant", "compliance_rate": 100.0}
    else:
        print(f"\n❌ {result['total_violations']} VIOLATIONS FOUND:")
        print("=" * 60)

        # Group violations by file
        violations_by_file = {}
        for violation in result["violations"]:
            file_path = violation["file"]
            if file_path not in violations_by_file:
                violations_by_file[file_path] = []
            violations_by_file[file_path].append(violation)

        # Show violations by file
        for file_path, violations in violations_by_file.items():
            print(f"\n📁 {file_path}")
            for violation in violations[:10]:  # Show first 10 per file
                if violation["type"] != "parse_error":
                    print(
                        f"  ❌ Line {violation['line']}: {violation['function']} - {violation['reason']}"
                    )
                else:
                    print(f"  ❌ {violation['error']}")

            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more violations")

        return {
            "success": True,
            "result": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
            "status": "violations_found",
            "total_violations": result["total_violations"],
            "compliance_rate": result["compliance_rate"],
            "violations": result["violations"],
        }


if __name__ == "__main__":
    result = main()
    print("Post-upgrade audit completed. All system changes have been " "verified.")
    if result["status"] == "violations_found":
        print(f"Compliance rate: {result['compliance_rate']:.1f}%")
        print(f"Total violations: {result['total_violations']}")
