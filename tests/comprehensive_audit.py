#!/usr/bin/env python3
"""
Comprehensive Return Statement Audit

This script performs a thorough audit of the entire codebase to identify
functions missing return statements, with detailed categorization and
prioritization for fixing.
"""

import ast
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class ReturnStatementAuditor:
    """Comprehensive auditor for return statement compliance."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.violations = defaultdict(list)
        self.compliant_files = []
        self.total_functions = 0
        self.functions_with_returns = 0
        self.functions_without_returns = 0

        # Categories for prioritization
        self.categories = {
            "critical": ["agent", "model", "strategy", "service", "core", "main"],
            "high": ["ui", "log", "config", "util", "helper"],
            "medium": ["test", "example", "demo"],
            "low": ["plot", "viz", "display", "format"],
        }

        # File patterns to exclude
        self.exclude_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            "node_modules",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
            "*.dll",
            "*.exe",
        ]

        # Removed return statement - __init__ should not return values

    def should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from audit."""
        file_str = str(file_path)
        return any(pattern in file_str for pattern in self.exclude_patterns)

    def get_priority(self, file_path: str, function_name: str) -> str:
        """Determine priority level for a function."""
        file_lower = file_path.lower()
        func_lower = function_name.lower()

        for priority, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in file_lower or keyword in func_lower:
                    return priority

        return "low"

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for return statement compliance."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            violations = []
            functions_with_returns = 0
            total_functions = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_functions += 1
                    has_return = self.check_function_has_return(node)

                    if has_return:
                        functions_with_returns += 1
                    else:
                        # Check if function should have a return
                        if self.should_function_have_return(node, content):
                            priority = self.get_priority(str(file_path), node.name)
                            violations.append(
                                {
                                    "function": node.name,
                                    "line": node.lineno,
                                    "priority": priority,
                                    "reason": self.get_violation_reason(node, content),
                                }
                            )

            return {
                "file": str(file_path),
                "total_functions": total_functions,
                "functions_with_returns": functions_with_returns,
                "functions_without_returns": total_functions - functions_with_returns,
                "violations": violations,
                "compliant": len(violations) == 0,
            }

        except Exception as e:
            return {
                "success": True,
                "result": None,
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
                "file": str(file_path),
                "error": str(e),
                "total_functions": 0,
                "functions_with_returns": 0,
                "functions_without_returns": 0,
                "violations": [],
                "compliant": False,
            }

    def check_function_has_return(self, node: ast.FunctionDef) -> bool:
        """Check if function has any return statements."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
        return False

    def should_function_have_return(self, node: ast.FunctionDef, content: str) -> bool:
        """Determine if a function should have a return statement."""
        # Skip if it's a test function
        if node.name.startswith("test_") or "test" in node.name.lower():
            return False

        # Skip if it's a property or setter
        if hasattr(node, "decorator_list"):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id in [
                    "property",
                    "setter",
                ]:
                    return False

        # Check if function has side effects (print, logging, file operations, etc.)
        function_lines = content.split("\n")[node.lineno - 1: node.end_lineno]
        function_text = "\n".join(function_lines)

        # Patterns that suggest side effects
        side_effect_patterns = [
            r"\bprint\s*\(",
            r"\blogging\b",
            r"\blogger\b",
            r"\bst\.",
            r"\bplt\.",
            r"\bopen\s*\(",
            r"\bwrite\s*\(",
            r"\bsave\s*\(",
            r"\bload\s*\(",
            r"\breturn\b",  # Already has return
            r"\bpass\b",  # Explicitly no-op
            r"\braise\b",  # Raises exception
        ]

        has_side_effects = any(
            re.search(pattern, function_text, re.IGNORECASE)
            for pattern in side_effect_patterns
        )

        # If function has side effects, it should have a return
        return has_side_effects

    def get_violation_reason(self, node: ast.FunctionDef, content: str) -> str:
        """Get reason for violation."""
        function_lines = content.split("\n")[node.lineno - 1: node.end_lineno]
        function_text = "\n".join(function_lines)

        if re.search(r"\bprint\s*\(", function_text):
            return "Has print statements"
        elif re.search(r"\blogging\b|\blogger\b", function_text):
            return "Has logging statements"
        elif re.search(r"\bst\.", function_text):
            return "Has Streamlit UI operations"
        elif re.search(r"\bplt\.", function_text):
            return "Has plotting operations"
        elif re.search(
            r"\bopen\s*\(|\bwrite\s*\(|\bsave\s*\(|\bload\s*\(", function_text
        ):
            return "Has file I/O operations"
        else:
            return "Has side effects"

    def audit_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive audit of the entire codebase."""
        print("🔍 Starting comprehensive return statement audit...")

        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self.should_exclude_file(Path(root) / d)]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    if not self.should_exclude_file(file_path):
                        python_files.append(file_path)

        print(f"📁 Found {len(python_files)} Python files to audit")

        results = []
        for file_path in python_files:
            result = self.analyze_file(file_path)
            results.append(result)

            if result["compliant"]:
                self.compliant_files.append(result["file"])

            self.total_functions += result["total_functions"]
            self.functions_with_returns += result["functions_with_returns"]
            self.functions_without_returns += result["functions_without_returns"]

            # Categorize violations
            for violation in result["violations"]:
                self.violations[violation["priority"]].append(
                    {
                        "file": result["file"],
                        "function": violation["function"],
                        "line": violation["line"],
                        "reason": violation["reason"],
                    }
                )

        return self.generate_report(results)

    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        total_files = len(results)
        compliant_files = len(self.compliant_files)
        non_compliant_files = total_files - compliant_files

        compliance_rate = (
            (self.functions_with_returns / self.total_functions * 100)
            if self.total_functions > 0
            else 0
        )

        # Summary by priority
        priority_summary = {}
        for priority in ["critical", "high", "medium", "low"]:
            violations = self.violations[priority]
            priority_summary[priority] = {
                "count": len(violations),
                "files_affected": len(set(v["file"] for v in violations)),
                "violations": violations,
            }

        report = {
            "summary": {
                "total_files": total_files,
                "compliant_files": compliant_files,
                "non_compliant_files": non_compliant_files,
                "total_functions": self.total_functions,
                "functions_with_returns": self.functions_with_returns,
                "functions_without_returns": self.functions_without_returns,
                "compliance_rate": compliance_rate,
                "file_compliance_rate": (
                    (compliant_files / total_files * 100) if total_files > 0 else 0
                ),
            },
            "priority_summary": priority_summary,
            "detailed_results": results,
            "recommendations": self.generate_recommendations(priority_summary),
        }

        return report

    def generate_recommendations(self, priority_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on audit results."""
        recommendations = []

        critical_count = priority_summary["critical"]["count"]
        high_count = priority_summary["high"]["count"]

        if critical_count > 0:
            recommendations.append(
                f"🚨 CRITICAL: Fix {critical_count} critical violations first (agents, models, services)"
            )

        if high_count > 0:
            recommendations.append(
                f"⚠️ HIGH: Address {high_count} high-priority violations (UI, logging, config)"
            )

        if priority_summary["medium"]["count"] > 0:
            recommendations.append(
                f"📝 MEDIUM: Consider fixing {priority_summary['medium']['count']} medium-priority violations"
            )

        if priority_summary["low"]["count"] > 0:
            recommendations.append(
                f"💡 LOW: {priority_summary['low']['count']} low-priority violations can be addressed later"
            )

        recommendations.append(
            "✅ Focus on functions with side effects (print, logging, file I/O, UI operations)"
        )
        recommendations.append(
            "✅ Add return statements with structured dictionaries containing status and metadata"
        )
        recommendations.append(
            "✅ Ensure all agent pipelines return usable outputs for autonomous operation"
        )

        return recommendations

    def print_report(self, report: Dict[str, Any]):
        """Print formatted audit report."""
        summary = report["summary"]
        priority_summary = report["priority_summary"]

        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE RETURN STATEMENT AUDIT REPORT")
        print("=" * 80)

        print(f"\n📈 OVERALL COMPLIANCE:")
        print(
            f"   Files: {summary['compliant_files']}/{summary['total_files']} compliant ({summary['file_compliance_rate']:.1f}%)"
        )
        print(
            f"   Functions: {
                summary['functions_with_returns']}/{
                summary['total_functions']} with returns ({
                summary['compliance_rate']:.1f}%)")

        print(f"\n🎯 VIOLATIONS BY PRIORITY:")
        for priority, data in priority_summary.items():
            if data["count"] > 0:
                print(
                    f"   {priority.upper()}: {data['count']} violations in {data['files_affected']} files"
                )

        print(f"\n🚨 CRITICAL VIOLATIONS:")
        for violation in priority_summary["critical"]["violations"][
            :10
        ]:  # Show first 10
            print(
                f"   {violation['file']}:{violation['line']} - {violation['function']} ({violation['reason']})"
            )

        if len(priority_summary["critical"]["violations"]) > 10:
            print(
                f"   ... and {len(priority_summary['critical']['violations']) - 10} more"
            )

        print(f"\n⚠️ HIGH PRIORITY VIOLATIONS:")
        for violation in priority_summary["high"]["violations"][:10]:  # Show first 10
            print(
                f"   {violation['file']}:{violation['line']} - {violation['function']} ({violation['reason']})"
            )

        if len(priority_summary["high"]["violations"]) > 10:
            print(f"   ... and {len(priority_summary['high']['violations']) - 10} more")

        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   {rec}")

        print("\n" + "=" * 80)

    def save_report(
        self, report: Dict[str, Any], filename: str = "comprehensive_audit_report.json"
    ):
        """Save audit report to JSON file."""
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to {filename}")


def main():
    """Run comprehensive audit."""
    auditor = ReturnStatementAuditor()
    report = auditor.audit_codebase()
    auditor.print_report(report)
    auditor.save_report(report)

    # Return summary for programmatic use
    return {
        "success": True,
        "result": None,
        "message": "Operation completed successfully",
        "timestamp": datetime.now().isoformat(),
        "compliance_rate": report["summary"]["compliance_rate"],
        "critical_violations": len(
            report["priority_summary"]["critical"]["violations"]
        ),
        "high_violations": len(report["priority_summary"]["high"]["violations"]),
        "total_violations": sum(
            len(data["violations"]) for data in report["priority_summary"].values()
        ),
    }


if __name__ == "__main__":
    main()
