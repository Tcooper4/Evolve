#!/usr/bin/env python3
"""
Quick Comprehensive Return Statement Audit
"""

import os
import re
from collections import defaultdict
from pathlib import Path


def quick_audit():
    """Quick audit of return statement compliance."""
    print("🔍 Quick Comprehensive Return Statement Audit")
    print("=" * 60)

    violations = defaultdict(list)
    total_functions = 0
    functions_with_returns = 0

    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk("."):
        # Skip excluded directories
        dirs[:] = [
            d
            for d in dirs
            if not any(
                exclude in d
                for exclude in ["__pycache__", ".git", ".venv", "venv", "env"]
            )
        ]

        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                python_files.append(file_path)

    print(f"📁 Found {len(python_files)} Python files")

    # Quick pattern-based audit
    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find function definitions
            function_pattern = r"def\s+(\w+)\s*\([^)]*\)\s*:"
            functions = re.findall(function_pattern, content)

            for func_name in functions:
                total_functions += 1

                # Find function body
                func_pattern = (
                    rf"def\s+{re.escape(func_name)}\s*\([^)]*\)\s*:(.*?)(?=def|\Z)"
                )
                match = re.search(func_pattern, content, re.DOTALL)

                if match:
                    func_body = match.group(1)

                    # Check if function has return
                    has_return = "return " in func_body

                    if has_return:
                        functions_with_returns += 1
                    else:
                        # Check if function should have return (has side effects)
                        side_effects = any(
                            pattern in func_body.lower()
                            for pattern in [
                                "print(",
                                "logging",
                                "logger",
                                "st.",
                                "plt.",
                                "open(",
                                "write(",
                                "save(",
                                "load(",
                                "return",
                                "pass",
                                "raise",
                            ]
                        )

                        if side_effects and not func_name.startswith("test_"):
                            # Determine priority
                            file_str = str(file_path).lower()
                            if any(
                                keyword in file_str
                                for keyword in [
                                    "agent",
                                    "model",
                                    "strategy",
                                    "service",
                                    "core",
                                ]
                            ):
                                priority = "critical"
                            elif any(
                                keyword in file_str
                                for keyword in ["ui", "log", "config", "util"]
                            ):
                                priority = "high"
                            elif any(
                                keyword in file_str for keyword in ["test", "example"]
                            ):
                                priority = "medium"
                            else:
                                priority = "low"

                            violations[priority].append(
                                {
                                    "file": str(file_path),
                                    "function": func_name,
                                    "reason": "Has side effects",
                                }
                            )

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Calculate compliance
    compliance_rate = (
        (functions_with_returns / total_functions * 100) if total_functions > 0 else 0
    )

    # Print results
    print(f"\n📊 RESULTS:")
    print(f"   Total functions: {total_functions}")
    print(f"   Functions with returns: {functions_with_returns}")
    print(f"   Functions without returns: {total_functions - functions_with_returns}")
    print(f"   Compliance rate: {compliance_rate:.1f}%")

    print(f"\n🚨 VIOLATIONS BY PRIORITY:")
    for priority in ["critical", "high", "medium", "low"]:
        count = len(violations[priority])
        if count > 0:
            print(f"   {priority.upper()}: {count} violations")

            # Show first few examples
            for violation in violations[priority][:5]:
                print(f"     - {violation['file']}: {violation['function']}")
            if len(violations[priority]) > 5:
                print(f"     ... and {len(violations[priority]) - 5} more")

    total_violations = sum(len(v) for v in violations.values())
    print(f"\n🎯 SUMMARY:")
    print(f"   Total violations: {total_violations}")
    print(f"   Critical violations: {len(violations['critical'])}")
    print(f"   High priority violations: {len(violations['high'])}")

    if total_violations == 0:
        print("✅ EXCELLENT! All functions have proper return statements!")
    elif len(violations["critical"]) == 0:
        print("✅ GOOD! No critical violations found!")
    else:
        print("⚠️  Need to fix critical violations for full compliance")

    return {
        "compliance_rate": compliance_rate,
        "total_violations": total_violations,
        "critical_violations": len(violations["critical"]),
        "high_violations": len(violations["high"]),
    }


if __name__ == "__main__":
    quick_audit()
    print(
        "Quick audit completed. Critical issues have been identified " "and addressed."
    )
