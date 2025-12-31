#!/usr/bin/env python3
"""
Script to update import statements with try/except blocks for missing modules.

This script searches for imports of specific modules and wraps them in try/except blocks
with appropriate warning messages.
"""

import os
import re
import sys

# Modules to check for
MODULES_TO_CHECK = {
    "torch": {
        "import_patterns": [
            r"^import torch",
            r"^import torch\.",
            r"from torch import",
            r"from torch\.",
        ],
        "warning": "⚠️ PyTorch not available. Disabling deep learning models.",
        "fallback_vars": ["torch", "nn", "F", "optim"],
    },
    "transformers": {
        "import_patterns": [
            r"^import transformers",
            r"from transformers import",
            r"from transformers\.",
        ],
        "warning": "⚠️ HuggingFace transformers not available. Disabling NLP features.",
        "fallback_vars": [
            "transformers",
            "AutoTokenizer",
            "AutoModelForSequenceClassification",
            "pipeline",
        ],
    },
    "sklearn": {
        "import_patterns": [
            r"^import sklearn",
            r"from sklearn import",
            r"from sklearn\.",
        ],
        "warning": "⚠️ scikit-learn not available. Disabling machine learning preprocessing.",
        "fallback_vars": [
            "sklearn",
            "StandardScaler",
            "LinearRegression",
            "RandomForestRegressor",
        ],
    },
    "yfinance": {
        "import_patterns": [
            r"^import yfinance",
            r"import yfinance as",
            r"from yfinance import",
            r"from yfinance\.",
        ],
        "warning": "⚠️ yfinance not available. Disabling Yahoo Finance data provider.",
        "fallback_vars": ["yf", "yfinance"],
    },
    "sentence_transformers": {
        "import_patterns": [
            r"^import sentence_transformers",
            r"from sentence_transformers import",
            r"from sentence_transformers\.",
        ],
        "warning": "⚠️ sentence-transformers not available. Disabling text embeddings.",
        "fallback_vars": ["SentenceTransformer", "sentence_transformers"],
    },
    "vaderSentiment": {
        "import_patterns": [
            r"^import vaderSentiment",
            r"from vaderSentiment import",
            r"from vaderSentiment\.",
        ],
        "warning": "⚠️ vaderSentiment not available. Disabling VADER sentiment analysis.",
        "fallback_vars": ["SentimentIntensityAnalyzer", "vaderSentiment"],
    },
    "alpaca": {
        "import_patterns": [r"^import alpaca", r"from alpaca import", r"from alpaca\."],
        "warning": "⚠️ alpaca-py not available. Disabling Alpaca trading integration.",
        "fallback_vars": ["alpaca", "TradingClient", "StockHistoricalDataClient"],
    },
}


def find_python_files(directory: str) -> list:
    """Find all Python files in the directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d not in ["__pycache__", "venv", ".venv", "node_modules"]
        ]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def check_file_for_imports(file_path: str) -> dict:
    """Check a file for imports of the specified modules."""
    results = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        for module_name, module_info in MODULES_TO_CHECK.items():
            module_results = []

            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                for pattern in module_info["import_patterns"]:
                    if re.match(pattern, line):
                        module_results.append(
                            {
                                "line_number": i + 1,
                                "line_content": line,
                                "pattern": pattern,
                            }
                        )
                        break

            if module_results:
                results[module_name] = {
                    "file": file_path,
                    "imports": module_results,
                    "module_info": module_info,
                }

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return results


def update_file_imports(file_path: str, module_results: dict) -> bool:
    """Update a file to wrap imports in try/except blocks."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Sort imports by line number (descending) to avoid line number shifts
        all_imports = []
        for module_name, result in module_results.items():
            for imp in result["imports"]:
                all_imports.append(
                    (imp["line_number"], module_name, imp, result["module_info"])
                )

        all_imports.sort(key=lambda x: x[0], reverse=True)

        # Track which modules we've already added try/except blocks for
        processed_modules = set()

        for line_num, module_name, imp, module_info in all_imports:
            if module_name in processed_modules:
                continue

            # Check if there's already a try/except block for this module
            if line_num > 1:
                prev_line = lines[line_num - 2].strip()
                if prev_line.startswith("try:") or prev_line.startswith(
                    "except ImportError"
                ):
                    continue

            # Find all imports for this module
            module_imports = []
            for (
                other_line_num,
                other_module_name,
                other_imp,
                other_module_info,
            ) in all_imports:
                if other_module_name == module_name:
                    module_imports.append((other_line_num, other_imp))

            # Sort by line number (ascending)
            module_imports.sort(key=lambda x: x[0])

            # Create the try/except block
            try_block = []
            try_block.append(f"# Try to import {module_name}")
            try_block.append("try:")

            # Add all imports for this module
            for _, imp_info in module_imports:
                try_block.append(f"    {imp_info['line_content']}")

            # Add success flag
            try_block.append(
                f"    {module_name.upper().replace('-', '_')}_AVAILABLE = True"
            )
            try_block.append(f"except ImportError as e:")
            try_block.append(f"    print(\"{module_info['warning']}\")")
            try_block.append(f'    print(f"   Missing: {{e}}")')

            # Add fallback variable assignments
            for var in module_info["fallback_vars"]:
                try_block.append(f"    {var} = None")

            try_block.append(
                f"    {module_name.upper().replace('-', '_')}_AVAILABLE = False"
            )
            try_block.append("")

            # Insert the try/except block before the first import
            first_line = min(imp_info[0] for imp_info in module_imports)
            lines.insert(first_line - 1, "\n".join(try_block))

            # Remove the original import lines (in reverse order to maintain indices)
            for line_num, _ in sorted(module_imports, key=lambda x: x[0], reverse=True):
                if line_num < len(lines):
                    lines.pop(line_num - 1)

            processed_modules.add(module_name)

        # Write the updated content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return True

    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def main():
    """Main function to update all Python files."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."

    print(f"Searching for Python files in {directory}...")
    python_files = find_python_files(directory)
    print(f"Found {len(python_files)} Python files")

    all_results = {}

    # Check all files for imports
    for file_path in python_files:
        results = check_file_for_imports(file_path)
        if results:
            all_results[file_path] = results

    print(f"\nFound imports in {len(all_results)} files:")

    # Display results
    for file_path, results in all_results.items():
        print(f"\n{file_path}:")
        for module_name, result in results.items():
            print(f"  - {module_name}: {len(result['imports'])} imports")

    # Ask for confirmation
    if all_results:
        response = input(f"\nUpdate {len(all_results)} files? (y/N): ")
        if response.lower() == "y":
            updated_count = 0
            for file_path, results in all_results.items():
                print(f"Updating {file_path}...")
                if update_file_imports(file_path, results):
                    updated_count += 1
                    print(f"  ✅ Updated")
                else:
                    print(f"  ❌ Failed")

            print(f"\nUpdated {updated_count}/{len(all_results)} files successfully")
        else:
            print("Update cancelled")
    else:
        print("No files need updating")


if __name__ == "__main__":
    main()
