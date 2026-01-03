#!/usr/bin/env python3
"""
Helper script to find remaining division bugs in the codebase.

This script searches for common unsafe division patterns that should be
replaced with safe_math utilities.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Patterns to search for
PATTERNS = {
    "RSI calculations": [
        r"rs\s*=\s*.*gain.*/\s*.*loss",
        r"rs\s*=\s*.*avg_gain.*/\s*.*avg_loss",
        r"rsi\s*=\s*100\s*-\s*\(100\s*/\s*\(1\s*\+\s*rs\)\)",
    ],
    "Returns calculations": [
        r"np\.diff\(.*\)\s*/\s*.*\[",
        r"returns\s*=\s*.*diff.*/\s*.*\[",
        r"returns\s*=\s*\(.*\s*-\s*.*\.shift\(\)\)\s*/\s*.*\.shift\(\)",
    ],
    "Drawdown calculations": [
        r"drawdown\s*=\s*\(.*\s*-\s*.*max.*\)\s*/\s*.*max",
        r"drawdown\s*=\s*\(.*cumulative.*\s*-\s*.*\)\s*/\s*.*",
    ],
    "Sharpe ratio": [
        r"sharpe\s*=\s*.*mean.*/\s*.*std",
        r"sharpe_ratio\s*=\s*.*mean.*/\s*.*std",
    ],
    "Sortino ratio": [
        r"sortino\s*=\s*.*mean.*/\s*.*downside",
    ],
    "MAPE": [
        r"mape\s*=\s*.*mean.*abs.*\(.*\s*-\s*.*\)\s*/\s*.*\)",
    ],
    "Normalization": [
        r"normalized\s*=\s*\(.*\s*-\s*.*\.min\(\)\)\s*/\s*\(.*\.max\(\)\s*-\s*.*\.min\(\)\)",
        r"z_scores\s*=\s*\(.*\s*-\s*.*\.mean\(\)\)\s*/\s*.*\.std\(\)",
    ],
    "Price momentum": [
        r"momentum\s*=\s*.*\s*/\s*.*\.shift\(.*\)\s*-\s*1",
        r"ratio\s*=\s*.*price\s*/\s*.*reference",
    ],
    "Bollinger position": [
        r"position\s*=\s*\(.*\s*-\s*.*lower.*\)\s*/\s*\(.*upper.*\s*-\s*.*lower.*\)",
    ],
    "General divisions": [
        r"[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[a-zA-Z_][a-zA-Z0-9_]*\s*/\s*[a-zA-Z_][a-zA-Z0-9_]*",
    ],
}


def find_patterns_in_file(filepath: Path, patterns: List[str]) -> List[Tuple[int, str, str]]:
    """Find patterns in a file and return matches with line numbers."""
    matches = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                # Skip comments and safe_math imports
                if line.strip().startswith("#") or "safe_math" in line:
                    continue
                # Skip safe_divide, safe_rsi, etc. calls
                if any(safe_func in line for safe_func in ["safe_divide", "safe_rsi", "safe_returns", 
                                                           "safe_drawdown", "safe_sharpe", "safe_sortino",
                                                           "safe_mape", "safe_normalize", "safe_kelly",
                                                           "safe_bollinger", "safe_price_momentum", "safe_calmar"]):
                    continue
                
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append((line_num, pattern, line.strip()))
                        break  # Only report once per line
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return matches


def scan_directory(directory: Path, exclude_dirs: List[str] = None) -> Dict[str, List[Tuple[Path, List[Tuple[int, str, str]]]]]:
    """Scan directory for division bugs."""
    if exclude_dirs is None:
        exclude_dirs = ["__pycache__", ".git", "node_modules", ".pytest_cache", "venv", "env"]
    
    results = {category: [] for category in PATTERNS.keys()}
    
    for root, dirs, files in os.walk(directory):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if not file.endswith(".py"):
                continue
            
            filepath = Path(root) / file
            
            # Skip safe_math.py itself
            if "safe_math.py" in str(filepath):
                continue
            
            for category, patterns in PATTERNS.items():
                matches = find_patterns_in_file(filepath, patterns)
                if matches:
                    results[category].append((filepath, matches))
    
    return results


def print_results(results: Dict[str, List[Tuple[Path, List[Tuple[int, str, str]]]]]):
    """Print scan results in a readable format."""
    total_files = 0
    total_matches = 0
    
    for category, file_matches in results.items():
        if not file_matches:
            continue
        
        print(f"\n{'='*80}")
        print(f"{category.upper()}")
        print(f"{'='*80}")
        
        category_files = len(file_matches)
        category_matches = sum(len(matches) for _, matches in file_matches)
        total_files += category_files
        total_matches += category_matches
        
        print(f"Found {category_matches} potential issues in {category_files} files:\n")
        
        for filepath, matches in file_matches:
            print(f"  {filepath}")
            for line_num, pattern, line in matches[:5]:  # Show first 5 matches per file
                print(f"    Line {line_num}: {line[:80]}")
            if len(matches) > 5:
                print(f"    ... and {len(matches) - 5} more matches")
            print()
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {total_matches} potential division bugs in {total_files} files")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    # Default to trading directory
    trading_dir = Path(__file__).parent.parent / "trading"
    
    if not trading_dir.exists():
        print(f"Error: {trading_dir} does not exist")
        return
    
    print("Scanning for division bugs...")
    print(f"Directory: {trading_dir}\n")
    
    results = scan_directory(trading_dir)
    print_results(results)
    
    # Save results to file
    output_file = Path(__file__).parent.parent / "division_bugs_report.txt"
    with open(output_file, "w") as f:
        f.write("Division Bugs Report\n")
        f.write("=" * 80 + "\n\n")
        
        for category, file_matches in results.items():
            if not file_matches:
                continue
            
            f.write(f"{category.upper()}\n")
            f.write("-" * 80 + "\n")
            
            for filepath, matches in file_matches:
                f.write(f"\n{filepath}\n")
                for line_num, pattern, line in matches:
                    f.write(f"  Line {line_num}: {line}\n")
            f.write("\n")
    
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    main()

