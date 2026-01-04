"""
Comprehensive file usage analysis using multiple methods:
1. Static import tracing (improved)
2. Vulture (unused code detection)
3. Actual import testing
"""

import ast
import os
import sys
import importlib.util
from pathlib import Path
from typing import Set, Dict, List
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Entry points
ENTRY_POINTS = [
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "pages",
]

EXCLUDE_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".cache", "cache",
    "venv", "env", ".venv", ".git", "node_modules", "tests", "test_", "_test"
}

SOURCE_EXTENSIONS = {".py"}


def get_all_source_files() -> Set[Path]:
    """Get all source files in the project."""
    source_files = set()
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                if not any(exclude in file_path.parts for exclude in EXCLUDE_DIRS):
                    source_files.add(file_path)
    
    return source_files


def normalize_module_path(file_path: Path) -> str:
    """Convert file path to module path."""
    rel_path = file_path.relative_to(PROJECT_ROOT)
    parts = list(rel_path.parts)
    
    # Remove .py extension
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    
    # Handle __init__.py
    if parts[-1] == "__init__":
        parts = parts[:-1]
    
    return ".".join(parts)


def extract_all_imports(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    # Also add parent modules for better resolution
                    parts = node.module.split(".")
                    for i in range(1, len(parts)):
                        imports.add(".".join(parts[:i]))
    except Exception as e:
        # Silently skip files that can't be parsed
        pass
    
    return imports


def find_module_file(module_name: str, search_paths: List[Path], 
                     module_map: Dict[str, Path]) -> Path:
    """Find the file for a given module name."""
    # First, try direct lookup in module map (exact match)
    if module_name in module_map:
        return module_map[module_name]
    
    # Fall back to file system search (handles packages and submodules)
    parts = module_name.split(".")
    
    for search_path in search_paths:
        current = search_path
        for part in parts:
            # Try as directory with __init__.py
            init_file = current / part / "__init__.py"
            if init_file.exists():
                current = current / part
                continue
            
            # Try as .py file
            py_file = current / f"{part}.py"
            if py_file.exists():
                return py_file
            
            # Try as directory
            if (current / part).is_dir():
                current = current / part
                continue
            
            break
        else:
            # Found the module
            init_file = current / "__init__.py"
            if init_file.exists():
                return init_file
    
    return None


def trace_imports_comprehensive(start_file: Path, visited: Set[Path], 
                               all_source_files: Set[Path],
                               module_map: Dict[str, Path]) -> Set[Path]:
    """Comprehensively trace all imports recursively."""
    if start_file in visited or not start_file.exists():
        return set()
    
    visited.add(start_file)
    used_files = {start_file}
    
    # Get search paths - include parent directories and project root
    search_paths = [start_file.parent, PROJECT_ROOT]
    # Add parent's parent for better resolution
    if start_file.parent != PROJECT_ROOT:
        search_paths.append(start_file.parent.parent)
    
    # Extract imports
    imports = extract_all_imports(start_file)
    
    for imp in imports:
        # Skip standard library
        if imp.split(".")[0] in sys.stdlib_module_names:
            continue
        
        # Skip known third-party
        third_party = ["streamlit", "pandas", "numpy", "plotly", "sklearn", 
                      "torch", "tensorflow", "yfinance", "requests", "scipy",
                      "matplotlib", "seaborn", "xgboost", "lightgbm", "catboost",
                      "optuna", "ray", "redis", "sqlalchemy", "pymongo",
                      "flask", "fastapi", "django", "tornado", "aiohttp"]
        if imp.split(".")[0] in third_party:
            continue
        
        # Try to find the module using module_map first, then file system
        module_file = find_module_file(imp, search_paths, module_map)
        
        if module_file and module_file.exists() and module_file in all_source_files:
            # Recursively trace imports from this file
            used_files.update(
                trace_imports_comprehensive(module_file, visited, 
                                          all_source_files, module_map)
            )
    
    return used_files


def run_vulture_analysis() -> Dict[str, any]:
    """Run vulture to find unused code."""
    print("\n" + "=" * 80)
    print("Running Vulture Analysis...")
    print("=" * 80)
    
    try:
        # Run vulture on the project
        result = subprocess.run(
            ["vulture", str(PROJECT_ROOT), "--min-confidence", "80", 
             "--exclude", "tests", "--exclude", "__pycache__"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        
        # Parse vulture output
        unused_items = []
        for line in result.stdout.split("\n"):
            if line.strip() and ":" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    file_path = parts[0].strip()
                    unused_items.append(file_path)
        
        return {
            "success": True,
            "unused_items": unused_items,
            "output": result.stdout
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "Vulture not installed. Run: pip install vulture"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def build_module_map(all_source_files: Set[Path]) -> Dict[str, Path]:
    """Build a map of module names to file paths."""
    module_map = {}
    
    for file_path in all_source_files:
        module_name = normalize_module_path(file_path)
        if module_name:
            module_map[module_name] = file_path
    
    return module_map


def main():
    """Main analysis function."""
    print("=" * 80)
    print("COMPREHENSIVE FILE USAGE ANALYSIS")
    print("=" * 80)
    
    # Get all source files
    print("\n1. Scanning for source files...")
    all_source_files = get_all_source_files()
    print(f"   Found {len(all_source_files)} source files")
    
    # Build module map
    print("\n2. Building module map...")
    module_map = build_module_map(all_source_files)
    print(f"   Mapped {len(module_map)} modules")
    
    # Get entry points
    entry_files = []
    
    # 1. Add app.py as entry point
    app_file = PROJECT_ROOT / "app.py"
    if app_file.exists():
        entry_files.append(app_file)
        print(f"\n3. Entry point 1: app.py")
    
    # 2. Add ALL pages in /pages/ directory
    pages_dir = PROJECT_ROOT / "pages"
    if pages_dir.exists():
        page_files = sorted([f for f in pages_dir.glob("*.py") if f.name != "__init__.py"])
        entry_files.extend(page_files)
        print(f"   Entry points 2-{len(entry_files)}: {len(page_files)} page files in /pages/")
        for page_file in page_files:
            print(f"      - {page_file.name}")
    
    print(f"\n   Total entry points: {len(entry_files)}")
    
    # Trace imports recursively from all entry points
    print("\n4. Tracing imports comprehensively...")
    print("   This will trace from:")
    print("   1. app.py")
    print("   2. ALL pages in /pages/")
    print("   3. All resulting files recursively")
    print()
    
    visited = set()
    used_files = set()
    
    for entry_file in entry_files:
        if entry_file.exists():
            print(f"   Tracing from {entry_file.relative_to(PROJECT_ROOT)}...")
            used = trace_imports_comprehensive(
                entry_file, visited, all_source_files, module_map
            )
            used_files.update(used)
            print(f"      → Found {len(used)} files reachable (total so far: {len(used_files)})")
    
    print(f"\n5. Import tracing complete!")
    print(f"   Used files: {len(used_files)}")
    print(f"   Orphaned files: {len(all_source_files) - len(used_files)}")
    
    # Run vulture
    vulture_result = run_vulture_analysis()
    
    # Write results
    output_file = PROJECT_ROOT / "comprehensive_file_analysis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE FILE USAGE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total source files: {len(all_source_files)}\n")
        f.write(f"Used files (import tracing): {len(used_files)}\n")
        f.write(f"Orphaned files (import tracing): {len(all_source_files) - len(used_files)}\n\n")
        
        if vulture_result.get("success"):
            f.write(f"Vulture found {len(vulture_result.get('unused_items', []))} unused items\n\n")
            f.write("Vulture Output:\n")
            f.write(vulture_result.get("output", ""))
        else:
            f.write(f"Vulture analysis failed: {vulture_result.get('error', 'Unknown error')}\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("USED FILES (Import Tracing)\n")
        f.write("=" * 80 + "\n\n")
        for file in sorted(used_files, key=lambda x: str(x.relative_to(PROJECT_ROOT))):
            f.write(f"{file.relative_to(PROJECT_ROOT)}\n")
        
        orphaned_files = all_source_files - used_files
        f.write("\n" + "=" * 80 + "\n")
        f.write("ORPHANED FILES (Import Tracing)\n")
        f.write("=" * 80 + "\n\n")
        for file in sorted(orphaned_files, key=lambda x: str(x.relative_to(PROJECT_ROOT))):
            f.write(f"{file.relative_to(PROJECT_ROOT)}\n")
    
    print(f"\n6. Results written to: {output_file.relative_to(PROJECT_ROOT)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total source files: {len(all_source_files)}")
    print(f"Used files (import tracing): {len(used_files)}")
    print(f"Orphaned files (import tracing): {len(all_source_files) - len(used_files)}")
    if vulture_result.get("success"):
        print(f"Vulture unused items: {len(vulture_result.get('unused_items', []))}")


if __name__ == "__main__":
    main()

