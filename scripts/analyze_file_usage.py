"""
Analyze which files are actually used by the Streamlit application.

Traces imports from app.py and all pages/ files to find:
1. All files that are imported/used
2. All orphaned files (not imported anywhere)
"""

import ast
import os
import sys
from pathlib import Path
from typing import Set, Dict, List
import importlib.util

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Entry points for the Streamlit app
ENTRY_POINTS = [
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "pages",
]

# Directories to exclude from analysis
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".cache",
    "cache",
    "venv",
    "env",
    ".venv",
    ".git",
    "node_modules",
    "tests",  # Test files are not part of production
    "test_",
    "_test",
}

# File extensions to analyze
SOURCE_EXTENSIONS = {".py"}


def is_excluded(path: Path) -> bool:
    """Check if a path should be excluded from analysis."""
    parts = path.parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith(".") and part != ".py":
            return True
    return False


def get_all_source_files() -> Set[Path]:
    """Get all source files in the project."""
    source_files = set()
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                if not is_excluded(file_path):
                    source_files.add(file_path)
    
    return source_files


def extract_imports(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return imports


def resolve_import(import_name: str, current_file: Path) -> Path:
    """Resolve an import name to a file path."""
    # Handle relative imports
    if import_name.startswith("."):
        # Relative import - resolve relative to current file
        parts = import_name.split(".")
        depth = len([p for p in parts if p == ""])
        base = current_file.parent
        for _ in range(depth - 1):
            base = base.parent
        module_name = parts[-1] if parts[-1] else current_file.stem
        return base / f"{module_name}.py"
    
    # Absolute import
    # Try to find in project
    project_parts = PROJECT_ROOT.parts
    current_parts = current_file.parts
    
    # Find common path
    common_len = 0
    for i, (p, c) in enumerate(zip(project_parts, current_parts)):
        if p == c:
            common_len = i + 1
        else:
            break
    
    # Try different resolution strategies
    import_parts = import_name.split(".")
    
    # Strategy 1: Direct file in same directory or parent
    for parent in [current_file.parent] + list(current_file.parents):
        if parent == PROJECT_ROOT or PROJECT_ROOT in parent.parents:
            test_path = parent / f"{import_parts[0]}.py"
            if test_path.exists():
                return test_path
            test_path = parent / import_parts[0] / "__init__.py"
            if test_path.exists():
                return test_path
            test_path = parent / import_parts[0] / f"{import_parts[-1]}.py"
            if test_path.exists():
                return test_path
    
    # Strategy 2: Search from project root
    search_path = PROJECT_ROOT
    for part in import_parts:
        test_path = search_path / part
        if test_path.is_dir():
            init_file = test_path / "__init__.py"
            if init_file.exists():
                search_path = test_path
                continue
            py_file = test_path.with_suffix(".py")
            if py_file.exists():
                return py_file
        else:
            py_file = test_path.with_suffix(".py")
            if py_file.exists():
                return py_file
            break
    
    return None


def trace_imports(start_file: Path, visited: Set[Path], all_source_files: Set[Path]) -> Set[Path]:
    """Recursively trace all imports from a starting file."""
    if start_file in visited or not start_file.exists():
        return set()
    
    visited.add(start_file)
    used_files = {start_file}
    
    # Extract imports
    imports = extract_imports(start_file)
    
    # Resolve each import
    for imp in imports:
        # Skip standard library and third-party imports
        if imp in sys.stdlib_module_names or imp in ["streamlit", "pandas", "numpy", "plotly"]:
            continue
        
        resolved = resolve_import(imp, start_file)
        if resolved and resolved in all_source_files:
            # Recursively trace this file
            used_files.update(trace_imports(resolved, visited, all_source_files))
    
    return used_files


def main():
    """Main analysis function."""
    print("=" * 80)
    print("File Usage Analysis for Streamlit Application")
    print("=" * 80)
    
    # Get all source files
    print("\n1. Scanning for source files...")
    all_source_files = get_all_source_files()
    print(f"   Found {len(all_source_files)} source files")
    
    # Get entry points
    entry_files = []
    if (PROJECT_ROOT / "app.py").exists():
        entry_files.append(PROJECT_ROOT / "app.py")
    
    pages_dir = PROJECT_ROOT / "pages"
    if pages_dir.exists():
        for page_file in pages_dir.glob("*.py"):
            if page_file.name != "__init__.py":
                entry_files.append(page_file)
    
    print(f"\n2. Found {len(entry_files)} entry points:")
    for ef in entry_files:
        print(f"   - {ef.relative_to(PROJECT_ROOT)}")
    
    # Trace imports from entry points
    print("\n3. Tracing imports...")
    visited = set()
    used_files = set()
    
    for entry_file in entry_files:
        if entry_file.exists():
            print(f"   Tracing from {entry_file.name}...")
            used = trace_imports(entry_file, visited, all_source_files)
            used_files.update(used)
            print(f"   Found {len(used)} files reachable from {entry_file.name}")
    
    print(f"\n4. Analysis complete!")
    print(f"   Total source files: {len(all_source_files)}")
    print(f"   Used files: {len(used_files)}")
    print(f"   Orphaned files: {len(all_source_files) - len(used_files)}")
    
    # Find orphaned files
    orphaned_files = all_source_files - used_files
    
    # Write results
    output_file = PROJECT_ROOT / "file_usage_analysis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FILE USAGE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total source files: {len(all_source_files)}\n")
        f.write(f"Used files: {len(used_files)}\n")
        f.write(f"Orphaned files: {len(orphaned_files)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("USED FILES\n")
        f.write("=" * 80 + "\n\n")
        for file in sorted(used_files, key=lambda x: str(x.relative_to(PROJECT_ROOT))):
            f.write(f"{file.relative_to(PROJECT_ROOT)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("ORPHANED FILES\n")
        f.write("=" * 80 + "\n\n")
        for file in sorted(orphaned_files, key=lambda x: str(x.relative_to(PROJECT_ROOT))):
            f.write(f"{file.relative_to(PROJECT_ROOT)}\n")
    
    print(f"\n5. Results written to: {output_file.relative_to(PROJECT_ROOT)}")
    
    # Print summary by directory
    print("\n6. Orphaned files by directory:")
    orphaned_by_dir = {}
    for file in orphaned_files:
        rel_path = file.relative_to(PROJECT_ROOT)
        dir_name = str(rel_path.parent)
        if dir_name not in orphaned_by_dir:
            orphaned_by_dir[dir_name] = []
        orphaned_by_dir[dir_name].append(rel_path.name)
    
    for dir_name in sorted(orphaned_by_dir.keys()):
        print(f"   {dir_name}: {len(orphaned_by_dir[dir_name])} files")


if __name__ == "__main__":
    main()

