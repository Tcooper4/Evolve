"""
Automated State Inventory Analysis Script

This script analyzes Python files to extract:
1. All classes and their mutable attributes
2. Scope classification (run-scoped, session-scoped, global)
3. Special patterns (singletons, file-backed state, caches)

Usage:
    python scripts/analyze_state_inventory.py [--path PATH] [--output OUTPUT]
"""

import ast
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compatibility: ast.unparse was added in Python 3.9
if sys.version_info >= (3, 9):
    def ast_unparse(node):
        try:
            return ast.unparse(node)
        except Exception:
            return _fallback_unparse(node)
else:
    def ast_unparse(node):
        return _fallback_unparse(node)

def _fallback_unparse(node):
    """Fallback AST unparsing for older Python versions or complex nodes."""
    try:
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_str = _fallback_unparse(node.value) if hasattr(node, 'value') else '?'
            attr_str = node.attr if hasattr(node, 'attr') else '?'
            return f"{value_str}.{attr_str}"
        elif isinstance(node, ast.Str):  # Python < 3.8
            return repr(node.s)
        elif isinstance(node, ast.Num):  # Python < 3.8
            return repr(node.n)
        elif isinstance(node, ast.NameConstant):  # Python < 3.8
            return repr(node.value)
        else:
            return f"<{type(node).__name__}>"
    except Exception:
        return f"<{type(node).__name__}>"


@dataclass
class AttributeInfo:
    """Information about a class attribute."""
    name: str
    scope: str  # "run-scoped", "session-scoped", "global", "unknown"
    type_hint: Optional[str] = None
    initial_value: Optional[str] = None
    is_file_path: bool = False
    is_cache: bool = False
    is_lock: bool = False
    is_queue: bool = False
    is_logger: bool = False
    is_config: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    file_path: str
    base_classes: List[str] = field(default_factory=list)
    attributes: List[AttributeInfo] = field(default_factory=list)
    is_singleton: bool = False
    has_file_backed_state: bool = False
    has_cache: bool = False
    singleton_pattern: Optional[str] = None
    file_paths: List[str] = field(default_factory=list)
    cache_attributes: List[str] = field(default_factory=list)


class StateAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze class state."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.classes: Dict[str, ClassInfo] = {}
        self.current_class: Optional[str] = None
        self.global_instances: Set[str] = set()  # Track global singleton instances
        self.file_operations: Set[str] = set()  # Track file operations
        
    def visit_Module(self, node: ast.Module):
        # First pass: find global singleton instances
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(item.value, ast.Call):
                            if isinstance(item.value.func, ast.Name):
                                # Pattern: _instance = ClassName()
                                if target.id.startswith('_') or target.id.islower():
                                    self.global_instances.add(target.id)
        
        # Second pass: analyze classes
        self.generic_visit(node)
        
    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name
        self.current_class = class_name
        
        class_info = ClassInfo(
            name=class_name,
            file_path=str(self.file_path),
            base_classes=[base.id if isinstance(base, ast.Name) else ast_unparse(base) 
                         for base in node.bases]
        )
        
        # Add class to dictionary BEFORE visiting so __init__ can access it
        self.classes[class_name] = class_info
        
        # Analyze class body
        self.generic_visit(node)
        
        # Check for singleton pattern
        class_info.is_singleton = self._check_singleton_pattern(node)
        
        # Check for file-backed state
        class_info.has_file_backed_state = len(class_info.file_paths) > 0
        
        # Check for cache
        class_info.has_cache = len(class_info.cache_attributes) > 0
        
        self.current_class = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "__init__":
            if self.current_class:
                # Analyze __init__ method for attributes
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        self._analyze_assignment(stmt)
                    elif isinstance(stmt, ast.AnnAssign):
                        self._analyze_ann_assignment(stmt)
        
        self.generic_visit(node)
    
    def _analyze_assignment(self, stmt: ast.Assign):
        """Analyze an assignment statement in __init__."""
        if not self.current_class:
            return
            
        for target in stmt.targets:
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id == "self":
                    attr_name = target.attr
                    attr_info = self._create_attribute_info(attr_name, stmt.value)
                    
                    if attr_info not in self.classes[self.current_class].attributes:
                        self.classes[self.current_class].attributes.append(attr_info)
    
    def _analyze_ann_assignment(self, stmt: ast.AnnAssign):
        """Analyze an annotated assignment (type hints)."""
        if not self.current_class:
            return
        
        try:
            if isinstance(stmt.target, ast.Attribute):
                if isinstance(stmt.target.value, ast.Name) and stmt.target.value.id == "self":
                    attr_name = stmt.target.attr
                    type_hint = None
                    if stmt.annotation:
                        try:
                            type_hint = ast_unparse(stmt.annotation)
                        except Exception:
                            type_hint = str(stmt.annotation)
                    
                    attr_info = AttributeInfo(
                        name=attr_name,
                        scope="unknown",
                        type_hint=type_hint
                    )
                    
                    if stmt.value:
                        attr_info = self._analyze_value(attr_info, stmt.value)
                    
                    if attr_info not in self.classes[self.current_class].attributes:
                        self.classes[self.current_class].attributes.append(attr_info)
        except Exception as e:
            logger.debug(f"Error analyzing annotated assignment: {e}")
    
    def _create_attribute_info(self, name: str, value: ast.AST) -> AttributeInfo:
        """Create AttributeInfo from attribute name and value."""
        attr_info = AttributeInfo(name=name, scope="unknown")
        
        # Analyze the value
        attr_info = self._analyze_value(attr_info, value)
        
        # Classify scope based on patterns
        attr_info.scope = self._classify_scope(name, value)
        
        return attr_info
    
    def _analyze_value(self, attr_info: AttributeInfo, value: ast.AST) -> AttributeInfo:
        """Analyze the value to detect patterns."""
        try:
            value_str = ast_unparse(value)
        except Exception:
            value_str = str(value)
        attr_info.initial_value = value_str
        
        # Check for file paths
        if self._is_file_path(value, value_str):
            attr_info.is_file_path = True
            attr_info.scope = "global"
            if self.current_class:
                self.classes[self.current_class].file_paths.append(attr_info.name)
        
        # Check for caches
        if self._is_cache(attr_info.name, value, value_str):
            attr_info.is_cache = True
            if self.current_class:
                self.classes[self.current_class].cache_attributes.append(attr_info.name)
        
        # Check for locks
        if self._is_lock(value, value_str):
            attr_info.is_lock = True
            attr_info.scope = "global"
        
        # Check for queues
        if self._is_queue(value, value_str):
            attr_info.is_queue = True
            attr_info.scope = "session-scoped"
        
        # Check for logger
        if self._is_logger(value, value_str):
            attr_info.is_logger = True
            attr_info.scope = "global"
        
        # Check for config
        if self._is_config(attr_info.name, value, value_str):
            attr_info.is_config = True
        
        return attr_info
    
    def _is_file_path(self, value: ast.AST, value_str: str) -> bool:
        """Check if value is a file path."""
        patterns = [
            r'Path\(["\']',
            r'["\'][^"\']*\.(json|pkl|pickle|csv|yaml|yml|txt|log)',
            r'os\.path\.join',
            r'pathlib\.Path',
        ]
        
        for pattern in patterns:
            if re.search(pattern, value_str, re.IGNORECASE):
                return True
        
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                if value.func.id == "Path":
                    return True
        
        return False
    
    def _is_cache(self, name: str, value: ast.AST, value_str: str) -> bool:
        """Check if attribute is a cache."""
        cache_keywords = ['cache', 'Cache', 'memory', 'Memory', 'store', 'Store']
        
        if any(keyword in name.lower() for keyword in cache_keywords):
            return True
        
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                if 'Cache' in value.func.id or 'cache' in value.func.id.lower():
                    return True
        
        if isinstance(value, ast.Dict):
            return True
        
        return False
    
    def _is_lock(self, value: ast.AST, value_str: str) -> bool:
        """Check if value is a lock."""
        lock_patterns = [
            r'threading\.(RLock|Lock)',
            r'asyncio\.Lock',
            r'FileLock',
            r'\.lock',
        ]
        
        for pattern in lock_patterns:
            if re.search(pattern, value_str):
                return True
        
        return False
    
    def _is_queue(self, value: ast.AST, value_str: str) -> bool:
        """Check if value is a queue."""
        queue_patterns = [
            r'Queue\(',
            r'asyncio\.Queue',
            r'queue\.Queue',
        ]
        
        for pattern in queue_patterns:
            if re.search(pattern, value_str):
                return True
        
        return False
    
    def _is_logger(self, value: ast.AST, value_str: str) -> bool:
        """Check if value is a logger."""
        logger_patterns = [
            r'logging\.getLogger',
            r'logger\s*=',
        ]
        
        for pattern in logger_patterns:
            if re.search(pattern, value_str):
                return True
        
        return False
    
    def _is_config(self, name: str, value: ast.AST, value_str: str) -> bool:
        """Check if attribute is configuration."""
        config_keywords = ['config', 'Config', 'settings', 'Settings']
        
        if any(keyword in name.lower() for keyword in config_keywords):
            return True
        
        return False
    
    def _classify_scope(self, name: str, value: ast.AST) -> str:
        """Classify attribute scope based on patterns."""
        value_str = ast_unparse(value)
        
        # Global indicators
        global_indicators = [
            'logger', 'log', 'logging',
            'Path(', 'pathlib',
            'threading.', 'asyncio.',
            'os.getenv', 'os.environ',
            'logging.getLogger',
        ]
        
        if any(indicator in value_str for indicator in global_indicators):
            return "global"
        
        # Session-scoped indicators
        session_indicators = [
            'history', 'History',
            'cache', 'Cache',
            'registry', 'Registry',
            'memory', 'Memory',
            'Queue', 'queue',
            'List[', 'Dict[',
            '[]', '{}',
        ]
        
        if any(indicator in value_str or indicator in name for indicator in session_indicators):
            return "session-scoped"
        
        # Run-scoped indicators
        run_indicators = [
            'data', 'Data',
            'result', 'Result',
            'metrics', 'Metrics',
            'trades', 'Trades',
            'positions', 'Positions',
        ]
        
        if any(indicator in name for indicator in run_indicators):
            return "run-scoped"
        
        # Default based on initialization pattern
        if isinstance(value, (ast.List, ast.Dict, ast.Set)):
            return "session-scoped"
        
        if isinstance(value, ast.Call):
            # If it's a constructor call, likely run-scoped
            return "run-scoped"
        
        return "unknown"
    
    def _check_singleton_pattern(self, node: ast.ClassDef) -> bool:
        """Check if class uses singleton pattern."""
        # Check for global instance variable
        for global_name in self.global_instances:
            # This is a heuristic - check if class name matches
            if node.name.lower() in global_name.lower() or global_name.lower() in node.name.lower():
                if self.current_class:
                    self.classes[self.current_class].singleton_pattern = f"Global instance: {global_name}"
                return True
        
        # Check for class-level _instance pattern
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "_instance" or target.id.startswith("_"):
                            if isinstance(item.value, ast.Constant) and item.value.value is None:
                                # Pattern: _instance = None
                                if self.current_class:
                                    self.classes[self.current_class].singleton_pattern = "Class-level _instance"
                                return True
        
        return False


def analyze_file(file_path: Path) -> Dict[str, ClassInfo]:
    """Analyze a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        analyzer = StateAnalyzer(file_path)
        analyzer.visit(tree)
        
        return analyzer.classes
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return {}


def find_python_files(root_path: Path, exclude_dirs: Optional[Set[str]] = None) -> List[Path]:
    """Find all Python files in the directory tree."""
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', '.git', 'node_modules', '.venv', 'venv', 'env', '.env'}
    
    python_files = []
    for path in root_path.rglob("*.py"):
        # Skip excluded directories
        if any(excluded in path.parts for excluded in exclude_dirs):
            continue
        python_files.append(path)
    
    return sorted(python_files)


def load_files_from_comprehensive_analysis(file_path: Path) -> List[Path]:
    """Load file paths from comprehensive_file_analysis.txt."""
    python_files = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the "USED FILES" section
        in_used_files = False
        files_not_found = []
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Start collecting after "USED FILES (Import Tracing)" header
            # Look for the exact section header (not the summary line)
            if line_stripped.upper() == "USED FILES (IMPORT TRACING)":
                in_used_files = True
                logger.debug(f"Found USED FILES section at line {line_num}")
                continue  # Skip the header line itself
            
            # Only process if we're in the USED FILES section
            if not in_used_files:
                continue
            
            # Stop at next major section (if any) - look for separator lines
            # But only stop if we've collected at least one file (to avoid stopping on the header separator)
            if line_stripped.startswith("=") and len(line_stripped) > 20 and len(python_files) > 0:
                # This is a section separator, stop collecting
                logger.debug(f"Found section separator at line {line_num}, stopping collection")
                break
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Collect Python files
            if line_stripped.endswith('.py'):
                # Path handles both forward and backslashes on Windows
                file_path_obj = Path(line_stripped)
                
                # Make path absolute if it's relative
                if not file_path_obj.is_absolute():
                    file_path_obj = Path.cwd() / file_path_obj
                
                if file_path_obj.exists():
                    python_files.append(file_path_obj)
                    if len(python_files) <= 3:
                        logger.info(f"Found file: {file_path_obj}")
                else:
                    files_not_found.append(line_stripped)
                    if len(files_not_found) <= 3:
                        logger.warning(f"File not found: {file_path_obj}")
        
        if files_not_found:
            logger.info(f"Total files not found: {len(files_not_found)} out of {len(files_not_found) + len(python_files)}")
    
    except Exception as e:
        logger.error(f"Error reading comprehensive_file_analysis.txt: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return sorted(set(python_files))  # Remove duplicates


def generate_markdown_report(all_classes: Dict[str, ClassInfo], output_path: Path):
    """Generate markdown report from analysis."""
    lines = [
        "# STATE INVENTORY - AUTOMATED ANALYSIS",
        "",
        "## Analysis Methodology",
        "- **Run-scoped**: State reset per execution/run",
        "- **Session-scoped**: State persists across multiple operations within a session",
        "- **Global**: Module-level or singleton instances shared across sessions",
        "- **Singleton**: Single instance pattern",
        "- **File-backed**: State persisted to files (JSON, CSV, pickle, etc.)",
        "- **Cache**: Caching mechanism",
        "",
        "---",
        "",
        "## CLASS STATE INVENTORY",
        "",
    ]
    
    class_num = 1
    for class_name, class_info in sorted(all_classes.items()):
        lines.append(f"### {class_num}. {class_name} ({class_info.file_path})")
        lines.append("")
        lines.append("| Attribute | Scope | Type | Notes |")
        lines.append("|-----------|-------|------|-------|")
        
        if not class_info.attributes:
            lines.append("| *(no attributes found)* | | | |")
        else:
            for attr in class_info.attributes:
                notes = []
                if attr.is_file_path:
                    notes.append("file path")
                if attr.is_cache:
                    notes.append("cache")
                if attr.is_lock:
                    notes.append("lock")
                if attr.is_queue:
                    notes.append("queue")
                if attr.is_logger:
                    notes.append("logger")
                if attr.is_config:
                    notes.append("config")
                
                notes_str = ", ".join(notes) if notes else ""
                type_str = attr.type_hint or (attr.initial_value[:50] + "..." if attr.initial_value and len(attr.initial_value) > 50 else attr.initial_value or "")
                
                lines.append(f"| `{attr.name}` | {attr.scope} | {type_str} | {notes_str} |")
        
        lines.append("")
        lines.append("**Special Patterns**:")
        
        special_patterns = []
        if class_info.is_singleton:
            special_patterns.append(f"- **Singleton**: {class_info.singleton_pattern or 'Detected singleton pattern'}")
        if class_info.has_file_backed_state:
            special_patterns.append(f"- **File-backed**: {', '.join(class_info.file_paths)}")
        if class_info.has_cache:
            special_patterns.append(f"- **Cache**: {', '.join(class_info.cache_attributes)}")
        
        if special_patterns:
            lines.extend(special_patterns)
        else:
            lines.append("- None")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        class_num += 1
    
    # Add summary
    lines.extend([
        "## SUMMARY STATISTICS",
        "",
        f"### Total Classes Analyzed",
        f"- **{len(all_classes)} classes** documented with complete state inventory",
        "",
        "### Scope Distribution",
        f"- **Run-scoped**: {sum(1 for c in all_classes.values() for a in c.attributes if a.scope == 'run-scoped')} attributes",
        f"- **Session-scoped**: {sum(1 for c in all_classes.values() for a in c.attributes if a.scope == 'session-scoped')} attributes",
        f"- **Global**: {sum(1 for c in all_classes.values() for a in c.attributes if a.scope == 'global')} attributes",
        f"- **Unknown**: {sum(1 for c in all_classes.values() for a in c.attributes if a.scope == 'unknown')} attributes",
        "",
        "### Special Patterns",
        f"- **Singletons**: {sum(1 for c in all_classes.values() if c.is_singleton)} identified",
        f"- **File-backed**: {sum(1 for c in all_classes.values() if c.has_file_backed_state)} classes with file persistence",
        f"- **Caches**: {sum(1 for c in all_classes.values() if c.has_cache)} cache implementations",
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Markdown report written to {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Python codebase for state inventory")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Root path to analyze (default: use comprehensive_file_analysis.txt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="STATE_INVENTORY_AUTOMATED.md",
        help="Output markdown file path (default: STATE_INVENTORY_AUTOMATED.md)"
    )
    parser.add_argument(
        "--comprehensive-file",
        type=str,
        default="comprehensive_file_analysis.txt",
        help="Path to comprehensive_file_analysis.txt (default: comprehensive_file_analysis.txt)"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=["__pycache__", ".git", "node_modules", ".venv", "venv", "env", ".env", "tests"],
        help="Directories to exclude (only used with --path)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    # Determine which files to analyze
    if args.path:
        # Use directory scanning mode
        root_path = Path(args.path)
        exclude_dirs = set(args.exclude)
        logger.info(f"Analyzing Python files in {root_path}")
        logger.info(f"Excluding directories: {exclude_dirs}")
        python_files = find_python_files(root_path, exclude_dirs)
    else:
        # Use comprehensive_file_analysis.txt
        comprehensive_file = Path(args.comprehensive_file)
        if not comprehensive_file.exists():
            logger.error(f"File not found: {comprehensive_file}")
            logger.info("Please provide --path or ensure comprehensive_file_analysis.txt exists")
            return
        
        logger.info(f"Loading files from {comprehensive_file}")
        python_files = load_files_from_comprehensive_analysis(comprehensive_file)
        logger.info(f"Found {len(python_files)} Python files from comprehensive_file_analysis.txt")
    
    all_classes: Dict[str, ClassInfo] = {}
    
    for file_path in python_files:
        logger.info(f"Analyzing {file_path}")
        classes = analyze_file(file_path)
        
        # Handle duplicate class names by prefixing with file path
        for class_name, class_info in classes.items():
            unique_name = f"{class_info.file_path}::{class_name}"
            all_classes[unique_name] = class_info
    
    logger.info(f"Found {len(all_classes)} classes")
    
    # Generate report
    generate_markdown_report(all_classes, output_path)
    
    # Also generate JSON for programmatic access
    json_output = output_path.with_suffix('.json')
    json_data = {
        class_name: {
            "name": class_info.name,
            "file_path": class_info.file_path,
            "base_classes": class_info.base_classes,
            "attributes": [
                {
                    "name": attr.name,
                    "scope": attr.scope,
                    "type_hint": attr.type_hint,
                    "initial_value": attr.initial_value,
                    "is_file_path": attr.is_file_path,
                    "is_cache": attr.is_cache,
                    "is_lock": attr.is_lock,
                    "is_queue": attr.is_queue,
                    "is_logger": attr.is_logger,
                    "is_config": attr.is_config,
                }
                for attr in class_info.attributes
            ],
            "is_singleton": class_info.is_singleton,
            "has_file_backed_state": class_info.has_file_backed_state,
            "has_cache": class_info.has_cache,
            "singleton_pattern": class_info.singleton_pattern,
            "file_paths": class_info.file_paths,
            "cache_attributes": class_info.cache_attributes,
        }
        for class_name, class_info in all_classes.items()
    }
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    logger.info(f"JSON report written to {json_output}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()

