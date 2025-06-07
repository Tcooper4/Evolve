import logging
from typing import Dict, List, Optional, Set
import json
from pathlib import Path
from datetime import datetime
import hashlib
import difflib
from dataclasses import dataclass, asdict
import re

@dataclass
class CodeContext:
    file_path: str
    content: str
    last_modified: str
    dependencies: List[str]
    imports: List[str]
    functions: List[str]
    classes: List[str]
    variables: List[str]
    comments: List[str]
    docstrings: List[str]
    type_hints: Dict[str, str]
    test_coverage: float
    complexity: float
    last_analyzed: str

class CodeContextManager:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.contexts: Dict[str, CodeContext] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.change_history: List[Dict] = []
        self.analysis_cache: Dict[str, Dict] = {}

    def setup_logging(self):
        """Configure logging for the code context manager."""
        log_path = Path("automation/logs/code_context")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "code_context.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def update_context(self, file_path: str, content: str) -> CodeContext:
        """
        Update the context for a file.
        
        Args:
            file_path: Path to the file
            content: New content of the file
        
        Returns:
            CodeContext: Updated context
        """
        # Calculate content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check if content has changed
        if file_path in self.contexts:
            old_hash = hashlib.md5(self.contexts[file_path].content.encode()).hexdigest()
            if old_hash == content_hash:
                return self.contexts[file_path]
        
        # Analyze code
        analysis = self._analyze_code(content)
        
        # Create new context
        context = CodeContext(
            file_path=file_path,
            content=content,
            last_modified=datetime.now().isoformat(),
            dependencies=analysis['dependencies'],
            imports=analysis['imports'],
            functions=analysis['functions'],
            classes=analysis['classes'],
            variables=analysis['variables'],
            comments=analysis['comments'],
            docstrings=analysis['docstrings'],
            type_hints=analysis['type_hints'],
            test_coverage=analysis['test_coverage'],
            complexity=analysis['complexity'],
            last_analyzed=datetime.now().isoformat()
        )
        
        # Update contexts
        self.contexts[file_path] = context
        
        # Update dependency graph
        self._update_dependency_graph(file_path, analysis['dependencies'])
        
        # Record change
        self._record_change(file_path, content)
        
        return context

    def _analyze_code(self, content: str) -> Dict:
        """Analyze code content."""
        # Use cached analysis if available
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.analysis_cache:
            return self.analysis_cache[content_hash]
        
        analysis = {
            'dependencies': self._extract_dependencies(content),
            'imports': self._extract_imports(content),
            'functions': self._extract_functions(content),
            'classes': self._extract_classes(content),
            'variables': self._extract_variables(content),
            'comments': self._extract_comments(content),
            'docstrings': self._extract_docstrings(content),
            'type_hints': self._extract_type_hints(content),
            'test_coverage': self._calculate_test_coverage(content),
            'complexity': self._calculate_complexity(content)
        }
        
        # Cache analysis
        self.analysis_cache[content_hash] = analysis
        return analysis

    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from code."""
        dependencies = []
        
        # Look for import statements
        import_pattern = r'^import\s+(\w+)'
        from_pattern = r'^from\s+(\w+)\s+import'
        
        for line in content.split('\n'):
            import_match = re.match(import_pattern, line.strip())
            from_match = re.match(from_pattern, line.strip())
            
            if import_match:
                dependencies.append(import_match.group(1))
            elif from_match:
                dependencies.append(from_match.group(1))
        
        return list(set(dependencies))

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        
        for line in content.split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                imports.append(line.strip())
        
        return imports

    def _extract_functions(self, content: str) -> List[str]:
        """Extract function definitions from code."""
        functions = []
        
        function_pattern = r'def\s+(\w+)\s*\('
        for match in re.finditer(function_pattern, content):
            functions.append(match.group(1))
        
        return functions

    def _extract_classes(self, content: str) -> List[str]:
        """Extract class definitions from code."""
        classes = []
        
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            classes.append(match.group(1))
        
        return classes

    def _extract_variables(self, content: str) -> List[str]:
        """Extract variable assignments from code."""
        variables = []
        
        # Look for variable assignments
        assignment_pattern = r'^\s*(\w+)\s*='
        for line in content.split('\n'):
            match = re.match(assignment_pattern, line.strip())
            if match:
                variables.append(match.group(1))
        
        return variables

    def _extract_comments(self, content: str) -> List[str]:
        """Extract comments from code."""
        comments = []
        
        # Look for single-line comments
        comment_pattern = r'#\s*(.+)$'
        for line in content.split('\n'):
            match = re.search(comment_pattern, line)
            if match:
                comments.append(match.group(1).strip())
        
        return comments

    def _extract_docstrings(self, content: str) -> List[str]:
        """Extract docstrings from code."""
        docstrings = []
        
        # Look for triple-quoted strings
        docstring_pattern = r'"""(.*?)"""'
        for match in re.finditer(docstring_pattern, content, re.DOTALL):
            docstrings.append(match.group(1).strip())
        
        return docstrings

    def _extract_type_hints(self, content: str) -> Dict[str, str]:
        """Extract type hints from code."""
        type_hints = {}
        
        # Look for type annotations
        type_pattern = r'(\w+):\s*(\w+)'
        for match in re.finditer(type_pattern, content):
            variable, type_hint = match.groups()
            type_hints[variable] = type_hint
        
        return type_hints

    def _calculate_test_coverage(self, content: str) -> float:
        """Calculate test coverage for code."""
        # This is a simplified calculation
        # In a real implementation, you would use a coverage tool
        test_pattern = r'def\s+test_'
        total_functions = len(self._extract_functions(content))
        test_functions = len(re.findall(test_pattern, content))
        
        if total_functions == 0:
            return 0.0
        
        return (test_functions / total_functions) * 100

    def _calculate_complexity(self, content: str) -> float:
        """Calculate code complexity."""
        # This is a simplified calculation
        # In a real implementation, you would use a complexity analyzer
        complexity = 0
        
        # Count control structures
        control_patterns = [
            r'\bif\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\btry\b',
            r'\bexcept\b',
            r'\bwith\b'
        ]
        
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, content))
        
        return complexity

    def _update_dependency_graph(self, file_path: str, dependencies: List[str]):
        """Update the dependency graph."""
        self.dependency_graph[file_path] = set(dependencies)

    def _record_change(self, file_path: str, new_content: str):
        """Record a change to a file."""
        if file_path in self.contexts:
            old_content = self.contexts[file_path].content
            diff = list(difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                lineterm=''
            ))
        else:
            diff = new_content.splitlines()
        
        self.change_history.append({
            'file_path': file_path,
            'timestamp': datetime.now().isoformat(),
            'diff': diff
        })

    def get_context(self, file_path: str) -> Optional[CodeContext]:
        """Get context for a file."""
        return self.contexts.get(file_path)

    def get_dependencies(self, file_path: str) -> Set[str]:
        """Get dependencies for a file."""
        return self.dependency_graph.get(file_path, set())

    def get_change_history(self, file_path: Optional[str] = None) -> List[Dict]:
        """Get change history for a file or all files."""
        if file_path:
            return [change for change in self.change_history if change['file_path'] == file_path]
        return self.change_history

    def get_affected_files(self, file_path: str) -> Set[str]:
        """Get files affected by changes to a file."""
        affected = set()
        
        # Find files that depend on the changed file
        for file, dependencies in self.dependency_graph.items():
            if file_path in dependencies:
                affected.add(file)
        
        return affected

    def clear_context(self, file_path: Optional[str] = None):
        """Clear context for a file or all files."""
        if file_path:
            self.contexts.pop(file_path, None)
            self.dependency_graph.pop(file_path, None)
        else:
            self.contexts.clear()
            self.dependency_graph.clear()
            self.change_history.clear()
            self.analysis_cache.clear() 