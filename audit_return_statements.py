#!/usr/bin/env python3
"""
Comprehensive audit script to verify all functions have proper return statements.
This script checks the entire Evolve codebase for agentic modularity compliance.
"""

import os
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReturnStatementAuditor:
    """Auditor for checking return statement compliance."""
    
    def __init__(self, exclude_dirs: Set[str] = None):
        self.exclude_dirs = exclude_dirs or {'.venv', 'venv', 'env', 'archive', 'legacy', 'test_coverage', '__pycache__', '.git', 'htmlcov', 'site-packages', 'dist', 'build', 'node_modules'}
        self.issues = []
        self.passing_functions = []
        self.exempt_functions = set()
        
    def audit_codebase(self, root_dir: str = ".") -> Dict[str, Any]:
        """Audit the entire codebase for return statement compliance."""
        logger.info("Starting comprehensive return statement audit...")
        
        python_files = self._find_python_files(root_dir)
        logger.info(f"Found {len(python_files)} Python files to audit")
        
        for file_path in python_files:
            self._audit_file(file_path)
        
        return {'success': True, 'result': self._generate_report(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _find_python_files(self, root_dir: str) -> List[str]:
        """Find all Python files in the codebase."""
        python_files = []
        for root, dirs, files in os.walk(root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _audit_file(self, file_path: str):
        """Audit a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            self._analyze_ast(file_path, tree, content)
            
        except Exception as e:
            logger.error(f"Error auditing {file_path}: {e}")
            self.issues.append({
                'file': file_path,
                'type': 'parse_error',
                'error': str(e)
            })
    
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
        if func_name == '__init__':
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
        
        if needs_return:
            self.issues.append({
                'file': file_path,
                'line': func_line,
                'function': func_name,
                'type': 'missing_return',
                'reason': 'Function has side effects or logging but no return statement'
            })
        else:
            self.passing_functions.append(f"{file_path}:{func_name}")
    
    def _has_return_statement(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has any return statements."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return True
        return False
    
    def _has_only_logging_statements(self, func_node: ast.FunctionDef, content: str) -> bool:
        """Check if function only contains logging/print statements."""
        lines = content.split('\n')
        func_start = func_node.lineno - 1
        func_end = func_node.end_lineno
        
        func_lines = lines[func_start:func_end]
        func_content = '\n'.join(func_lines)
        
        # Check for logging patterns
        logging_patterns = [
            r'logger\.',
            r'print\(',
            r'st\.',
            r'logging\.',
            r'console\.',
            r'print\s*\('
        ]
        
        has_logging = any(re.search(pattern, func_content) for pattern in logging_patterns)
        
        # Check if function has other meaningful operations
        has_other_ops = self._has_meaningful_operations(func_node)
        
        return has_logging and not has_other_ops
    
    def _has_meaningful_operations(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has meaningful operations beyond logging."""
        meaningful_nodes = [
            ast.Assign, ast.AugAssign, ast.Call, ast.If, ast.For, ast.While,
            ast.Try, ast.With, ast.Raise, ast.Return, ast.Yield
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
            if attr_name in ['info', 'warning', 'error', 'debug', 'critical']:
                return True
        elif isinstance(call_node.func, ast.Name):
            if call_node.func.id == 'print':
                return True
        return False
    
    def _has_side_effects(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has side effects."""
        side_effect_patterns = [
            'open(', 'write(', 'read(', 'save(', 'load(',
            'requests.', 'urllib.', 'http.',
            'subprocess.', 'os.system(',
            'redis.', 'database.', 'sqlite.',
            'st.', 'streamlit.'
        ]
        
        func_str = ast.unparse(func_node)
        return any(pattern in func_str for pattern in side_effect_patterns)
    
    def _function_needs_return(self, func_node: ast.FunctionDef, has_return: bool, 
                             has_only_logging: bool, has_side_effects: bool) -> bool:
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
        return_indicators = ['get', 'fetch', 'load', 'read', 'calculate', 'compute', 'process', 'validate', 'check']
        
        if any(indicator in func_name for indicator in return_indicators):
            return True
        
        return False
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of findings."""
        total_issues = len(self.issues)
        total_passing = len(self.passing_functions)
        total_exempt = len(self.exempt_functions)
        
        # Group issues by file
        issues_by_file = {}
        for issue in self.issues:
            file_path = issue['file']
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)
        
        # Sort files by number of issues
        sorted_files = sorted(issues_by_file.items(), key=lambda x: len(x[1]), reverse=True)
        
        return {
            'summary': {
                'total_issues': total_issues,
                'total_passing': total_passing,
                'total_exempt': total_exempt,
                'compliance_rate': f"{((total_passing + total_exempt) / (total_issues + total_passing + total_exempt) * 100):.1f}%"
            },
            'issues_by_file': sorted_files,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for fixing issues."""
        recommendations = [
            "1. Add structured return statements to all functions with side effects",
            "2. Implement consistent error handling with return status",
            "3. Add return statements to logging functions for status reporting",
            "4. Consider adding return statements to utility functions for better integration",
            "5. Review and update function documentation to reflect return values"
        ]
        return recommendations

def main():
    """Main function to run the audit."""
    auditor = ReturnStatementAuditor()
    result = auditor.audit_codebase()
    
    if result['success']:
        report = result['result']
        print("\n" + "="*80)
        print("RETURN STATEMENT AUDIT REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\nSUMMARY:")
        print(f"  Total Issues: {summary['total_issues']}")
        print(f"  Passing Functions: {summary['total_passing']}")
        print(f"  Exempt Functions: {summary['total_exempt']}")
        print(f"  Compliance Rate: {summary['compliance_rate']}")
        
        if summary['total_issues'] > 0:
            print(f"\nTOP FILES WITH ISSUES:")
            for file_path, issues in report['issues_by_file'][:10]:
                print(f"  {file_path}: {len(issues)} issues")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print("\n" + "="*80)
    else:
        print(f"Audit failed: {result['message']}")

if __name__ == "__main__":
    main() 