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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReturnStatementAuditor:
    """Auditor for checking return statement compliance."""
    
    def __init__(self, exclude_dirs: Set[str] = None):
        self.exclude_dirs = exclude_dirs or {'archive', 'legacy', 'test_coverage', '__pycache__', '.git'}
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
        
        return self._generate_report()
    
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
        return_indicators = [
            'get_', 'fetch_', 'load_', 'read_', 'parse_', 'calculate_',
            'compute_', 'generate_', 'create_', 'build_', 'make_',
            'render_', 'display_', 'show_', 'plot_', 'draw_'
        ]
        
        func_name = func_node.name
        if any(indicator in func_name for indicator in return_indicators):
            return True
        
        return False
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        total_functions = len(self.passing_functions) + len(self.issues) + len(self.exempt_functions)
        
        report = {
            'summary': {
                'total_functions_audited': total_functions,
                'passing_functions': len(self.passing_functions),
                'functions_with_issues': len(self.issues),
                'exempt_functions': len(self.exempt_functions),
                'compliance_rate': (len(self.passing_functions) + len(self.exempt_functions)) / total_functions if total_functions > 0 else 0
            },
            'issues': self.issues,
            'passing_functions': self.passing_functions,
            'exempt_functions': list(self.exempt_functions),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for fixing issues."""
        recommendations = []
        
        if self.issues:
            recommendations.append("🔧 FUNCTIONS NEEDING RETURN STATEMENTS:")
            for issue in self.issues:
                recommendations.append(f"  - {issue['file']}:{issue['function']} (line {issue['line']})")
        
        recommendations.append("✅ AGENTIC MODULARITY STATUS:")
        if len(self.issues) == 0:
            recommendations.append("  🎉 FULL COMPLIANCE: All functions return structured outputs")
            recommendations.append("  🚀 System meets ChatGPT-like autonomous architecture standards")
        else:
            recommendations.append(f"  ⚠️  {len(self.issues)} functions need return statements")
            recommendations.append("  🔄 System needs updates for full agentic modularity")
        
        return recommendations

def main():
    """Run the comprehensive audit."""
    auditor = ReturnStatementAuditor()
    report = auditor.audit_codebase()
    
    # Print summary
    print("\n" + "="*80)
    print("🔍 EVOLVE CODEBASE RETURN STATEMENT AUDIT")
    print("="*80)
    
    summary = report['summary']
    print(f"\n📊 AUDIT SUMMARY:")
    print(f"  Total functions audited: {summary['total_functions_audited']}")
    print(f"  ✅ Passing functions: {summary['passing_functions']}")
    print(f"  ⚠️  Functions with issues: {summary['functions_with_issues']}")
    print(f"  🔄 Exempt functions (__init__): {summary['exempt_functions']}")
    print(f"  📈 Compliance rate: {summary['compliance_rate']:.1%}")
    
    # Print recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    # Print detailed issues if any
    if report['issues']:
        print(f"\n🔧 DETAILED ISSUES:")
        for issue in report['issues']:
            print(f"  📁 {issue['file']}:{issue['function']} (line {issue['line']})")
            print(f"     Reason: {issue['reason']}")
    
    print("\n" + "="*80)
    
    # Return exit code based on compliance
    if summary['compliance_rate'] >= 0.95:
        print("🎉 EXCELLENT: Codebase meets agentic modularity standards!")
        return 0
    elif summary['compliance_rate'] >= 0.90:
        print("✅ GOOD: Codebase mostly compliant, minor improvements needed")
        return 0
    else:
        print("⚠️  NEEDS WORK: Significant improvements needed for full compliance")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 