#!/usr/bin/env python3
"""
POST-UPGRADE RETURN AUDIT

Comprehensive audit script to scan the Evolve codebase for return statement compliance.
"""

import os
import re
import ast
import json
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReturnStatementAuditor:
    """Auditor for return statement compliance."""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.violations = []
        self.excluded_dirs = {'archive', 'legacy', 'test_coverage', '__pycache__', '.git', '.venv', 'htmlcov'}
        self.critical_methods = {
            'run', 'execute', 'select_model', 'train', 'predict', 'evaluate',
            'process', 'handle', 'analyze', 'optimize', 'backtest', 'trade',
            'generate', 'create', 'build', 'setup', 'initialize', 'configure'
        }
        
    def should_skip_file(self, filepath: Path) -> bool:
        """Check if file should be skipped."""
        # Skip excluded directories
        for part in filepath.parts:
            if part in self.excluded_dirs:
                return True
        
        # Skip non-Python files
        if not filepath.suffix == '.py':
            return True
            
        return False
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase."""
        python_files = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Remove excluded directories from dirs list
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    if not self.should_skip_file(filepath):
                        python_files.append(filepath)
        
        return python_files
    
    def analyze_function_returns(self, filepath: Path) -> List[Dict]:
        """Analyze return statements in a Python file."""
        violations = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                # Skip files with syntax errors
                return violations
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    violation = self.check_function_compliance(node, filepath, content)
                    if violation:
                        violations.append(violation)
        
        except Exception as e:
            logger.warning(f"Error analyzing {filepath}: {e}")
        
        return violations
    
    def check_function_compliance(self, func_node: ast.FunctionDef, filepath: Path, content: str) -> Optional[Dict]:
        """Check if a function complies with return statement requirements."""
        # Skip __init__ methods unless they have side effects
        if func_node.name == '__init__':
            return self.check_init_method(func_node, filepath, content)
        
        # Get function body
        func_lines = content.split('\n')[func_node.lineno-1:func_node.end_lineno]
        func_body = '\n'.join(func_lines)
        
        # Check for return statement
        has_return = self.has_return_statement(func_node)
        
        # Check for print/logging statements
        has_print_logging = self.has_print_or_logging(func_node)
        
        # Check if it's a critical method
        is_critical = func_node.name in self.critical_methods
        
        # Determine violation type
        violation = None
        
        if not has_return:
            if has_print_logging or is_critical:
                violation = {
                    'file': str(filepath),
                    'function': func_node.name,
                    'line': func_node.lineno,
                    'issue': 'missing_return_with_side_effects',
                    'severity': 'high' if is_critical else 'medium',
                    'details': f"Function '{func_node.name}' has no return statement but contains {'print/logging statements' if has_print_logging else 'critical method logic'}"
                }
            else:
                # Check if function has any logic beyond simple assignments
                if self.has_significant_logic(func_node):
                    violation = {
                        'file': str(filepath),
                        'function': func_node.name,
                        'line': func_node.lineno,
                        'issue': 'missing_return_with_logic',
                        'severity': 'medium',
                        'details': f"Function '{func_node.name}' has no return statement but contains significant logic"
                    }
        
        elif has_return and is_critical:
            # Check if critical methods return structured output
            if not self.returns_structured_output(func_node):
                violation = {
                    'file': str(filepath),
                    'function': func_node.name,
                    'line': func_node.lineno,
                    'issue': 'critical_method_non_structured_return',
                    'severity': 'high',
                    'details': f"Critical method '{func_node.name}' should return structured output (dict with status)"
                }
        
        return violation
    
    def check_init_method(self, func_node: ast.FunctionDef, filepath: Path, content: str) -> Optional[Dict]:
        """Check __init__ method compliance."""
        # Check if __init__ has side effects beyond simple assignments
        has_side_effects = self.has_side_effects(func_node)
        
        if has_side_effects:
            return {
                'file': str(filepath),
                'function': func_node.name,
                'line': func_node.lineno,
                'issue': 'init_with_side_effects_no_return',
                'severity': 'medium',
                'details': f"__init__ method has side effects but no return statement"
            }
        
        return None
    
    def has_return_statement(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has a return statement."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return True
        return False
    
    def has_print_or_logging(self, func_node: ast.FunctionDef) -> bool:
        """Check if function contains print or logging statements."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['print', 'logger']:
                        return True
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'logger':
                        return True
        return False
    
    def has_significant_logic(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has significant logic beyond simple assignments."""
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                return True
            elif isinstance(node, ast.Call):
                return True
        return False
    
    def has_side_effects(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has side effects."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                return True
            elif isinstance(node, (ast.If, ast.For, ast.While)):
                return True
        return False
    
    def returns_structured_output(self, func_node: ast.FunctionDef) -> bool:
        """Check if function returns structured output (dict with status)."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Dict):
                    # Check if dict contains 'status' key
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and key.value == 'status':
                            return True
                elif isinstance(node.value, ast.Call):
                    # Check if returning a function call that might return structured output
                    if isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                        if any(keyword in func_name.lower() for keyword in ['status', 'result', 'response']):
                            return True
        return False
    
    def run_audit(self) -> Dict:
        """Run the complete audit."""
        logger.info("Starting return statement audit...")
        
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        total_violations = 0
        
        for filepath in python_files:
            violations = self.analyze_function_returns(filepath)
            self.violations.extend(violations)
            total_violations += len(violations)
            
            if violations:
                logger.info(f"Found {len(violations)} violations in {filepath}")
        
        # Categorize violations
        high_severity = [v for v in self.violations if v['severity'] == 'high']
        medium_severity = [v for v in self.violations if v['severity'] == 'medium']
        low_severity = [v for v in self.violations if v['severity'] == 'low']
        
        audit_result = {
            'summary': {
                'total_files_analyzed': len(python_files),
                'total_violations': total_violations,
                'high_severity': len(high_severity),
                'medium_severity': len(medium_severity),
                'low_severity': len(low_severity),
                'compliance_status': 'compliant' if total_violations == 0 else 'non_compliant'
            },
            'violations': self.violations,
            'violations_by_type': {
                'missing_return_with_side_effects': len([v for v in self.violations if v['issue'] == 'missing_return_with_side_effects']),
                'missing_return_with_logic': len([v for v in self.violations if v['issue'] == 'missing_return_with_logic']),
                'critical_method_non_structured_return': len([v for v in self.violations if v['issue'] == 'critical_method_non_structured_return']),
                'init_with_side_effects_no_return': len([v for v in self.violations if v['issue'] == 'init_with_side_effects_no_return'])
            }
        }
        
        return audit_result
    
    def generate_report(self, audit_result: Dict) -> str:
        """Generate a human-readable report."""
        summary = audit_result['summary']
        violations = audit_result['violations']
        
        report = []
        report.append("=" * 80)
        report.append("POST-UPGRADE RETURN AUDIT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append(f"  Files analyzed: {summary['total_files_analyzed']}")
        report.append(f"  Total violations: {summary['total_violations']}")
        report.append(f"  High severity: {summary['high_severity']}")
        report.append(f"  Medium severity: {summary['medium_severity']}")
        report.append(f"  Low severity: {summary['low_severity']}")
        report.append(f"  Compliance status: {summary['compliance_status'].upper()}")
        report.append("")
        
        if violations:
            report.append("VIOLATIONS FOUND:")
            report.append("-" * 40)
            
            # Group by severity
            for severity in ['high', 'medium', 'low']:
                severity_violations = [v for v in violations if v['severity'] == severity]
                if severity_violations:
                    report.append(f"\n{severity.upper()} SEVERITY VIOLATIONS:")
                    for violation in severity_violations:
                        report.append(f"  {violation['file']}:{violation['line']} - {violation['function']}")
                        report.append(f"    Issue: {violation['issue']}")
                        report.append(f"    Details: {violation['details']}")
                        report.append("")
        else:
            report.append("✅ All return statements are compliant!")
        
        report.append("=" * 80)
        
        return '\n'.join(report)

def main():
    """Main function to run the audit."""
    auditor = ReturnStatementAuditor()
    audit_result = auditor.run_audit()
    
    # Generate and print report
    report = auditor.generate_report(audit_result)
    print(report)
    
    # Save detailed results
    with open('post_upgrade_audit_results.json', 'w') as f:
        json.dump(audit_result, f, indent=2)
    
    # Return appropriate exit code
    if audit_result['summary']['total_violations'] == 0:
        print("\n✅ AUDIT PASSED: All return statements are compliant")
        return 0
    else:
        print(f"\n❌ AUDIT FAILED: {audit_result['summary']['total_violations']} violations found")
        return 1

if __name__ == "__main__":
    exit(main()) 