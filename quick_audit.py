#!/usr/bin/env python3
"""
Quick Post-Upgrade Return Statement Audit

Fast audit to identify critical functions missing return statements.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def find_python_files(root_dir: str = ".") -> List[str]:
    """Find Python files excluding archive/legacy directories."""
    python_files = []
    exclude_dirs = {'.venv', 'venv', '__pycache__', '.git', 'archive', 'legacy', 'test_coverage'}
    
    for root, dirs, files in os.walk(root_dir):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def check_function_returns(file_path: str) -> List[Dict[str, Any]]:
    """Check if functions in a file have proper return statements."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        violations = []
        
        # Find function definitions
        for i, line in enumerate(lines):
            # Match function definitions
            func_match = re.match(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
            if func_match:
                func_name = func_match.group(1)
                func_line = i + 1
                
                # Skip __init__ methods
                if func_name == '__init__':
                    continue
                
                # Find the end of the function
                func_end = len(lines)
                indent_level = len(line) - len(line.lstrip())
                
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == '':
                        continue
                    current_indent = len(lines[j]) - len(lines[j].lstrip())
                    if current_indent <= indent_level and lines[j].strip():
                        func_end = j
                        break
                
                # Extract function content
                func_content = '\n'.join(lines[i:func_end])
                
                # Check if function has return statement
                has_return = 'return ' in func_content or 'return\n' in func_content
                
                # Check if function has side effects
                side_effects = any(pattern in func_content for pattern in [
                    'logger.', 'print(', 'st.', 'open(', 'write(', 'requests.',
                    'redis.', 'subprocess.', 'os.system(', 'logging.',
                    'save(', 'load(', 'send(', 'post(', 'get(',
                    'display(', 'show(', 'plot(', 'render(', 'draw(',
                    'execute(', 'run(', 'start(', 'stop(', 'create(',
                    'update(', 'set_', 'add_', 'remove_', 'delete_',
                    'log_', 'notify_', 'alert_', 'send_', 'write_'
                ])
                
                # Check if function name suggests it should return something
                return_indicators = [
                    'get_', 'fetch_', 'load_', 'read_', 'parse_', 'calculate_',
                    'compute_', 'generate_', 'create_', 'build_', 'make_',
                    'render_', 'display_', 'show_', 'plot_', 'draw_',
                    'analyze_', 'process_', 'transform_', 'convert_',
                    'validate_', 'check_', 'verify_', 'test_',
                    'run_', 'execute_', 'select_', 'choose_',
                    'log_', 'save_', 'export_', 'publish_'
                ]
                
                should_return = any(indicator in func_name for indicator in return_indicators)
                
                # Function needs return if it has side effects or should return something
                needs_return = (side_effects or should_return) and not has_return
                
                if needs_return:
                    reason = "side effects" if side_effects else "should return based on name"
                    violations.append({
                        'file': file_path,
                        'line': func_line,
                        'function': func_name,
                        'reason': reason,
                        'side_effects': side_effects,
                        'should_return': should_return
                    })
        
        return violations
        
    except Exception as e:
        return [{
            'file': file_path,
            'line': 0,
            'function': 'parse_error',
            'reason': f'Error parsing file: {str(e)}',
            'side_effects': False,
            'should_return': False
        }]

def main():
    """Main audit function."""
    print("🔍 QUICK POST-UPGRADE RETURN STATEMENT AUDIT")
    print("=" * 60)
    
    # Find Python files
    python_files = find_python_files()
    print(f"Found {len(python_files)} Python files to audit")
    
    all_violations = []
    files_with_violations = 0
    
    for file_path in python_files:
        violations = check_function_returns(file_path)
        if violations:
            all_violations.extend(violations)
            files_with_violations += 1
            print(f"\n📁 {file_path}")
            for violation in violations:
                if violation['function'] != 'parse_error':
                    print(f"  ❌ Line {violation['line']}: {violation['function']} - {violation['reason']}")
                else:
                    print(f"  ❌ {violation['reason']}")
    
    print("\n" + "=" * 60)
    print("📊 AUDIT RESULTS")
    print("=" * 60)
    print(f"Files audited: {len(python_files)}")
    print(f"Files with violations: {files_with_violations}")
    print(f"Total violations: {len(all_violations)}")
    
    if all_violations:
        print(f"\n❌ VIOLATIONS FOUND - SYSTEM NOT FULLY COMPLIANT")
        print("=" * 60)
        
        # Show top violations by category
        side_effect_violations = [v for v in all_violations if v.get('side_effects') and v['function'] != 'parse_error']
        name_based_violations = [v for v in all_violations if v.get('should_return') and not v.get('side_effects') and v['function'] != 'parse_error']
        
        print(f"Functions with side effects missing returns: {len(side_effect_violations)}")
        print(f"Functions that should return based on name: {len(name_based_violations)}")
        
        print(f"\n🎯 TOP PRIORITY FIXES (Side Effects):")
        for violation in side_effect_violations[:20]:
            print(f"  - {violation['file']}:{violation['line']} - {violation['function']}")
        
        if len(side_effect_violations) > 20:
            print(f"  ... and {len(side_effect_violations) - 20} more")
        
        return {
            'status': 'violations_found',
            'total_violations': len(all_violations),
            'side_effect_violations': len(side_effect_violations),
            'name_based_violations': len(name_based_violations),
            'files_with_violations': files_with_violations
        }
    else:
        print("\n✅ ALL RETURN STATEMENTS ARE COMPLIANT!")
        return {'status': 'compliant'}

if __name__ == "__main__":
    result = main()
    print(f"\nFinal result: {result['status']}") 