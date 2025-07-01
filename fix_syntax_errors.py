#!/usr/bin/env python3
"""
Script to fix critical syntax errors in the codebase.
"""

import os
import re
import glob

def fix_file(file_path):
    """Fix syntax errors in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix malformed return statements with structured dictionaries
        # Pattern: return {'success': True, 'result': value, 'message': '...', 'timestamp': '...'}
        # Replace with: return value
        
        # Fix return statements that should return simple values
        patterns_to_fix = [
            # Fix return statements in functions that should return simple values
            (r'return \{\'success\': True, \'result\': ([^,]+), \'message\': [^}]+\}', r'return \1'),
            (r'return \{\'success\': True, \'result\': ([^,]+), \'message\': [^}]+\}', r'return \1'),
            
            # Fix malformed return statements with missing parentheses
            (r'return \{.*\'result\': ([^,]+), \'low\', \'message\': [^}]+\}', r'return \1, \'low\''),
            
            # Fix return statements with missing colons
            (r'return \{.*\'result\': ([^,]+), \'message\': [^}]+\}', r'return \1'),
        ]
        
        for pattern, replacement in patterns_to_fix:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        # Fix specific syntax errors
        content = content.replace("return UpdateResult(", 
                                 "return UpdateResult(")
        
        content = content.replace("return {", 
                                 "return {")
        
        # Fix unexpected indents by removing inappropriate return statements
        content = re.sub(r'(\s+)return \{\s*"success": True,\s*"message": "[^"]*",\s*"timestamp": [^}]+\s*\}\s*$', '', content, flags=re.MULTILINE)
        
        # Fix dictionary syntax errors
        content = re.sub(r'(\w+):\s*([^,}]+),?\s*(\w+):', r'\1: \2,\n        \3:', content)
        
        # Fix closing parenthesis errors
        content = re.sub(r'\)\s*\}\s*$', '))', content, flags=re.MULTILINE)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix syntax errors."""
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip virtual environments and cache directories
        if any(skip in root for skip in ['.venv', 'venv', '__pycache__', '.git', 'node_modules']):
            continue
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    for file_path in python_files:
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main() 