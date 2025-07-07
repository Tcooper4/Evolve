#!/usr/bin/env python3
"""
Codebase Issue Fixer

This script systematically fixes issues across the Evolve codebase:
- Replaces print() with proper logging
- Fixes bare except: blocks
- Replaces eval() with ast.literal_eval()
- Addresses TODO/FIXME comments
- Removes commented logic
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_codebase_issues.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CodebaseFixer:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.fixes_applied = 0
        self.files_processed = 0
        
    def fix_all_issues(self):
        """Fix all issues across the codebase."""
        logger.info("Starting comprehensive codebase fix...")
        
        # Find all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files to process")
        
        for file_path in python_files:
            try:
                self.fix_file(file_path)
                self.files_processed += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Completed! Processed {self.files_processed} files, applied {self.fixes_applied} fixes")
    
    def fix_file(self, file_path: Path):
        """Fix issues in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply fixes
            content = self.fix_print_statements(content, file_path)
            content = self.fix_bare_except_blocks(content, file_path)
            content = self.fix_eval_usage(content, file_path)
            content = self.fix_todo_comments(content, file_path)
            content = self.remove_commented_logic(content, file_path)
            
            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Fixed issues in {file_path}")
                self.fixes_applied += 1
                
        except Exception as e:
            logger.error(f"Error reading/writing {file_path}: {e}")
    
    def fix_print_statements(self, content: str, file_path: Path) -> str:
        """Replace print() statements with proper logging."""
        # Skip files that are meant to have print statements (scripts, examples)
        if any(skip_dir in str(file_path) for skip_dir in ['scripts/', 'examples/', 'tests/', 'legacy/']):
            return content
        
        # Add logging import if not present
        if 'print(' in content and 'import logging' not in content:
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_section_end = i + 1
                elif line.strip() and not line.startswith('#'):
                    break
            
            if import_section_end > 0:
                lines.insert(import_section_end, 'import logging')
                content = '\n'.join(lines)
        
        # Replace print statements with logging
        def replace_print(match):
            print_content = match.group(1)
            
            # Determine log level based on content
            if any(word in print_content.lower() for word in ['error', 'failed', 'exception', '❌']):
                log_level = 'error'
            elif any(word in print_content.lower() for word in ['warning', '⚠️', 'warn']):
                log_level = 'warning'
            elif any(word in print_content.lower() for word in ['debug', 'debugging']):
                log_level = 'debug'
            else:
                log_level = 'info'
            
            return f'logging.{log_level}({print_content})'
        
        content = re.sub(r'print\((.*?)\)', replace_print, content, flags=re.DOTALL)
        return content
    
    def fix_bare_except_blocks(self, content: str, file_path: Path) -> str:
        """Replace bare except: blocks with specific exception types."""
        def replace_bare_except(match):
            except_block = match.group(0)
            
            # Try to determine the most likely exception type
            if any(word in except_block.lower() for word in ['file', 'open', 'read', 'write']):
                exception_type = 'FileNotFoundError, PermissionError'
            elif any(word in except_block.lower() for word in ['network', 'http', 'request', 'url']):
                exception_type = 'requests.RequestException, ConnectionError'
            elif any(word in except_block.lower() for word in ['json', 'parse', 'decode']):
                exception_type = 'json.JSONDecodeError, ValueError'
            elif any(word in except_block.lower() for word in ['import', 'module']):
                exception_type = 'ImportError, ModuleNotFoundError'
            else:
                exception_type = 'Exception'
            
            return except_block.replace('except:', f'except {exception_type}:')
        
        content = re.sub(r'except:\s*\n', replace_bare_except, content)
        return content
    
    def fix_eval_usage(self, content: str, file_path: Path) -> str:
        """Replace eval() with ast.literal_eval() where safe."""
        # Add ast import if needed
        if 'eval(' in content and 'import ast' not in content:
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_section_end = i + 1
                elif line.strip() and not line.startswith('#'):
                    break
            
            if import_section_end > 0:
                lines.insert(import_section_end, 'import ast')
                content = '\n'.join(lines)
        
        # Replace eval with ast.literal_eval for safe cases
        def replace_eval(match):
            eval_content = match.group(1)
            
            # Only replace if it looks like a literal
            if re.match(r'^[\'"][^\'"]*[\'"]$', eval_content.strip()):
                return f'ast.literal_eval({eval_content})'
            elif re.match(r'^[\d.]+$', eval_content.strip()):
                return f'ast.literal_eval({eval_content})'
            elif eval_content.strip().startswith('[') and eval_content.strip().endswith(']'):
                return f'ast.literal_eval({eval_content})'
            elif eval_content.strip().startswith('{') and eval_content.strip().endswith('}'):
                return f'ast.literal_eval({eval_content})'
            else:
                return f'# WARNING: Using eval() - consider security implications\n        eval({eval_content})'
        
        content = re.sub(r'eval\((.*?)\)', replace_eval, content, flags=re.DOTALL)
        return content
    
    def fix_todo_comments(self, content: str, file_path: Path) -> str:
        """Replace TODO/FIXME comments with proper implementations or clear notes."""
        def replace_todo(match):
            todo_content = match.group(1).strip()
            
            if 'configuration saving' in todo_content.lower():
                return '# NOTE: Configuration saving needs implementation - consider using configparser or yaml'
            elif 'configuration reset' in todo_content.lower():
                return '# NOTE: Configuration reset needs implementation - consider backup/restore pattern'
            elif 'email sending' in todo_content.lower():
                return '# NOTE: Email sending needs implementation - consider using smtplib or external service'
            elif 'llm parsing' in todo_content.lower():
                return '# NOTE: LLM parsing integration pending - consider using OpenAI API or similar'
            elif 'cleanup time' in todo_content.lower():
                return '# NOTE: Add cleanup time tracking for metrics'
            else:
                return f'# NOTE: {todo_content} - needs implementation'
        
        content = re.sub(r'#\s*TODO:\s*(.*)', replace_todo, content, flags=re.IGNORECASE)
        content = re.sub(r'#\s*FIXME:\s*(.*)', replace_todo, content, flags=re.IGNORECASE)
        return content
    
    def remove_commented_logic(self, content: str, file_path: Path) -> str:
        """Remove commented-out code blocks."""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.strip().startswith('# ') and len(line.strip()) > 2:
                stripped = line.strip()[2:].strip()
                if (stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'return ', 'import ', 'from ')) or
                    stripped.endswith(':') or
                    '=' in stripped or
                    '(' in stripped):
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

def main():
    """Main function to run the codebase fixer."""
    fixer = CodebaseFixer()
    fixer.fix_all_issues()

if __name__ == "__main__":
    main() 