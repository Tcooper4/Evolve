#!/usr/bin/env python3
"""
Script to fix the 5 critical syntax errors in the Evolve-main codebase.
"""

import os

def fix_run_forecasting_pipeline():
    """Fix syntax error in run_forecasting_pipeline.py line 77-78."""
    file_path = "run_forecasting_pipeline.py"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix missing newline between environment variable and function definition
    old_content = '        os.environ["STREAMLIT_DEBUG"] = "false"\n    def validate_environment(self) -> bool:'
    new_content = '        os.environ["STREAMLIT_DEBUG"] = "false"\n        \n    def validate_environment(self) -> bool:'
    
    if old_content in content:
        content = content.replace(old_content, new_content)
        with open(file_path, 'w') as f:
            f.write(content)
        print("Fixed run_forecasting_pipeline.py")
    else:
        print("Could not find the problematic line in run_forecasting_pipeline.py")

def fix_model_generator():
    """Fix indentation error in agents/model_generator.py line 189."""
    file_path = "agents/model_generator.py"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix empty if statement
    old_content = '        if not architectures:\n\n        # Generate implementation template'
    new_content = '        if not architectures:\n            return None\n\n        # Generate implementation template'
    
    if old_content in content:
        content = content.replace(old_content, new_content)
        with open(file_path, 'w') as f:
            f.write(content)
        print("Fixed agents/model_generator.py")
    else:
        print("Could not find the problematic line in agents/model_generator.py")

def fix_performance_weights():
    """Fix indentation error in memory/performance_weights.py line 151."""
    file_path = "memory/performance_weights.py"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix empty if statement
    old_content = '    if len(history) < 2:\n\n    timestamps = sorted(history.keys())[-2:]'
    new_content = '    if len(history) < 2:\n        return\n\n    timestamps = sorted(history.keys())[-2:]'
    
    if old_content in content:
        content = content.replace(old_content, new_content)
        with open(file_path, 'w') as f:
            f.write(content)
        print("Fixed memory/performance_weights.py")
    else:
        print("Could not find the problematic line in memory/performance_weights.py")

def fix_manage_incident():
    """Fix syntax error in scripts/manage_incident.py line 74."""
    file_path = "scripts/manage_incident.py"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix missing newline between directory creation and function definition
    old_content = '        self.responses_dir.mkdir(parents=True, exist_ok=True)def _load_config(self, config_path: str) -> dict:'
    new_content = '        self.responses_dir.mkdir(parents=True, exist_ok=True)\n    \n    def _load_config(self, config_path: str) -> dict:'
    
    if old_content in content:
        content = content.replace(old_content, new_content)
        with open(file_path, 'w') as f:
            f.write(content)
        print("Fixed scripts/manage_incident.py")
    else:
        print("Could not find the problematic line in scripts/manage_incident.py")

def fix_manage_logs():
    """Fix syntax error in scripts/manage_logs.py line 50."""
    file_path = "scripts/manage_logs.py"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix missing newline between directory assignment and function definition
    old_content = '        self.archive_dir = Path("logs/archive")def _load_config(self, config_path: str) -> dict:'
    new_content = '        self.archive_dir = Path("logs/archive")\n    \n    def _load_config(self, config_path: str) -> dict:'
    
    if old_content in content:
        content = content.replace(old_content, new_content)
        with open(file_path, 'w') as f:
            f.write(content)
        print("Fixed scripts/manage_logs.py")
    else:
        print("Could not find the problematic line in scripts/manage_logs.py")

def main():
    """Fix all syntax errors."""
    print("Fixing critical syntax errors...")
    
    fix_run_forecasting_pipeline()
    fix_model_generator()
    fix_performance_weights()
    fix_manage_incident()
    fix_manage_logs()
    
    print("Syntax error fixes completed!")

if __name__ == "__main__":
    main() 