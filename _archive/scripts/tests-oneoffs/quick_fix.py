#!/usr/bin/env python3
"""
Quick fix for critical syntax errors.
"""

import re


def fix_updater_agent():
    """Fix syntax errors in updater_agent.py."""
    with open("trading/agents/updater_agent.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Fix the malformed return statement
    content = content.replace(
        "return {'success': True, 'result': 'ensemble_adjust', 'low', 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}",
        "return 'ensemble_adjust', 'low'",
    )

    # Fix other malformed return statements
    content = re.sub(
        r"return \{'success': True, 'result': ([^,]+), 'message': [^}]+\}",
        r"return \1",
        content,
    )

    with open("trading/agents/updater_agent.py", "w", encoding="utf-8") as f:
        f.write(content)

    print("Fixed updater_agent.py")


def fix_updater_utils():
    """Fix syntax errors in updater/utils.py."""
    with open("trading/agents/updater/utils.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Fix malformed return statements
    content = re.sub(
        r"return \{'success': True, 'result': ([^,]+), ([^,]+), 'message': [^}]+\}",
        r"return \1, \2",
        content,
    )

    content = re.sub(
        r"return \{'success': True, 'result': ([^,]+), 'message': [^}]+\}",
        r"return \1",
        content,
    )

    with open("trading/agents/updater/utils.py", "w", encoding="utf-8") as f:
        f.write(content)

    print("Fixed updater/utils.py")


if __name__ == "__main__":
    fix_updater_agent()
    fix_updater_utils()
