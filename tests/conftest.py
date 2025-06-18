"""Pytest configuration file."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add core directory to Python path
core_dir = project_root / 'core'
if core_dir.exists():
    sys.path.insert(0, str(core_dir)) 