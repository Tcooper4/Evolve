# Streamlit Pages Path Fix

## Issue

Streamlit pages run in a **separate context** from `app.py`, so the `sys.path` fix in `app.py` doesn't apply to the pages. This causes imports to fail in pages even though they work in `app.py`.

## The Problem

When Streamlit loads a page file (e.g., `pages/3_Trade_Execution.py`):
- It runs in a different Python context
- The `sys.path` modifications in `app.py` don't apply
- Imports like `from execution.broker_adapter import ...` fail
- Warnings appear: `WARNING:root:BrokerAdapter not available: No module named 'execution.broker_adapter'`

## Solution

Add the same path fix to each page file that needs to import from the project root:

```python
import sys
from pathlib import Path

# Add project root to Python path for imports (Streamlit pages run in separate context)
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

**Note:** `Path(__file__).parent.parent` because:
- `__file__` = `pages/3_Trade_Execution.py`
- `.parent` = `pages/`
- `.parent.parent` = project root

## Files Fixed

1. ✅ **`pages/3_Trade_Execution.py`** - Added path fix at the top
2. ✅ **`pages/4_Portfolio.py`** - Added path fix at the top

## Why This Is Needed

Streamlit's architecture:
- `app.py` runs in one Python process
- Each page in `pages/` runs in its own context
- Each context has its own `sys.path`
- Path modifications don't carry over between contexts

By adding the path fix to each page, we ensure:
- ✅ Imports work correctly in all pages
- ✅ No more "module not found" warnings
- ✅ Consistent behavior across all Streamlit pages

## Verification

After this fix:
- `from execution.broker_adapter import BrokerAdapter` should work
- `from execution.execution_agent import ExecutionAgent` should work
- `from portfolio.allocator import PortfolioAllocator` should work
- All other project imports should work in pages

