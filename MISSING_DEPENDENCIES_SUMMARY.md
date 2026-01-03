# Missing Dependencies Summary

Based on the codebase analysis, here are the Python packages that are **missing from `pyproject.toml`** but are used in the code:

## Critical Dependencies (Required for Core Features)

### 1. **PyYAML** (`pyyaml`)
- **Used in:** `core/orchestrator/task_orchestrator.py`
- **Purpose:** YAML configuration file parsing
- **Install:** `pip install pyyaml`
- **Status:** ❌ Missing from `pyproject.toml`

### 2. **schedule**
- **Used in:** 
  - `core/orchestrator/task_scheduler.py`
  - `trading/agents/updater/scheduler.py`
  - `meta/meta_controller.py`
- **Purpose:** Task scheduling functionality
- **Install:** `pip install schedule`
- **Status:** ❌ Missing from `pyproject.toml`

### 3. **statsmodels**
- **Used in:** `trading/models/arima_model.py`
- **Purpose:** ARIMA time series forecasting
- **Install:** `pip install statsmodels`
- **Status:** ❌ Missing from `pyproject.toml`

### 4. **arch**
- **Used in:** 
  - `trading/models/garch_model.py`
  - `trading/risk/risk_manager.py`
- **Purpose:** GARCH volatility modeling
- **Install:** `pip install arch`
- **Status:** ❌ Missing from `pyproject.toml`

## Optional Dependencies (For Advanced Features)

### 5. **flaml**
- **Used in:** `agents/model_innovation_agent.py`
- **Purpose:** AutoML for model discovery
- **Install:** `pip install flaml`
- **Status:** ❌ Missing from `pyproject.toml`
- **Note:** Has fallback if not available

### 6. **autoformer-pytorch**
- **Used in:** `trading/models/autoformer_model.py`
- **Purpose:** Autoformer transformer model for time series
- **Install:** `pip install autoformer-pytorch`
- **Status:** ❌ Missing from `pyproject.toml`
- **Note:** Has fallback if not available

### 7. **catboost**
- **Used in:** `trading/models/catboost_model.py`
- **Purpose:** CatBoost gradient boosting
- **Status:** ✅ Present in `requirements.txt` but ❌ Missing from `pyproject.toml`

## Already Installed (in requirements.txt but not pyproject.toml)

These are in `requirements.txt` but should be added to `pyproject.toml` for consistency:

- `catboost>=1.2.8` (in requirements.txt line 33)

## Import Warnings (NOT Missing Dependencies)

These warnings are **NOT** about missing Python packages - they're about import path issues that are already handled:

1. **`WARNING:__main__:UI components import error: No module named 'ui.page_renderer'`**
   - ✅ File exists: `ui/page_renderer.py`
   - ✅ Handled with try/except in `app.py`
   - This is likely a Python path issue, not a missing package

2. **`WARNING:__main__:Task Orchestrator not available: No module named 'utils.common_helpers'`**
   - ✅ File exists: `utils/common_helpers.py`
   - ✅ Handled with fallback in `core/orchestrator/task_orchestrator.py`
   - This is likely a Python path issue, not a missing package

3. **`WARNING:__main__:Agent Controller not available: No module named 'agents.agent_controller'`**
   - ✅ File exists: `agents/agent_controller.py`
   - ✅ Handled with try/except in `agents/__init__.py`
   - This is likely a Python path issue, not a missing package

## Quick Install Command

To install all missing dependencies:

```bash
pip install pyyaml schedule statsmodels arch flaml autoformer-pytorch
```

## Recommended: Update pyproject.toml

Add these to the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "pyyaml>=6.0.0",           # For YAML config parsing
    "schedule>=1.2.0",         # For task scheduling
    "statsmodels>=0.14.0",     # For ARIMA models
    "arch>=6.0.0",             # For GARCH models
    "flaml>=2.0.0",            # For AutoML (optional)
    "autoformer-pytorch>=0.1.0",  # For Autoformer (optional)
    "catboost>=1.2.0",         # For CatBoost models
]
```

## Verification

After installing, verify with:

```python
import yaml
import schedule
import statsmodels
from arch import arch_model
from flaml import AutoML
from autoformer_pytorch import Autoformer
from catboost import CatBoostRegressor

print("✅ All dependencies installed!")
```

