
# ✅ FINAL CLEANUP CHECKLIST – EVOLVE SYSTEM (REPAIR ALL OUTSTANDING ISSUES)

---

## 🔥 1. PRINT TO LOGGING CONVERSION (HIGH PRIORITY)

- [ ] Search every Python file for `print(` statements.
- [ ] Replace each with:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("message here")
```
- [ ] Use `.debug()` for debugging, `.info()` for status, `.error()` for errors.

---

## 🧼 2. ADD MISSING DOCSTRINGS

- [ ] Every `class` and `def` must have a docstring.
- [ ] Use clear descriptions for:
  - Purpose of the function
  - Each argument and return type
  - Example:
```python
def compute_sharpe(returns: pd.Series) -> float:
    '''
    Computes the Sharpe ratio.

    Args:
        returns (pd.Series): Series of returns

    Returns:
        float: Annualized Sharpe ratio
    '''
```

---

## 🧠 3. ADD TYPE HINTS

- [ ] Add type hints to **all function arguments** and **return types**.
- [ ] Use:
  - `-> None` if no return
  - `-> pd.DataFrame`, `-> str`, etc. as appropriate
- [ ] Import any types from `typing` as needed.

---

## 🧩 4. MODULARIZATION

- [ ] Break up `main.py` and `launch_production.py`:
  - Move routing logic to `agents/router.py`
  - Move UI to `ui/streamlit_ui.py`
  - Move fallback classes to `fallback/` folder (one per file)
  - Move forecast logic to `models/forecast_engine.py`

---

## 🧪 5. INTEGRITY CHECK

- [ ] After refactor, run all files and ensure there are:
  - No broken imports
  - No crashes on load
  - All fallback logic still works
- [ ] Do a dry run forecast with sample prompt to confirm system still functions

---

✅ You must not hallucinate any new logic or delete working features. Only apply what is listed here.
