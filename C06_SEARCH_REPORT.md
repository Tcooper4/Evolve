# C06: Multi-Asset Portfolio Support - Search Report

**Date:** 2024-12-19
**Status:** SEARCH COMPLETE - Ready for Implementation

---

## 🔍 SEARCH RESULTS

### Files Analyzed:
1. `portfolio/__init__.py` - Module exports
2. `portfolio/allocator.py` - Portfolio allocation strategies
3. `portfolio/risk_manager.py` - Risk management
4. `trading/portfolio/portfolio_manager.py` - Main portfolio manager

---

## 📋 FINDINGS

### 1. **Current Single-Symbol Assumptions:**

#### `trading/portfolio/portfolio_manager.py`:
- **Line 227-238:** `open_position()` method takes `symbol: str` (single string)
- **Line 54-72:** `Position` dataclass has `symbol: str` (single symbol per position)
- **Line 343-422:** `update_positions()` accepts `prices: Dict[str, float]` - can handle multiple symbols but not initialized as a portfolio
- **Line 424-437:** `_calculate_position_size()` takes `symbol: str` (single symbol)

**Current Behavior:**
- Portfolio can have multiple positions (multiple symbols)
- But each position is tracked individually
- No portfolio-level initialization with symbol list
- No portfolio-level allocation across assets
- No correlation matrix for multi-asset optimization

#### `portfolio/allocator.py`:
- **Line 111-116:** `allocate_portfolio()` takes `assets: List[AssetMetrics]` - **ALREADY SUPPORTS MULTIPLE ASSETS**
- Uses ticker/symbol from AssetMetrics objects
- Already has multi-asset allocation strategies

#### `portfolio/risk_manager.py`:
- **Line 343-422:** `calculate_portfolio_risk()` takes `weights: Dict[str, float]` - **ALREADY SUPPORTS MULTIPLE ASSETS**
- **Line 386-471:** `generate_rebalancing_actions()` works with `Dict[str, float]` positions

---

## 🎯 IMPLEMENTATION REQUIREMENTS

### What Needs to Change:

1. **PortfolioManager.__init__()** - Add `symbols: List[str]` parameter
2. **PortfolioManager** - Add `self.symbols: List[str]` attribute
3. **PortfolioManager** - Add `initialize_portfolio(symbols: List[str])` method
4. **PortfolioManager** - Add `get_symbols() -> List[str]` method
5. **PortfolioManager** - Add `get_all_positions() -> Dict[str, List[Position]]` method (grouped by symbol)
6. **PortfolioManager** - Add correlation matrix calculation
7. **PortfolioManager** - Add portfolio-level allocation logic

### What Already Works:
- ✅ `PortfolioAllocator` already supports multiple assets
- ✅ `PortfolioRiskManager` already supports multiple assets
- ✅ Position tracking can handle multiple symbols
- ✅ `update_positions()` accepts multiple prices

---

## 📝 SPECIFIC CODE LOCATIONS

### File: `trading/portfolio/portfolio_manager.py`

**Line 150-195:** `__init__()` method
- Currently: No symbol initialization
- Needs: `symbols: Optional[List[str]] = None` parameter
- Needs: `self.symbols: List[str] = []` attribute

**Line 227-298:** `open_position()` method
- Currently: `symbol: str` (single)
- Status: OK - can be called multiple times for different symbols
- No change needed

**Line 343-422:** `update_positions()` method
- Currently: `prices: Dict[str, float]` (already multi-symbol)
- Status: OK - already supports multiple symbols
- No change needed

**NEW METHODS NEEDED:**
- `initialize_portfolio(symbols: List[str], initial_capital: float) -> None`
- `get_symbols() -> List[str]`
- `get_all_positions() -> Dict[str, List[Position]]`
- `calculate_correlation_matrix() -> pd.DataFrame`
- `get_portfolio_allocation() -> Dict[str, float]`

---

## ✅ IMPLEMENTATION STRATEGY

1. Add `symbols` attribute to `PortfolioManager`
2. Add `initialize_portfolio()` method
3. Add helper methods for multi-asset operations
4. Integrate with existing `PortfolioAllocator` for allocation
5. Add correlation matrix calculation
6. Update position tracking to support portfolio view

---

## 🧪 TEST SCENARIO

```python
from trading.portfolio.portfolio_manager import PortfolioManager

pm = PortfolioManager()
symbols = ['AAPL', 'MSFT', 'GOOGL']
pm.initialize_portfolio(symbols=symbols, initial_capital=100000)

# Test multi-asset initialization
assert pm.get_symbols() == ['AAPL', 'MSFT', 'GOOGL']

# Test position tracking
positions = pm.get_all_positions()
assert isinstance(positions, dict)
assert all(symbol in positions for symbol in symbols)

# Test correlation matrix
corr_matrix = pm.calculate_correlation_matrix()
assert corr_matrix.shape == (3, 3)
```

---

## 📊 SUMMARY

**Current State:**
- System can handle multiple positions (multiple symbols)
- But lacks portfolio-level initialization
- No portfolio-level allocation management
- No correlation matrix calculation

**Required Changes:**
- Add portfolio initialization with symbol list
- Add portfolio-level methods
- Add correlation matrix calculation
- Integrate with existing allocator

**Complexity:** MEDIUM
**Breaking Changes:** None (backward compatible)
**Files to Modify:** 1 file (`trading/portfolio/portfolio_manager.py`)

---

**READY FOR IMPLEMENTATION** ✅

