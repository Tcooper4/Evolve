# yfinance Parameter Fixes (`threads`, `proxy`, and `progress`)

## Issue

The errors `PriceHistory.history() got an unexpected keyword argument 'threads'`, `PriceHistory.history() got an unexpected keyword argument 'proxy'`, and `PriceHistory.history() got an unexpected keyword argument 'progress'` occurred because newer versions of yfinance have removed these parameters from the `history()` method.

## Root Cause

In `trading/data/data_loader.py`, the code was passing `threads=request.threads`, `proxy=request.proxy`, and `progress=request.progress` to `ticker_obj.history()`, but these parameters are no longer supported in recent versions of yfinance.

**Pattern:** Newer yfinance versions have removed several parameters that were previously supported. The current supported parameters are:
- `period` or `start`/`end` (date range)
- `interval` (data frequency)
- `auto_adjust` (adjust for splits/dividends)
- `prepost` (include pre/post market data)
- `actions` (include dividends/splits)
- `repair` (repair missing data)
- `timeout` (request timeout)
- `raise_errors` (error handling)

## Fix Applied

Removed the `threads`, `proxy`, and `progress` parameters from both `history()` calls in `trading/data/data_loader.py`:

**Before:**
```python
data = ticker_obj.history(
    start=request.start_date,
    end=request.end_date,
    interval=request.interval,
    auto_adjust=request.auto_adjust,
    prepost=request.prepost,
    threads=request.threads,   # ❌ Removed
    proxy=request.proxy,        # ❌ Removed
    progress=request.progress,  # ❌ Removed
)
```

**After:**
```python
data = ticker_obj.history(
    start=request.start_date,
    end=request.end_date,
    interval=request.interval,
    auto_adjust=request.auto_adjust,
    prepost=request.prepost,
)
```

## Impact

- ✅ Data loading now works with current versions of yfinance
- ✅ The `threads`, `proxy`, and `progress` fields in `DataLoadRequest` are still present but no longer used (for backward compatibility)
- ✅ No functional impact:
  - yfinance handles threading internally
  - Proxy configuration should be set via environment variables (`HTTP_PROXY`/`HTTPS_PROXY`) if needed
  - Progress indication is no longer available but data loading still works

## Verification

The fix has been applied to:
- Line 479-485: `history()` call with start/end dates (removed `threads`, `proxy`, and `progress`)
- Line 488-494: `history()` call with period (removed `threads`, `proxy`, and `progress`)

Both calls now work correctly with current yfinance versions.

## Other Files Checked

Checked all other `history()` calls in the codebase:
- ✅ `trading/data/providers/yfinance_provider.py` - Uses only `start`, `end`, `interval` (valid)
- ✅ `fallback/data_feed.py` - Uses only `start`, `end`, `interval` (valid)
- ✅ `data/streaming_pipeline.py` - Uses only `start`, `end`, `interval` (valid)
- ✅ `data/live_data_feed.py` - Uses only `start`, `end`, `period`, `interval` (valid)
- ✅ `pages/6_Performance.py` - Uses only `period` (valid)
- ✅ `trading/agents/market_regime_agent.py` - Uses only `period` (valid)
- ✅ `trading/data/data_loader.py` line 634 - Uses only `period` (valid)

All other `history()` calls in the codebase are using only supported parameters.

## Note on Proxy Configuration

If proxy support is needed, configure it via environment variables:
- `HTTP_PROXY` or `HTTPS_PROXY` - yfinance will use these automatically
- Or configure at the system/network level

