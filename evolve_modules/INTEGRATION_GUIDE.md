# Evolve Platform — New Modules Integration Guide
## Built externally, ready for integration

---

## Files Included

```
evolve_modules/
├── trading/
│   ├── analysis/
│   │   ├── econometric_diagnostics.py   ← Full econometric test suite
│   │   ├── chart_pattern_detector.py    ← H&S, double top/bottom, triangles, S/R
│   │   ├── ml_score_trainer.py          ← True ML-based AI scoring
│   │   └── macro_factors.py             ← Fed/yield curve/VIX/credit macro signals
│   └── validation/
│       └── walk_forward_utils.py        ← Proper walk-forward validator (replaces stub)
├── utils/
│   └── risk_metrics.py                  ← VaR, CVaR, Kelly Criterion, Sharpe/Sortino/Calmar
└── agents/
    └── briefing/
        └── morning_briefing.py          ← Autonomous morning briefing pipeline
```

---

## Integration Instructions

### 1. walk_forward_utils.py
**Replace:** `trading/validation/walk_forward_utils.py` (current file is a stub)
**Wire into:** `pages/5_Backtest.py` Walk-Forward tab

```python
# In pages/5_Backtest.py, add a Walk-Forward tab:
from trading.validation.walk_forward_utils import WalkForwardValidator

with tab_walkforward:
    symbol = st.text_input("Symbol", "AAPL")
    model = st.selectbox("Model", ["xgboost", "ridge", "arima", "catboost"])
    if st.button("Run Walk-Forward Validation"):
        with st.spinner("Running validation..."):
            import yfinance as yf
            hist = yf.Ticker(symbol).history(period="2y")
            wfv = WalkForwardValidator(model_name=model, symbol=symbol)
            results = wfv.run(hist, train_window=252, test_window=63, step_size=21)
            summary = wfv.get_summary()
            st.json(summary)
            df = wfv.get_dataframe()
            if not df.empty:
                st.dataframe(df)
```

---

### 2. econometric_diagnostics.py
**Drop into:** `trading/analysis/econometric_diagnostics.py` (new file)
**Wire into:** `pages/2_Analyze.py` — add "Diagnostics" tab

```python
# In pages/2_Analyze.py, add to tab list:
from trading.analysis.econometric_diagnostics import EconometricDiagnostics

# Inside a "Diagnostics" tab:
diag = EconometricDiagnostics(symbol, hist)
diag.render_streamlit()  # Full UI rendered automatically
```

---

### 3. chart_pattern_detector.py
**Drop into:** `trading/analysis/chart_pattern_detector.py` (new file)
**Wire into:** `pages/2_Analyze.py` chart section

```python
from trading.analysis.chart_pattern_detector import ChartPatternDetector

detector = ChartPatternDetector(symbol, hist)
# Option A: Standalone UI
detector.render_streamlit()

# Option B: Overlay on existing Plotly chart
fig = detector.render_streamlit(fig=existing_fig)  # returns updated fig
```

---

### 4. ml_score_trainer.py
**Drop into:** `trading/analysis/ml_score_trainer.py` (new file)
**Wire into:** `trading/analysis/ai_score.py`

Two-step integration:

**Step A — Train the model (one-time, or scheduled weekly):**
```python
from trading.analysis.ml_score_trainer import MLScoreTrainer
trainer = MLScoreTrainer()
result = trainer.train(universe=["AAPL","MSFT","GOOGL",...])  # SP100 list
print(result)  # Shows accuracy metrics
```

**Step B — Add ML score to compute_ai_score() in ai_score.py:**
```python
# At the bottom of compute_ai_score(), before returning:
try:
    from trading.analysis.ml_score_trainer import MLScoreTrainer
    trainer = MLScoreTrainer()
    ml_result = trainer.predict(symbol, hist)
    if ml_result.get("ml_score") is not None:
        # Blend ML score with rules-based score (50/50)
        ml_score = ml_result["ml_score"]
        overall = round((overall + ml_score) / 2, 1)
        signals.append({
            "name": "ML Score",
            "value": ml_score,
            "impact": "positive" if ml_score > 6 else "negative" if ml_score < 4 else "neutral",
            "description": (
                f"ML model predicts {ml_result.get('predicted_7d_return', 0):+.1f}% "
                f"7-day return ({ml_result.get('direction', 'NEUTRAL')})"
            ),
        })
except Exception:
    pass  # Fall back to rules-based only
```

---

### 5. risk_metrics.py
**Drop into:** `utils/risk_metrics.py` (new file)
**Wire into:** `pages/4_Trade.py` risk section and `pages/5_Backtest.py`

```python
from utils.risk_metrics import (
    calculate_var, calculate_cvar,
    kelly_criterion, kelly_from_returns,
    compute_performance_metrics,
    render_risk_metrics_streamlit,
)

# Full dashboard in one call:
render_risk_metrics_streamlit(
    returns=portfolio_returns,
    symbol="Portfolio",
    portfolio_value=10000.0,
    benchmark_returns=spy_returns,
)

# Or individual calculations:
var = calculate_var(returns, confidence=0.95, portfolio_value=10000)
kelly = kelly_from_returns(returns)
metrics = compute_performance_metrics(returns, spy_returns)
print(metrics.to_dict())
print(f"Grade: {metrics.grade()}")
```

---

### 6. macro_factors.py
**Drop into:** `trading/analysis/macro_factors.py` (new file)
**Wire into:** `trading/analysis/ai_score.py` fundamental section

```python
# In ai_score.py fundamental section, after valuation overlay:
try:
    from trading.analysis.macro_factors import MacroFactors
    macro = MacroFactors()
    macro_adj = macro.get_ai_score_adjustment(sector=sector)
    fundamental_score = min(10.0, max(0.0,
        fundamental_score + macro_adj.get("score_adjustment", 0)
    ))
    signals.extend(macro_adj.get("signals", []))
except Exception:
    pass
```

---

### 7. morning_briefing.py
**Drop into:** `agents/briefing/morning_briefing.py` (new file, new directory)
**Wire into:** `pages/6_Chat.py` autonomous mode

```python
# In pages/6_Chat.py:
from agents.briefing.morning_briefing import MorningBriefing

# Add toggle in sidebar or top of page:
autonomous_mode = st.toggle("🤖 Autonomous Mode", key="autonomous_mode")

if autonomous_mode:
    briefing = MorningBriefing(universe="sp100", min_ai_score=6.5)
    briefing.render_streamlit()
else:
    # Normal chat interface
    pass
```

---

## Session Map — What This Replaces

| Original Session | Module Built Here | Cursor Integration Session |
|---|---|---|
| S57 (walk-forward) | walk_forward_utils.py | 1 session to wire into Backtest page |
| S65-S66 (ML score training) | ml_score_trainer.py | 1 session to wire into ai_score.py |
| S67 (ML score wiring) | Included in ml_score_trainer.py | Part of above session |
| S69 (morning briefing) | morning_briefing.py | 1 session to wire into Chat page |
| S72-S73 (econometrics) | econometric_diagnostics.py | 1 session to wire into Analyze page |
| S74 (factor model) | Covered by macro_factors.py | Part of macro session |
| S75 (VaR/CVaR) | risk_metrics.py | 1 session to wire into Trade page |
| S76 (Kelly) | Included in risk_metrics.py | Part of above session |
| S77 (performance metrics) | Included in risk_metrics.py | Part of above session |
| S83 (chart patterns) | chart_pattern_detector.py | 1 session to wire into Analyze page |
| S81 (macro factors) | macro_factors.py | 1 session to wire into AI Score |

**Sessions saved: ~15-18 sessions**
**Integration sessions needed: ~7 sessions**
**Net reduction: ~8-11 sessions**

---

## Notes

1. All files use the same logging pattern as existing codebase
2. All files use `_col_map = {c.lower(): c for c in data.columns}` for column normalization
3. All external calls are wrapped in try/except
4. All Streamlit rendering uses `render_streamlit()` methods for easy wiring
5. All files degrade gracefully — if a dependency is missing, they return empty state with a message
6. The ML Score trainer saves its model to `.cache/ml_score/` — add this to .gitignore
