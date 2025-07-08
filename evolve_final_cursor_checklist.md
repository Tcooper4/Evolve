
# ✅ Final Codebase Cleanup & Polish Checklist for Cursor

## 🔥 Critical Fixes
- [ ] Remove or archive all deprecated or legacy files:
  - `app.py`
  - `core/agents/chat_agent.py`
  - All contents of `/legacy/`
- [ ] Verify no `import` statements reference these files (check `__init__.py`, routers, test scripts, etc.)

## 🧹 Code Hygiene
- [ ] Remove all wildcard imports (e.g., `from module import *`)
- [ ] Replace all `print()` statements with proper logging (or remove entirely)
- [ ] Remove or resolve all `TODO`, `FIXME`, and `debug` comments
- [ ] Delete commented-out blocks of legacy or unused code

## 📄 Documentation
- [ ] Add docstrings to all public functions, classes, and key modules
- [ ] Add module-level docstrings where missing
- [ ] Use consistent formatting for docstrings (Google or NumPy style)

## 🧠 Logic Improvements
- [ ] Validate that model loading, signal generation, and backtesting all respect prompt routing
- [ ] Ensure all models (Prophet, XGBoost, LSTM, etc.) can be selected and weighted in hybrid ensemble dynamically
- [ ] Ensure strategy toggles (SMA, RSI, MACD, Bollinger) work without crashing on slider changes

## 🧪 Testing
- [ ] Create a `/tests` folder with:
  - [ ] Unit tests for each model
  - [ ] Strategy signal tests
  - [ ] Prompt routing tests
  - [ ] Backtest logic tests
- [ ] Aim for at least 80% test coverage

## 💻 UI & UX Polish
- [ ] Upgrade `unified_interface.py` to:
  - [ ] Use ChatGPT-style layout (sidebar input, chat-style logs, tabs for Forecast, Backtest, Strategy)
  - [ ] Add animation (Framer Motion or Streamlit-native)
  - [ ] Style buttons, cards, sliders with consistent rounded corners, padding, shadows
- [ ] Display:
  - [ ] Forecast plots with confidence intervals
  - [ ] Model weights and accuracy (MSE, Sharpe) side-by-side
  - [ ] Backtest summary stats (Sharpe, Return, Max Drawdown, Win Rate)
  - [ ] Buy/sell markers clearly on strategy charts
- [ ] Make all outputs interactive (zoom, hover)

## 🚀 Deployment Prep
- [ ] Ensure `.env` is respected for API keys
- [ ] Remove dev-only keys or test toggles
- [ ] Confirm correct entry point is `unified_interface.py`
- [ ] Add `Procfile` or `streamlit run` instructions for deployment
