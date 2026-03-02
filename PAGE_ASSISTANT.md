# Page-Aware AI Assistant

Lightweight sidebar component that lets users ask the active LLM about the current page. Each page has its own assistant (own session state keys), so they do not interfere with each other or with the main Chat.

## Components

### 1. `trading/services/page_assistant_service.py`

- **`get_page_context(page_name: str, session_state: Any) -> str`**
  - Returns a short context string describing what the user is currently looking at on that page.
  - Used to build the system prompt for the page assistant.
  - Output is kept concise (~1500 chars) to avoid token overflow.

**Page-specific context:**

| Page               | Context summary |
|--------------------|-----------------|
| **Strategy Testing** | Loaded data (symbol, rows), recent backtest (strategy name, total return, Sharpe) from `backtest_results` / `backtest_strategy` / `loaded_data` / `backtest_symbol`. |
| **Risk Management**  | Current risk snapshot from `risk_manager.current_metrics` (VaR95, volatility, max drawdown, Sharpe), `risk_limits`, and length of `risk_history`. |
| **Portfolio**        | Position summary from `portfolio_manager.get_position_summary()` or `get_all_positions()` (open/total counts, symbols). |
| **Forecasting**      | `symbol`, `forecast_data` row count, `forecast_horizon`, `current_model`, and whether `current_forecast_result` exists (last forecast run). |

If a page is not recognized or an error occurs, returns a generic line: `"User is on the {page_name} page. No additional context available."`

### 2. `ui/page_assistant.py`

- **`render_page_assistant(page_name: str) -> None`**
  - Renders an expander at the bottom of the page: **"💬 Ask AI about this page"**.
  - Contains:
    - A text input (placeholder: e.g. "What does this backtest result mean?").
    - A **Submit** button.
  - On submit:
    1. Calls `get_page_context(page_name, st.session_state)` to build context.
    2. Builds system prompt: *"You are an expert trading assistant. The user is currently on the [page_name] page. Here is their current context: [context]. Answer their question concisely in plain English."*
    3. Calls `call_active_llm_chat(system_prompt, context_block="", conversation_messages=[], user_message=user_question.strip(), max_tokens=1024)` from `agents.llm.active_llm_calls`.
    4. Stores the response in session state and displays it under **Response:**.
    5. Clears the input and reruns so the response is visible.

**Session state keys (per page):**

- Prefix: `page_assistant_<page_slug>` where `<page_slug>` is the page name with spaces → `_` and `&` → `and`.
- Keys:
  - `{prefix}_input`: last input text (cleared after submit).
  - `{prefix}_response`: last AI response text.

Examples: `page_assistant_Forecasting_input`, `page_assistant_Strategy_Testing_response`, `page_assistant_Risk_Management_response`, `page_assistant_Portfolio_response`. Main Chat does not use these keys.

## Integration

Each of the four pages calls the assistant at the **bottom** of the script:

| Page file                    | Import | Call |
|-----------------------------|--------|------|
| `pages/2_Forecasting.py`    | `from ui.page_assistant import render_page_assistant` | `render_page_assistant("Forecasting")` |
| `pages/3_Strategy_Testing.py` | same | `render_page_assistant("Strategy Testing")` |
| `pages/6_Risk_Management.py` | same | `render_page_assistant("Risk Management")` |
| `pages/5_Portfolio.py`      | same | `render_page_assistant("Portfolio")` |

No changes to trading logic; only the assistant UI and context wiring were added.

## Dependencies

- **Streamlit** (e.g. 1.28+ for `st.rerun()`).
- **Active LLM**: same as main app (`config.llm_config`, `agents.llm.active_llm_calls.call_active_llm_chat`). If the LLM is not configured or the call fails, the component shows an error message.

## Usage

1. Open any of the four pages (Forecasting, Strategy Testing, Risk Management, Portfolio).
2. Scroll to the bottom and expand **"💬 Ask AI about this page"**.
3. Type a question (e.g. "What does this backtest result mean?", "Why is my VaR so high?").
4. Click **Submit**. The assistant uses the current page context and the active LLM to answer in plain English.
