# Home Page Redesign & Page Assistant Updates

## 1. Home Page Redesign (pages/0_Home.py)

### Changes made

- **Metric row at top:** Added a row of `st.metric()` tiles for SPY and AAPL showing current price and weekly change % (delta). Uses `market_data` from the briefing service; only rendered when market data is available.
- **Charts with date axis:** The briefing service (`trading/services/home_briefing_service.py`) now includes a `dates` list in each symbol’s `market_data` (formatted as `"Mar 1"`, `"Feb 28"`, etc. from the provider’s DataFrame index). The Home page builds the line chart from a DataFrame whose index is these dates so the x-axis shows real dates instead of 0.0, 1.0, 2.0.
- **Briefing text container:** The main briefing text is wrapped in `st.container(border=True)` for a bordered, card-like block.
- **Cards:** Replaced plain expanders with `st.container(border=True)` cards: bold headline, detail as caption (no expander), and optional price chart with date axis when `dates` are present.
- **Follow-up input:** Replaced the small caption with a clear label: **"Ask the AI anything about your portfolio or the markets:"** and kept the text input below it.
- **Page assistant:** Added `render_page_assistant("Home")` at the bottom of the page (sidebar).

### Files touched

- `pages/0_Home.py` — metrics, briefing container, cards with border, chart index from `dates`, follow-up label, page assistant.
- `trading/services/home_briefing_service.py` — `_get_market_data()` now adds `dates` (and keeps `series`) per symbol for chart axes.

---

## 2. Page Assistant: Sidebar Chat Panel (ui/page_assistant.py)

### Changes made

- **Sidebar placement:** The assistant is rendered in `st.sidebar` inside an expander **"💬 Page Assistant"** with `expanded=True` so it stays open by default.
- **Conversation history:** Each page’s assistant keeps a list of `{role, content}` in `st.session_state` under `{prefix}_history`. Messages are shown above the input.
- **Checkbox "Allow AI to suggest changes":** When checked, the system prompt includes: *"You may suggest specific UI actions the user can take on this page."* When unchecked: *"Provide explanations and analysis only, do not suggest making changes."* State is stored in `{prefix}_allow_suggestions`.
- **Send button:** Replaced "Submit" with "Send". On send, the user message and AI response are appended to history; the input is cleared and the app reruns.
- **Same backend:** Still uses `call_active_llm_chat()` from `agents.llm.active_llm_calls` and `get_page_context()` / `get_full_context_summary()` from `trading.services.page_assistant_service`.

### Session state keys (per page)

- `page_assistant_{slug}_history` — list of `{"role": "user"|"assistant", "content": str}`.
- `page_assistant_{slug}_allow_suggestions` — bool.
- `page_assistant_{slug}_input` — current input text.
- `page_assistant_{slug}_submit` — button key.

---

## 3. Page Assistant Context for New Pages (trading/services/page_assistant_service.py)

Added context helpers and routing for:

- **Chat** — `_chat_context()`: recent conversation length from `messages` / `chat_messages`.
- **Trade Execution** — `_trade_execution_context()`: execution mode and pending orders count.
- **Performance** — `_performance_context()`: strategy performance metrics / performance data loaded.
- **Model Lab** — `_model_lab_context()`: current/selected model and whether training is in progress.
- **Home** — `_home_context()`: briefing loaded and number of cards.

`get_page_context()` now dispatches on these page names and returns the corresponding context string.

---

## 4. Page Assistant Wiring on Additional Pages

`render_page_assistant(page_name)` is called at the end of:

| Page file                 | Page name passed     |
|---------------------------|----------------------|
| pages/1_Chat.py           | `"Chat"`             |
| pages/4_Trade_Execution.py| `"Trade Execution"`   |
| pages/7_Performance.py    | `"Performance"`       |
| pages/8_Model_Lab.py      | `"Model Lab"`        |
| pages/0_Home.py           | `"Home"`             |

Existing pages (Forecasting, Strategy Testing, Risk Management, Portfolio) were already calling `render_page_assistant()`; they now use the same sidebar expander implementation.

---

## Summary

- **Home:** Metric row (SPY/AAPL), date-axis charts, bordered briefing and cards, clearer follow-up label, sidebar assistant.
- **Page assistant:** Sidebar expander, conversation history, "Allow AI to suggest changes" checkbox, same LLM and context services.
- **Context:** New helpers for Chat, Trade Execution, Performance, Model Lab, and Home.
- **Wiring:** Assistant added to Chat, Trade Execution, Performance, Model Lab, and Home.
