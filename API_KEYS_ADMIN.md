# API Keys Section in Admin (API_KEYS_ADMIN)

## Overview

The **API Keys** area in **Admin → Configuration** (pages/11_Admin.py) now includes:

1. **AI Provider Keys** – LLM API keys at the top.  
2. **Trading** – Alpaca keys.  
3. **Data Provider API Keys** – Existing data-provider keys (Alpha Vantage, Finnhub, Polygon, etc.).

Keys are **never shown**; only whether each key is set is indicated (e.g. with a green check). Saving writes to the project `.env` via `python-dotenv`’s `set_key()`.

---

## 1. AI Provider Keys (top of API Keys)

Subsection: **AI Provider Keys**

| Env var | Label   | Provider (for Test) |
|---------|---------|----------------------|
| `ANTHROPIC_API_KEY` | Claude   | `claude` |
| `OPENAI_API_KEY`    | GPT-4    | `gpt4`   |
| `GOOGLE_API_KEY`    | Gemini   | `gemini` |
| `MOONSHOT_API_KEY`  | Kimi     | `kimi`   |
| `HUGGINGFACE_API_KEY` | HuggingFace | `huggingface` |

For each key:

- **Status** – Green check (✅) if set in environment (or `HF_TOKEN` for HuggingFace); otherwise “—”.  
- **Input** – Password field, no value shown; placeholder `"•••••••• (set)"` if set, else `"Enter key..."`.  
- **Save** – Writes the current input to `.env` with `dotenv.set_key()` and reloads env with `load_dotenv(..., override=True)`.  
- **Test** – Calls `get_provider_status(provider)` from `agents/llm/active_llm_calls.py` and shows success/warning (no key values displayed).

If `python-dotenv` is not available, a note is shown and Save is ineffective.

---

## 2. Trading subsection

Subsection: **Trading**

| Env var | Label              |
|---------|--------------------|
| `ALPACA_API_KEY`    | Alpaca API Key    |
| `ALPACA_SECRET_KEY` | Alpaca Secret Key |

Same pattern: status (set/not set), masked password input, Save to `.env`. No Test button (no provider-status helper for Alpaca in the same way).

---

## 3. Data Provider API Keys

Existing **Data Provider API Keys** block is unchanged in structure (Alpha Vantage, Finnhub, Polygon, OpenAI, NewsAPI, Reddit). The new **AI Provider Keys** and **Trading** blocks were added **above** it so LLM and trading keys are at the top.

---

## 4. Implementation details

- **`.env` path** – `Path(__file__).resolve().parent.parent / ".env"` (project root), using `_project_root` already defined at top of 11_Admin.py.  
- **Never display values** – Inputs for AI and Trading keys use `value=""`; placeholders indicate “set” vs “enter key”.  
- **Save** – `from dotenv import set_key, load_dotenv`; `set_key(str(_env_path), env_var, new_val)` then `load_dotenv(_env_path, override=True)` so the process sees the new key.  
- **Test** – `from agents.llm.active_llm_calls import get_provider_status`; `get_provider_status(provider)`; result shown as success/warning text only.

---

## 5. Anthropic fallback (home briefing)

In **trading/services/home_briefing_service.py**, the briefing no longer fails when `anthropic` is missing:

- **Try** – `from anthropic import Anthropic`.  
- **If ImportError** – Skip Claude call and return a **fallback briefing** that:  
  - States that the AI briefing is unavailable and instructs to install `anthropic` and set `ANTHROPIC_API_KEY`.  
  - Still includes **memory context**, **market data**, and **risk snapshot** as plain text (truncated where needed).  
  - Returns a single card suggesting to set up the API key.

So the Home page always gets a useful response (market data + memory summary) even without the Anthropic API.

---

## Files touched

| File | Change |
|------|--------|
| **pages/11_Admin.py** | New “AI Provider Keys” and “Trading” subsections at top of API Keys expander; password inputs, status, Save via set_key, Test via get_provider_status. |
| **trading/services/home_briefing_service.py** | Try/except around `anthropic` import; fallback briefing with context when Anthropic is not installed. |
