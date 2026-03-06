# Per-user persistent storage (beta)

Per-user API keys and session data for a small beta user set. Each user has their own keys and preferences stored securely.

## Overview

- **Storage**: SQLite DB at `data/users.db` (one row per `session_id`).
- **Secrets**: API keys are encrypted at rest with Fernet (key from `EVOLVE_ENCRYPTION_KEY` in `.env`).
- **Session identity**: A unique `session_id` (e.g. `secrets.token_hex(16)`) identifies a user; it is persisted in Streamlit `st.session_state`, URL query param `?sid=...`, and optionally browser `localStorage` via `evolve_session_id`.

## Files

| File | Purpose |
|------|--------|
| `config/user_store.py` | DB init, encrypt/decrypt helpers, `save_user_keys` / `load_user_keys`, `save_user_preferences` / `load_user_preferences`. |
| `components/onboarding.py` | Streamlit onboarding: form for OpenAI/Anthropic/News API keys and preferred LLM; persist `session_id`; "Reset my keys" to show form again. |
| `app.py` | Calls `init_user_db()`, `check_onboarding()`; if onboarding complete, loads user keys and injects them into `os.environ` for the session. |
| `requirements_deploy.txt` | Adds `cryptography` for Fernet. |

## Security

- **Encryption**: Fernet symmetric encryption. Key is in `.env` as `EVOLVE_ENCRYPTION_KEY`. If missing on first run, a key is generated and appended to `.env`.
- **DB**: `data/users.db` (and `-shm` / `-wal`) are in `.gitignore` and must never be committed.
- **Env**: Do not commit `.env` or `EVOLVE_ENCRYPTION_KEY`; keep them local or in a secure deploy secret store.

## Flow

1. **First visit**: No `session_id` → one is generated, stored in session state and URL (`?sid=...`), and written to `localStorage` via an HTML snippet. Onboarding form is shown (OpenAI key required; Anthropic and News optional; preferred LLM: openai/anthropic).
2. **Submit**: Keys and preferences are saved via `save_user_keys` / `save_user_preferences`. Success message: "Your keys are saved. You won't need to enter them again on this device."
3. **Later visits**: Same browser/URL with `?sid=...` (or `localStorage` restored) → `load_user_keys(session_id)` returns stored keys → app injects them into `os.environ` and continues without showing the form.
4. **Reset**: "Reset my keys" clears the onboarding state and shows the form again; resubmitting overwrites stored keys.

## Dependencies

- `cryptography` (see `requirements_deploy.txt`). Install:  
  `pip install -r requirements_deploy.txt`  
  (On some Linux system Pythons you may need `--break-system-packages`.)

## Compile check

```bash
py -3.10 -m py_compile config/user_store.py components/onboarding.py
```

Both modules compile successfully.
