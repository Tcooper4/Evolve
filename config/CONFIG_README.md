# Configuration — Single Source of Truth

**P3 fix (AUDIT_REPORT.md 3.1):** This document defines the single source of truth for application and trading configuration.

## Preferred entry point

- **Use `config.app_config.get_config()`** (or `from config import get_config`) for all application settings. It loads YAML from `config/` and overrides with environment variables.
- **Use `config.Config`** for a simplified property-based interface over the same `AppConfig` instance.

```python
from config.app_config import get_config
config = get_config()
# config.server, config.logging, config.database, config.strategies, config.risk, etc.
```

## Trading mode (paper vs live)

- **Execution / broker:** Paper vs live is determined by:
  - **`execution.live_trading_interface.LiveTradingInterface(mode=...)`**: `mode="simulated"` | `"paper"` | `"live"`. Use `"live"` only for real money; `"paper"` for Alpaca paper.
  - **`execution.broker_adapter`**: Set config key `"paper": True` or `"paper": False` explicitly (do not infer from URL).
- **Env:** Optional override for some settings (e.g. `LOG_LEVEL`, `REDIS_HOST`). See `config.app_config.AppConfig._apply_env_overrides()` and `env.example` for supported variables.
- **Do not** rely on `LIVE_TRADING` env to flip behavior when code has explicitly set paper/live; constructor/config is the source of truth (see P1 fixes).

## Database and Redis

- **Database URL:** Use `trading.database.connection.get_database_url()` (reads `DB_*`, `SQLITE_PATH` from env). Same process should use the same config; avoid mixing with other config loaders for DB.
- **Shutdown:** Call `trading.database.connection.close_database()` on application shutdown so connections are closed cleanly (see P4.2).

## Other config modules

- **`config/config_loader.py`**: Loads and validates YAML sections; use for advanced validation or reload. Prefer `get_config()` for normal runtime config.
- **`config/settings.py`**: Legacy/env-only settings; new code should prefer `get_config()`.
- **`trading/config/*`**: Trading-specific overrides; import from root `config` when possible so one loader owns the source of truth.

## Env vars (summary)

See `env.example` for the full list. Key ones for trading/live:

| Variable        | Purpose                          | Live vs paper |
|----------------|-----------------------------------|---------------|
| `LIVE_TRADING` | Not used to override constructor | P1: ignored   |
| Alpaca config  | `paper: True/False` in config     | Set explicitly |
| `DB_*`         | PostgreSQL connection            | Same for all  |
| `SQLITE_PATH`  | SQLite path when no DB password   | Same for all  |
