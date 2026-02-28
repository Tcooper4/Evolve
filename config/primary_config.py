"""
Primary configuration entry point (single source of truth).

P3 fix (AUDIT_REPORT.md 3.1): Import config from here so one loader owns
application and trading settings. See config/CONFIG_README.md.
"""

from typing import Optional

from config.app_config import AppConfig, get_config

__all__ = ["get_config", "get_primary_config", "AppConfig"]


def get_primary_config() -> AppConfig:
    """
    Return the primary application config (single source of truth).
    Same as get_config(); name exists for clarity in docs.
    """
    return get_config()


def get_trading_mode_from_env() -> Optional[str]:
    """
    Read trading mode from env for display/documentation only.
    Do NOT use to override constructor args; config/code is source of truth.
    Returns: "paper" | "live" | None (None = use config/code).
    """
    import os
    v = os.getenv("TRADING_MODE") or os.getenv("LIVE_TRADING")
    if not v:
        return None
    v = v.strip().lower()
    if v in ("true", "1", "live"):
        return "live"
    if v in ("false", "0", "paper"):
        return "paper"
    return None
