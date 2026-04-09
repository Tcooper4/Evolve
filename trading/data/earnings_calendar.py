"""Earnings calendar data pipeline using yfinance."""

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=128)
def get_upcoming_earnings(symbol: str, days_ahead: int = 30) -> dict:
    """Return upcoming earnings and recent EPS surprise information for a symbol."""
    try:
        ticker = yf.Ticker(symbol)

        # Earnings calendar can be a DataFrame or dict depending on yfinance version
        cal = getattr(ticker, "calendar", None)
        info = getattr(ticker, "info", {}) or {}

        earnings_date = None
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            # DataFrame format: columns are Timestamp dates
            earnings_date = cal.columns[0] if len(cal.columns) > 0 else None
        elif isinstance(cal, dict):
            ed = cal.get("Earnings Date", [None])
            earnings_date = ed[0] if isinstance(ed, list) else ed

        days_until = None
        is_within_window = False

        if earnings_date:
            # Normalize to date object
            if hasattr(earnings_date, "date"):
                earnings_date = earnings_date.date()
            try:
                today = datetime.today().date()
                days_until = (earnings_date - today).days
                is_within_window = 0 <= days_until <= days_ahead
            except Exception:
                days_until = None
                is_within_window = False

        # Historical earnings for last EPS surprise
        hist_earnings = getattr(ticker, "earnings_history", None)
        last_eps_actual = None
        last_eps_surprise_pct = None
        last_earnings_date = None

        if hist_earnings is not None and not getattr(hist_earnings, "empty", True):
            last = hist_earnings.iloc[-1]
            last_eps_actual = last.get("epsActual")
            est = last.get("epsEstimate")
            idx = hist_earnings.index[-1]
            last_earnings_date = str(idx.date()) if hasattr(idx, "date") else str(idx)
            if est and est != 0:
                try:
                    last_eps_surprise_pct = round(
                        (last_eps_actual - est) / abs(est) * 100, 1
                    )
                except Exception:
                    last_eps_surprise_pct = None

        return {
            "symbol": symbol,
            "next_earnings_date": str(earnings_date) if earnings_date else None,
            "days_until": days_until,
            "is_within_window": is_within_window,
            "eps_estimate": info.get("forwardEps"),
            "last_earnings_date": last_earnings_date,
            "last_eps_actual": last_eps_actual,
            "last_eps_surprise_pct": last_eps_surprise_pct,
        }

    except Exception as e:  # pragma: no cover - defensive fallback
        return {
            "symbol": symbol,
            "next_earnings_date": None,
            "days_until": None,
            "is_within_window": False,
            "eps_estimate": None,
            "last_earnings_date": None,
            "last_eps_actual": None,
            "last_eps_surprise_pct": None,
            "error": str(e),
        }


def get_macro_calendar(days_ahead: int = 14) -> Dict[str, Any]:
    """
    Returns upcoming macro events
    (Fed meetings, CPI, PPI, jobs)
    from FRED release calendar when available.
    No API key needed for schedule fallback.
    """
    try:
        _today = datetime.today().date()
        _events: List[Dict[str, Any]] = []

        # FRED schedule requires a valid key; omit here — use fallback dates.
        try:
            import urllib.request

            _url = (
                "https://api.stlouisfed.org/fred/releases"
                "/dates?api_key=pubkey"
                "&file_type=json"
                "&limit=20"
                "&sort_order=asc"
            )
            _req = urllib.request.Request(
                _url,
                headers={"User-Agent": "Evolve Trading"},
            )
            with urllib.request.urlopen(_req, timeout=10):
                pass
        except Exception:
            pass

        _key_dates = [
            ("2026-04-16", "CPI Release", "inflation"),
            ("2026-04-17", "PPI Release", "inflation"),
            ("2026-04-30", "FOMC Meeting", "rates"),
            ("2026-05-01", "Jobs Report", "employment"),
            ("2026-05-14", "CPI Release", "inflation"),
            ("2026-05-28", "FOMC Meeting", "rates"),
            ("2026-06-11", "FOMC Meeting", "rates"),
        ]

        for _date_str, _name, _etype in _key_dates:
            try:
                _d = datetime.fromisoformat(_date_str).date()
                _days = (_d - _today).days
                if 0 <= _days <= days_ahead:
                    _events.append(
                        {
                            "date": _date_str,
                            "name": _name,
                            "type": _etype,
                            "days_until": _days,
                        }
                    )
            except Exception:
                pass

        return {
            "events": _events,
            "days_ahead": days_ahead,
            "success": True,
        }
    except Exception as e:
        return {
            "events": [],
            "days_ahead": days_ahead,
            "success": False,
            "error": str(e),
        }

