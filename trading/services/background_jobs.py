# -*- coding: utf-8 -*-
"""Server-side background loop for paper limit fills and alert checks.

Correctness: fills/alerts must evaluate even with zero browsers open.
Notification is secondary (NotificationHub). Kill switch:
``EVOLVE_BACKGROUND_JOBS=0|false|off|no``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Env kill switch (default ON). Conservative interval matches delayed quotes.
DEFAULT_INTERVAL_SEC = 45.0
_KILL_VALUES = {"0", "false", "off", "no", "disabled"}

_task: Optional[asyncio.Task] = None
_stop: Optional[asyncio.Event] = None


def background_jobs_enabled() -> bool:
    raw = (os.getenv("EVOLVE_BACKGROUND_JOBS", "1") or "1").strip().lower()
    return raw not in _KILL_VALUES


def is_us_equity_session_open(now: Optional[datetime] = None) -> bool:
    """Weekday + MarketHours (incl. extended sessions when configured)."""
    try:
        import pytz
        from trading.utils.time_utils import MarketHours

        tz = pytz.timezone("America/New_York")
        dt = now.astimezone(tz) if now is not None else datetime.now(tz)
        if dt.weekday() >= 5:
            return False
        return MarketHours(timezone="America/New_York").is_market_open(dt)
    except Exception as e:
        logger.debug("is_us_equity_session_open failed, treating closed: %s", e)
        return False


def username_from_session_id(session_id: str) -> str:
    sid = (session_id or "").strip()
    if sid.startswith("user:"):
        return sid[5:]
    return sid


def collect_target_session_ids() -> List[str]:
    """Users with open limits and/or saved alerts (union, stable order)."""
    seen: Set[str] = set()
    out: List[str] = []

    try:
        from trading.portfolio.paper_portfolio import list_users_with_open_limit_orders

        for uid in list_users_with_open_limit_orders():
            if uid and uid not in seen:
                seen.add(uid)
                out.append(uid)
    except Exception as e:
        logger.warning("background_jobs: open-order scan failed: %s", e)

    try:
        from config.user_store import list_session_ids_with_alerts

        for sid in list_session_ids_with_alerts():
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    except Exception as e:
        logger.warning("background_jobs: alert-user scan failed: %s", e)

    return out


def run_limit_checks_for_user(user_id: str) -> List[Dict[str, Any]]:
    from trading.portfolio.paper_portfolio import PaperPortfolio

    return PaperPortfolio(user_id=user_id).check_limit_orders() or []


def run_alert_checks_for_user(session_id: str) -> List[Dict[str, Any]]:
    from trading.services.alert_checker import check_alerts_for_user

    return check_alerts_for_user(session_id) or []


async def _publish(username: str, payload: Dict[str, Any]) -> None:
    try:
        from trading.services.notification_hub import notification_hub

        await notification_hub.publish(username, payload)
    except Exception as e:
        logger.debug("background_jobs: notify %s failed: %s", username, e)


async def background_tick() -> Dict[str, int]:
    """One scan cycle. Safe to call from tests (no market-hours gate here)."""
    fills_n = 0
    alerts_n = 0
    targets = collect_target_session_ids()
    if not targets:
        return {"users": 0, "fills": 0, "alerts": 0}

    for session_id in targets:
        username = username_from_session_id(session_id)
        try:
            filled = await asyncio.to_thread(run_limit_checks_for_user, session_id)
            for order in filled:
                fills_n += 1
                await _publish(username, {
                    "type": "limit_fill",
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "filled_price": order.get("filled_price"),
                    "order_id": order.get("id"),
                    "message": (
                        f"Limit {order.get('side')} {order.get('symbol')} "
                        f"filled @ {order.get('filled_price')}"
                    ),
                })
        except Exception as e:
            logger.warning("background_jobs: limits %s: %s", session_id, e)

        try:
            from trading.services.alert_push_policy import (
                should_push_alert_notification,
            )

            triggered = await asyncio.to_thread(
                run_alert_checks_for_user, session_id
            )
            for row in triggered:
                # Execution already happened inside check_alerts_for_user
                # (one-shot stamp). Push is a separate, optional channel.
                alerts_n += 1
                do_push, reason = should_push_alert_notification(username, row)
                if not do_push:
                    logger.debug(
                        "background_jobs: alert push skipped (%s) %s %s",
                        reason, username, row.get("symbol"),
                    )
                    continue
                await _publish(username, {
                    "type": "alert_trigger",
                    "symbol": row.get("symbol"),
                    "condition": row.get("condition"),
                    "threshold": row.get("threshold"),
                    "alert_id": row.get("alert_id"),
                    "current_price": row.get("current_price"),
                    "mode": row.get("mode") or "action",
                    "message": (
                        f"Alert {row.get('symbol')} "
                        f"{row.get('condition')} {row.get('threshold')}"
                    ),
                })
        except Exception as e:
            logger.warning("background_jobs: alerts %s: %s", session_id, e)

    return {"users": len(targets), "fills": fills_n, "alerts": alerts_n}


async def _loop(interval_sec: float) -> None:
    assert _stop is not None
    logger.info(
        "background_jobs: started (interval=%.0fs, kill=EVOLVE_BACKGROUND_JOBS=0)",
        interval_sec,
    )
    # Also emit on uvicorn's logger so Docker logs always show the start line.
    logging.getLogger("uvicorn.error").info(
        "background_jobs: started (interval=%.0fs)", interval_sec
    )
    while not _stop.is_set():
        try:
            if is_us_equity_session_open():
                stats = await background_tick()
                if stats["fills"] or stats["alerts"]:
                    logger.info(
                        "background_jobs: tick users=%s fills=%s alerts=%s",
                        stats["users"], stats["fills"], stats["alerts"],
                    )
            else:
                logger.debug("background_jobs: market closed — skip tick")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("background_jobs: tick error: %s", e)
        try:
            await asyncio.wait_for(_stop.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass
    logger.info("background_jobs: stopped")


async def start_background_jobs(
    interval_sec: float = DEFAULT_INTERVAL_SEC,
) -> None:
    """Start the single global loop (idempotent per process)."""
    global _task, _stop
    if not background_jobs_enabled():
        logger.info(
            "background_jobs: disabled via EVOLVE_BACKGROUND_JOBS=%s",
            os.getenv("EVOLVE_BACKGROUND_JOBS"),
        )
        logging.getLogger("uvicorn.error").info(
            "background_jobs: disabled via EVOLVE_BACKGROUND_JOBS=%s",
            os.getenv("EVOLVE_BACKGROUND_JOBS"),
        )
        return
    if _task is not None and not _task.done():
        logger.debug("background_jobs: already running — skip start")
        return
    _stop = asyncio.Event()
    _task = asyncio.create_task(
        _loop(interval_sec), name="evolve-background-jobs"
    )


async def stop_background_jobs() -> None:
    global _task, _stop
    if _stop is not None:
        _stop.set()
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("background_jobs: stop await: %s", e)
    _task = None
    _stop = None
