# -*- coding: utf-8 -*-
"""In-process user-scoped WebSocket notification fan-out.

Execution (fills / alerts) happens in background_jobs regardless of
connections; this hub only delivers UI notifications when a browser is
subscribed via /ws/notifications.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class NotificationHub:
    """Map username -> connected notification websockets."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sockets: Dict[str, Set[WebSocket]] = {}

    async def connect(self, username: str, ws: WebSocket) -> None:
        key = (username or "").strip()
        if not key:
            return
        async with self._lock:
            self._sockets.setdefault(key, set()).add(ws)
        logger.debug("notification_hub: %s connected (%d)", key,
                     len(self._sockets.get(key, ())))

    async def disconnect(self, username: str, ws: WebSocket) -> None:
        key = (username or "").strip()
        async with self._lock:
            group = self._sockets.get(key)
            if not group:
                return
            group.discard(ws)
            if not group:
                self._sockets.pop(key, None)

    async def publish(self, username: str, payload: Dict[str, Any]) -> int:
        """Push JSON payload to all sockets for username. Returns sent count."""
        key = (username or "").strip()
        if not key:
            return 0
        async with self._lock:
            targets = list(self._sockets.get(key, ()))
        sent = 0
        dead: list = []
        for ws in targets:
            try:
                await ws.send_json(payload)
                sent += 1
            except Exception as e:
                logger.debug("notification_hub: send failed for %s: %s", key, e)
                dead.append(ws)
        for ws in dead:
            await self.disconnect(key, ws)
        return sent


# Process-wide singleton — one hub per uvicorn worker.
notification_hub = NotificationHub()
