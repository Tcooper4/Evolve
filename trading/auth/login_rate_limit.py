# -*- coding: utf-8 -*-
"""Login failed-attempt rate limiter (per username + per IP).

Design (typo-friendly, grows under abuse)
-----------------------------------------
* First ``MAX_FREE_FAILURES`` consecutive bad passwords → no cooldown
  (a typo or two should not burn minutes).
* On the next failure, a short cooldown starts (``BASE_COOLDOWN_SEC``).
* Each *new* lockout after a prior one doubles the cooldown, capped at
  ``MAX_COOLDOWN_SEC``. Successful login clears both username and IP state.
* During cooldown, attempts are rejected immediately (HTTP 429) with a
  clear remaining-seconds message — no silent hang, no bcrypt work.

In-process only (single API worker). Persisting across restarts is
intentionally out of scope for this pass.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Typo budget before any lockout.
MAX_FREE_FAILURES = 5
# First lockout: brief pause (not multi-minute).
BASE_COOLDOWN_SEC = 15.0
# Subsequent lockouts after release + more failures.
COOLDOWN_GROWTH = 2.0
MAX_COOLDOWN_SEC = 900.0  # 15 minutes


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after_sec: float = 0.0
    detail: str = ""
    failures: int = 0


@dataclass
class _Bucket:
    failures: int = 0
    cooldown_until: float = 0.0
    lockouts: int = 0  # how many times a cooldown was armed


class LoginRateLimiter:
    """Thread-safe per-username and per-IP failed-login tracker."""

    def __init__(
        self,
        *,
        max_free_failures: int = MAX_FREE_FAILURES,
        base_cooldown_sec: float = BASE_COOLDOWN_SEC,
        cooldown_growth: float = COOLDOWN_GROWTH,
        max_cooldown_sec: float = MAX_COOLDOWN_SEC,
        clock=None,
    ) -> None:
        self.max_free_failures = int(max_free_failures)
        self.base_cooldown_sec = float(base_cooldown_sec)
        self.cooldown_growth = float(cooldown_growth)
        self.max_cooldown_sec = float(max_cooldown_sec)
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._by_user: Dict[str, _Bucket] = {}
        self._by_ip: Dict[str, _Bucket] = {}

    def _now(self) -> float:
        return float(self._clock())

    def _cooldown_length(self, lockouts_after_arm: int) -> float:
        # lockouts_after_arm is 1-based count after arming this lockout
        n = max(1, int(lockouts_after_arm))
        length = self.base_cooldown_sec * (self.cooldown_growth ** (n - 1))
        return min(self.max_cooldown_sec, length)

    def _active_cooldown(self, bucket: _Bucket, now: float) -> float:
        remaining = float(bucket.cooldown_until) - now
        return remaining if remaining > 0 else 0.0

    def check(
        self,
        username: str,
        ip: str,
    ) -> RateLimitDecision:
        """Return whether a login attempt may proceed (no hang)."""
        user = (username or "").strip().lower() or "?"
        ip_key = (ip or "").strip() or "unknown"
        now = self._now()
        with self._lock:
            for key, store, label in (
                (user, self._by_user, "username"),
                (ip_key, self._by_ip, "IP"),
            ):
                bucket = store.get(key) or _Bucket()
                rem = self._active_cooldown(bucket, now)
                if rem > 0:
                    sec = int(rem) + (1 if rem > int(rem) else 0)
                    return RateLimitDecision(
                        allowed=False,
                        retry_after_sec=rem,
                        failures=bucket.failures,
                        detail=(
                            f"Too many failed login attempts for this {label}. "
                            f"Try again in {sec} second"
                            f"{'' if sec == 1 else 's'}."
                        ),
                    )
            return RateLimitDecision(allowed=True)

    def record_failure(self, username: str, ip: str) -> RateLimitDecision:
        """Count a failed password check; may arm / extend cooldown."""
        user = (username or "").strip().lower() or "?"
        ip_key = (ip or "").strip() or "unknown"
        now = self._now()
        with self._lock:
            worst = RateLimitDecision(allowed=True)
            for key, store in ((user, self._by_user), (ip_key, self._by_ip)):
                bucket = store.get(key) or _Bucket()
                # Still in an active cooldown — leave state; caller should
                # have used check() first, but be defensive.
                rem = self._active_cooldown(bucket, now)
                if rem > 0:
                    store[key] = bucket
                    worst = RateLimitDecision(
                        allowed=False,
                        retry_after_sec=rem,
                        failures=bucket.failures,
                        detail=(
                            f"Too many failed login attempts. "
                            f"Try again in {int(rem) + 1} seconds."
                        ),
                    )
                    continue

                bucket.failures += 1
                if bucket.failures > self.max_free_failures:
                    bucket.lockouts += 1
                    length = self._cooldown_length(bucket.lockouts)
                    bucket.cooldown_until = now + length
                    rem = length
                    decision = RateLimitDecision(
                        allowed=False,
                        retry_after_sec=rem,
                        failures=bucket.failures,
                        detail=(
                            f"Too many failed login attempts "
                            f"({bucket.failures} failures). "
                            f"Account temporarily locked for "
                            f"{int(rem)} second"
                            f"{'' if int(rem) == 1 else 's'}."
                        ),
                    )
                    if (not worst.allowed) or rem >= worst.retry_after_sec:
                        worst = decision
                store[key] = bucket
            return worst if not worst.allowed else RateLimitDecision(
                allowed=True,
                failures=max(
                    (self._by_user.get(user) or _Bucket()).failures,
                    (self._by_ip.get(ip_key) or _Bucket()).failures,
                ),
            )

    def record_success(self, username: str, ip: str) -> None:
        """Clear counters after a verified login."""
        user = (username or "").strip().lower() or "?"
        ip_key = (ip or "").strip() or "unknown"
        with self._lock:
            self._by_user.pop(user, None)
            self._by_ip.pop(ip_key, None)

    def snapshot(
        self, username: str, ip: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Test helper: (user_bucket_dict, ip_bucket_dict)."""
        user = (username or "").strip().lower() or "?"
        ip_key = (ip or "").strip() or "unknown"
        with self._lock:
            def as_dict(b: Optional[_Bucket]) -> Dict[str, float]:
                if not b:
                    return {"failures": 0, "cooldown_until": 0, "lockouts": 0}
                return {
                    "failures": float(b.failures),
                    "cooldown_until": float(b.cooldown_until),
                    "lockouts": float(b.lockouts),
                }

            return as_dict(self._by_user.get(user)), as_dict(
                self._by_ip.get(ip_key)
            )


# Process-wide limiter used by the API.
login_rate_limiter = LoginRateLimiter()


__all__ = [
    "MAX_FREE_FAILURES",
    "BASE_COOLDOWN_SEC",
    "MAX_COOLDOWN_SEC",
    "RateLimitDecision",
    "LoginRateLimiter",
    "login_rate_limiter",
]
