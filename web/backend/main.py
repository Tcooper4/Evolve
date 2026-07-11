# -*- coding: utf-8 -*-
"""Evolve web API — the FastAPI backend for the React frontend.

This is the first vertical slice of the Streamlit -> React migration.
Design principles:

* **One identity system.** Authentication verifies against the SAME
  SQLite accounts database the Streamlit login gate uses
  (trading/auth/accounts.py), and every request runs with
  EVOLVE_SESSION_ID-style identity (``user:<name>``) so the per-user
  plumbing built for multi-user mode — API-key resolver, memory scoping,
  watchlist — works identically here. Creating an account with
  scripts/manage_users.py grants access to BOTH frontends.
* **Reuse the verified backend.** Endpoints call the same trading/*
  modules the Streamlit app uses; no logic is duplicated.
* **Streamlit keeps running.** This API is additive; nothing in the
  existing app changes.

Run:  uvicorn web.backend.main:app --port 8000
Env:  EVOLVE_AUTH_SECRET (JWT signing; falls back to data/.auth_secret)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24


def _secret() -> str:
    env = os.getenv("EVOLVE_AUTH_SECRET")
    if env:
        return env
    try:
        from trading.auth.gate import _cookie_secret

        return _cookie_secret()
    except Exception:
        import secrets

        return secrets.token_hex(32)


app = FastAPI(title="Evolve API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("EVOLVE_CORS_ORIGINS",
                            "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    display_name: str


class QuoteResponse(BaseModel):
    symbol: str
    price: Optional[float] = None
    prev_close: Optional[float] = None
    change_pct: Optional[float] = None
    volume: Optional[float] = None


class Candle(BaseModel):
    time: str  # ISO date — lightweight-charts consumes {time, open, high, low, close}
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoryResponse(BaseModel):
    symbol: str
    interval: str
    candles: List[Candle]


class WatchlistItem(BaseModel):
    symbol: str
    note: Optional[str] = None


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

@app.post("/api/auth/token", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """Issue a JWT for valid credentials from the shared accounts DB."""
    from trading.auth import accounts

    username = (form.username or "").strip().lower()
    if not accounts.authenticate(username, form.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    display = username
    for u in accounts.list_users():
        if u["username"] == username:
            display = u["display_name"]
            break
    claims = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS),
    }
    token = jwt.encode(claims, _secret(), algorithm=ALGORITHM)
    return TokenResponse(access_token=token, username=username,
                         display_name=display)


def current_user(token: str = Depends(oauth2)) -> str:
    """Resolve the JWT to a username and install the per-request identity
    so all the multi-user plumbing (API keys, memory, watchlist) scopes
    to this user exactly as it does under the Streamlit gate."""
    cred_err = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise cred_err
    except JWTError:
        raise cred_err
    os.environ["EVOLVE_SESSION_ID"] = f"user:{username}"
    return username


# --------------------------------------------------------------------------
# Market data (reusing the verified backend modules)
# --------------------------------------------------------------------------

@app.get("/api/quote/{symbol}", response_model=QuoteResponse)
def quote(symbol: str, user: str = Depends(current_user)) -> QuoteResponse:
    import yfinance as yf

    from trading.data.ticker_resolver import normalize_ticker

    sym = normalize_ticker(symbol)
    try:
        info = yf.Ticker(sym).fast_info
        price = getattr(info, "last_price", None)
        prev = getattr(info, "previous_close", None)
        change = ((price - prev) / prev * 100) if price and prev else None
        return QuoteResponse(symbol=sym, price=price, prev_close=prev,
                             change_pct=change,
                             volume=getattr(info, "last_volume", None))
    except Exception as e:  # noqa: BLE001 - degrade to empty quote
        logger.warning("quote failed for %s: %s", sym, e)
        return QuoteResponse(symbol=sym)


@app.get("/api/history/{symbol}", response_model=HistoryResponse)
def history(symbol: str, period: str = "6mo", interval: str = "1d",
            user: str = Depends(current_user)) -> HistoryResponse:
    import yfinance as yf

    from trading.data.ticker_resolver import normalize_ticker

    sym = normalize_ticker(symbol)
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval)
        if df is None or df.empty:
            return HistoryResponse(symbol=sym, interval=interval, candles=[])
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
        candles = [
            Candle(time=idx.strftime("%Y-%m-%d"),
                   open=float(r["Open"]), high=float(r["High"]),
                   low=float(r["Low"]), close=float(r["Close"]),
                   volume=float(r.get("Volume", 0.0)))
            for idx, r in df.iterrows()
        ]
        return HistoryResponse(symbol=sym, interval=interval, candles=candles)
    except Exception as e:  # noqa: BLE001
        logger.warning("history failed for %s: %s", sym, e)
        return HistoryResponse(symbol=sym, interval=interval, candles=[])


# --------------------------------------------------------------------------
# Watchlist (user-scoped through the existing manager)
# --------------------------------------------------------------------------

@app.get("/api/watchlist")
def get_watchlist(user: str = Depends(current_user)) -> List[Dict[str, Any]]:
    from trading.data.watchlist import WatchlistManager

    return WatchlistManager(user_id=f"user:{user}").get_all()


@app.post("/api/watchlist")
def add_watchlist(item: WatchlistItem,
                  user: str = Depends(current_user)) -> Dict[str, Any]:
    from trading.data.watchlist import WatchlistManager

    WatchlistManager(user_id=f"user:{user}").add_ticker(item.symbol,
                                                        note=item.note)
    return {"ok": True, "symbol": item.symbol.upper()}


@app.delete("/api/watchlist/{symbol}")
def remove_watchlist(symbol: str,
                     user: str = Depends(current_user)) -> Dict[str, Any]:
    from trading.data.watchlist import WatchlistManager

    WatchlistManager(user_id=f"user:{user}").remove_ticker(symbol)
    return {"ok": True}


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "app": "evolve-api"}
