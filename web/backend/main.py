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


# --------------------------------------------------------------------------
# Analysis / Scanner / Backtest / Chat / Settings (page-parity endpoints)
# --------------------------------------------------------------------------

class ScoreResponse(BaseModel):
    symbol: str
    score: Optional[float] = None
    grade: Optional[str] = None
    signals: Dict[str, Any] = {}
    error: Optional[str] = None


@app.get("/api/score/{symbol}", response_model=ScoreResponse)
def ai_score(symbol: str, user: str = Depends(current_user)) -> ScoreResponse:
    """AI Score with per-signal breakdown (same engine as the Analyze page)."""
    from trading.analysis.ai_score import compute_ai_score
    from trading.data.ticker_resolver import normalize_ticker

    sym = normalize_ticker(symbol)
    try:
        r = compute_ai_score(sym) or {}
        return ScoreResponse(
            symbol=sym,
            score=r.get("score") or r.get("ai_score"),
            grade=r.get("grade"),
            signals=r.get("signals") or r.get("breakdown") or {},
            error=r.get("error"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("score failed for %s: %s", sym, e)
        return ScoreResponse(symbol=sym, error=str(e))


class ScanRequest(BaseModel):
    filters: List[str] = []
    max_results: int = 15
    min_quick_score: float = 6.0


@app.post("/api/scan")
def scan(req: ScanRequest, user: str = Depends(current_user)) -> Dict[str, Any]:
    from trading.analysis.market_scanner import scan_market

    try:
        return scan_market(
            filters=req.filters or None,
            max_results=req.max_results,
            min_quick_score=req.min_quick_score,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("scan failed: %s", e)
        return {"success": False, "error": str(e), "results": []}


class BacktestRequest(BaseModel):
    symbol: str = "SPY"
    strategy: str = "RSIStrategy"
    period: str = "1y"
    params: Dict[str, Any] = {}


@app.post("/api/backtest")
def backtest(req: BacktestRequest,
             user: str = Depends(current_user)) -> Dict[str, Any]:
    """One strategy backtest via the optimizer's evaluation bridge - the
    same execution-verified path the Backtest tab uses."""
    import trading.strategies  # populate registry (filesystem discovery)
    import yfinance as yf

    from trading.data.ticker_resolver import normalize_ticker
    from trading.optimization.strategy_backtest_objective import evaluate_params

    sym = normalize_ticker(req.symbol)
    try:
        df = yf.Ticker(sym).history(period=req.period, interval="1d")
        if df is None or df.empty:
            return {"success": False, "error": "no data", "symbol": sym}
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
        result = evaluate_params(req.strategy, df, req.params or {})
        out = dict(result) if isinstance(result, dict) else {"metric": result}
        out.update({"success": True, "symbol": sym, "strategy": req.strategy})
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("backtest failed: %s", e)
        return {"success": False, "error": str(e), "symbol": sym}


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest, user: str = Depends(current_user)) -> Dict[str, Any]:
    """v1 chat: the user's own memory context + their own LLM key.
    (The full tool-calling agent loop stays in the Streamlit chat for now;
    noted in web/README.md.)"""
    try:
        from trading.memory.memory_store import get_memory_store
        from trading.services.chat_nl_service import get_memory_context
        from agents.llm.active_llm_calls import call_active_llm_simple

        store = get_memory_store()
        try:
            ctx = get_memory_context(store)
        except Exception:
            ctx = ""
        prompt = (
            f"{ctx}\n\nUser: {req.message}\nAssistant:"
            if ctx else req.message
        )
        reply = call_active_llm_simple(prompt, max_tokens=800) or ""
        if not reply:
            return {"success": False,
                    "error": "No LLM configured - add an API key in Settings."}
        return {"success": True, "reply": reply}
    except Exception as e:  # noqa: BLE001
        logger.warning("chat failed: %s", e)
        return {"success": False, "error": str(e)}


class KeysRequest(BaseModel):
    anthropic: Optional[str] = None
    openai: Optional[str] = None
    news: Optional[str] = None


@app.get("/api/settings/keys")
def get_keys(user: str = Depends(current_user)) -> Dict[str, bool]:
    """Which keys the CURRENT USER has stored (never the values)."""
    from config.user_store import load_user_keys, load_user_api_keys

    uid = f"user:{user}"
    keys: Dict[str, str] = {}
    keys.update(load_user_api_keys(uid) or {})
    keys.update(load_user_keys(uid) or {})
    return {
        "anthropic": bool(keys.get("ANTHROPIC_API_KEY")),
        "openai": bool(keys.get("OPENAI_API_KEY")),
        "news": bool(keys.get("NEWS_API_KEY")),
    }


@app.post("/api/settings/keys")
def save_keys(req: KeysRequest,
              user: str = Depends(current_user)) -> Dict[str, Any]:
    from config.user_store import load_user_api_keys, save_user_api_keys

    uid = f"user:{user}"
    keys = dict(load_user_api_keys(uid) or {})
    if req.anthropic is not None:
        keys["ANTHROPIC_API_KEY"] = req.anthropic
    if req.openai is not None:
        keys["OPENAI_API_KEY"] = req.openai
    if req.news is not None:
        keys["NEWS_API_KEY"] = req.news
    save_user_api_keys(uid, keys)
    try:
        from config.llm_config import reset_llm_config

        reset_llm_config(uid)  # new keys take effect immediately, this user only
    except Exception:
        pass
    return {"ok": True}


# --------------------------------------------------------------------------
# Live quote stream (websocket) + production static serving
# --------------------------------------------------------------------------

from fastapi import WebSocket, WebSocketDisconnect  # noqa: E402


@app.websocket("/ws/quote/{symbol}")
async def quote_stream(ws: WebSocket, symbol: str) -> None:
    """Push a quote every few seconds. Token via ?token= query param
    (browsers can't set headers on websockets)."""
    import asyncio

    token = ws.query_params.get("token", "")
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        username = payload.get("sub")
        assert username
    except Exception:
        await ws.close(code=4401)
        return
    await ws.accept()
    os.environ["EVOLVE_SESSION_ID"] = f"user:{username}"

    import yfinance as yf

    from trading.data.ticker_resolver import normalize_ticker

    sym = normalize_ticker(symbol)
    try:
        while True:
            try:
                info = yf.Ticker(sym).fast_info
                price = getattr(info, "last_price", None)
                prev = getattr(info, "previous_close", None)
                await ws.send_json({
                    "symbol": sym,
                    "price": price,
                    "change_pct": ((price - prev) / prev * 100)
                    if price and prev else None,
                })
            except Exception:
                await ws.send_json({"symbol": sym, "price": None,
                                    "change_pct": None})
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


# Serve the built frontend in production: uvicorn web.backend.main:app
# then open http://localhost:8000 - no separate frontend server needed.
_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_dist):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
