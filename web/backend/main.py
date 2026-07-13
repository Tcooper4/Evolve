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
def history(symbol: str, period: str = "6mo", interval: str = "",
            user: str = Depends(current_user)) -> HistoryResponse:
    import yfinance as yf

    from trading.data.ticker_resolver import normalize_ticker

    sym = normalize_ticker(symbol)
    per = (period or "6mo").strip().lower()
    user_iv = (interval or "").strip().lower()
    # Denser bars on short lookbacks so 1M/3M don't look fat vs 6M/1Y.
    presets = {
        "1d": ("1d", "5m"),
        "5d": ("5d", "15m"),
        "1w": ("5d", "15m"),
        "1mo": ("1mo", "1h"),
        "3mo": ("3mo", "1h"),
        "6mo": ("6mo", "1d"),
        "1y": ("1y", "1d"),
        "2y": ("2y", "1d"),
        "5y": ("5y", "1wk"),
        "max": ("max", "1wk"),
        "all": ("max", "1wk"),
    }
    allowed_iv = {
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk",
    }
    if per in presets:
        per, default_iv = presets[per]
    else:
        default_iv = "1d"
    iv = user_iv if user_iv in allowed_iv else default_iv
    if iv == "1m" and per not in ("1d", "5d"):
        per = "5d"
    try:
        df = yf.Ticker(sym).history(period=per, interval=iv)
        if df is None or df.empty:
            return HistoryResponse(symbol=sym, interval=iv, candles=[])
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
        candles = []
        intraday = iv.endswith("m") or iv in ("1h", "60m", "90m")
        for idx, r in df.iterrows():
            t = (
                idx.strftime("%Y-%m-%dT%H:%M:%S")
                if intraday
                else idx.strftime("%Y-%m-%d")
            )
            candles.append(Candle(
                time=t,
                open=float(r["Open"]), high=float(r["High"]),
                low=float(r["Low"]), close=float(r["Close"]),
                volume=float(r.get("Volume", 0.0) or 0.0),
            ))
        # Yahoo often reports Volume=0 on the still-forming bar. Infer it as
        # session volume minus sum of completed same-day bars so soft-poll
        # legend / histogram can show live volume accumulating.
        if intraday and candles and float(candles[-1].volume or 0) <= 0:
            try:
                session_vol = float(
                    getattr(yf.Ticker(sym).fast_info, "last_volume", 0) or 0
                )
                last_day = candles[-1].time[:10]
                prior = sum(
                    float(c.volume or 0)
                    for c in candles[:-1]
                    if c.time[:10] == last_day
                )
                inferred = max(0.0, session_vol - prior)
                if inferred > 0:
                    last = candles[-1]
                    candles[-1] = Candle(
                        time=last.time,
                        open=last.open,
                        high=last.high,
                        low=last.low,
                        close=last.close,
                        volume=inferred,
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("live bar volume infer failed for %s: %s", sym, e)
        return HistoryResponse(symbol=sym, interval=iv, candles=candles)
    except Exception as e:  # noqa: BLE001
        logger.warning("history failed for %s: %s", sym, e)
        return HistoryResponse(symbol=sym, interval=iv, candles=[])


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
    short_score: Optional[float] = None
    short_grade: Optional[str] = None
    technical_score: Optional[float] = None
    momentum_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    fundamental_score: Optional[float] = None
    summary: Optional[str] = None
    last_price: Optional[float] = None
    signals: Dict[str, Any] = {}
    signal_list: List[Dict[str, Any]] = []
    error: Optional[str] = None


def _score_signals_as_dict(raw: Any) -> Dict[str, Any]:
    """Normalize compute_ai_score signals (list of {name, value, ...}) to a dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        key = str(name)
        val = item.get("value", item.get("score"))
        try:
            out[key] = float(val) if val is not None else item
        except (TypeError, ValueError):
            out[key] = item
    return out


@app.get("/api/score/{symbol}", response_model=ScoreResponse)
def ai_score(symbol: str, mode: str = "long",
             user: str = Depends(current_user)) -> ScoreResponse:
    """AI Score with per-signal breakdown (same engine as the Analyze page)."""
    from config.user_store import load_user_preferences
    from trading.analysis.ai_score import compute_ai_score, compute_short_score
    from trading.data.ticker_resolver import normalize_ticker

    sym = normalize_ticker(symbol)
    try:
        prefs = load_user_preferences(f"user:{user}") or {}
        scoring_style = prefs.get("scoring_style") or "Balanced (default)"
        r = compute_ai_score(sym, scoring_style=scoring_style) or {}
        _score = r.get("overall_score")
        if _score is None:
            _score = r.get("score", r.get("ai_score"))
        sig_raw = r.get("signals") or r.get("breakdown") or []
        short = {}
        try:
            short = compute_short_score(sym, None, ai_result=r) or {}
        except Exception as e:
            logger.debug("short score skipped: %s", e)
        # mode=short surfaces short_score as the primary ring value
        primary = _score
        primary_grade = r.get("grade")
        if (mode or "long").lower() == "short":
            primary = short.get("short_score", primary)
            primary_grade = short.get("grade", primary_grade)
        return ScoreResponse(
            symbol=sym,
            score=primary,
            grade=primary_grade,
            short_score=short.get("short_score"),
            short_grade=short.get("grade"),
            technical_score=r.get("technical_score"),
            momentum_score=r.get("momentum_score"),
            sentiment_score=r.get("sentiment_score"),
            fundamental_score=r.get("fundamental_score"),
            summary=r.get("summary") or short.get("summary"),
            last_price=r.get("last_price"),
            signals=_score_signals_as_dict(sig_raw),
            signal_list=sig_raw if isinstance(sig_raw, list) else [],
            error=r.get("error") or short.get("error"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("score failed for %s: %s", sym, e)
        return ScoreResponse(symbol=sym, error=str(e))


class ScanRequest(BaseModel):
    filters: List[str] = []
    max_results: int = 15
    min_quick_score: float = 6.0
    universe: str = "sp100"
    custom_tickers: List[str] = []


@app.post("/api/scan")
def scan(req: ScanRequest, user: str = Depends(current_user)) -> Dict[str, Any]:
    from trading.analysis.market_scanner import _get_universe, scan_market
    from trading.data.ticker_resolver import normalize_ticker

    try:
        custom = [
            normalize_ticker(t)
            for t in (req.custom_tickers or [])
            if t and str(t).strip()
        ]
        tickers = custom if custom else _get_universe(req.universe or "sp100")
        result = scan_market(
            filters=req.filters or None,
            universe=tickers,
            max_results=req.max_results,
            min_quick_score=req.min_quick_score,
        ) or {}
        err = result.get("error")
        return {
            **result,
            "success": not err,
            "error": err,
            "results": result.get("results") or [],
            "universe": "custom" if custom else (req.universe or "sp100").strip().lower(),
            "universe_size": len(tickers),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("scan failed: %s", e)
        return {"success": False, "error": str(e), "results": []}


class BacktestRequest(BaseModel):
    symbol: str = "SPY"
    strategy: str = "RSIStrategy"
    period: str = "1y"
    params: Dict[str, Any] = {}
    force_defaults: bool = False
    cost_bps: float = 5.0


class ModelBacktestRequest(BaseModel):
    symbol: str = "SPY"
    model: str = "xgboost"
    period: str = "1y"
    horizon: int = 5


def _downsample(rows: List[Dict[str, Any]], max_n: int = 400) -> List[Dict[str, Any]]:
    """Evenly sample points so long histories stay chartable without silent tail-only truncation."""
    n = len(rows)
    if n <= max_n:
        return rows
    out: List[Dict[str, Any]] = []
    for i in range(max_n):
        idx = int(round(i * (n - 1) / (max_n - 1)))
        out.append(rows[idx])
    return out


def _trades_from_position(
    position,
    close,
    equity,
) -> List[Dict[str, Any]]:
    """Round-trip trades from a long-only position series."""
    trades: List[Dict[str, Any]] = []
    in_pos = False
    entry_t = None
    entry_px = 0.0
    entry_eq = 1.0
    for idx in position.index:
        pos = float(position.loc[idx])
        if not in_pos and abs(pos) > 1e-9:
            in_pos = True
            entry_t = idx
            entry_px = float(close.loc[idx])
            entry_eq = float(equity.loc[idx]) or 1.0
        elif in_pos and abs(pos) < 1e-9:
            exit_px = float(close.loc[idx])
            exit_eq = float(equity.loc[idx]) or entry_eq
            pret = (exit_px / entry_px - 1.0) if entry_px else 0.0
            eret = (exit_eq / entry_eq - 1.0) if entry_eq else 0.0
            trades.append({
                "entry": str(entry_t)[:10],
                "exit": str(idx)[:10],
                "entry_price": round(entry_px, 4),
                "exit_price": round(exit_px, 4),
                "price_return": round(pret, 4),
                "equity_return": round(eret, 4),
                "won": eret > 0,
            })
            in_pos = False
    if in_pos and entry_t is not None:
        last = position.index[-1]
        exit_px = float(close.loc[last])
        exit_eq = float(equity.loc[last]) or entry_eq
        pret = (exit_px / entry_px - 1.0) if entry_px else 0.0
        eret = (exit_eq / entry_eq - 1.0) if entry_eq else 0.0
        trades.append({
            "entry": str(entry_t)[:10],
            "exit": str(last)[:10] + " (open)",
            "entry_price": round(entry_px, 4),
            "exit_price": round(exit_px, 4),
            "price_return": round(pret, 4),
            "equity_return": round(eret, 4),
            "won": eret > 0,
            "open": True,
        })
    return trades


@app.post("/api/backtest")
def backtest(req: BacktestRequest,
             user: str = Depends(current_user)) -> Dict[str, Any]:
    """One strategy backtest via the optimizer's evaluation bridge - the
    same execution-verified path the Backtest tab uses."""
    import trading.strategies  # populate registry (filesystem discovery)
    import yfinance as yf

    from trading.data.ticker_resolver import normalize_ticker
    from trading.optimization.strategy_backtest_objective import (
        evaluate_params,
        normalize_ohlcv,
        run_strategy,
        signals_to_position,
        signals_to_returns,
    )
    from trading.optimization.strategy_param_spaces import get_default_params

    sym = normalize_ticker(req.symbol)
    cost_bps = float(req.cost_bps if req.cost_bps is not None else 5.0)
    try:
        params = dict(req.params or {})
        used_saved = False
        if req.force_defaults:
            params = get_default_params(req.strategy) or {}
        elif not params:
            try:
                from trading.services.self_tune import get_adopted_params

                adopted = get_adopted_params(req.strategy, sym)
                if adopted:
                    params = adopted
                    used_saved = True
            except Exception as e:
                logger.debug("adopted params load skipped: %s", e)
        if not params:
            params = get_default_params(req.strategy) or {}

        df = yf.Ticker(sym).history(period=req.period, interval="1d")
        if df is None or df.empty:
            return {"success": False, "error": "no data", "symbol": sym}
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
        result = evaluate_params(req.strategy, df, params, cost_bps=cost_bps)
        out = dict(result) if isinstance(result, dict) else {"metric": result}
        try:
            signals = run_strategy(req.strategy, df, params)
            ndata = normalize_ohlcv(df)
            close = ndata["close"].astype(float)
            rets = signals_to_returns(signals, close, cost_bps=cost_bps)
            equity = (1.0 + rets.fillna(0.0)).cumprod()
            position = signals_to_position(signals, close.index, signal_mode="hold")

            equity_rows = [
                {"time": idx.strftime("%Y-%m-%d"), "value": float(equity.loc[idx])}
                for idx in close.index
            ]
            buy_hold = close / float(close.iloc[0] or 1.0)
            comparison = [
                {
                    "time": idx.strftime("%Y-%m-%d"),
                    "actual": float(buy_hold.loc[idx]),
                    "predicted": float(equity.loc[idx]),
                    "position": float(position.loc[idx]),
                }
                for idx in close.index
            ]
            trades = _trades_from_position(position, close, equity)
            closed = [t for t in trades if not t.get("open")]
            trade_wins = sum(1 for t in closed if t.get("won"))
            trade_win_rate = (trade_wins / len(closed)) if closed else None

            out["equity"] = _downsample(equity_rows)
            out["comparison"] = _downsample(comparison)
            out["trades"] = trades[-40:]
            out["kind"] = "strategy"
            out["in_market_bars"] = int((position.abs() > 1e-9).sum())
            out["total_bars"] = int(len(close))
            out["buy_hold_return"] = float(buy_hold.iloc[-1] - 1.0)
            out["trade_win_rate"] = (
                round(float(trade_win_rate), 3) if trade_win_rate is not None else None
            )
            out["n_trades"] = len(closed)
            # Clarify day-level win_rate from risk_metrics
            if "win_rate" in out:
                out["day_win_rate"] = out.pop("win_rate")
        except Exception as e:
            logger.debug("equity curve skipped: %s", e)
            out["equity"] = []
            out["comparison"] = []
            out["trades"] = []
            out["kind"] = "strategy"
        out.update({
            "success": True,
            "symbol": sym,
            "strategy": req.strategy,
            "period": req.period,
            "params_used": params,
            "used_saved_params": used_saved,
            "force_defaults": bool(req.force_defaults),
            "cost_bps": cost_bps,
        })
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("backtest failed: %s", e)
        return {"success": False, "error": str(e), "symbol": sym}


@app.post("/api/backtest/model")
def backtest_model(req: ModelBacktestRequest,
                   user: str = Depends(current_user)) -> Dict[str, Any]:
    """Walk-forward forecast backtest for a single model (not a rule strategy)."""
    import yfinance as yf

    from trading.data.ticker_resolver import normalize_ticker
    from trading.validation.walk_forward_utils import WalkForwardValidator

    sym = normalize_ticker(req.symbol)
    model = (req.model or "xgboost").strip().lower()
    # Keep interactive runs tractable — LSTM is too slow for this button
    allowed = {
        "arima", "xgboost", "ridge", "catboost", "prophet", "garch", "hybrid",
    }
    if model not in allowed:
        return {
            "success": False,
            "error": f"Model '{model}' not supported for interactive backtest. "
                     f"Use one of: {', '.join(sorted(allowed))}",
            "symbol": sym,
        }
    period = (req.period or "1y").strip().lower()
    # Period → train / test / step sized for UI latency
    presets = {
        "6mo": (80, 15, 20),
        "1y": (120, 21, 30),
        "2y": (180, 21, 42),
        "5y": (252, 42, 63),
    }
    train_w, test_w, step = presets.get(period, presets["1y"])
    horizon = max(1, min(int(req.horizon or 5), 10))
    try:
        df = yf.Ticker(sym).history(period=period, interval="1d")
        if df is None or df.empty:
            return {"success": False, "error": "no data", "symbol": sym}
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)

        # Cap windows so a UI click finishes in reasonable time
        max_windows = 6
        while True:
            n = 0
            i = train_w
            while i + test_w <= len(df):
                n += 1
                i += step
            if n <= max_windows or step >= test_w * 3:
                break
            step = int(step * 1.4)

        wfv = WalkForwardValidator(model_name=model, symbol=sym, window_type="expanding")
        result = wfv.run(
            df,
            train_window=train_w,
            test_window=test_w,
            step_size=step,
            horizon=horizon,
            # LEAKAGE FIX (2026-07): the purge/embargo mechanism was added
            # to WalkForwardValidator and verified correct in isolation
            # (tests/test_routing_validation.py's research harness used
            # it), but this LIVE route - the one the fold-strip UI users
            # actually see - was never updated to pass it, so real
            # results were still leakage-prone with the default purge=0.
            # A gap of `horizon` bars between train end and test start is
            # the standard baseline (see the regime-detection literature
            # this project reviewed): without it, any feature or label
            # that reaches even one bar past the train boundary can peek
            # into the forecast horizon itself.
            purge=horizon,
        )
        perf = result.model_performance or {}
        if perf.get("error") and not result.windows:
            return {
                "success": False,
                "symbol": sym,
                "model": model,
                "error": str(perf.get("error")),
            }

        # Direction-following paper equity (long when forecast steps up, else flat)
        equity: List[Dict[str, Any]] = []
        try:
            eq = 1.0
            equity.append({"time": "start", "value": eq})
            for w in result.windows:
                preds = list(w.predictions or [])
                acts = list(w.actuals or [])
                for j in range(1, min(len(preds), len(acts))):
                    prev_a = float(acts[j - 1])
                    cur_a = float(acts[j])
                    if prev_a == 0:
                        continue
                    pred_up = float(preds[j]) >= float(preds[j - 1])
                    act_ret = (cur_a / prev_a) - 1.0
                    eq *= (1.0 + act_ret) if pred_up else 1.0
                    equity.append({
                        "time": f"w{w.window_index}-{j}",
                        "value": float(eq),
                    })
            if len(equity) < 2:
                eq = 1.0
                equity = [{"time": "start", "value": 1.0}]
                for w in result.windows:
                    da = float(w.directional_accuracy) if w.directional_accuracy == w.directional_accuracy else 0.5
                    eq *= 1.0 + (da - 0.5) * 0.02 * max(1, len(w.predictions or []))
                    equity.append({
                        "time": str(w.test_end)[:10],
                        "value": float(eq),
                    })
        except Exception as e:
            logger.debug("model equity skipped: %s", e)
            equity = []

        # Predicted vs actual path (concatenated OOS steps across windows)
        comparison: List[Dict[str, Any]] = []
        try:
            step_i = 0
            for w in result.windows:
                preds = list(w.predictions or [])
                acts = list(w.actuals or [])
                for j in range(min(len(preds), len(acts))):
                    comparison.append({
                        "i": step_i,
                        "window": w.window_index,
                        "predicted": float(preds[j]),
                        "actual": float(acts[j]),
                    })
                    step_i += 1
            # Cap for UI payload size (even sample, keep full span)
            comparison = _downsample(comparison, 400)
        except Exception as e:
            logger.debug("model comparison skipped: %s", e)
            comparison = []

        da = perf.get("mean_directional_accuracy")
        equity = _downsample(equity, 400) if equity else []

        # FOLD VISIBILITY (2026-07): per-window results were computed and
        # then aggregated away - stability ACROSS time is the whole point
        # of walk-forward, so expose each fold for the UI to show.
        folds: List[Dict[str, Any]] = []
        try:
            for w in result.windows:
                folds.append({
                    "window": int(w.window_index),
                    "train_start": str(w.train_start)[:10],
                    "train_end": str(w.train_end)[:10],
                    "test_start": str(w.test_start)[:10],
                    "test_end": str(w.test_end)[:10],
                    "mae": float(w.mae) if w.mae == w.mae else None,
                    "mape": float(w.mape) if w.mape == w.mape else None,
                    "directional_accuracy": (
                        float(w.directional_accuracy)
                        if w.directional_accuracy == w.directional_accuracy
                        else None
                    ),
                })
        except Exception as e:
            logger.debug("folds skipped: %s", e)

        return {
            "success": True,
            "kind": "model",
            "folds": folds,
            "folds_note": (
                None if folds else
                "No per-window fold detail — need enough history for ≥2 "
                "walk-forward windows. Headline averages can hide regime luck."
            ),
            "symbol": sym,
            "model": model,
            "period": period,
            "n_windows": perf.get("n_windows") or len(result.windows),
            "mean_mae": perf.get("mean_mae"),
            "mean_mape": perf.get("mean_mape"),
            "mean_directional_accuracy": da,
            "hit_rate_5pct": perf.get("hit_rate_5pct"),
            "hit_rate_10pct": perf.get("hit_rate_10pct"),
            "consistency_score": perf.get("consistency_score"),
            "directional_accuracy": da,
            "illustrative_return": (equity[-1]["value"] - 1.0) if equity else None,
            "equity": equity,
            "equity_note": (
                "Illustrative only — long when forecast steps up, else flat. "
                "Not a real trading backtest."
            ),
            "comparison": comparison,
            "windows": [
                {
                    "window": w.window_index,
                    "test_start": str(w.test_start)[:10],
                    "test_end": str(w.test_end)[:10],
                    "mape": round(float(w.mape), 2) if w.mape == w.mape else None,
                    "directional_accuracy": (
                        round(float(w.directional_accuracy), 3)
                        if w.directional_accuracy == w.directional_accuracy else None
                    ),
                }
                for w in result.windows[:12]
            ],
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("model backtest failed: %s", e)
        return {"success": False, "error": str(e), "symbol": sym}


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest, user: str = Depends(current_user)) -> Dict[str, Any]:
    """Full chat turn through the SHARED tool-calling brain
    (trading/services/chat_turn.py) - the same loop the Streamlit chat
    uses: memory context + Agent Skills + platform tools (scan, score,
    forecast, news, risk, patterns, backtests, options sentiment), with
    graceful fallbacks. Both frontends now behave identically."""
    try:
        from trading.services.chat_turn import run_chat_turn

        return run_chat_turn(req.message)
    except Exception as e:  # noqa: BLE001
        logger.warning("chat failed: %s", e)
        return {"success": False, "error": str(e)}


class KeysRequest(BaseModel):
    anthropic: Optional[str] = None
    openai: Optional[str] = None
    news: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    twitter_bearer: Optional[str] = None


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
        "reddit": bool(
            keys.get("REDDIT_CLIENT_ID") and keys.get("REDDIT_CLIENT_SECRET")
        ),
        "twitter": bool(keys.get("TWITTER_BEARER_TOKEN")),
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
    if req.reddit_client_id is not None:
        keys["REDDIT_CLIENT_ID"] = req.reddit_client_id
    if req.reddit_client_secret is not None:
        keys["REDDIT_CLIENT_SECRET"] = req.reddit_client_secret
    if req.twitter_bearer is not None:
        keys["TWITTER_BEARER_TOKEN"] = req.twitter_bearer
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

from web.backend.parity_routes import build_router as _build_parity_router

app.include_router(_build_parity_router(current_user))


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


# (frontend static mount moved to the END of this module - a
# catch-all '/' mount registered mid-file swallows every route added
# after it; found when /api/portfolio returned index.html.)


# --------------------------------------------------------------------------
# Paper portfolio
# --------------------------------------------------------------------------

class TradeRequest(BaseModel):
    symbol: str
    side: str  # buy | sell
    quantity: float
    price: Optional[float] = None


@app.get("/api/portfolio")
def portfolio(user: str = Depends(current_user)) -> Dict[str, Any]:
    from trading.portfolio.paper_portfolio import PaperPortfolio

    return PaperPortfolio(user_id=f"user:{user}").get_summary()


@app.post("/api/portfolio/trade")
def portfolio_trade(req: TradeRequest,
                    user: str = Depends(current_user)) -> Dict[str, Any]:
    from trading.portfolio.paper_portfolio import PaperPortfolio

    price = req.price
    if price is None:
        try:
            import yfinance as yf

            from trading.data.ticker_resolver import normalize_ticker

            p = yf.Ticker(normalize_ticker(req.symbol)).fast_info.last_price
            price = float(p) if p else None
        except Exception:
            price = None
        if price is None:
            return {"success": False,
                    "error": "no live price - provide one explicitly"}
    return PaperPortfolio(user_id=f"user:{user}").record_trade(
        req.symbol, req.side, req.quantity, price
    )


@app.get("/api/portfolio/trades")
def portfolio_trades(user: str = Depends(current_user)) -> List[Dict[str, Any]]:
    from trading.portfolio.paper_portfolio import PaperPortfolio

    return PaperPortfolio(user_id=f"user:{user}").get_trades()


# Serve the built frontend in production: uvicorn web.backend.main:app
# then open http://localhost:8000 - no separate frontend server needed.
_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_dist):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
