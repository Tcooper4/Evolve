"""
SEC EDGAR Integration
======================
Free alternative data from SEC filings.
No API key required for JSON/history endpoints.

Uses:
- https://www.sec.gov/files/company_tickers.json
- https://data.sec.gov/submissions/CIK{cik}.json

Provides CIK resolution, latest 10-K/10-Q metadata, MD&A text excerpt,
and filing sentiment (LLM when configured, else keyword heuristic).
"""

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {
    "User-Agent": "Evolve Trading Platform contact@evolve.local",
    "Accept": "application/json",
}

_CIK_CACHE: Dict[str, str] = {}
_FILING_CACHE: Dict[str, dict] = {}
_SEC_CACHE: Dict[str, Any] = {}


def _sleep_rate_limit() -> None:
    time.sleep(0.11)


def _edgar_get_json(url: str) -> Optional[Any]:
    """GET JSON from SEC with light rate limiting."""
    try:
        _sleep_rate_limit()
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception as e:
        logger.debug("EDGAR JSON fetch failed %s: %s", url, e)
        return None


def _edgar_get_text(url: str) -> Optional[str]:
    try:
        _sleep_rate_limit()
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=25) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("EDGAR text fetch failed %s: %s", url, e)
        return None


def get_cik(ticker: str) -> Optional[str]:
    """Return 10-digit zero-padded CIK for a US equity ticker."""
    ticker = ticker.upper().strip()
    if not ticker:
        return None
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    tickers_data = _edgar_get_json(tickers_url)
    if not isinstance(tickers_data, dict):
        return None

    for entry in tickers_data.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("ticker", "")).upper() != ticker:
            continue
        cik_raw = entry.get("cik_str")
        if cik_raw is None:
            return None
        cik = str(int(cik_raw)).zfill(10)
        _CIK_CACHE[ticker] = cik
        return cik
    return None


def get_latest_filing(
    ticker: str,
    form_type: str = "10-Q",
) -> Optional[dict]:
    """Metadata for the most recent filing of ``form_type`` (10-K, 10-Q, …)."""
    cache_key = f"{ticker.upper().strip()}_{form_type}"
    if cache_key in _FILING_CACHE:
        cached = _FILING_CACHE[cache_key]
        if time.time() - float(cached.get("_cached_at", 0)) < 86400:
            return {k: v for k, v in cached.items() if k != "_cached_at"}

    cik = get_cik(ticker)
    if not cik:
        return None

    cik_padded = cik.zfill(10)
    url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
    data = _edgar_get_json(url)
    if not isinstance(data, dict):
        return None

    filings = data.get("filings", {})
    recent = filings.get("recent", {})
    if not isinstance(recent, dict):
        return None

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    n = min(len(forms), len(dates), len(accessions))
    for i in range(n):
        if forms[i] != form_type:
            continue
        acc = str(accessions[i])
        acc_nd = acc.replace("-", "")
        cik_int = str(int(cik_padded))
        prim = (
            str(primary_docs[i])
            if i < len(primary_docs) and primary_docs[i]
            else None
        )
        if not prim:
            continue
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{acc_nd}/{prim}"
        )
        result = {
            "ticker": ticker.upper().strip(),
            "cik": cik_padded,
            "cik_int": cik_int,
            "form": form_type,
            "date": str(dates[i]),
            "accession": acc,
            "primary_document": prim,
            "document_url": doc_url,
            "url": doc_url,
            "_cached_at": time.time(),
        }
        _FILING_CACHE[cache_key] = result
        return {k: v for k, v in result.items() if k != "_cached_at"}

    return None


def get_mda_text(
    ticker: str,
    max_chars: int = 3000,
) -> Optional[str]:
    """Truncated plain text from latest 10-Q or 10-K primary document."""
    try:
        filing = get_latest_filing(ticker, "10-Q") or get_latest_filing(ticker, "10-K")
        if not filing:
            return None
        doc_url = filing.get("document_url") or filing.get("url")
        if not doc_url:
            return None

        html = _edgar_get_text(doc_url)
        if not html:
            return None

        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        mda_patterns = [
            r"management.{0,40}discussion",
            r"results of operations",
        ]
        mda_start = -1
        low = text.lower()
        for pat in mda_patterns:
            match = re.search(pat, low)
            if match:
                mda_start = match.start()
                break

        if mda_start >= 0:
            extract = text[mda_start : mda_start + max_chars]
        else:
            mid = len(text) // 3
            extract = text[mid : mid + max_chars]

        return extract.strip() or None
    except Exception as e:
        logger.debug("MD&A extraction failed for %s: %s", ticker, e)
        return None


def get_sec_sentiment(
    ticker: str,
    llm_client: Optional[Callable[[str], str]] = None,
) -> dict:
    """Filing sentiment -1..1; ``source`` llm | keyword | no_filing | unavailable."""
    result: Dict[str, Any] = {
        "ticker": ticker,
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "key_themes": [],
        "filing_date": None,
        "source": "unavailable",
    }

    try:
        mda_text = get_mda_text(ticker)
        if not mda_text:
            result["source"] = "no_filing"
            return result

        filing = get_latest_filing(ticker, "10-Q") or get_latest_filing(ticker, "10-K")
        if filing:
            result["filing_date"] = filing.get("date")

        if llm_client:
            try:
                prompt = (
                    f"Analyze this excerpt from {ticker}'s SEC filing MD&A. "
                    "Return JSON with:\n"
                    "- sentiment_score: float -1.0 to 1.0 (bullish outlook = positive)\n"
                    "- key_themes: list of up to 3 short strings\n"
                    "- outlook: one of positive/neutral/negative\n\n"
                    f"Text:\n{mda_text[:2000]}\n\nReturn only valid JSON."
                )
                response = llm_client(prompt)
                if response:
                    cleaned = (
                        response.strip()
                        .removeprefix("```json")
                        .removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )
                    parsed = json.loads(cleaned)
                    result["sentiment_score"] = float(parsed.get("sentiment_score", 0))
                    result["key_themes"] = list(parsed.get("key_themes", []) or [])[:5]
                    lbl = str(parsed.get("outlook", "neutral")).lower()
                    result["sentiment_label"] = (
                        lbl if lbl in ("positive", "neutral", "negative") else "neutral"
                    )
                    result["source"] = "llm"
                    return result
            except Exception as e:
                logger.debug("LLM SEC analysis failed: %s", e)

        text_lower = mda_text.lower()
        positive_words = [
            "growth",
            "increase",
            "strong",
            "improved",
            "record",
            "exceeded",
            "profitable",
            "expansion",
            "robust",
            "momentum",
            "outperform",
            "confident",
        ]
        negative_words = [
            "decline",
            "decrease",
            "loss",
            "challenge",
            "risk",
            "uncertain",
            "headwind",
            "pressure",
            "weak",
            "impairment",
            "restructur",
            "lawsuit",
        ]

        pos_count = sum(text_lower.count(w) for w in positive_words)
        neg_count = sum(text_lower.count(w) for w in negative_words)
        total = pos_count + neg_count

        if total > 0:
            score = (pos_count - neg_count) / total
            result["sentiment_score"] = float(max(-1.0, min(1.0, score)))
            result["sentiment_label"] = (
                "positive"
                if score > 0.1
                else "negative"
                if score < -0.1
                else "neutral"
            )
            result["source"] = "keyword"

        return result
    except Exception as e:
        logger.warning("SEC sentiment failed for %s: %s", ticker, e)
        result["source"] = "unavailable"
        return result


def _build_llm_client() -> Optional[Callable[[str], str]]:
    try:
        from config.llm_config import get_llm_config

        cfg = get_llm_config()
        if not (cfg.has_openai() or cfg.has_anthropic()):
            return None

        def _llm(prompt: str) -> str:
            from agents.llm.active_llm_calls import call_active_llm_simple

            return str(call_active_llm_simple(prompt))

        return _llm
    except Exception as e:
        logger.debug("SEC LLM client unavailable: %s", e)
        return None


def _get_sec_signal_impl(ticker: str) -> dict:
    """Uncached SEC signal bundle (network + optional LLM)."""
    try:
        llm_fn = _build_llm_client()
        sentiment = get_sec_sentiment(ticker, llm_client=llm_fn)
        return {
            "sec_sentiment": float(sentiment.get("sentiment_score", 0.0)),
            "sec_label": str(sentiment.get("sentiment_label", "neutral")),
            "sec_themes": list(sentiment.get("key_themes", []) or []),
            "sec_filing_date": sentiment.get("filing_date"),
            "sec_source": str(sentiment.get("source", "unavailable")),
        }
    except Exception as e:
        logger.debug("SEC signal failed for %s: %s", ticker, e)
        return {
            "sec_sentiment": 0.0,
            "sec_label": "unavailable",
            "sec_themes": [],
            "sec_filing_date": None,
            "sec_source": "error",
        }


_SEC_SIGNAL_CACHE: Dict[str, Tuple[dict, float]] = {}

try:
    import streamlit as st

    @st.cache_data(ttl=86400, show_spinner=False)
    def _streamlit_cached_sec_signal(ticker: str) -> dict:
        return _get_sec_signal_impl(ticker)

except ImportError:
    _streamlit_cached_sec_signal = None  # type: ignore[misc, assignment]


def get_sec_signal(ticker: str) -> dict:
    """
    Trading-oriented bundle from SEC analysis.

    Keys: sec_sentiment, sec_label, sec_themes, sec_filing_date, sec_source.
    Cached 24h in-process and via Streamlit cache_data when applicable.
    """
    t = str(ticker).upper().strip()
    hit = _SEC_SIGNAL_CACHE.get(t)
    if hit:
        result, ts = hit
        if time.time() - ts < 86400:
            return dict(result)

    out: dict
    if _streamlit_cached_sec_signal is not None:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            if get_script_run_ctx() is not None:
                out = dict(_streamlit_cached_sec_signal(t))
            else:
                out = dict(_get_sec_signal_impl(t))
        except Exception:
            out = dict(_get_sec_signal_impl(t))
    else:
        out = dict(_get_sec_signal_impl(t))

    _SEC_SIGNAL_CACHE[t] = (out, time.time())
    return dict(out)


def get_institutional_ownership(ticker: str) -> dict:
    """
    Fetches 13F institutional ownership
    data from SEC EDGAR.
    Returns top holders, ownership
    concentration, and recent changes.
    No API key required.
    """
    sym = (ticker or "").strip().upper()
    if not sym:
        return _neutral_institutional(sym)

    _cache_key = f"13f_{sym}"
    _now = time.time()
    if (
        _cache_key in _SEC_CACHE
        and _now - _SEC_CACHE.get(f"{_cache_key}_ts", 0) < 14400
    ):
        return dict(_SEC_CACHE[_cache_key])

    try:
        if not get_cik(sym):
            return _neutral_institutional(sym)

        import yfinance as yf

        _t = yf.Ticker(sym)
        _holders = None
        try:
            _holders = _t.institutional_holders
        except Exception:
            pass

        if _holders is None or (
            hasattr(_holders, "empty") and _holders.empty
        ):
            return _neutral_institutional(sym)

        _top: List[dict] = []
        for _, row in _holders.head(10).iterrows():
            try:
                _holder = row.get(
                    "Holder",
                    row.get("Name", "Unknown"),
                )
                _sh = row.get(
                    "Shares",
                    row.get("Value", 0),
                )
                _pct = row.get(
                    "% Out",
                    row.get("pctHeld", 0),
                )
                _pct_f = float(_pct or 0)
                if _pct_f > 1.0:
                    _pct_f = _pct_f / 100.0
                _top.append(
                    {
                        "holder": str(_holder),
                        "shares": int(float(_sh)) if _sh is not None else 0,
                        "pct_out": _pct_f,
                    }
                )
            except Exception:
                continue

        _total_pct = sum(h["pct_out"] for h in _top)

        _signal = "NEUTRAL"
        _strength = 5.0
        if _total_pct > 0.70:
            _signal = "BUY"
            _strength = 7.0
        elif _total_pct > 0.50:
            _signal = "BUY"
            _strength = 6.0
        elif _total_pct < 0.20:
            _signal = "SELL"
            _strength = 4.0

        result = {
            "symbol": sym,
            "top_holders": _top,
            "institutional_pct": round(_total_pct, 3),
            "n_holders": len(_top),
            "signal": _signal,
            "signal_strength": _strength,
            "source": "yfinance_13f",
            "success": True,
        }
        _SEC_CACHE[_cache_key] = result
        _SEC_CACHE[f"{_cache_key}_ts"] = _now
        return result

    except Exception as e:
        logger.debug(
            "Institutional ownership failed for %s: %s",
            sym,
            e,
        )
        return _neutral_institutional(sym)


def _neutral_institutional(sym: str) -> dict:
    return {
        "symbol": sym,
        "top_holders": [],
        "institutional_pct": 0.0,
        "n_holders": 0,
        "signal": "NEUTRAL",
        "signal_strength": 5.0,
        "source": "unavailable",
        "success": False,
    }


def get_earnings_transcript_sentiment(ticker: str) -> dict:
    """
    Fetches most recent earnings call
    transcript from SEC EDGAR 8-K
    filings and extracts sentiment
    using LLM when available, else
    keyword heuristic.
    Returns management tone, key themes,
    and forward guidance sentiment.
    """
    sym = (ticker or "").strip().upper()
    if not sym:
        return _neutral_transcript(sym)

    _cache_key = f"transcript_{sym}"
    _now = time.time()
    if (
        _cache_key in _SEC_CACHE
        and _now - _SEC_CACHE.get(f"{_cache_key}_ts", 0) < 86400
    ):
        return dict(_SEC_CACHE[_cache_key])

    try:
        cik = get_cik(sym)
        if not cik:
            return _neutral_transcript(sym)

        _sub_url = f"{EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
        _data = _edgar_get_json(_sub_url)
        if not _data:
            return _neutral_transcript(sym)

        _filings = _data.get("filings", {}).get("recent", {})
        _forms = _filings.get("form", [])
        _dates = _filings.get("filingDate", [])

        _8k_idx = None
        for i, f in enumerate(_forms):
            if f == "8-K":
                _8k_idx = i
                break

        _filing_date = (
            str(_dates[_8k_idx])
            if _8k_idx is not None and _8k_idx < len(_dates)
            else "unknown"
        )

        _text = get_mda_text(sym)
        if not _text or len(_text) < 100:
            return _neutral_transcript(sym)

        _sentiment = "neutral"
        _themes: List[str] = []
        _guidance = "unclear"
        _score_adj = 0.0

        _parsed_llm = False
        try:
            from agents.llm.active_llm_calls import (
                call_active_llm_simple,
            )

            _prompt = (
                f"Analyze this earnings filing excerpt for {sym}. "
                "In 3 sentences: 1) Management tone "
                "(positive/neutral/negative), "
                "2) Key themes (growth/risk/uncertainty), "
                "3) Forward guidance sentiment. "
                "Reply in JSON: "
                '{"tone": str, "themes": [str], '
                '"guidance": str, "score_adj": float '
                "between -1.5 and 1.5}\n\n"
                f"Text: {_text[:2000]}"
            )
            _resp = call_active_llm_simple(
                _prompt,
                max_tokens=200,
            )
            if _resp:
                _clean = (
                    _resp.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
                _parsed = json.loads(_clean)
                _sentiment = str(
                    _parsed.get("tone", "neutral"),
                )
                _themes = list(_parsed.get("themes", []) or [])
                _guidance = str(
                    _parsed.get("guidance", "unclear"),
                )
                _score_adj = float(_parsed.get("score_adj", 0.0))
                _parsed_llm = True
        except Exception:
            pass

        if not _parsed_llm:
            _text_lower = _text.lower()
            _pos = sum(
                1
                for w in [
                    "growth",
                    "strong",
                    "exceeded",
                    "raised",
                    "confident",
                    "record",
                ]
                if w in _text_lower
            )
            _neg = sum(
                1
                for w in [
                    "headwind",
                    "uncertain",
                    "missed",
                    "lowered",
                    "concern",
                    "difficult",
                ]
                if w in _text_lower
            )
            if _pos > _neg + 1:
                _sentiment = "positive"
                _score_adj = 0.5
            elif _neg > _pos + 1:
                _sentiment = "negative"
                _score_adj = -0.5

        result = {
            "symbol": sym,
            "filing_date": _filing_date,
            "sentiment": _sentiment,
            "themes": _themes,
            "guidance": _guidance,
            "score_adj": _score_adj,
            "source": "sec_edgar_8k",
            "success": True,
        }
        _SEC_CACHE[_cache_key] = result
        _SEC_CACHE[f"{_cache_key}_ts"] = _now
        return result

    except Exception as e:
        logger.debug(
            "Transcript sentiment failed for %s: %s",
            sym,
            e,
        )
        return _neutral_transcript(sym)


def _neutral_transcript(sym: str) -> dict:
    return {
        "symbol": sym,
        "filing_date": None,
        "sentiment": "neutral",
        "themes": [],
        "guidance": "unclear",
        "score_adj": 0.0,
        "source": "unavailable",
        "success": False,
    }
