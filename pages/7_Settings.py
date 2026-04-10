# -*- coding: utf-8 -*-
"""
Settings page — Watchlist, Alerts, System (from Alerts, Admin, Watchlist).
"""
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js
from trading.data.ticker_resolver import resolve_ticker

try:
    from trading.utils.notification_system import NotificationSystem
except Exception:
    NotificationSystem = None
try:
    import psutil
except Exception:
    psutil = None
try:
    from config.user_store import load_user_preferences, save_user_preferences
except Exception:
    load_user_preferences = save_user_preferences = None

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

st.title("⚙️ Settings")
st.caption("Watchlist, alerts, and system configuration")

tab_wl, tab_keys, tab_alerts, tab_track, tab_research, tab_admin = st.tabs([
    "Watchlist",
    "🔑 API Keys",
    "Alerts",
    "Performance",
    "Research preferences",
    "System",
])

with tab_wl:
    st.subheader("Watchlist")
    try:
        if load_user_preferences:
            prefs = load_user_preferences(st.session_state.get("evolve_session_id", "") or "default")
            if prefs:
                st.caption("User preferences loaded from user_store.")
        from components.watchlist_widget import render_watchlist
        render_watchlist()
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_keys:
    st.subheader("API Keys")
    st.caption(
        "Your keys are encrypted and stored "
        "locally. They persist across sessions "
        "and are never shared."
    )
    try:
        from utils.session_utils import (
            get_stable_user_id
        )
        from config.user_store import (
            save_user_api_keys,
            load_user_api_keys,
        )
        _uid = st.session_state.get("evolve_session_id") or get_stable_user_id()
        _saved = load_user_api_keys(_uid) or {}

        # Show masked existing keys
        _has_anthropic = bool(
            _saved.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        _has_openai = bool(
            _saved.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        _has_news = bool(
            _saved.get("NEWS_API_KEY")
            or os.environ.get("NEWS_API_KEY")
        )
        _has_reddit = bool(
            _saved.get("REDDIT_CLIENT_ID")
            or os.environ.get("REDDIT_CLIENT_ID")
        )

        st.markdown("#### OpenAI")
        if _has_openai:
            st.success("✅ OpenAI key saved")
        openai_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-... (leave blank to keep existing)",
            key="settings_openai_key",
            help="Chat, commentary, morning briefing narrative.",
        )

        st.markdown("#### Anthropic (optional)")
        if _has_anthropic:
            st.success("✅ Anthropic key saved")
        anthropic_key = st.text_input(
            "Anthropic API key",
            type="password",
            placeholder="sk-ant-... (leave blank to keep existing)",
            key="settings_anthropic_key",
            help="Alternative LLM — Claude for chat and agents.",
        )

        st.markdown("#### News API (optional)")
        if _has_news:
            st.success("✅ News API key saved")
        news_key = st.text_input(
            "News API key",
            type="password",
            placeholder="NewsAPI key (leave blank to keep existing)",
            key="settings_news_key",
        )

        st.markdown("#### Reddit (optional)")
        if _has_reddit:
            st.success("✅ Reddit app credentials saved")
        reddit_client_id = st.text_input(
            "Reddit client ID",
            type="password",
            placeholder="Reddit app client ID",
            key="settings_reddit_id",
        )
        reddit_secret = st.text_input(
            "Reddit client secret",
            type="password",
            placeholder="Reddit app secret",
            key="settings_reddit_secret",
        )

        _pref_llm = (
            load_user_preferences(_uid) or {}
        ).get("preferred_llm_provider") or "openai"
        provider = st.selectbox(
            "Preferred LLM provider",
            ["openai", "anthropic"],
            index=(0 if _pref_llm == "openai" else 1),
            help="Default provider for chat and agents.",
        )

        if st.button(
            "Save API Keys",
            key="settings_save_keys"
        ):
            keys_to_save = {}
            if anthropic_key.strip():
                keys_to_save[
                    "ANTHROPIC_API_KEY"
                ] = anthropic_key.strip()
            if openai_key.strip():
                keys_to_save[
                    "OPENAI_API_KEY"
                ] = openai_key.strip()
            if news_key.strip():
                keys_to_save["NEWS_API_KEY"] = news_key.strip()
            if reddit_client_id.strip():
                keys_to_save["REDDIT_CLIENT_ID"] = reddit_client_id.strip()
            if reddit_secret.strip():
                keys_to_save["REDDIT_CLIENT_SECRET"] = reddit_secret.strip()
            if keys_to_save:
                save_user_api_keys(
                    _uid, keys_to_save
                )
                for k, v in keys_to_save.items():
                    st.session_state[f"user_key_{k}"] = v
                _is_cloud = (
                    os.environ.get("STREAMLIT_SHARING_MODE")
                    or os.environ.get("IS_STREAMLIT_CLOUD")
                    or not os.path.exists(".env")
                )
                if _is_cloud:
                    from config.user_store import inject_user_keys_to_session
                    inject_user_keys_to_session(_uid)
                else:
                    from config.user_store import inject_user_keys_to_env
                    inject_user_keys_to_env(_uid)
                try:
                    from config.user_store import save_user_preferences, load_user_preferences
                    _p = load_user_preferences(_uid) or {}
                    save_user_preferences(
                        _uid,
                        {**_p, "preferred_llm_provider": provider},
                    )
                except Exception as _e:
                    st.caption(f"Preference note: {_e}")
                st.success(
                    "Keys saved and activated. "
                    "They will load automatically "
                    "next time you open the app."
                )
            else:
                st.info(
                    "No new keys entered. "
                    "Existing keys unchanged."
                )

        # Clear keys option
        if st.button(
            "Clear Saved Keys",
            key="settings_clear_keys"
        ):
            save_user_api_keys(_uid, {
                "ANTHROPIC_API_KEY": "",
                "OPENAI_API_KEY": "",
                "NEWS_API_KEY": "",
                "REDDIT_CLIENT_ID": "",
                "REDDIT_CLIENT_SECRET": "",
            })
            st.warning("Keys cleared.")

    except Exception as e:
        st.caption(
            f"API key storage unavailable: {e}"
        )

with tab_alerts:
    st.subheader("Alerts")
    st.caption(
        "Saved price and score alerts (stored with your user preferences)."
    )
    try:
        from utils.session_utils import get_stable_user_id
        from utils.dataframe_utils import normalize_for_display

        import pandas as pd

        _sid = get_stable_user_id()
        if not _sid:
            st.caption(
                "Sign in or complete onboarding so alerts can be saved."
            )
        elif not (load_user_preferences and save_user_preferences):
            st.caption("User preferences storage is not available.")
        else:
            _prefs = load_user_preferences(_sid) or {}
            _alerts = list(_prefs.get("evolve_alerts", []))
            if not isinstance(_alerts, list):
                _alerts = []

            _cond_labels = {
                "price_above": "Price Above",
                "price_below": "Price Below",
                "ai_score_above": "AI Score Above",
                "pct_change": "% Change",
            }

            if _alerts:
                st.markdown("#### Saved alerts")
                _rows = []
                for _a in _alerts:
                    if not isinstance(_a, dict):
                        continue
                    _rows.append({
                        "Symbol": _a.get("symbol", ""),
                        "Condition": _cond_labels.get(
                            _a.get("condition", ""),
                            _a.get("condition", ""),
                        ),
                        "Threshold": _a.get("threshold", ""),
                        "Created": _a.get("created_at", "")[:19]
                        if _a.get("created_at")
                        else "",
                        "id": _a.get("id", ""),
                    })
                if _rows:
                    _adf = pd.DataFrame(_rows)
                    _display = _adf.drop(columns=["id"], errors="ignore")
                    st.dataframe(
                        normalize_for_display(_display),
                        width="stretch",
                        hide_index=True,
                    )

                _del_opts = [
                    f"{r.get('Symbol', '')} — {r.get('Condition', '')} "
                    f"(threshold {r.get('Threshold', '')})"
                    for r in _rows
                    if r.get("id")
                ]
                _ids = [r.get("id") for r in _rows if r.get("id")]
                if _del_opts and _ids:
                    _pick = st.selectbox(
                        "Remove alert",
                        list(range(len(_ids))),
                        format_func=lambda i: _del_opts[i],
                        key="settings_alert_delete_pick",
                    )
                    if st.button(
                        "Delete selected alert",
                        key="settings_alert_delete_btn",
                    ):
                        _rid = _ids[_pick]
                        _alerts = [
                            _x for _x in _alerts
                            if isinstance(_x, dict)
                            and _x.get("id") != _rid
                        ]
                        _prefs["evolve_alerts"] = _alerts
                        try:
                            save_user_preferences(_sid, _prefs)
                            st.success("Alert removed.")
                            st.rerun()
                        except Exception as _se:
                            logger.warning(
                                "settings: save after delete failed: %s",
                                _se,
                            )
                            st.caption(
                                f"Could not save removal: {_se}"
                            )
            else:
                st.info("No saved alerts yet.")

            st.markdown("---")
            st.markdown("#### New alert")
            _sym = st.text_input(
                "Symbol",
                value="AAPL",
                key="settings_alert_symbol",
            ).strip().upper()
            _sym = resolve_ticker(_sym, validate=False)
            _cond_ui = st.selectbox(
                "Condition",
                [
                    "Price Above",
                    "Price Below",
                    "AI Score Above",
                    "% Change",
                ],
                key="settings_alert_condition",
            )
            _cond_key = {
                "Price Above": "price_above",
                "Price Below": "price_below",
                "AI Score Above": "ai_score_above",
                "% Change": "pct_change",
            }[_cond_ui]
            _thr = st.number_input(
                "Threshold value",
                value=150.0
                if _cond_key in ("price_above", "price_below")
                else (6.5 if _cond_key == "ai_score_above" else 3.0),
                step=0.25,
                key="settings_alert_threshold",
            )
            if st.button("Save alert", key="settings_alert_save", type="primary"):
                if not _sym:
                    st.warning("Enter a symbol.")
                else:
                    _new = {
                        "id": str(uuid.uuid4()),
                        "symbol": _sym,
                        "condition": _cond_key,
                        "threshold": float(_thr),
                        "created_at": datetime.now().isoformat(),
                    }
                    _alerts.append(_new)
                    _prefs["evolve_alerts"] = _alerts
                    try:
                        save_user_preferences(_sid, _prefs)
                        st.success(f"Saved alert for {_sym}.")
                        st.rerun()
                    except Exception as _se:
                        logger.warning(
                            "settings: save alert failed: %s", _se
                        )
                        st.caption(f"Could not save alert: {_se}")
    except Exception as e:
        logger.warning("settings alerts tab failed: %s", e)
        st.caption(f"Alerts unavailable: {e}")

with tab_track:
    st.subheader("Performance tracking")
    st.caption(
        "Tracked recommendations from Deep Dive — live P&L vs entry, "
        "target, and stop."
    )
    try:
        from utils.session_utils import get_stable_user_id
        from utils.dataframe_utils import normalize_for_display
        import pandas as pd
        from trading.services.recommendation_tracker import (
            RecommendationTracker,
        )

        _tsid = get_stable_user_id()
        if not _tsid:
            st.info(
                "Complete onboarding so recommendations can be tied to "
                "your account."
            )
        else:
            _trk = RecommendationTracker()
            _clr_c, _ = st.columns([1, 3])
            with _clr_c:
                if st.button(
                    "Clear closed recommendations",
                    key="rec_clear_closed",
                    help="Remove closed rows from stored tracking history.",
                ):
                    try:
                        _nrm = _trk.clear_closed_recommendations(_tsid)
                        st.success(
                            f"Removed {_nrm} closed recommendation(s)."
                        )
                        st.rerun()
                    except Exception as _cle:
                        logger.warning(
                            "settings: clear closed recs failed: %s", _cle
                        )
                        st.caption(f"Could not clear: {_cle}")
            _summ = _trk.get_performance_summary(_tsid)
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.metric(
                    "Tracked (open)",
                    str(_summ.get("total_recommendations") or 0),
                )
            with t2:
                _wr = _summ.get("win_rate")
                st.metric(
                    "Win rate (target vs stop)",
                    f"{float(_wr)*100:.1f}%"
                    if _wr is not None
                    else "—",
                )
            with t3:
                _ar = _summ.get("avg_return_pct")
                st.metric(
                    "Avg P&L %",
                    f"{_ar:.2f}" if _ar is not None else "—",
                )
            with t4:
                st.caption(
                    f"Best {_summ.get('best_trade_pct')} · "
                    f"worst {_summ.get('worst_trade_pct')}"
                )
            _outs = _trk.check_outcomes(_tsid)
            if _outs:
                _odf = pd.DataFrame(_outs)
                st.dataframe(
                    normalize_for_display(_odf),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(
                    "No open tracked picks. Use **Track recommendation** "
                    "on a ticker’s Deep Dive card."
                )
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_research:
    st.markdown("### Briefing & scanner preferences")
    st.caption(
        "Customize your morning briefing and scanner defaults."
    )
    try:
        from utils.session_utils import get_stable_user_id
        from config.user_store import load_user_preferences, save_user_preferences

        _uid = st.session_state.get("evolve_session_id") or get_stable_user_id()
        _prefs = load_user_preferences(_uid) or {}
    except Exception:
        _uid = ""
        _prefs = {}

    universe_opts = [
        "Top 25 (fastest, ~30s)",
        "SP100 (balanced, ~60s)",
        "NASDAQ100 (tech-heavy)",
        "SP500 (broadest, ~3min)",
    ]
    style_opts = [
        "Balanced (default)",
        "Momentum-heavy",
        "Technical-heavy",
        "Fundamental-heavy",
    ]

    _def_u = "Top 25 (fastest, ~30s)"
    _stored_u = _prefs.get("briefing_universe", _def_u)
    _u_idx = (
        universe_opts.index(_stored_u) if _stored_u in universe_opts else 0
    )

    _def_s = "Balanced (default)"
    _stored_s = _prefs.get("scoring_style", _def_s)
    _s_idx = style_opts.index(_stored_s) if _stored_s in style_opts else 0

    min_score = st.slider(
        "Minimum AI Score for briefing",
        min_value=4.0,
        max_value=8.0,
        value=float(_prefs.get("min_ai_score", 5.5)),
        step=0.5,
        key="pref_min_score",
        help="Lower = more stocks shown. Higher = only strongest signals.",
    )

    universe_choice = st.selectbox(
        "Briefing scan universe",
        options=universe_opts,
        index=_u_idx,
        key="pref_universe",
    )

    scoring_style = st.radio(
        "Score weighting style",
        options=style_opts,
        index=_s_idx,
        key="pref_scoring",
        horizontal=True,
    )

    direction_opts = [
        "Bullish only (BUY signals)",
        "Both directions",
        "Bearish only (short signals)",
    ]
    _def_d = "Bullish only (BUY signals)"
    _stored_d = _prefs.get("opportunity_direction", _def_d)
    _d_idx = (
        direction_opts.index(_stored_d) if _stored_d in direction_opts else 0
    )
    opportunity_direction = st.radio(
        "Show opportunities",
        options=direction_opts,
        index=_d_idx,
        key="pref_direction",
        horizontal=True,
        help="Bullish only filters out picks where forecast target is below entry.",
    )

    ALL_SECTORS = [
        "Technology",
        "Healthcare",
        "Finance",
        "Energy",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Industrial",
        "Communication Services",
        "Materials",
        "Real Estate",
        "Utilities",
    ]
    _stored_sectors = _prefs.get("preferred_sectors") or []
    if not isinstance(_stored_sectors, list):
        _stored_sectors = []
    preferred_sectors = st.multiselect(
        "Focus sectors",
        options=ALL_SECTORS,
        default=[s for s in _stored_sectors if s in ALL_SECTORS],
        key="pref_sectors",
        help="Leave empty to scan all sectors. Select to limit briefing candidates.",
    )

    watchlist_only = st.toggle(
        "Briefing from watchlist only",
        value=bool(_prefs.get("watchlist_only", False)),
        key="pref_watchlist_only",
        help="When ON, briefing scans your saved watchlist tickers only.",
    )
    if watchlist_only:
        st.caption(
            "Your watchlist tickers will be scanned instead of the universe. "
            "Add tickers under Watchlist to include them."
        )

    st.markdown("---")
    st.markdown("### 📊 Signal IC Status")
    st.caption(
        "Dynamic signal weights activate after 30+ scored observations per symbol "
        "or 100+ globally."
    )
    try:
        import os
        import sqlite3

        from trading.analysis.signal_score_store import (
            _DB_PATH,
            _TABLE,
            fill_forward_returns,
        )

        if os.path.exists(_DB_PATH):
            with sqlite3.connect(_DB_PATH) as _c:
                _n = _c.execute(
                    f"SELECT COUNT(*) FROM {_TABLE}"
                ).fetchone()[0]
                _n_filled = _c.execute(
                    f"SELECT COUNT(*) FROM {_TABLE} "
                    "WHERE return_7d IS NOT NULL"
                ).fetchone()[0]
            st.metric("Scored observations", _n)
            st.metric("With realized returns", _n_filled)
            if st.button(
                "Fill forward returns now",
                key="fill_returns",
            ):
                _filled = fill_forward_returns()
                st.success(
                    f"Filled {_filled} return observations"
                )
        else:
            st.caption(
                "No data yet — score some tickers in Analyze to begin "
                "accumulating data."
            )
    except Exception as _ie:
        st.caption(f"IC status unavailable: {_ie}")

    st.markdown("---")
    if st.button("Save preferences", key="save_research_prefs", type="primary"):
        try:
            from config.user_store import save_user_preferences as _save_rp
            from utils.session_utils import get_stable_user_id as _gid

            _uid_save = st.session_state.get("evolve_session_id") or _gid()
            _save_rp(
                _uid_save,
                {
                    **_prefs,
                    "min_ai_score": min_score,
                    "briefing_universe": universe_choice,
                    "scoring_style": scoring_style,
                    "opportunity_direction": opportunity_direction,
                    "preferred_sectors": preferred_sectors,
                    "watchlist_only": watchlist_only,
                },
            )
            st.success("Preferences saved. Refresh Home to apply.")
        except Exception as e:
            st.caption(f"Could not save: {e}")

with tab_admin:
    st.subheader("System")
    try:
        st.caption(
            "Process metrics, optional packages, and cache footprint."
        )

        if psutil:
            _cpu = psutil.cpu_percent(interval=0.1)
            _mem = psutil.virtual_memory()
            st.markdown("#### Resource usage")
            _m1, _m2 = st.columns(2)
            with _m1:
                st.metric("CPU usage", f"{_cpu:.1f}%")
            with _m2:
                st.metric("RAM usage", f"{_mem.percent:.1f}%")
            try:
                _root = (
                    os.environ.get("SystemDrive", "C:") + "\\"
                    if os.name == "nt"
                    else "/"
                )
                _disk = psutil.disk_usage(_root)
                st.metric(
                    "Disk usage",
                    f"{_disk.percent:.1f}%",
                    help=f"Root: {_root}",
                )
            except Exception as _de:
                logger.warning("settings: disk usage failed: %s", _de)
                st.caption(f"Disk metrics unavailable: {_de}")
        else:
            st.caption(
                "Install psutil for CPU, memory, and disk metrics."
            )

        st.markdown("#### App")
        st.metric("Version", "v3.19.0")

        st.markdown("#### Optional packages")
        try:
            import importlib.metadata as _imd

            _nf_ver = _imd.version("neuralforecast")
            _nf_label = (
                f"neuralforecast {_nf_ver} "
                f"(N-BEATS, N-HiTS, PatchTST, TFT)"
            )
            _nf_ok = True
        except Exception:
            _nf_label = (
                "neuralforecast (N-BEATS, N-HiTS, PatchTST, TFT)"
            )
            _nf_ok = False
        _opt = [
            ("shap", "shap"),
            (_nf_label, "__nf__"),
            ("faiss", "faiss"),
            ("cvxpy", "cvxpy"),
            ("pandas_ta", "pandas_ta"),
        ]
        _oc1, _oc2 = st.columns(2)
        for _i, (_label, _mod) in enumerate(_opt):
            if _mod == "__nf__":
                _ok = _nf_ok
            else:
                _ok = False
                try:
                    __import__(_mod)
                    _ok = True
                except Exception:
                    pass
            _target = _oc1 if _i % 2 == 0 else _oc2
            with _target:
                st.write(
                    f"{'✅' if _ok else '❌'} {_label}"
                )

        st.markdown("#### Model cache (.cache/)")
        _cache_root = project_root / ".cache"
        _n_files = 0
        _size_b = 0
        try:
            if _cache_root.exists():
                for _p in _cache_root.rglob("*"):
                    if _p.is_file():
                        _n_files += 1
                        try:
                            _size_b += _p.stat().st_size
                        except OSError as _oe:
                            logger.debug(
                                "settings: cache stat skip: %s", _oe
                            )
            _mb = _size_b / (1024 * 1024)
            st.metric("Cache files", f"{_n_files}")
            st.metric("Approx. size", f"{_mb:.2f} MB")
        except Exception as _ce:
            logger.warning("settings: cache scan failed: %s", _ce)
            st.caption(f"Could not scan .cache: {_ce}")

        st.markdown("---")
        st.markdown("#### Platform Monitoring")
        try:
            import sys as _sys
            from pathlib import Path as _Path

            _mroot = project_root / "_archive" / "monitoring"
            _mon_ok = False
            if _mroot.is_dir():
                _mp = str(_mroot)
                if _mp not in _sys.path:
                    _sys.path.insert(0, _mp)
                try:
                    from health_check import HealthChecker

                    _hc = HealthChecker()
                    _health = _hc.check_system_health()
                    st.json(
                        {
                            k: _health[k]
                            for k in ("timestamp", "status", "uptime_seconds")
                            if k in _health
                        }
                    )
                    _comps = _health.get("components") or {}
                    if _comps:
                        st.caption("Components")
                        st.json(_comps)
                    _mon_ok = True
                except Exception as _me:
                    logger.warning("settings: monitoring failed: %s", _me)
            if not _mon_ok:
                st.info(
                    "Monitoring module not yet available."
                )
        except Exception as _me2:
            st.caption(f"Monitoring unavailable: {_me2}")

        st.markdown("---")
        st.markdown("**ML Score Model**")
        try:
            from trading.analysis.ml_score_trainer import (
                MLScoreTrainer,
                SP100_SAMPLE,
            )

            _ml_dir = project_root / ".cache" / "ml_score"
            _ml_model = _ml_dir / "ml_score_model.joblib"
            _ml_meta_f = _ml_dir / "ml_score_meta.json"
            model_exists = _ml_model.is_file()
            _ml_meta = {}
            if _ml_meta_f.is_file():
                try:
                    _ml_meta = json.loads(
                        _ml_meta_f.read_text(encoding="utf-8")
                    )
                except Exception as _je:
                    logger.debug("settings: ml meta read: %s", _je)
            if model_exists and _ml_meta.get("training_date"):
                _vda = _ml_meta.get("val_directional_accuracy")
                _da_pct = (
                    f"{float(_vda)*100:.1f}%"
                    if _vda is not None
                    else "—"
                )
                st.success("ML Score model is trained and active.")
                st.caption(
                    f"Training date: {_ml_meta.get('training_date', '')[:19]} · "
                    f"Directional accuracy (val): {_da_pct} · "
                    f"n_samples: {_ml_meta.get('n_samples', '—')}"
                )
            else:
                st.warning(
                    "AI Score is using rules-based scoring. Train the ML "
                    "model to enable blended ML scoring."
                )
            _uni_choice = st.selectbox(
                "Training universe size",
                [
                    "Quick (20 stocks)",
                    "Standard (50 stocks)",
                    "Full SP100 (100 stocks)",
                ],
                key="ml_train_universe",
            )
            _sp100_fp = project_root / "data" / "universes" / "sp100.json"
            _train_uni = []
            try:
                _raw_u = json.loads(
                    _sp100_fp.read_text(encoding="utf-8")
                )
                if isinstance(_raw_u, list):
                    if "Quick" in _uni_choice:
                        _train_uni = [str(x) for x in _raw_u[:20]]
                    elif "Standard" in _uni_choice:
                        _train_uni = [str(x) for x in _raw_u[:50]]
                    else:
                        _train_uni = [str(x) for x in _raw_u[:100]]
            except Exception as _ue:
                logger.warning("settings: universe load failed: %s", _ue)
            if st.button("Train ML Score Model", key="train_ml_score"):
                _n = len(_train_uni) if _train_uni else len(SP100_SAMPLE)
                _prog_ml = st.progress(0, text="Preparing…")
                _ml_status = st.empty()

                def _ml_progress(sym: str, cur: int, total: int) -> None:
                    if total <= 0:
                        return
                    p = min(float(cur) / float(total), 1.0)
                    _prog_ml.progress(p, text=f"{sym} ({cur}/{total})")
                    _ml_status.caption(
                        f"Building dataset: {sym} — {cur} of {total} tickers"
                    )

                with st.spinner(
                    "Training ML Score model on historical data… "
                    f"(~3–10 min for {_n} stocks)"
                ):
                    try:
                        trainer = MLScoreTrainer()
                        result = trainer.train(
                            universe=_train_uni if _train_uni else None,
                            progress_callback=_ml_progress,
                        )
                    except Exception as _te:
                        result = {"error": str(_te)}
                _prog_ml.progress(1.0, text="Done")
                if result.get("error"):
                    st.error(f"Training failed: {result['error']}")
                else:
                    import time as _time_ml

                    st.session_state["ml_score_trained_at"] = _time_ml.time()
                    _da = float(
                        result.get("val_directional_accuracy") or 0
                    )
                    _r2 = result.get("val_r2")
                    _nf = result.get("n_features", "?")
                    st.success(
                        f"ML Score model trained. Features: {_nf}, "
                        f"R²: {_r2}"
                    )
                    st.caption(
                        f"Samples: {result.get('n_samples')} · "
                        f"Validation — directional accuracy: "
                        f"{_da*100:.1f}% · "
                        f"n_samples: {result.get('n_samples')}"
                    )
        except Exception as e:
            st.caption(f"ML Score training unavailable: {e}")

        st.markdown("---")
        st.markdown("**Model Hyperparameter Optimization**")
        try:
            import yfinance as yf

            from trading.optimization.optuna_optimizer import get_optimizer

            st.caption(
                "Optimize XGBoost hyperparameters using Bayesian search (Optuna) "
                "on a short SPY feature window."
            )
            opt_model = st.selectbox(
                "Model to optimize",
                ["xgboost", "ridge", "catboost"],
                key="opt_model_select",
            )
            if st.button("Run Optimization", key="optuna_run_btn"):
                if opt_model != "xgboost":
                    st.caption(
                        "Only the XGBoost + Optuna path is wired here; "
                        "choose xgboost for a live run."
                    )
                else:
                    with st.spinner(
                        "Optimizing xgboost… this may take several minutes"
                    ):
                        try:
                            _h = yf.Ticker("SPY").history(period="2y")
                            if _h.empty:
                                st.warning("No SPY data.")
                            else:
                                import pandas as pd

                                _c = _h["Close"].astype(float)
                                _df = pd.DataFrame(
                                    {
                                        "y": _c.pct_change().shift(-1),
                                        "x1": _c.shift(1),
                                        "x2": _c.shift(2),
                                    }
                                ).dropna()
                                X = _df[["x1", "x2"]]
                                y = _df["y"]
                                optimizer = get_optimizer()
                                result = optimizer.optimize_xgboost(
                                    X, y, n_trials=min(40, 80)
                                )
                                if result and not result.get("error"):
                                    st.success(
                                        f"Best score (RMSE): "
                                        f"{result.get('best_score', 'N/A')}"
                                    )
                                    st.json(result.get("best_params", {}))
                                else:
                                    st.caption(
                                        f"Optimization failed: "
                                        f"{result.get('error', 'unknown')}"
                                    )
                        except Exception as oe:
                            st.caption(f"Optimization failed: {oe}")
        except Exception as e:
            st.caption(f"Optimizer unavailable: {e}")

        st.markdown("---")
        st.markdown("**System Backup & Recovery**")
        try:
            import asyncio

            from trading.recovery.disaster_recovery_manager import (
                DisasterRecoveryManager,
            )

            drm = DisasterRecoveryManager()
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("Create Backup", key="backup_btn"):
                    try:
                        asyncio.run(drm.create_backup())
                        st.success("Backup created")
                    except Exception as be:
                        st.caption(f"Backup failed: {be}")
            with col_b2:
                if st.button("List Backups", key="list_backups_btn"):
                    try:
                        backups = drm.list_backups()
                        if backups:
                            st.json(
                                [b.backup_id for b in backups[:5]]
                            )
                        else:
                            st.caption("No backups found")
                    except Exception as le:
                        st.caption(f"Could not list backups: {le}")
        except Exception as re:
            st.caption(f"Recovery system unavailable: {re}")

        if st.button(
            "Clear session & reload",
            key="settings_session_reboot",
            help="Clears Streamlit session state and cached data, then reloads.",
        ):
            try:
                st.cache_data.clear()
            except Exception as _x:
                logger.warning("settings: cache_data.clear: %s", _x)
            try:
                st.session_state.clear()
            except Exception as _x:
                logger.warning("settings: session_state.clear: %s", _x)
            st.rerun()
    except Exception as e:
        logger.warning("settings system tab failed: %s", e)
        st.caption(f"System panel unavailable: {e}")


# Page Assistant
try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Settings")
except Exception:
    pass
