# -*- coding: utf-8 -*-
"""
Settings page — Watchlist, Alerts, System (from Alerts, Admin, Watchlist).
"""
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

tab_wl, tab_keys, tab_alerts, tab_admin = st.tabs([
    "Watchlist",
    "🔑 API Keys",
    "Alerts",
    "System"
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
            inject_user_keys_to_env,
        )
        _uid = get_stable_user_id()
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

        st.markdown("#### Anthropic (Claude)")
        if _has_anthropic:
            st.success("✅ Anthropic key saved")
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-... (leave blank to keep existing)",
            key="settings_anthropic_key"
        )

        st.markdown("#### OpenAI")
        if _has_openai:
            st.success("✅ OpenAI key saved")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-... (leave blank to keep existing)",
            key="settings_openai_key"
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
            if keys_to_save:
                save_user_api_keys(
                    _uid, keys_to_save
                )
                for k, v in keys_to_save.items():
                    st.session_state[f"user_key_{k}"] = v
                inject_user_keys_to_env(_uid)
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
        _opt = [
            ("shap", "shap"),
            ("neuralforecast", "neuralforecast"),
            ("faiss", "faiss"),
            ("cvxpy", "cvxpy"),
            ("pandas_ta", "pandas_ta"),
        ]
        _oc1, _oc2 = st.columns(2)
        for _i, (_label, _mod) in enumerate(_opt):
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
            from trading.analysis.ml_score_trainer import MLScoreTrainer

            model_exists = (
                Path(".cache/ml_score/ml_score_model.joblib").exists()
            )
            if model_exists:
                st.success("ML Score model is trained and active")
            else:
                st.warning(
                    "ML Score model not trained — using rules-based scoring only"
                )
            if st.button("Train ML Score Model", key="train_ml_score"):
                with st.spinner(
                    "Training on default universe... (may take several minutes)"
                ):
                    trainer = MLScoreTrainer()
                    result = trainer.train()
                    if result.get("error"):
                        st.error(f"Training failed: {result['error']}")
                    else:
                        _da = result.get("val_directional_accuracy") or 0
                        st.success(
                            f"Trained on {result.get('n_samples')} samples. "
                            f"Directional accuracy: {_da*100:.1f}%"
                        )
        except Exception as e:
            st.caption(f"ML Score training unavailable: {e}")

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
