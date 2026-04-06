"""
Evolve Trading Platform - Streamlit entry point.

Minimal launcher: page config, logging, environment loading, and sidebar branding.
All features live in the multipage app (pages/0_Home.py, pages/1_Chat.py, etc.).
Run: streamlit run app.py
"""

import atexit
import logging
import os
import sys
import warnings
from pathlib import Path

# Ensure local cache directories exist for models
os.makedirs(".cache", exist_ok=True)
os.makedirs(".cache/lstm", exist_ok=True)

# Suppress TensorFlow/keras warnings from optional dependencies
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    for _name in ("tensorflow", "tf_keras", "keras"):
        logging.getLogger(_name).setLevel(logging.ERROR)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tf_keras")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="keras")

import streamlit as st

# Project root on path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _shutdown():
    try:
        from trading.database.connection import close_database
        close_database()
    except Exception as _e:
        print(f"Shutdown warning: {_e}", file=sys.stderr)
    try:
        from trading.memory import close_memory_store
        close_memory_store()
    except Exception as _e:
        print(f"Shutdown warning: {_e}", file=sys.stderr)


atexit.register(_shutdown)

# Logging
try:
    from config.logging_config import setup_logging, get_logger
    setup_logging(config={"level": "INFO", "file": "logs/trading_system.log", "max_size": 10 * 1024 * 1024, "backup_count": 5, "console": True, "log_rotation": True})
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

logger.info("Evolve Trading System starting.")

# Warnings
warnings.filterwarnings("ignore")
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

# Environment
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Environment variables loaded from .env")
except ImportError:
    pass
except Exception as e:
    logger.error("Error loading .env: %s", e)

# Page config
st.set_page_config(
    page_title="Evolve AI Trading",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Per-user onboarding: init DB, check keys; inject into env if complete
from config.user_store import init_user_db, load_user_keys, inject_user_keys_to_env
from components.onboarding import check_onboarding

init_user_db()

# Inject persisted user API keys
try:
    from utils.session_utils import (
        get_stable_user_id
    )
    _stable_uid = get_stable_user_id()
    try:
        import os as _os
        _is_cloud = (
            _os.environ.get("STREAMLIT_SHARING_MODE") or
            _os.environ.get("IS_STREAMLIT_CLOUD") or
            not _os.path.exists(".env")
        )
        if _is_cloud:
            from config.user_store import inject_user_keys_to_session
            inject_user_keys_to_session(_stable_uid)
        else:
            inject_user_keys_to_env(_stable_uid)
    except Exception as _e:
        pass
    if _stable_uid and "evolve_session_id" not in st.session_state:
        st.session_state["evolve_session_id"] = _stable_uid
except Exception as _e:
    pass  # Never block app load for this


def _market_status() -> str:
    """US/Eastern session label for sidebar (approximate NYSE-style hours)."""
    import datetime

    import pytz

    now = datetime.datetime.now(pytz.timezone("US/Eastern"))
    wd = now.weekday()
    h, m = now.hour, now.minute
    mins = h * 60 + m
    if wd >= 5:
        return "Market closed · Weekend"
    if 240 <= mins < 270:
        return "Pre-market · 4:00–4:30 AM ET"
    if 270 <= mins < 570:
        return "Pre-market · Open"
    if 570 <= mins < 960:
        return "Market open · NYSE/NASDAQ"
    if 960 <= mins < 1200:
        return "After-hours · Open"
    return "Market closed"


# ── Global ticker search (sidebar, above nav) ───────
if "global_search_ticker" not in st.session_state:
    st.session_state["global_search_ticker"] = ""

_home = st.Page(
    "pages/1_Dashboard.py",
    title="Home",
    icon="🏠",
    default=True,
)
_analyze = st.Page("pages/2_Analyze.py", title="Analyze", icon="📊")
_scanner = st.Page("pages/3_Scanner.py", title="Scanner", icon="🔍")
_trade = st.Page("pages/4_Trade.py", title="Trade", icon="💼")
_backtest = st.Page("pages/5_Backtest.py", title="Backtest", icon="⏮")
_chat = st.Page("pages/6_Chat.py", title="Chat", icon="💬")
_settings = st.Page("pages/7_Settings.py", title="Settings", icon="⚙️")

_pg = st.navigation(
    {
        "": [_home],
        "Advanced tools": [
            _analyze,
            _scanner,
            _trade,
            _backtest,
            _chat,
        ],
        "System": [_settings],
    },
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.markdown("### Evolve")
    st.caption("Trading copilot")
    _gsearch = st.text_input(
        "Search ticker",
        placeholder="/ to search any ticker...",
        key="global_search_ticker",
        label_visibility="collapsed",
    )
    if _gsearch and len(_gsearch.strip()) >= 1:
        _sym = _gsearch.strip().upper()
        if st.sidebar.button(
            "Open " + _sym + " on Home",
            key="global_search_go",
            use_container_width=True,
        ):
            st.session_state["analyze_ticker"] = _sym
            st.session_state["deep_dive_ticker"] = _sym
            st.switch_page("pages/1_Dashboard.py")
    try:
        st.caption("Market status: **" + _market_status() + "**")
    except Exception:
        st.caption("Market status: —")
    st.markdown("---")
    try:
        st.page_link(
            "pages/1_Dashboard.py",
            label="🏠 Home",
        )
        with st.expander(
            "Advanced tools",
            expanded=False,
        ):
            st.page_link(
                "pages/2_Analyze.py",
                label="📊 Analyze",
            )
            st.page_link(
                "pages/3_Scanner.py",
                label="🔍 Scanner",
            )
            st.page_link(
                "pages/4_Trade.py",
                label="💼 Trade",
            )
            st.page_link(
                "pages/5_Backtest.py",
                label="⏮ Backtest",
            )
            st.page_link(
                "pages/6_Chat.py",
                label="💬 Chat",
            )
        st.page_link(
            "pages/7_Settings.py",
            label="⚙️ Settings",
        )
    except Exception:
        if st.button(
            "🏠 Home",
            key="nav_home",
        ):
            st.switch_page(
                "pages/1_Dashboard.py",
            )
        if st.button(
            "📊 Analyze",
            key="nav_analyze",
        ):
            st.switch_page(
                "pages/2_Analyze.py",
            )
        if st.button(
            "🔍 Scanner",
            key="nav_scanner",
        ):
            st.switch_page(
                "pages/3_Scanner.py",
            )
        if st.button(
            "⚙️ Settings",
            key="nav_settings",
        ):
            st.switch_page(
                "pages/7_Settings.py",
            )

session_id = check_onboarding()

# Always inject API keys from user store on every run
try:
    _sid = session_id or st.session_state.get("evolve_session_id", "") or ""
    import os as _os2
    _is_cloud2 = (
        _os2.environ.get("STREAMLIT_SHARING_MODE") or
        _os2.environ.get("IS_STREAMLIT_CLOUD") or
        not _os2.path.exists(".env")
    )
    if _is_cloud2:
        from config.user_store import inject_user_keys_to_session
        inject_user_keys_to_session(_sid)
    else:
        inject_user_keys_to_env(_sid)
except Exception:
    # Keys may not be set yet; continue without failing
    pass

if not session_id:
    st.stop()

try:
    from utils.session_utils import prune_streamlit_session_cache

    prune_streamlit_session_cache(max_tickers=10)
except Exception:
    pass

# ── Inject theme globally ────────────────────────────
try:
    from components.theme import inject_theme
    inject_theme()
except Exception:
    pass

# Optional: initialize notification service, audit logger, and LLM processor for pages that use them
if "notification_service" not in st.session_state:
    try:
        from trading.utils.notification_system import NotificationSystem
        st.session_state.notification_service = NotificationSystem()
    except Exception:
        st.session_state.notification_service = None
if "audit_logger" not in st.session_state:
    try:
        from trading.logs.audit_logger import audit_logger
        st.session_state.audit_logger = audit_logger
    except Exception:
        st.session_state.audit_logger = None
if "llm_processor" not in st.session_state:
    try:
        from trading.nlp.llm_processor import LLMProcessor
        st.session_state.llm_processor = LLMProcessor()
    except Exception:
        st.session_state.llm_processor = None

_pg.run()
