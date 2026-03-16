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
    initial_sidebar_state="auto",
)

# Per-user onboarding: init DB, check keys; inject into env if complete
from config.user_store import init_user_db, load_user_keys, inject_user_keys_to_env
from components.onboarding import check_onboarding

init_user_db()
session_id = check_onboarding()

# Always inject API keys from user store into os.environ on every run
try:
    inject_user_keys_to_env(session_id or st.session_state.get("evolve_session_id", "") or "")
except Exception:
    # Keys may not be set yet; continue without failing
    pass

if not session_id:
    st.stop()

# ── Global ticker search (sidebar) ──────────────────
if "global_search_ticker" not in st.session_state:
    st.session_state["global_search_ticker"] = ""

with st.sidebar:
    _gsearch = st.text_input(
        "Search ticker",
        placeholder="/ to search any ticker...",
        key="global_search_ticker",
        label_visibility="collapsed",
    )
    if _gsearch and len(_gsearch.strip()) >= 1:
        _sym = _gsearch.strip().upper()
        if st.sidebar.button(
            "Analyze " + _sym,
            key="global_search_go",
            use_container_width=True,
        ):
            st.session_state["analyze_ticker"] = _sym
            st.switch_page("pages/2_Analyze.py")
    st.markdown("---")

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

# Main area when this script is the active page
st.markdown("# Welcome to Evolve")
st.markdown("Use the **sidebar** to open **Dashboard**, **Analyze**, **Scanner**, **Trade**, **Backtest**, **Chat**, **Settings**, or the legacy pages.")
st.info("👉 Select a page from the sidebar to get started. New pages: 1_Dashboard, 2_Analyze, 3_Scanner, 4_Trade, 5_Backtest, 6_Chat, 7_Settings.")
