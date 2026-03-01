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
    except Exception:
        pass
    try:
        from trading.memory import close_memory_store
        close_memory_store()
    except Exception:
        pass


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

# Sidebar branding
with st.sidebar:
    st.markdown("## 🚀 Evolve AI")
    st.caption("Autonomous Trading Intelligence")
    st.markdown("---")

# Main area when this script is the active page
st.markdown("# Welcome to Evolve")
st.markdown("Use the **sidebar** to open **Home**, **Chat**, **Forecasting**, **Strategy Testing**, and other pages.")
st.info("👉 Select a page from the sidebar to get started.")
