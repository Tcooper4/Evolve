"""
Centralized LLM API configuration (single source of truth).

AGENT_UPGRADE: All OpenAI and Anthropic client initialization should use
get_llm_config() so keys and model names come from config/app_config and env.
See P3P4_FIXES.md and config/CONFIG_README.md.

Global LLM selector: get_active_llm() / set_active_llm() use MemoryStore preference
key 'active_llm' to choose provider and model for the entire app (Admin page).
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
logger = logging.getLogger(__name__)


class _NoKeyError(Exception):
    """No API key configured for this provider — normal when users have not added keys."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"{provider} API key not configured")


def llm_available() -> bool:
    """True if at least one of Anthropic or OpenAI keys is available (session, env, or LLMConfig)."""
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return True
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return True
    try:
        cfg = get_llm_config()
        if cfg.anthropic_api_key or cfg.openai_api_key:
            return True
    except Exception:
        pass
    try:
        import streamlit as st

        if (st.session_state.get("user_key_ANTHROPIC_API_KEY") or "").strip():
            return True
        if (st.session_state.get("user_key_OPENAI_API_KEY") or "").strip():
            return True
    except Exception:
        pass
    return False


def require_llm(feature_name: str = "This feature") -> bool:
    """
    Use at the top of auto-run LLM UI blocks. Shows a single Streamlit info when no key.
    Returns True if an LLM can be called; False otherwise.
    """
    if llm_available():
        return True
    try:
        import streamlit as st

        if not st.session_state.get("_evolve_llm_setup_notice_shown"):
            st.session_state["_evolve_llm_setup_notice_shown"] = True
            st.info(
                f"🔑 {feature_name} requires an API key. Enter yours in "
                "**Settings → API Keys** or on the welcome screen."
            )
    except Exception:
        pass
    return False


# Primary Claude model for Evolve agent layer (default when no preference set)
CLAUDE_PRIMARY_MODEL = "claude-sonnet-4-20250514"

# Supported providers for global LLM selector
LLM_PROVIDERS = ["claude", "gpt4", "gemini", "ollama", "huggingface", "kimi"]

# Default models per provider (used when user selects provider but no model)
DEFAULT_MODELS: Dict[str, str] = {
    "claude": CLAUDE_PRIMARY_MODEL,
    "gpt4": "gpt-4o",
    "gemini": "gemini-1.5-pro",
    "ollama": "llama3",
    "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
    "kimi": "kimi-k2-5",
}

# HuggingFace sub-modes: "inference" = Inference API (cloud, free tier rate-limited), "local" = transformers pipeline (offline)
HUGGINGFACE_MODES = ["inference", "local"]

# Display names for sidebar / Admin
PROVIDER_DISPLAY_NAMES: Dict[str, str] = {
    "claude": "Claude",
    "gpt4": "GPT-4",
    "gemini": "Gemini",
    "ollama": "Ollama",
    "huggingface": "HuggingFace",
    "kimi": "Kimi K2.5",
}


@dataclass
class LLMConfig:
    """LLM API keys and model names from config + env."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    primary_model: str = CLAUDE_PRIMARY_MODEL  # Claude for reasoning
    google_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    moonshot_api_key: Optional[str] = None

    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    def has_google(self) -> bool:
        return bool(self.google_api_key)

    def has_huggingface(self) -> bool:
        return bool(self.huggingface_api_key)

    def has_moonshot(self) -> bool:
        return bool(self.moonshot_api_key)


_config: Optional[LLMConfig] = None


def _normalize_openai_key(raw: Optional[str]) -> Optional[str]:
    """Strip surrounding whitespace and quotes from OPENAI_API_KEY. Return None if empty."""
    if raw is None:
        return None
    s = raw.strip().strip('"').strip("'").strip()
    if s == "" or s.lower() == "null":
        return None
    return s


def _normalize_anthropic_key(raw: Optional[str]) -> Optional[str]:
    """Strip surrounding whitespace and quotes from ANTHROPIC_API_KEY. Return None if empty."""
    if raw is None:
        return None
    s = raw.strip().strip('"').strip("'").strip()
    if s == "" or s.lower() == "null":
        return None
    if s and not s.startswith("sk-ant-"):
        logger.debug(
            "ANTHROPIC_API_KEY may be malformed — expected format: sk-ant-..."
        )
    return s


def get_llm_config() -> LLMConfig:
    """Return LLM config from app config and env (single source of truth)."""
    global _config
    if _config is not None:
        return _config
    try:
        from config.app_config import get_config

        get_config()  # ensure config loaded
        openai_key = _normalize_openai_key(os.getenv("OPENAI_API_KEY"))
        anthropic_key = _normalize_anthropic_key(os.getenv("ANTHROPIC_API_KEY"))
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None
        hf_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or None
        moonshot_key = os.getenv("MOONSHOT_API_KEY") or None
        primary = os.getenv("EVOLVE_PRIMARY_LLM_MODEL", CLAUDE_PRIMARY_MODEL)
        _config = LLMConfig(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            primary_model=primary,
            google_api_key=google_key,
            huggingface_api_key=hf_key,
            moonshot_api_key=moonshot_key,
        )
        return _config
    except Exception as e:
        logger.warning(f"Could not load app config for LLM, using env only: {e}")
        openai_key = _normalize_openai_key(os.getenv("OPENAI_API_KEY"))
        _config = LLMConfig(
            openai_api_key=openai_key,
            anthropic_api_key=_normalize_anthropic_key(os.getenv("ANTHROPIC_API_KEY")),
            primary_model=os.getenv("EVOLVE_PRIMARY_LLM_MODEL", CLAUDE_PRIMARY_MODEL),
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None,
            huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or None,
            moonshot_api_key=os.getenv("MOONSHOT_API_KEY") or None,
        )
        return _config


def get_active_llm() -> Tuple[str, str, Dict[str, Any]]:
    """
    Return (provider, model, options) for the currently selected app-wide LLM.
    Reads from MemoryStore preference key 'active_llm'. Defaults to Claude if not set.
    options may contain e.g. huggingface_mode: "inference" | "local".
    """
    try:
        from trading.memory import get_memory_store

        store = get_memory_store()
        raw = store.get_preference("active_llm")
        if raw is None:
            return "claude", CLAUDE_PRIMARY_MODEL, {}
        if isinstance(raw, dict):
            provider = (raw.get("provider") or "claude").lower()
            model = raw.get("model") or DEFAULT_MODELS.get(provider, CLAUDE_PRIMARY_MODEL)
            options = raw.get("options") or {}
            if provider not in LLM_PROVIDERS:
                provider = "claude"
                model = CLAUDE_PRIMARY_MODEL
                options = {}
            return provider, model, options
        return "claude", CLAUDE_PRIMARY_MODEL, {}
    except Exception as e:
        logger.warning(f"get_active_llm failed, defaulting to Claude: {e}")
        return "claude", CLAUDE_PRIMARY_MODEL, {}


def set_active_llm(provider: str, model: str, **options: Any) -> None:
    """
    Save the selected provider, model, and optional options to MemoryStore (preference key 'active_llm').
    options may include e.g. huggingface_mode="inference"|"local".
    """
    provider = (provider or "claude").lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from {LLM_PROVIDERS}")
    try:
        from trading.memory import get_memory_store

        store = get_memory_store()
        payload: Dict[str, Any] = {"provider": provider, "model": model or DEFAULT_MODELS.get(provider, "")}
        if options:
            payload["options"] = options
        store.upsert_preference("active_llm", payload)
    except Exception as e:
        logger.error(f"set_active_llm failed: {e}")
        raise


def get_openai_client():
    """Return an OpenAI client instance using centralized config. Caller must handle ImportError."""
    from openai import OpenAI

    # Per-user session key takes priority over shared env (safe on Streamlit Cloud)
    api_key = None
    try:
        import streamlit as st
        api_key = st.session_state.get("user_key_OPENAI_API_KEY") or None
    except Exception:
        pass
    if not api_key:
        api_key = get_llm_config().openai_api_key
    if not api_key:
        raise _NoKeyError("openai")
    return OpenAI(api_key=api_key)


def get_anthropic_client():
    """Return an Anthropic client instance using centralized config. Caller must handle ImportError."""
    import anthropic

    # Per-user session key takes priority over shared env (safe on Streamlit Cloud)
    api_key = None
    try:
        import streamlit as st
        api_key = st.session_state.get("user_key_ANTHROPIC_API_KEY") or None
    except Exception:
        pass
    if not api_key:
        api_key = get_llm_config().anthropic_api_key
    if not api_key:
        raise _NoKeyError("anthropic")
    return anthropic.Anthropic(api_key=api_key)
