"""
Streamlit onboarding: optional API keys, preferred LLM, persist by session_id.
Session ID is derived from API key hash when LLM keys exist; otherwise an
anonymous id (time + random). Keys are optional — onboarding always completes.
"""

import hashlib
import os as _os
import random
import time
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from config.user_store import load_user_keys, load_user_preferences, save_user_keys, save_user_preferences


def _new_anon_session_id() -> str:
    return hashlib.sha256(
        f"{time.time()}{random.random()}".encode()
    ).hexdigest()[:16]


def _persist_session_id_to_local_storage(session_id: str) -> None:
    """Best-effort: write session_id to browser localStorage (may fail in Cloud iframe)."""
    if not session_id:
        return
    components.html(
        f"""
        <script>
        try {{
            localStorage.setItem('evolve_session_id', '{session_id}');
        }} catch (e) {{}}
        </script>
        """,
        height=0,
    )


def _finalize_onboarding(
    session_id: str,
    keys: dict,
    preferred_llm: str,
) -> None:
    save_user_keys(session_id, keys)
    prev = load_user_preferences(session_id) or {}
    save_user_preferences(
        session_id,
        {
            **prev,
            "preferred_llm_provider": preferred_llm,
            "onboarding_completed": True,
            "onboarding_done": True,
        },
    )
    st.session_state["evolve_session_id"] = session_id
    st.query_params["sid"] = session_id
    st.session_state["evolve_show_form"] = False
    st.session_state["evolve_onboarding_done"] = True
    _is_cloud = (
        _os.environ.get("STREAMLIT_SHARING_MODE")
        or _os.environ.get("IS_STREAMLIT_CLOUD")
        or not _os.path.exists(".env")
    )
    if _is_cloud:
        from config.user_store import inject_user_keys_to_session

        inject_user_keys_to_session(session_id)
    else:
        from config.user_store import inject_user_keys_to_env

        inject_user_keys_to_env(session_id)


def _render_onboarding_form(session_id: Optional[str]) -> bool:
    """
    Render onboarding UI. Returns True when user completed this run (save or skip).
    """
    st.markdown("# Evolve")
    st.markdown(
        "Institutional-grade algorithmic trading "
        "and ML research platform. "
        "Powered by 10 forecasting models, "
        "real-time market scanning, and an AI "
        "copilot that runs every morning."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Works without API keys**")
        st.caption(
            "Price charts · AI Score · "
            "10-model forecasts · Market scanner · "
            "Backtesting · Walk-forward validation · "
            "Options flow · Risk metrics"
        )
    with col2:
        st.markdown("**Requires OpenAI or Anthropic**")
        st.caption(
            "Chat agent · Morning briefing narrative · "
            "AI commentary · Deep dive analysis · "
            "Natural language queries"
        )
    with col3:
        st.markdown("**Optional enhancements**")
        st.caption(
            "News API → premium headlines · "
            "Reddit → social sentiment · "
            "Anthropic → Claude as LLM provider"
        )

    st.markdown("---")

    st.markdown("### API keys")
    st.caption(
        "Enter keys to unlock AI features. "
        "All keys are encrypted and stored "
        "locally per session."
    )

    openai_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        key="onboarding_openai",
        help="Required for chat, commentary, and "
        "morning briefing narrative. "
        "Get yours at platform.openai.com",
    )
    anthropic_key = st.text_input(
        "Anthropic API key (optional)",
        type="password",
        placeholder="sk-ant-...",
        key="onboarding_anthropic",
        help="Alternative to OpenAI. "
        "Uses Claude for all LLM features. "
        "Get yours at console.anthropic.com",
    )
    news_key = st.text_input(
        "News API key (optional)",
        type="password",
        placeholder="Your NewsAPI key",
        key="onboarding_news",
        help="Enhances news aggregator with "
        "premium sources. "
        "Free tier at newsapi.org",
    )
    reddit_client_id = st.text_input(
        "Reddit client ID (optional)",
        type="password",
        placeholder="Reddit app client ID",
        key="onboarding_reddit_id",
        help="Enables real Reddit sentiment "
        "instead of rate-limited public API. "
        "Create app at reddit.com/prefs/apps",
    )
    reddit_secret = st.text_input(
        "Reddit client secret (optional)",
        type="password",
        placeholder="Reddit app secret",
        key="onboarding_reddit_secret",
        help="Required with Reddit client ID",
    )
    provider = st.selectbox(
        "Preferred LLM provider",
        ["openai", "anthropic"],
        key="onboarding_provider",
        help="Which LLM to use for chat and "
        "analysis. Falls back to the other "
        "if primary is unavailable.",
    )

    if not openai_key.strip() and not anthropic_key.strip():
        st.caption(
            "You can continue without API keys — "
            "charts, forecasts, scanner, and backtesting stay fully available. "
            "Chat and AI narrative features stay disabled until you add a key in "
            "Settings or here."
        )

    col_save, col_skip = st.columns([2, 1])
    with col_save:
        save_btn = st.button(
            "Save keys and continue",
            type="primary",
            key="onboarding_save",
            use_container_width=True,
        )
    with col_skip:
        skip_btn = st.button(
            "Continue without keys",
            key="onboarding_skip",
            use_container_width=True,
            help="Core platform works without keys",
        )

    if skip_btn:
        anon_id = _new_anon_session_id()
        keys = {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "NEWS_API_KEY": "",
            "REDDIT_CLIENT_ID": "",
            "REDDIT_CLIENT_SECRET": "",
        }
        _pref = st.session_state.get("onboarding_provider") or "openai"
        _finalize_onboarding(anon_id, keys, str(_pref))
        _persist_session_id_to_local_storage(anon_id)
        return True

    if save_btn:
        oa = (openai_key or "").strip()
        ant = (anthropic_key or "").strip()
        if oa or ant:
            api_key = oa or ant
            new_sid = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        else:
            new_sid = _new_anon_session_id()

        keys = {
            "OPENAI_API_KEY": oa,
            "ANTHROPIC_API_KEY": ant,
            "NEWS_API_KEY": (news_key or "").strip(),
            "REDDIT_CLIENT_ID": (reddit_client_id or "").strip(),
            "REDDIT_CLIENT_SECRET": (reddit_secret or "").strip(),
        }
        _finalize_onboarding(new_sid, keys, provider)
        _persist_session_id_to_local_storage(new_sid)
        st.success(
            "Your preferences are saved. "
            "Bookmark this URL to return on this device."
        )
        return True

    return False


def check_onboarding() -> Optional[str]:
    """
    Run onboarding flow. Return session_id when the user may enter the app.
    Return None only while the onboarding UI must block the rest of the app.
    """
    if st.session_state.get("evolve_force_onboarding"):
        st.session_state["evolve_force_onboarding"] = False
        st.session_state["evolve_onboarding_done"] = False
        st.session_state["evolve_show_form"] = True

    sid = st.query_params.get("sid") or st.session_state.get("evolve_session_id")
    if sid:
        st.session_state["evolve_session_id"] = sid
        st.query_params["sid"] = sid
        _persist_session_id_to_local_storage(sid)

    if st.session_state.get("evolve_onboarding_done"):
        return sid

    if not st.session_state.get("evolve_show_form") and sid:
        prefs = load_user_preferences(sid)
        if prefs.get("onboarding_completed") or prefs.get("onboarding_done"):
            return sid
        existing = load_user_keys(sid)
        if existing.get("OPENAI_API_KEY") or existing.get("ANTHROPIC_API_KEY"):
            return sid

    if _render_onboarding_form(sid):
        sid = st.session_state.get("evolve_session_id")
        _persist_session_id_to_local_storage(sid or "")
        return sid

    components.html(
        """
        <script>
        try {
            const sid = localStorage.getItem('evolve_session_id');
            if (sid) {
                window.parent.postMessage({type: 'evolve_session', session_id: sid}, '*');
            }
        } catch (e) {}
        </script>
        """,
        height=0,
    )

    st.markdown("---")
    if st.button("Reset my keys"):
        st.session_state["evolve_force_onboarding"] = True
        st.session_state["evolve_onboarding_done"] = False
        st.rerun()
    return None
