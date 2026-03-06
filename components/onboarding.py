"""
Streamlit onboarding for beta: collect API keys and preferred LLM, persist by session_id.
Session ID is stored in st.session_state, URL query param (?sid=xxx), and optionally localStorage.
"""

import secrets
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from config.user_store import load_user_keys, save_user_keys, save_user_preferences


def _ensure_session_id() -> str:
    """Get or create session_id; persist to session_state and URL."""
    sid = st.session_state.get("evolve_session_id") or st.query_params.get("sid")
    if not sid:
        sid = secrets.token_hex(16)
        st.session_state["evolve_session_id"] = sid
        st.query_params["sid"] = sid
    return sid


def _persist_session_id_to_local_storage(session_id: str) -> None:
    """Write session_id to browser localStorage via embedded HTML."""
    components.html(
        f"""
        <script>
        localStorage.setItem('evolve_session_id', '{session_id}');
        </script>
        """,
        height=0,
    )


def _render_onboarding_form(session_id: str) -> bool:
    """Render the keys form. Returns True if user submitted successfully."""
    st.markdown("### Beta onboarding")
    st.caption("Enter your API keys once. They are stored encrypted per device/session.")

    with st.form("onboarding_keys"):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Required for GPT models.",
        )
        anthropic_key = st.text_input(
            "Anthropic API Key (optional)",
            type="password",
            placeholder="sk-ant-...",
            help="For Claude models.",
        )
        news_key = st.text_input(
            "News API Key (optional)",
            type="password",
            placeholder="Your News API key",
        )
        preferred_llm = st.selectbox(
            "Preferred LLM provider",
            options=["openai", "anthropic"],
            index=0,
            help="Default provider for chat and agents.",
        )
        submitted = st.form_submit_button("Save keys")

    if submitted:
        if not openai_key and not anthropic_key:
            st.error("Please provide at least one of OpenAI or Anthropic API key.")
            return False
        keys = {
            "OPENAI_API_KEY": (openai_key or "").strip(),
            "ANTHROPIC_API_KEY": (anthropic_key or "").strip(),
            "NEWS_API_KEY": (news_key or "").strip(),
        }
        save_user_keys(session_id, keys)
        save_user_preferences(session_id, {"preferred_llm_provider": preferred_llm})
        st.query_params["sid"] = session_id
        st.session_state["evolve_show_form"] = False
        st.success("Your keys are saved. You won't need to enter them again on this device.")
        st.session_state["evolve_onboarding_done"] = True
        return True
    return False


def check_onboarding() -> Optional[str]:
    """
    Run onboarding flow. Return session_id if onboarding is complete (keys saved),
    otherwise render the onboarding UI and return None.
    """
    # Reset requested: show form again (ignore stored keys for this run)
    if st.session_state.get("evolve_force_onboarding"):
        st.session_state["evolve_force_onboarding"] = False
        st.session_state["evolve_onboarding_done"] = False
        st.session_state["evolve_show_form"] = True

    session_id = _ensure_session_id()
    _persist_session_id_to_local_storage(session_id)

    # Already completed this run (e.g. after form submit)
    if st.session_state.get("evolve_onboarding_done"):
        return session_id

    # Check if we already have keys for this session (unless user clicked Reset)
    if not st.session_state.get("evolve_show_form"):
        existing = load_user_keys(session_id)
        if existing and (existing.get("OPENAI_API_KEY") or existing.get("ANTHROPIC_API_KEY")):
            return session_id

    # Show onboarding form
    if _render_onboarding_form(session_id):
        return session_id

    # Optional: script to read session_id from localStorage and post to parent (for iframe/embed)
    components.html(
        """
        <script>
        const sid = localStorage.getItem('evolve_session_id');
        if (sid) {
            window.parent.postMessage({type: 'evolve_session', session_id: sid}, '*');
        }
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
