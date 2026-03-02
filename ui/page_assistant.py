"""
Lightweight page-aware AI assistant sidebar component.

Renders in the sidebar an expander "💬 Page Assistant" (open by default) with
conversation history, text input, Send button, and checkbox "Allow AI to suggest changes".
Uses call_active_llm_chat() with page context from get_page_context().
Each page uses its own session state keys so assistants do not interfere with each other or the main Chat.
"""

import streamlit as st

from trading.services.page_assistant_service import get_page_context, get_full_context_summary


def _session_key_prefix(page_name: str) -> str:
    """Unique prefix for this page's assistant (independent of other pages and main Chat)."""
    slug = page_name.replace(" ", "_").replace("&", "and")
    return f"page_assistant_{slug}"


def render_page_assistant(page_name: str) -> None:
    """
    Render the Page Assistant in the sidebar: expander at the bottom "💬 Page Assistant"
    (expanded by default), with conversation history, text input, Send button, and
    checkbox "Allow AI to suggest changes". State is stored per page in st.session_state.
    """
    prefix = _session_key_prefix(page_name)
    input_key = f"{prefix}_input"
    history_key = f"{prefix}_history"
    allow_suggestions_key = f"{prefix}_allow_suggestions"
    clear_input_key = f"{prefix}_clear_input"

    if clear_input_key in st.session_state and st.session_state[clear_input_key]:
        st.session_state[input_key] = ""
        del st.session_state[clear_input_key]
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    if allow_suggestions_key not in st.session_state:
        st.session_state[allow_suggestions_key] = False
    if input_key not in st.session_state:
        st.session_state[input_key] = ""

    # Render in sidebar: expander at bottom
    with st.sidebar:
        with st.expander("💬 Page Assistant", expanded=True):
            st.caption(f"Context: **{page_name}**")

            # Conversation history (this page session)
            history = st.session_state[history_key]
            for msg in history:
                if msg.get("role") == "user":
                    st.markdown(f"**You:** {msg.get('content', '')}")
                else:
                    st.markdown(f"**AI:** {msg.get('content', '')}")
                st.markdown("---")

            user_question = st.text_input(
                "Your question",
                placeholder="e.g. What does this backtest result mean?",
                key=input_key,
                label_visibility="collapsed",
            )
            allow_suggestions = st.checkbox(
                "Allow AI to suggest changes",
                key=allow_suggestions_key,
            )

            if st.button("Send", key=f"{prefix}_submit"):
                if not (user_question and user_question.strip()):
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Asking AI..."):
                        try:
                            from agents.llm.active_llm_calls import call_active_llm_chat

                            context = get_page_context(page_name, st.session_state)
                            full_summary = get_full_context_summary()
                            context_block = context + ("\n\nCross-app context: " + full_summary if full_summary else "")
                            suggestion_instruction = (
                                "You may suggest specific UI actions the user can take on this page."
                                if st.session_state.get(allow_suggestions_key, False)
                                else "Provide explanations and analysis only, do not suggest making changes."
                            )
                            system_prompt = (
                                f"You are an expert trading assistant. The user is currently on the **{page_name}** page. "
                                f"Here is their current context: {context_block}\n\n"
                                f"Answer their question concisely in plain English. Be specific to their data when relevant. {suggestion_instruction}"
                            )
                            response = call_active_llm_chat(
                                system_prompt=system_prompt,
                                context_block="",
                                conversation_messages=history,
                                user_message=user_question.strip(),
                                max_tokens=1024,
                            )
                            st.session_state[history_key] = history + [
                                {"role": "user", "content": user_question.strip()},
                                {"role": "assistant", "content": response or ""},
                            ]
                            st.session_state[clear_input_key] = True
                        except Exception as e:
                            st.session_state[history_key] = history + [
                                {"role": "user", "content": user_question.strip()},
                                {"role": "assistant", "content": f"Error: {e}"},
                            ]
                            st.session_state[clear_input_key] = True
                        st.rerun()
