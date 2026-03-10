"""
Memory Management Page (Streamlit)

AGENT_MEMORY_LAYER:
- View / edit / delete MemoryStore entries
- Clear memory by type (short_term / long_term / preference)
"""

import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from trading.memory import get_memory_store
from trading.memory.memory_store import MemoryType


def _to_json_text(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str)
    except Exception:
        return str(value)


def _parse_json_text(text: str) -> Any:
    t = (text or "").strip()
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        return t


st.title("🧠 Memory Store")

try:
    store = get_memory_store()
    st.caption(f"SQLite: `{store.db_path}` • Session: `{store.session_id}`")
except Exception as e:
    st.error(f"MemoryStore unavailable: {e}")
    st.stop()

col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
with col_a:
    memory_type = st.selectbox(
        "Memory type",
        options=[MemoryType.SHORT_TERM, MemoryType.LONG_TERM, MemoryType.PREFERENCE],
        format_func=lambda x: x.value,
    )
with col_b:
    namespace = st.text_input("Namespace (optional)", value="")
with col_c:
    category = st.text_input("Category (optional)", value="")
with col_d:
    limit = st.number_input("Limit", min_value=10, max_value=5000, value=500, step=10)

session_id: Optional[str] = None
if memory_type == MemoryType.SHORT_TERM:
    session_id = st.text_input("Session id (short_term)", value=store.session_id)

try:
    entries = store.list(
        memory_type,
        namespace=namespace.strip() or None,
        category=category.strip() or None,
        session_id=session_id,
        limit=int(limit),
    )
except Exception as e:
    st.error(f"Error listing memory entries: {e}")
    entries = []

df = pd.DataFrame(
    [
        {
            "id": e.id,
            "memory_type": e.memory_type,
            "namespace": e.namespace,
            "session_id": e.session_id,
            "key": e.key,
            "category": e.category,
            "value": _to_json_text(e.value),
            "metadata": _to_json_text(e.metadata),
            "created_at": e.created_at,
            "updated_at": e.updated_at,
        }
        for e in entries
    ]
)

st.subheader("Entries")
if df.empty:
    st.info("No entries found for the selected filters.")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Edit / Delete")

selected_id = st.text_input("Entry id", value="")
if selected_id:
    try:
        entry = store.get(selected_id.strip())
    except Exception as e:
        st.error(f"Error loading entry: {e}")
        entry = None
else:
    entry = None

if entry is None and selected_id:
    st.warning("Entry not found.")

if entry is not None:
    edit_col1, edit_col2 = st.columns([1, 1])
    with edit_col1:
        new_key = st.text_input("Key", value=entry.key or "")
        new_category = st.text_input("Category", value=entry.category or "")
        new_value_text = st.text_area("Value (JSON)", value=_to_json_text(entry.value), height=220)
    with edit_col2:
        new_metadata_text = st.text_area(
            "Metadata (JSON)",
            value=_to_json_text(entry.metadata),
            height=220,
        )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        if st.button("💾 Update", type="primary"):
            value_obj = _parse_json_text(new_value_text)
            meta_obj = _parse_json_text(new_metadata_text) or {}
            try:
                ok = store.update_entry(
                    entry.id,
                    value=value_obj,
                    key=new_key.strip() or None,
                    category=new_category.strip() or None,
                    metadata=meta_obj if isinstance(meta_obj, dict) else {"metadata": meta_obj},
                )
                if ok:
                    st.success("Updated.")
                else:
                    st.error("Update failed.")
            except Exception as e:
                st.error(f"Update failed: {e}")
    with btn_col2:
        if st.button("🗑️ Delete"):
            try:
                if store.delete_entry(entry.id):
                    st.success("Deleted.")
                else:
                    st.error("Delete failed.")
            except Exception as e:
                st.error(f"Delete failed: {e}")
    with btn_col3:
        st.caption("Tip: copy the `id` from the table above and paste it here to edit.")

st.divider()
st.subheader("Add new entry")

add_col1, add_col2 = st.columns([1, 1])
with add_col1:
    add_type = st.selectbox(
        "New entry type",
        options=[MemoryType.SHORT_TERM, MemoryType.LONG_TERM, MemoryType.PREFERENCE],
        format_func=lambda x: x.value,
        key="add_type",
    )
    add_namespace = st.text_input("Namespace", value="global", key="add_namespace")
    add_key = st.text_input("Key (optional)", value="", key="add_key")
    add_category = st.text_input("Category (optional)", value="", key="add_category")
with add_col2:
    add_value_text = st.text_area("Value (JSON)", value="{}", height=180, key="add_value")
    add_meta_text = st.text_area("Metadata (JSON)", value="{}", height=180, key="add_meta")

if st.button("➕ Add entry"):
    val = _parse_json_text(add_value_text)
    meta = _parse_json_text(add_meta_text) or {}
    try:
        entry_id = store.add(
            add_type,
            namespace=add_namespace.strip() or "global",
            key=add_key.strip() or None,
            category=add_category.strip() or None,
            value=val,
            metadata=meta if isinstance(meta, dict) else {"metadata": meta},
            session_id=store.session_id if add_type == MemoryType.SHORT_TERM else None,
        )
        st.success(f"Added entry `{entry_id}`.")
    except Exception as e:
        st.error(f"Add failed: {e}")

st.divider()
st.subheader("Clear memory")

clear_col1, clear_col2 = st.columns([1, 2])
with clear_col1:
    clear_type = st.selectbox(
        "Type to clear",
        options=[MemoryType.SHORT_TERM, MemoryType.LONG_TERM, MemoryType.PREFERENCE],
        format_func=lambda x: x.value,
        key="clear_type",
    )
with clear_col2:
    clear_session = None
    if clear_type == MemoryType.SHORT_TERM:
        clear_session = st.text_input("Session id to clear", value=store.session_id, key="clear_session")
    if st.button("🧹 Clear", type="secondary"):
        try:
            count = store.clear(clear_type, session_id=clear_session)
            st.success(f"Cleared {count} entries from {clear_type.value}.")
        except Exception as e:
            st.error(f"Clear failed: {e}")

