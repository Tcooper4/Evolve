"""
NLP Tester: Streamlit page for PromptProcessor and LLMProcessor
- Input: Natural language query
- Output: Parsed entities (JSON) and LLM response (standard + streaming)
- Sidebar: Model, temperature, moderation, streaming toggles
- Diagnostics: Log viewer, edge case simulation
"""
import json
import os

import streamlit as st

from trading.nlp.llm_processor import LLMProcessor
from trading.nlp.prompt_processor import PromptProcessor

# Paths
LOG_PATH = "trading/nlp/logs/nlp_debug.log"
ENTITY_PATTERNS_PATH = "trading/nlp/configs/entity_patterns.json"

# --- Sidebar Controls ---
st.sidebar.title("NLP Tester Controls")
model = st.sidebar.selectbox("LLM Model", ["gpt-4", "gpt-3.5-turbo"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
moderation = st.sidebar.checkbox("Enable Moderation", value=True)
streaming = st.sidebar.checkbox("Enable Streaming Response", value=True)

# Diagnostics
st.sidebar.markdown("---")
st.sidebar.subheader("Diagnostics")
show_log = st.sidebar.checkbox("Show Debug Log Viewer", value=False)
edge_case = st.sidebar.selectbox(
    "Simulate Edge Case",
    ["None", "Missing Ticker", "Ambiguous Timeframe", "Multiple Actions"],
)

# --- Main UI ---
st.title("🧠 NLP Prompt & LLM Tester")
st.write("Test PromptProcessor and LLMProcessor with real-time feedback.")

# Input
query = st.text_area(
    "Enter a natural language query:",
    "Show me the forecast for AAPL next week using the LSTM model.",
)

if st.button("Process Query"):
    # --- PromptProcessor ---
    processor = PromptProcessor()
    entities = processor.extract_entities(query)
    intent = None
    # Simulate edge cases
    if edge_case == "Missing Ticker":
        entities.pop("ticker", None)
    elif edge_case == "Ambiguous Timeframe":
        entities["timeframe"] = ["next week", "next month"]
    elif edge_case == "Multiple Actions":
        entities["action"] = ["buy", "sell"]
    # Intent classification (optional)
    if hasattr(processor, "classify_intent"):
        intent = processor.classify_intent(query)
    # --- LLMProcessor ---
    llm = LLMProcessor(
        {
            "model": model,
            "temperature": temperature,
            "moderation": moderation,
            "max_tokens": 512,
        }
    )
    # Standard response
    try:
        response = llm.process(query)
    except Exception as e:
        response = f"Error: {e}"
    # Streaming response
    stream_chunks = []
    if streaming:
        try:
            for chunk in llm.process_stream(query):
                stream_chunks.append(chunk)
        except Exception as e:
            stream_chunks = [f"Error: {e}"]
    # --- Display Results ---
    st.subheader("Parsed Entities")
    st.json(entities)
    if intent:
        st.info(f"**Intent:** {intent}")
    st.subheader("LLM Response (Standard)")
    st.write(response)
    if streaming:
        st.subheader("LLM Response (Streaming)")
        st.write("".join(stream_chunks))

# --- Diagnostics: Log Viewer ---
if show_log:
    st.markdown("---")
    st.subheader("NLP Debug Log Viewer")
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            log_lines = f.readlines()[-200:]
        st.text("".join(log_lines))
    else:
        st.warning(f"Log file not found: {LOG_PATH}")

# --- Entity Patterns Viewer ---
with st.expander("Show Entity Patterns (JSON)"):
    if os.path.exists(ENTITY_PATTERNS_PATH):
        with open(ENTITY_PATTERNS_PATH, "r", encoding="utf-8") as f:
            patterns = json.load(f)
        st.json(patterns)
    else:
        st.warning(f"Entity patterns file not found: {ENTITY_PATTERNS_PATH}")
