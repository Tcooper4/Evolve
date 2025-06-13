import pandas as pd
import streamlit as st


def app() -> None:
    st.title("Forecast")

    symbol = st.text_input("Symbol", "AAPL")
    days = st.number_input("Forecast days", min_value=1, max_value=30, value=5)

    if st.button("Run Forecast"):
        try:
            data = st.session_state.market_data.get_data(symbol)
        except Exception as e:
            st.error(f"Unable to load data: {e}")
            return

        if data is None or data.empty:
            st.warning("No data available")
            return

        features = st.session_state.feature_engineer.engineer_features(data)
        try:
            preds = st.session_state.lstm_model.predict(features)
            idx = pd.date_range(data.index[-1], periods=len(preds) + 1, closed="right")
            forecast = pd.Series(preds, index=idx)
            st.line_chart(forecast)
        except Exception as e:
            st.error(f"Forecast failed: {e}")

