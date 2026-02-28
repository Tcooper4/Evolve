# Data (Project Root)

Project-level data feeds, registry files, and validation results. Market data and preprocessing live under **`trading/data/`**.

## Contents

- **Live/streaming:** `live_data_feed.py`, `live_feed.py`, `streaming_pipeline.py`
- **Sentiment:** `sentiment/sentiment_fetcher.py`
- **Registry/state:** `agent_registry.json`, `leaderboard/`, `optimization/optimization_history.json`
- **Validation:** `validation_results/` (e.g. validation run outputs)

Database connection and URL are handled by **`trading.database.connection`**; see root `config/CONFIG_README.md` for DB env vars.

## Market Data and Preprocessing

For data providers, preprocessing, and feature engineering, see **`trading/data/`** (e.g. `trading/data/data_provider.py`, `trading/data/preprocessing.py`, `trading/data/providers/`).
