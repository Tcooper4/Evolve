# Codebase Audit

**Scope:** All directories and files excluding `evolve_venv/`, `.venv/`, `__pycache__/`, `.git/`, `.cache/`, `data/`, `logs/`, `.pytest_cache/`.  
**ACTIVE/ORPHAN rule:** For each `.py` file, search codebase for module stem (filename without extension); exclude the file itself. Zero other references → ORPHAN; otherwise → ACTIVE.

---

## pages/

| File | Summary | Status |
|------|---------|--------|
| pages/__init__.py | Package init for pages | ACTIVE |
| pages/0_Home.py | Live market events monitor, watchlist scanner, featured event chart, morning briefing | ACTIVE |
| pages/1_Chat.py | Chat interface with natural-language interface and multi-agent routing | ACTIVE |
| pages/2_Forecasting.py | Forecasting and market analysis; quick/advanced forecast, AI model selection, multi-model comparison | ACTIVE |
| pages/3_Strategy_Testing.py | Strategy development and backtest UI; quick backtest, visual builder, code editor, ensemble creation | ACTIVE |
| pages/4_Trade_Execution.py | Trade execution and order management; market/limit/bracket orders, execution analytics | ACTIVE |
| pages/5_Portfolio.py | Portfolio and positions; overview, attribution, optimization, rebalancing | ACTIVE |
| pages/6_Risk_Management.py | Risk management; VaR, Monte Carlo, stress testing | ACTIVE |
| pages/7_Performance.py | Performance and history; strategy performance, trade history, health monitoring | ACTIVE |
| pages/8_Model_Lab.py | Model laboratory; training, optimization, explainability, registry | ACTIVE |
| pages/9_Reports.py | Reports and exports; templates, custom builder, PDF/Excel/HTML | ACTIVE |
| pages/10_Alerts.py | Alerts and notifications; price/signal/risk/portfolio alerts, multi-channel | ACTIVE |
| pages/11_Admin.py | System administration; config, users, API keys, agents, health, logs | ACTIVE |
| pages/12_Memory.py | Memory management; view/edit/delete MemoryStore entries by type | ACTIVE |

---

## trading/models/

| File | Summary | Status |
|------|---------|--------|
| trading/models/__init__.py | Exports BaseModel, LSTMModel, ForecastRouter, model registry, etc. | ACTIVE |
| trading/models/base_model.py | Base class for all ML models with common fit/predict/validation | ACTIVE |
| trading/models/forecast_router.py | Router for selecting and managing forecasting models by data/perf/preferences | ACTIVE |
| trading/models/arima_model.py | ARIMA model for time series forecasting with auto_arima support | ACTIVE |
| trading/models/lstm_model.py | LSTM-based forecasting model with robust error handling | ACTIVE |
| trading/models/xgboost_model.py | XGBoost forecasting model | ACTIVE |
| trading/models/prophet_model.py | Prophet forecasting model | ACTIVE |
| trading/models/ridge_model.py | Ridge regression model for forecasting | ACTIVE |
| trading/models/garch_model.py | GARCH model for volatility forecasting | ACTIVE |
| trading/models/tcn_model.py | TCN (Temporal Convolutional Network) model | ACTIVE |
| trading/models/catboost_model.py | CatBoost forecasting model | ACTIVE |
| trading/models/ensemble_model.py | Ensemble forecasting model | ACTIVE |
| trading/models/neuralforecast_models.py | NeuralForecast-based models | ACTIVE |
| trading/models/model_registry.py | Model registry and registration | ACTIVE |
| trading/models/model_utils.py | Model utilities | ACTIVE |
| trading/models/dataset.py | Dataset utilities for models | ACTIVE |
| trading/models/confidence_generator.py | Confidence generation for forecasts | ACTIVE |
| trading/models/forecast_explainability.py | Forecast explainability | ACTIVE |
| trading/models/forecast_features.py | Forecast feature engineering | ACTIVE |
| trading/models/forecast_normalizer.py | Forecast normalization | ACTIVE |
| trading/models/multi_model_aggregator.py | Multi-model aggregation | ACTIVE |
| trading/models/transformer_wrapper.py | Transformer model wrapper | ACTIVE |
| trading/models/advanced/__init__.py | Advanced model subpackage init | ACTIVE |
| trading/models/advanced/gnn/__init__.py | GNN subpackage; exports GNNForecaster | ACTIVE |
| trading/models/advanced/gnn/gnn_model.py | GNN (Graph Neural Network) model for time series | ACTIVE |
| trading/models/advanced/ensemble/__init__.py | Ensemble subpackage; exports EnsembleForecaster | ACTIVE |
| trading/models/advanced/lstm/__init__.py | LSTM subpackage init | ACTIVE |
| trading/models/advanced/rl/__init__.py | RL subpackage init | ACTIVE |
| trading/models/advanced/rl/strategy_optimizer.py | RL-based strategy optimizer | ACTIVE |
| trading/models/advanced/tcn/__init__.py | TCN subpackage init | ACTIVE |
| trading/models/advanced/transformer/__init__.py | Transformer subpackage init | ACTIVE |
| trading/models/advanced/transformer/time_series_transformer.py | Transformer model for time series forecasting | ACTIVE |
| trading/models/timeseries/__init__.py | Timeseries subpackage init | ACTIVE |

---

## trading/forecasting/

| File | Summary | Status |
|------|---------|--------|
| trading/forecasting/__init__.py | Package init (empty or minimal) | ACTIVE |
| trading/forecasting/hybrid_model.py | Hybrid ensemble with auto-updated weights and risk-aware weighting | ACTIVE |
| trading/forecasting/hybrid_model_selector.py | Selector for hybrid model configuration | ACTIVE |
| trading/forecasting/forecast_postprocessor.py | Postprocessing for forecasts | ACTIVE |
| trading/forecasting/postprocess.py | Dynamic EWMA alpha tuning for forecast processing | ACTIVE |

---

## trading/agents/

| File | Summary | Status |
|------|---------|--------|
| trading/agents/__init__.py | Package init; exports agents and registry | ACTIVE |
| trading/agents/agent_manager.py | Manages pluggable agents; registration, execution, retry/backoff | ACTIVE |
| trading/agents/agent_registry.py | Agent registration and lookup | ACTIVE |
| trading/agents/base_agent_interface.py | Base agent interface and types | ACTIVE |
| trading/agents/agent_leaderboard.py | Agent leaderboard and ranking | ACTIVE |
| trading/agents/commentary_agent.py | Commentary generation agent | ACTIVE |
| trading/agents/data_quality_agent.py | Data quality agent | ACTIVE |
| trading/agents/enhanced_prompt_router.py | Enhanced prompt routing | ACTIVE |
| trading/agents/execution_risk_agent.py | Execution risk agent | ACTIVE |
| trading/agents/execution_risk_control_agent.py | Execution risk control agent | ACTIVE |
| trading/agents/market_regime_agent.py | Market regime detection agent | ACTIVE |
| trading/agents/meta_strategy_agent.py | Meta strategy agent | ACTIVE |
| trading/agents/model_builder_agent.py | Model building agent | ACTIVE |
| trading/agents/model_evaluator_agent.py | Model evaluation agent | ACTIVE |
| trading/agents/model_improver_agent.py | Model improvement agent | ACTIVE |
| trading/agents/model_optimizer_agent.py | Model optimization agent | ACTIVE |
| trading/agents/model_selector_agent.py | Model selection agent | ACTIVE |
| trading/agents/model_synthesizer_agent.py | Model synthesis agent | ACTIVE |
| trading/agents/performance_critic_agent.py | Performance critic agent | ACTIVE |
| trading/agents/regime_detection_agent.py | Regime detection agent | ACTIVE |
| trading/agents/research_agent.py | Research agent | ACTIVE |
| trading/agents/strategy_dispatcher.py | Strategy dispatch | ACTIVE |
| trading/agents/strategy_improver_agent.py | Strategy improvement agent | ACTIVE |
| trading/agents/strategy_selector_agent.py | Strategy selection agent | ACTIVE |
| trading/agents/strategy_switcher.py | Strategy switching | ACTIVE |
| trading/agents/updater_agent.py | Updater agent | ACTIVE |
| trading/agents/task_dashboard.py | Task dashboard UI/logic | ACTIVE |
| trading/agents/task_memory.py | Task memory for agents | ACTIVE |
| trading/agents/intent_detector.py | Intent detection for prompts | ACTIVE |
| trading/agents/routing_engine.py | Routing engine for requests | ACTIVE |
| trading/agents/rl_trainer.py | RL trainer for agents | ACTIVE |
| trading/agents/prompt_templates.py | Prompt templates | ACTIVE |
| trading/agents/prompt_response_validator.py | Prompt response validation | ACTIVE |
| trading/agents/launch_execution_agent.py | Standalone launcher for Execution Agent service | ORPHAN |
| trading/agents/launch_leaderboard_dashboard.py | Launcher for agent leaderboard dashboard (Streamlit) | ORPHAN |
| trading/agents/demo_risk_controls.py | Demo for execution risk controls (run as script) | ORPHAN |
| trading/agents/demo_leaderboard.py | Demo leaderboard analytics (run as script) | ORPHAN |
| trading/agents/demo_pluggable_agents.py | Demo for pluggable agents | ORPHAN |
| trading/agents/test_integration.py | Integration test helper for agents (not pytest) | ORPHAN |
| trading/agents/test_execution_agent.py | Standalone test script for execution agent | ORPHAN |
| trading/agents/execution/__init__.py | Execution subpackage init | ACTIVE |
| trading/agents/execution/execution_agent.py | Execution agent implementation | ACTIVE |
| trading/agents/execution/execution_providers.py | Execution providers (broker adapters) | ACTIVE |
| trading/agents/execution/execution_models.py | Execution data models | ACTIVE |
| trading/agents/execution/risk_controls.py | Risk controls for execution | ACTIVE |
| trading/agents/execution/risk_calculator.py | Risk calculator for execution | ACTIVE |
| trading/agents/execution/position_manager.py | Position manager for execution | ACTIVE |
| trading/agents/execution/trade_signals.py | Trade signals for execution | ACTIVE |
| trading/agents/optimization/__init__.py | Optimization subpackage; exports BacktestIntegration | ACTIVE |
| trading/agents/optimization/backtest_integration.py | Backtest integration for agent optimization | ACTIVE |
| trading/agents/optimization/strategy_optimizer.py | Strategy optimizer for agents | ACTIVE |
| trading/agents/optimization/parameter_validator.py | Parameter validation for optimization | ACTIVE |
| trading/agents/optimization/performance_analyzer.py | Performance analyzer for optimization | ACTIVE |
| trading/agents/upgrader/__init__.py | Upgrader subpackage init | ACTIVE |
| trading/agents/upgrader/agent.py | Upgrader agent | ACTIVE |
| trading/agents/upgrader/scheduler.py | Upgrader scheduler | ACTIVE |
| trading/agents/upgrader/utils.py | Upgrader utilities | ACTIVE |

---

## agents/

| File | Summary | Status |
|------|---------|--------|
| agents/__init__.py | Package init | ACTIVE |
| agents/agent_config.py | Agent configuration | ACTIVE |
| agents/agent_controller.py | Agent controller | ACTIVE |
| agents/registry.py | Agent registry | ACTIVE |
| agents/task_router.py | Task routing | ACTIVE |
| agents/orchestrator.py | Orchestration | ACTIVE |
| agents/mock_agent.py | Mock agent for tests | ACTIVE |
| agents/implementations/__init__.py | Implementation helpers package | ACTIVE |
| agents/implementations/implementation_generator.py | Implementation generator | ACTIVE |
| agents/implementations/model_benchmarker.py | Model benchmarker | ACTIVE |
| agents/implementations/research_fetcher.py | Research fetcher | ACTIVE |
| agents/llm/__init__.py | LLM integration package | ACTIVE |
| agents/llm/agent.py | LLM agent | ACTIVE |
| agents/llm/llm_interface.py | LLM interface | ACTIVE |
| agents/llm/memory.py | LLM memory | ACTIVE |
| agents/llm/model_loader.py | Model loader for LLM | ACTIVE |
| agents/llm/tools.py | LLM tools | ACTIVE |
| agents/llm/llm_summary.py | LLM summary (strategy/market commentary) | ACTIVE |
| agents/llm/active_llm_calls.py | Active LLM calls tracking | ACTIVE |
| agents/llm_providers/__init__.py | LLM providers package | ACTIVE |
| agents/llm_providers/local_provider.py | Local LLM provider | ACTIVE |
| agents/llm_providers/anthropic_provider.py | Anthropic LLM provider | ACTIVE |

---

## trading/data/

| File | Summary | Status |
|------|---------|--------|
| trading/data/__init__.py | Data management and processing; exports DataLoader, providers, preprocessing | ACTIVE |
| trading/data/data_provider.py | Unified data provider interface with fallback and caching | ACTIVE |
| trading/data/data_loader.py | Data loader for market data; single/multi ticker, validation | ACTIVE |
| trading/data/data_listener.py | Data listener and real-time feed | ACTIVE |
| trading/data/preprocessing.py | DataPreprocessor, FeatureEngineering, DataValidator, DataScaler | ACTIVE |
| trading/data/news_fetcher.py | News fetching for context | ACTIVE |
| trading/data/macro_data_integration.py | Macro data integration and economic indicators | ACTIVE |
| trading/data/external_signals.py | External signal integration (news, social, macro, options flow) | ACTIVE |
| trading/data/providers/__init__.py | Data providers package | ACTIVE |
| trading/data/providers/base_provider.py | Base data provider interface | ACTIVE |
| trading/data/providers/yfinance_provider.py | YFinance data provider | ACTIVE |
| trading/data/providers/fallback_provider.py | Fallback data provider | ACTIVE |
| trading/data/providers/alpha_vantage_provider.py | Alpha Vantage data provider | ACTIVE |

---

## trading/strategies/

| File | Summary | Status |
|------|---------|--------|
| trading/strategies/__init__.py | Exports strategies and registry | ACTIVE |
| trading/strategies/base_strategy.py | Abstract base class for all trading strategies | ACTIVE |
| trading/strategies/bollinger_strategy.py | Bollinger Bands strategy | ACTIVE |
| trading/strategies/macd_strategy.py | MACD strategy | ACTIVE |
| trading/strategies/rsi_strategy.py | RSI strategy | ACTIVE |
| trading/strategies/rsi_signals.py | RSI signal generation | ACTIVE |
| trading/strategies/rsi_utils.py | RSI utilities | ACTIVE |
| trading/strategies/sma_strategy.py | SMA strategy | ACTIVE |
| trading/strategies/atr_strategy.py | ATR strategy | ACTIVE |
| trading/strategies/cci_strategy.py | CCI strategy | ACTIVE |
| trading/strategies/ensemble.py | Ensemble strategy | ACTIVE |
| trading/strategies/ensemble_methods.py | Ensemble methods | ACTIVE |
| trading/strategies/hybrid_engine.py | Hybrid strategy engine | ACTIVE |
| trading/strategies/multi_strategy_hybrid_engine.py | Multi-strategy hybrid engine | ACTIVE |
| trading/strategies/strategy_runner.py | Strategy runner | ACTIVE |
| trading/strategies/strategy_router.py | Strategy router | ACTIVE |
| trading/strategies/strategy_manager.py | Strategy manager | ACTIVE |
| trading/strategies/strategy_comparison.py | Strategy comparison | ACTIVE |
| trading/strategies/strategy_ranking.py | Strategy ranking | ACTIVE |
| trading/strategies/strategy_composer.py | Strategy composer | ACTIVE |
| trading/strategies/strategy_synthesizer.py | Strategy synthesizer | ACTIVE |
| trading/strategies/strategy_fallback.py | Strategy fallback | ACTIVE |
| trading/strategies/strategy_chain_router.py | Strategy chain router | ACTIVE |
| trading/strategies/strategy_implementations.py | Strategy implementations | ACTIVE |
| trading/strategies/registry.py | Strategy registry | ACTIVE |
| trading/strategies/gatekeeper.py | Strategy gatekeeper | ACTIVE |
| trading/strategies/validation.py | Strategy validation | ACTIVE |
| trading/strategies/parameter_validator.py | Parameter validation for strategies | ACTIVE |
| trading/strategies/custom_strategy_handler.py | Custom strategy handler | ACTIVE |
| trading/strategies/enhanced_strategy_engine.py | Enhanced strategy engine | ACTIVE |
| trading/strategies/breakout_strategy_engine.py | Breakout strategy engine | ACTIVE |
| trading/strategies/pairs_trading_engine.py | Pairs trading engine | ACTIVE |
| trading/strategies/composite_strategy.py | Composite strategy | ACTIVE |
| trading/strategies/adaptive_selector.py | Adaptive strategy selector | ACTIVE |

---

## trading/portfolio/

| File | Summary | Status |
|------|---------|--------|
| trading/portfolio/__init__.py | Package init | ACTIVE |
| trading/portfolio/portfolio_manager.py | Portfolio manager with LLM commentary and optimization integration | ACTIVE |
| trading/portfolio/portfolio_simulator.py | Portfolio simulation | ACTIVE |
| trading/portfolio/position_sizer.py | Position sizing | ACTIVE |
| trading/portfolio/llm_utils.py | LLM utilities for daily commentary and trade rationale | ACTIVE |

---

## trading/analysis/

| File | Summary | Status |
|------|---------|--------|
| trading/analysis/__init__.py | Package init | ACTIVE |
| trading/analysis/market_monitor.py | Live market events monitor; scans watchlist for volume/price spikes | ACTIVE |
| trading/analysis/chart_builder.py | Chart building for events | ACTIVE |
| trading/analysis/event_news_fetcher.py | Fetch news around events | ACTIVE |
| trading/analysis/news_ranker.py | Rank news by relevance | ACTIVE |

---

## trading/memory/

| File | Summary | Status |
|------|---------|--------|
| trading/memory/__init__.py | Package init; exports get_memory_store, close_memory_store | ACTIVE |
| trading/memory/agent_memory.py | Persistent memory for agent decisions, outcomes, history | ACTIVE |
| trading/memory/agent_memory_manager.py | Agent memory manager | ACTIVE |
| trading/memory/memory_store.py | Memory store (short/long-term, preference) | ACTIVE |
| trading/memory/state_manager.py | State management | ACTIVE |
| trading/memory/performance_memory.py | Performance memory | ACTIVE |
| trading/memory/performance_weights.py | Performance weights | ACTIVE |
| trading/memory/performance_logger.py | Performance logging | ACTIVE |
| trading/memory/model_log.py | Model logging | ACTIVE |
| trading/memory/model_monitor.py | Model monitoring | ACTIVE |
| trading/memory/agent_logger.py | Agent logging | ACTIVE |
| trading/memory/agent_thoughts_logger.py | Agent thoughts logging | ACTIVE |
| trading/memory/long_term_performance_tracker.py | Long-term performance tracking | ACTIVE |
| trading/memory/persistent_memory.py | Persistent memory | ACTIVE |
| trading/memory/prompt_feedback_memory.py | Prompt feedback memory | ACTIVE |
| trading/memory/strategy_logger.py | Strategy logging | ACTIVE |
| trading/memory/visualize_memory.py | Memory visualization | ACTIVE |
| trading/memory/logger_utils.py | Logger utilities | ACTIVE |
| trading/memory/goals/__init__.py | Goals subpackage init | ACTIVE |
| trading/memory/goals/status.py | Goals status | ACTIVE |

---

## trading/nlp/

| File | Summary | Status |
|------|---------|--------|
| trading/nlp/__init__.py | Package init | ACTIVE |
| trading/nlp/prompt_processor.py | Entity/intent processing and ProcessedPrompt dataclass | ACTIVE |
| trading/nlp/llm_processor.py | LLM processor for NLP | ACTIVE |
| trading/nlp/nl_interface.py | Natural language interface | ACTIVE |
| trading/nlp/prompt_bridge.py | Prompt bridge | ACTIVE |
| trading/nlp/response_formatter.py | Response formatting | ACTIVE |
| trading/nlp/sentiment_classifier.py | Sentiment classification | ACTIVE |
| trading/nlp/sentiment_processor.py | Sentiment processing | ACTIVE |
| trading/nlp/sentiment_bridge.py | Sentiment bridge | ACTIVE |
| trading/nlp/sandbox_nlp.py | Terminal-based NLP sandbox for PromptProcessor/LLMProcessor (run as script) | ORPHAN |

---

## trading/utils/

| File | Summary | Status |
|------|---------|--------|
| trading/utils/__init__.py | Package init; exports CacheManager, safe_math, etc. | ACTIVE |
| trading/utils/config_manager.py | Configuration management for the trading system | ACTIVE |
| trading/utils/cache_manager.py | Cache manager and cache_result decorator | ACTIVE |
| trading/utils/data_utils.py | Data utilities | ACTIVE |
| trading/utils/data_transformer.py | Data transformation | ACTIVE |
| trading/utils/data_validation.py | Data validation | ACTIVE |
| trading/utils/safe_math.py | Safe math (e.g. safe_sharpe_ratio, safe_divide) | ACTIVE |
| trading/utils/safe_indicators.py | Safe technical indicators | ACTIVE |
| trading/utils/logging.py | Logging utilities | ACTIVE |
| trading/utils/logging_utils.py | Logging utilities | ACTIVE |
| trading/utils/error_handling.py | Error handling | ACTIVE |
| trading/utils/error_logger.py | Error logging | ACTIVE |
| trading/utils/metrics.py | Metrics | ACTIVE |
| trading/utils/performance_metrics.py | Performance metrics | ACTIVE |
| trading/utils/performance.py | Performance utilities | ACTIVE |
| trading/utils/model_evaluation.py | Model evaluation | ACTIVE |
| trading/utils/model_monitoring.py | Model monitoring | ACTIVE |
| trading/utils/forecast_formatter.py | Forecast formatting | ACTIVE |
| trading/utils/feature_engineering.py | Feature engineering | ACTIVE |
| trading/utils/signal_generation.py | Signal generation | ACTIVE |
| trading/utils/signal_merger.py | Signal merging | ACTIVE |
| trading/utils/signal_scorer.py | Signal scoring | ACTIVE |
| trading/utils/time_utils.py | Time utilities | ACTIVE |
| trading/utils/validation_utils.py | Validation utilities | ACTIVE |
| trading/utils/env_manager.py | Environment management | ACTIVE |
| trading/utils/config_utils.py | Config utilities | ACTIVE |
| trading/utils/diagnostics.py | Diagnostics | ACTIVE |
| trading/utils/auto_repair.py | Auto-repair | ACTIVE |
| trading/utils/visualization.py | Visualization | ACTIVE |
| trading/utils/system_status.py | System status | ACTIVE |
| trading/utils/system_startup.py | System startup | ACTIVE |
| trading/utils/gpu_utils.py | GPU utilities | ACTIVE |
| trading/utils/redis_cache.py | Redis cache | ACTIVE |
| trading/utils/redis_utils.py | Redis utilities | ACTIVE |
| trading/utils/notification_system.py | Notification system | ACTIVE |
| trading/utils/notifications.py | Notifications | ACTIVE |
| trading/utils/reasoning_service.py | Reasoning service | ACTIVE |
| trading/utils/reasoning_logger.py | Reasoning logger | ACTIVE |
| trading/utils/reasoning_display.py | Reasoning display | ACTIVE |
| trading/utils/safe_executor.py | Safe executor | ACTIVE |
| trading/utils/reward_function.py | Reward function | ACTIVE |
| trading/utils/memory_logger.py | Memory logger | ACTIVE |
| trading/utils/monitor.py | Monitor | ACTIVE |
| trading/utils/prompt_formatter.py | Prompt formatting | ACTIVE |
| trading/utils/metrics/scorer.py | Scorer | ACTIVE |
| trading/utils/metrics/__init__.py | Metrics subpackage init | ACTIVE |
| trading/utils/launch_reasoning_service.py | Launch reasoning service | ACTIVE |
| trading/utils/demo_reasoning.py | Demo reasoning | ACTIVE |
| trading/utils/test_reasoning.py | Test reasoning | ACTIVE |

---

## trading/optimization/

| File | Summary | Status |
|------|---------|--------|
| trading/optimization/__init__.py | Package init | ACTIVE |
| trading/optimization/backtest_optimizer.py | Backtest optimizer with walk-forward and regime detection | ACTIVE |
| trading/optimization/base_optimizer.py | Base optimizer | ACTIVE |
| trading/optimization/strategy_optimizer.py | Strategy optimization | ACTIVE |
| trading/optimization/portfolio_optimizer.py | Portfolio optimization | ACTIVE |
| trading/optimization/optuna_optimizer.py | Optuna-based optimization | ACTIVE |
| trading/optimization/optuna_tuner.py | Optuna tuner | ACTIVE |
| trading/optimization/optimizer_factory.py | Optimizer factory | ACTIVE |
| trading/optimization/performance_logger.py | Performance logger | ACTIVE |
| trading/optimization/strategy_selection_agent.py | Strategy selection agent | ACTIVE |
| trading/optimization/core_optimizer.py | Core optimizer | ACTIVE |
| trading/optimization/bayesian_optimizer.py | Bayesian optimization | ACTIVE |
| trading/optimization/genetic_optimizer.py | Genetic optimization | ACTIVE |
| trading/optimization/grid_search_optimizer.py | Grid search optimizer | ACTIVE |
| trading/optimization/pso_optimizer.py | PSO optimizer | ACTIVE |
| trading/optimization/ray_optimizer.py | Ray-based optimization | ACTIVE |
| trading/optimization/rsi_optimizer.py | RSI parameter optimization | ACTIVE |
| trading/optimization/self_tuning_optimizer.py | Self-tuning optimizer | ACTIVE |
| trading/optimization/forecasting_integration.py | Forecasting integration | ACTIVE |
| trading/optimization/optimization_visualizer.py | Optimization visualization | ACTIVE |
| trading/optimization/utils/__init__.py | Utils subpackage init | ACTIVE |
| trading/optimization/utils/consolidator.py | Consolidation utilities | ACTIVE |
| trading/optimization/core/__init__.py | Core subpackage init | ACTIVE |
| trading/optimization/strategies/__init__.py | Strategies subpackage init | ACTIVE |
| trading/optimization/visualization/__init__.py | Visualization subpackage init | ACTIVE |

---

## trading/feature_engineering/

| File | Summary | Status |
|------|---------|--------|
| trading/feature_engineering/__init__.py | Package init | ACTIVE |
| trading/feature_engineering/feature_engineer.py | Feature engineering with pandas_ta and sklearn (PCA, etc.) | ACTIVE |
| trading/feature_engineering/indicators.py | Technical indicators | ACTIVE |
| trading/feature_engineering/macro_feature_engineering.py | Macro feature engineering | ACTIVE |
| trading/feature_engineering/utils.py | Feature engineering utilities | ACTIVE |

---

## config/

| File | Summary | Status |
|------|---------|--------|
| config/__init__.py | Package init | ACTIVE |
| config/config.py | Configuration utilities and simplified interface | ACTIVE |
| config/app_config.py | Application configuration from YAML and env | ACTIVE |
| config/config_loader.py | Centralized config loading with validation and env support | ACTIVE |
| config/primary_config.py | Primary configuration entry point (single source of truth) | ACTIVE |
| config/settings.py | Settings module | ACTIVE |
| config/logging_config.py | Logging configuration and setup | ACTIVE |
| config/llm_config.py | LLM configuration | ACTIVE |
| config/market_analysis_config.py | Market analysis config loader from YAML | ACTIVE |
| config/user_store.py | User preference/key storage | ACTIVE |
| config/app_config.yaml | YAML config for application | CONFIG |
| config/config.json | JSON config | CONFIG |
| config/settings.json | JSON settings | CONFIG |
| config/logging_config.yaml | YAML config for logging | CONFIG |
| config/market_analysis_config.yaml | YAML config for market analysis | CONFIG |
| config/backtest.yaml | YAML config for backtest | CONFIG |
| config/forecasting.yaml | YAML config for forecasting | CONFIG |
| config/model_registry.yaml | YAML config for model registry | CONFIG |
| config/optimizer_config.yaml | YAML config for optimizer | CONFIG |
| config/strategies.yaml | YAML config for strategies | CONFIG |
| config/system_config.yaml | YAML config for system | CONFIG |
| config/task_schedule.yaml | YAML config for task schedule | CONFIG |
| config/trigger_thresholds.json | JSON config for trigger thresholds | CONFIG |
| config/user_roles.yaml | YAML config for user roles | CONFIG |
| config/xgboost_config.yaml | YAML config for XGBoost | CONFIG |
| config/prometheus.yml | YAML config for Prometheus | CONFIG |
| config/pytest.ini | Pytest config | CONFIG |
| config/strategy_registry.json | JSON strategy registry | CONFIG |
| config/CONFIG_README.md | Config documentation | CONFIG |

---

## components/

| File | Summary | Status |
|------|---------|--------|
| components/__init__.py | Package init | ACTIVE |
| components/onboarding.py | Streamlit onboarding for beta; collect API keys and preferred LLM, persist by session_id | ACTIVE |

---

## tests/

| File | Summary | Status |
|------|---------|--------|
| tests/__init__.py | Package init; test discovery | ACTIVE |
| tests/conftest.py | Pytest configuration and shared fixtures | ACTIVE |
| tests/test_agent_manager.py | Tests for Enhanced Agent Manager | ACTIVE |
| tests/test_agent_registry.py | Tests for agent registry | ACTIVE |
| tests/test_imports.py | Import tests | ACTIVE |
| tests/test_basic_imports.py | Basic import tests | ACTIVE |
| tests/test_production_modules.py | Production module tests | ACTIVE |
| tests/test_production_readiness.py | Production readiness tests | ACTIVE |
| tests/test_forecasting/* | Forecasting tests (test_arima, test_hybrid, test_lstm, test_models, test_prophet) | ACTIVE |
| tests/test_optimization/* | Optimization tests (backtest_optimizer, hyperparameter_tuner, portfolio_optimizer, strategy_optimizer) | ACTIVE |
| tests/test_agents/* | Agent tests (router, leaderboard, base_interface, ensemble_voter, execution_risk_controls, etc.) | ACTIVE |
| tests/test_strategies/* | Strategy tests (bollinger, ensemble, macd, rsi, sma) | ACTIVE |
| tests/test_risk/* | Risk tests (test_risk_manager) | ACTIVE |
| tests/unit/* | Unit tests (forecast_router, hybrid_forecaster, lstm/xgboost/arima/prophet forecaster, signals, backtester, etc.) | ACTIVE |
| tests/analysis/*, tests/integration/*, tests/nlp/*, tests/optimization/*, tests/report/*, tests/full_pipeline/* | Other test packages | ACTIVE |
| tests/comprehensive_audit.py, tests/targeted_audit.py, tests/simple_audit.py, tests/check_system.py, tests/audit_return_statements.py | One-off audit/check scripts | ACTIVE / ORPHAN per pytest collect |

---

## Root-level files

| File | Summary | Status |
|------|---------|--------|
| app.py | Evolve Trading Platform — Streamlit entry point; page config, logging, env, sidebar | ACTIVE |
| main.py | Main entry point; CLI, health checks, launch options | ACTIVE |
| system_resilience.py | System resilience utilities | ACTIVE |
| setup.py | Package setup | ACTIVE |
| requirements.txt | Python dependency list | CONFIG |
| .env | Environment variables (excluded from audit scope in walk) | CONFIG |
| .gitignore | Git ignore rules | CONFIG |

---

### Orphaned Files (candidates for deletion)

- **trading/agents/launch_execution_agent.py** — Standalone script to run the Execution Agent; no other module imports it. Safe to delete if execution is only started via the app or another entry point.
- **trading/agents/launch_leaderboard_dashboard.py** — Launches the leaderboard dashboard as a script; no imports from the rest of the codebase. Safe to delete if the dashboard is always run from the main app or a different launcher.
- **trading/agents/demo_risk_controls.py** — Demo script for risk controls; only self-references. Safe to delete if demos are no longer needed.
- **trading/agents/demo_leaderboard.py** — Demo for leaderboard analytics; only self-references. Safe to delete if demos are no longer needed.
- **trading/agents/demo_pluggable_agents.py** — Demo for pluggable agents; not imported elsewhere. Safe to delete if demos are consolidated or removed.
- **trading/agents/test_integration.py** — Integration test helper (not a pytest test); listed in tests/__init__ but not imported as a module. Safe to delete if integration tests use other entry points.
- **trading/agents/test_execution_agent.py** — Standalone test script for the execution agent; not imported. Safe to delete if execution agent is tested via pytest only.
- **trading/nlp/sandbox_nlp.py** — Terminal-based NLP sandbox; run as `python sandbox_nlp.py` only; no other file imports it. Safe to delete if the NLP sandbox is no longer used for experimentation.

Any files under **_dead_code/** (if present) are not referenced by the active codebase and are candidates for deletion after confirming no external docs or runbooks reference them. One-off scripts under **scripts/** with zero references (e.g. analyze_file_usage.py, find_division_bugs.py) are safe to delete if not run manually or from CI.

---

### Stub Files (candidates for completion or deletion)

- **trading/forecasting/__init__.py** — Empty or minimal; fine as a package marker. No implementation needed.
- No other files in the audited directories (pages/, trading/models/, trading/forecasting/, trading/agents/, agents/, trading/data/, trading/strategies/, trading/portfolio/, trading/analysis/, trading/memory/, trading/nlp/, trading/utils/, trading/optimization/, trading/feature_engineering/, config/, components/, tests/, root) were identified as mostly pass/TODO/empty. Some modules (e.g. demo/launch scripts) have narrow use; worth implementing or keeping only if the feature is on the product roadmap.

---

### Duplicate Files (candidates for consolidation)

- **trading/utils/cache_manager.py** vs **trading/services/cache_manager.py** — Both define a CacheManager. `trading.utils.cache_manager` is the main export (used by trading/utils/__init__.py and decorators). `trading.services.cache_manager` is used by trading/services/quant_gpt.py for service-level caching. **Recommendation:** Keep `trading/utils/cache_manager.py` as the canonical implementation; refactor quant_gpt to use it or a thin wrapper.
- **utils/performance_metrics.py** vs **trading/utils/performance_metrics.py** — Overlapping performance metrics helpers; app and trading may use different roots. **Recommendation:** Audit call sites and consolidate into one module (e.g. trading.utils.performance_metrics) and re-export from utils if needed for backward compatibility.
- **config/config.py** vs **config/app_config.py** vs **config/primary_config.py** — Not duplicates; different roles (simplified interface, AppConfig/loading, single entry point). **Recommendation:** Keep all three; ensure primary_config remains the documented single entry.
- **llm/llm_summary.py** (if at repo root) vs **agents/llm/llm_summary.py** — Different behavior (strategy commentary vs market summary). **Recommendation:** Keep both; consider renaming or moving one into a shared package to avoid confusion.
