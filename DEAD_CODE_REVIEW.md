# Dead Code Review — _dead_code/

This document reviews every file under `_dead_code/`. For each subdirectory we list every file with a one-paragraph summary, then answer:

1. **Unique logic?** — Does it contain any logic not found elsewhere in the active codebase?
2. **Worth salvaging?** — Any feature ideas, algorithms, or UI patterns worth salvaging?
3. **Truly dead?** — Is it just broken/replaced code with no salvageable value?

---

## _dead_code/agents/

### File list and summaries

| File | Summary |
|------|--------|
| **multimodal_agent.py** | Visual reasoning agent: generates plots (Matplotlib/Plotly), passes images to vision models (GPT-4V or BLIP), produces natural-language insights on equity curve, drawdown, and performance. Includes optional audio (librosa, speech_recognition) and document (docx) handling. |
| **prompt_router_agent.py** | Routes user requests by request type, agent capabilities, historical performance, and load. Defines RequestType/AgentCapability enums, AgentInfo/capability matching, load balancing, fallback handling, and performance tracking. |
| **model_creator_agent.py** | Dynamically creates, validates, tests, and evaluates ML models (sklearn, XGBoost, LightGBM, PyTorch) with backtesting and automatic model management. |
| **walk_forward_agent.py** | Walk-forward validation and rolling retraining: RollingRetrainConfig (train/test windows, step size, retrain frequency), model_factory integration, prevents data leakage and simulates live deployment. |
| **updater/scheduler.py** | Scheduler for periodic model updates and reweighting checks using `schedule`; runs in a background thread with start/stop and a check callback. |
| **updater/agent.py** | UpdaterAgent: periodic model updates, performance monitoring, drift detection, ensemble reweighting. Uses TaskMemory, UpdateScheduler, and utils (e.g. detect_model_drift, get_ensemble_weights). Imports reference `trading.scheduler` and `trading.utils` that may have moved. |
| **updater/utils.py** | Helpers for updater: check_model_performance (MSE/Sharpe/drawdown thresholds), drift detection, update validation. |
| **updater/__init__.py** | Re-exports UpdaterAgent from .agent. |
| **task_agent.py** | Recursive task execution with TaskType/ActionType enums, TaskContext, performance monitoring, retry logic, and integration with memory.prompt_log and trading.memory.agent_logger. |
| **strategy_research_agent.py** | Internet-based strategy discovery: arXiv, SSRN, GitHub, QuantConnect; extracts strategy logic, saves discovered strategies, schedules backtests with schedule/BeautifulSoup/requests. |
| **model_innovation_agent.py** | AutoML model discovery and evaluation with FLAML/Optuna; compares against ensemble, updates model registry, hybrid weight optimization. Uses utils.cache_utils and utils.weight_registry (may not exist in active tree). |
| **model_generator_agent.py** | Auto-evolutionary model generator: discovers and implements models from research papers (arXiv), benchmarks, integrates with trading.base_agent (legacy import). |
| **model_generator.py** | Standalone ModelDiscoveryAgent (non–BaseAgent): arXiv search, architecture patterns (transformer, LSTM, etc.), discovers and persists models to models/discovered_models.json. |
| **model_discovery_agent.py** | BaseAgent version of model discovery: predefined fallback models, validation threshold, discovery/validation/history. |
| **prompt_agent.py** | Enhanced prompt agent: Hugging Face classification, GPT-4 structured parser, JSON schema validation, fallback chain, confidence scoring, optional Redis memory, PromptTraceLogger fallback. |
| **llm/quant_gpt_commentary_agent.py** | QuantGPT commentary: trade explanation, overfitting detection, regime analysis, counterfactual analysis, risk assessment, performance attribution; CommentaryType enum and LLMInterface/MarketAnalyzer/AgentMemory integration. |
| **swarm_orchestrator.py** | Coordinates multiple agents (Strategy, Risk, Execution, etc.) as async jobs; SwarmConfig (Redis or SQLite), SwarmTask, AgentType/AgentStatus. Imports WalkForwardAgent and RegimeDetectionAgent from trading.agents. |
| **forecast_dispatcher.py** | Dispatches forecasts with fallback (max retries, fallback model order, NaN threshold), confidence intervals, consensus checking across LSTM/XGBoost/Prophet/ARIMA/Ensemble. |
| **meta_learner.py** | Meta-learning agent: store_experience, learn_from_experiences, get_recommendation; MetaLearningExperience and adaptation to changing conditions. |
| **self_tuning_optimizer_agent.py** | Self-tuning optimizer: triggers (performance_decline, regime_change, scheduled, volatility_spike), Bayesian/Genetic optimization via core_optimizer, constraint handling. |
| **rolling_retraining_agent.py** | Rolling retraining + walk-forward: RetrainingConfig (frequency, lookback, performance_threshold), joblib/sklearn, feature importance and score tracking. |
| **task_delegation_agent.py** | Task delegation: delegate_task, delegate_workflow, task status, cancel, agent registration; TaskPriority/TaskStatus, integrates with AgentMemory and log_agent_thought. |
| **self_improving_agent.py** | Self-improving agent: performance_history, improvement_metrics, confidence; thin execute() with placeholder logic. |
| **prompt_clarification_agent.py** | Detects ambiguous prompts (AmbiguityType: multiple_strategies, vague_request, contradictory, missing_context, etc.), asks clarification questions with options. |
| **optimization/optimizer_agent.py** | Optimizer agent: parameter_validator, strategy_optimizer, backtest_integration, performance_analyzer; strategy combination and parameter optimization. |
| **nlp_agent.py** | NLP agent with spaCy, transformers, TextBlob for prompt parsing and model routing. |
| **meta_tuner_agent.py** | Autonomous hyperparameter tuning with skopt gp_minimize and Optuna; LSTM/XGBoost/RSI support, tuning history storage. |
| **meta_research_agent.py** | Automated research discovery and model evaluation; aiohttp/BeautifulSoup, ModelSelectorAgent and ModelRegistry integration. |
| **meta_learning_feedback_agent.py** | Meta-learning feedback: process_feedback, ensemble weights, model performance summary; integrates ModelSelectorAgent, ModelRegistry, Bayesian/Genetic optimizers. |
| **meta_agent_orchestrator.py** | Orchestrates multiple agents with AgentCall (timeout, retries, fallback_agent), OrchestrationResult, error handling and retries. |
| **meta_agent.py** | Meta-agent with memory store (JSON/SQLite): ModelStatus, PerformanceMetric, ModelPerformance tracking for upgraded models. |
| **prompt_router.py** | Prompt processing and routing: RequestType (including INVESTMENT), PromptContext, ProcessedPrompt, compound prompt parsing and sub-task dispatch; no BaseAgent. |

### 1. Unique logic not found elsewhere?

- **Yes, potentially unique:**  
  - **multimodal_agent.py** — Vision (GPT-4V/BLIP) + plot → insight pipeline; audio/doc handling. Active codebase has commentary and charts but not this unified vision-audio-doc agent.  
  - **prompt_clarification_agent.py** — Explicit ambiguity detection and clarification questions (AmbiguityType, options). Active NLP/intent may not have this flow.  
  - **fallback_llm (see fallback/)** — Negation detection (NegationType, action skipping) is a distinct idea; worth checking if active NL layer has it.  
  - **strategy_research_agent.py** — Structured discovery from arXiv/SSRN/GitHub/QuantConnect + scheduling backtests; active code may have strategy research but not this full pipeline.  
  - **model_innovation_agent.py** — FLAML + registry update + hybrid weight optimization in one agent; active model lab/optimization may be split differently.  
  - **meta_learner.py** / **meta_learning_feedback_agent.py** — Store experience / learn / recommend loop; active memory/agents may not implement this meta-learning loop.  
  - **self_tuning_optimizer_agent.py** — Trigger-based (performance_decline, regime_change, volatility_spike) auto-optimization; active optimization may be more manual/scheduled.

- **Partially overlapping:**  
  - Walk-forward / rolling retraining: active code has `trading/validation/walk_forward_utils.py` and optuna/tuners; dead_code has fuller agent wrappers (WalkForwardAgent, RollingRetrainingAgent).  
  - Forecast dispatch with fallback/consensus: active code has forecast_router and hybrid model; dead_code ForecastDispatcher is an agent-shaped version with explicit consensus config.  
  - Prompt routing: active `enhanced_prompt_router` and routing_engine; dead_code has multiple variants (prompt_router_agent, prompt_router, prompt_agent) with different feature sets.

### 2. Feature ideas / algorithms / UI patterns worth salvaging?

- **Worth salvaging:**  
  - **Multimodal agent:** “Chart → image → vision model → narrative” for equity curve/drawdown/performance.  
  - **Prompt clarification:** Ambiguity types and clarification-with-options before executing.  
  - **Negation/action-skip** in fallback_llm (see fallback section).  
  - **Strategy research pipeline:** arXiv/SSRN/GitHub/QuantConnect discovery + backtest scheduling.  
  - **Meta-learning:** Experience storage and recommendation loop.  
  - **Trigger-based self-tuning:** Performance/regime/volatility triggers for optimization.  
  - **Swarm orchestration:** Redis/SQLite-backed multi-agent coordination (if not already in active core).  
  - **Consensus forecasting:** Agreement threshold + confidence/performance weighting (if not in active hybrid).

- **Reference only:**  
  - Updater scheduler/agent/utils: pattern (periodic drift check + reweight) is useful; implementation depends on current trading.scheduler and trading.utils.  
  - QuantGPT commentary agent: commentary types (overfitting, counterfactual, regime) as a checklist for active commentary.

### 3. Truly just broken/replaced with no value?

- **Mostly replaced, minimal unique value:**  
  - **self_improving_agent.py** — Very thin; active self-improvement (if any) would need a fresh design.  
  - **meta_agent_orchestrator.py** — Simple orchestrator; active agent_manager and routing likely supersede it.  
  - **model_generator.py** vs **model_discovery_agent.py** — Two discovery implementations; keep one pattern, not both.  
  - **task_agent.py** — Rich task/context model but depends on memory.prompt_log and old agent_logger; would need porting to current task/memory APIs.  
  - **updater/agent.py** — Depends on trading.scheduler and trading.utils that may have moved; logic (drift, reweight) is the salvageable part.

- **Not broken, but superseded by design:**  
  - **prompt_router.py** vs **prompt_router_agent.py** vs **prompt_agent.py** — Active code has consolidated routing; these are legacy variants.  
  - **forecast_dispatcher.py** — Replaced by forecast_router + hybrid model in practice.  
  - **optimization/optimizer_agent.py** — Replaced by active optimization agents and backtest_integration.

---

## _dead_code/archive/

### File list and summaries

| File | Summary |
|------|--------|
| **legacy_tests/test_modular_refactor.py** | Tests for an old modular architecture: core imports (DataRequest, EventBus, IAgent, ServiceContainer, PluginManager, etc.), dependency injection, event system, plugin system, interface definitions. |
| **legacy_tests/final_integration_test.py** | Integration test: system resilience, “complete system integration”; explicitly skips deprecated UnifiedInterface v2 and old interface components. |
| **legacy_tests/fix_imports.py** | Validates trading module imports and specific submodules (trading.optimization, trading.risk, trading.portfolio, trading.agents, trading.utils) and expected class names. |

### 1. Unique logic?

- No. These are tests and import-validation scripts for an old layout (core, interfaces, plugins). The *ideas* (DI container, event bus, plugin manager) may exist elsewhere; the code itself is tied to old modules.

### 2. Worth salvaging?

- Only as **documentation** of what the old modular design expected (interfaces, events, plugins). Could inform a future refactor; not runnable as-is without the old core.

### 3. Truly dead?

- **Yes.** Broken by current structure; superseded by current tests and imports. Safe to delete after extracting any desired design notes.

---

## _dead_code/examples/

### File list and summaries

| File | Summary |
|------|--------|
| **README.md** | Documentation for examples: forecasting, backtesting, ensemble, strategy comparison, portfolio, risk analysis; code snippets and Jupyter notebook references (notebooks not present in _dead_code). |
| **task_orchestrator_example.py** | Demonstrates TaskOrchestrator (create, start, monitor, error handling); imports from core.task_orchestrator; uses mock agents. |
| **walk_forward_backtest_example.py** | Walk-forward backtest usage example. |
| **strategy_research_example.py** | Strategy research agent usage. |
| **task_agent_example.py** | Task agent usage. |
| **sentiment_analysis_example.py** | Sentiment analysis example. |
| **strategy_combo_example.py** | Strategy combination example. |
| **safe_json_saving_example.py** | Safe JSON saving usage. |
| **prompt_router_example.py** | Prompt router usage. |
| **risk_aware_hybrid_model_example.py** | Risk-aware hybrid model: HybridModel with Sharpe/drawdown/MSE weighting, MockModel, sample data generation, plotting. |
| **portfolio_management_example.py** | Portfolio management usage. |
| **prompt_examples_demo.py** | Prompt examples demo. |
| **optuna_tuner_example.py** | Optuna tuner usage. |
| **monte_carlo_simulation_example.py** | Monte Carlo simulation example. |
| **model_innovation_example.py** | Model innovation usage. |
| **model_performance_logging_example.py** | Model performance logging example. |
| **forecasting_example.py** | Basic forecasting example. |
| **meta_controller_example.py** | Meta controller usage. |
| **explainability_example.py** | Explainability (e.g. SHAP/LIME) example. |
| **example_caching_usage.py** | Caching usage example. |
| **ensemble_strategy_example.py** | Ensemble strategy example. |
| **example_async_strategy_runner.py** | Async strategy runner example. |
| **enhanced_arima_example.py** | Enhanced ARIMA example. |
| **enhanced_cost_backtest_example.py** | Enhanced cost backtest example. |

### 1. Unique logic?

- Mostly **no**. They are usage examples for components that either exist in the active codebase (forecasting, backtest, ensemble, portfolio, risk, optuna, walk-forward, etc.) or for removed components (task orchestrator, task agent, old prompt router).  
- **risk_aware_hybrid_model_example.py** is a clear, self-contained demo of risk-aware weighting (Sharpe/drawdown/MSE) with mocks; the *pattern* is in active hybrid model, but the example is still a good tutorial.

### 2. Worth salvaging?

- **Yes, as templates/tutorials:**  
  - **risk_aware_hybrid_model_example.py** — Good candidate to move to `docs/` or `examples/` in the active tree as a risk-aware hybrid tutorial.  
  - **README.md** — Structure and snippet ideas can be reused for an active “Examples” or “Quick start” doc; update imports and paths.  
  - Other examples: only if you reintroduce the corresponding feature (e.g. task orchestrator); otherwise keep as reference in _dead_code or discard.

### 3. Truly dead?

- **Mostly.** Many depend on old imports (core.task_orchestrator, old modules).  
- **Not dead:** README.md (doc only) and risk_aware_hybrid_model_example.py (self-contained pattern demo). Rest are broken or redundant unless you restore the underlying components.

---

## _dead_code/fallback/

### File list and summaries

| File | Summary |
|------|--------|
| **__init__.py** | Package init; exports FallbackAgentHub, FallbackDataFeed, FallbackPromptRouter, FallbackModelMonitor, FallbackPortfolioManager, FallbackStrategySelector, FallbackMarketRegimeAgent, FallbackHybridEngine, FallbackQuantGPT, FallbackReportExporter, create_fallback_components. |
| **data_feed.py** | FallbackDataFeed: mock + yfinance fallback, get_historical_data and basic data operations when primary sources fail. |
| **strategy_selector.py** | FallbackStrategySelector: predefined RSI/MACD/Bollinger configs with performance dicts and best_regime/risk_level. |
| **quant_gpt.py** | FallbackQuantGPT: template-based commentary (forecast/strategy/portfolio) with placeholder substitution when primary QuantGPT is down. |
| **prompt_router.py** | FallbackPromptRouter: keyword-based routing patterns (forecast, strategy, portfolio, etc.) to agent names and confidence. |
| **portfolio_manager.py** | FallbackPortfolioManager: mock portfolio (cash, positions, PnL) for degraded mode. |
| **model_monitor.py** | FallbackModelMonitor: mock model performance (lstm, xgboost, prophet) and trust_level. |
| **market_regime_agent.py** | FallbackMarketRegimeAgent: simple regime thresholds (trending, mean_reversion, volatile, sideways) and classification. |
| **hybrid_engine.py** | FallbackHybridEngine: minimal RSI/MACD/Bollinger execution stubs. |
| **fallback_model.py** | FallbackModel: train/predict stubs; predictions based on recent trend when primary models fail. |
| **agent_hub.py** | FallbackAgentHub: keyword-based route(prompt) to agent name and confidence. |
| **strategy_logger.py** | FallbackStrategyLogger: in-memory decision list and mock decision generation. |
| **report_exporter.py** | FallbackReportExporter: export_report to JSON/text and ensure export dir. |
| **fallback_strategy.py** | FallbackStrategy: FallbackSignal dataclass and generate_signals stub. |
| **fallback_llm.py** | Fallback LLM with negation detection: ActionType, NegationType (explicit, implicit, conditional, temporal), NegationPattern, action-scope and “skip” handling. |

### 1. Unique logic?

- **Yes:**  
  - **fallback_llm.py** — Negation detection (don’t/not/avoid/skip/unless) and action-scope so certain actions are skipped. This pattern is not clearly duplicated in the active codebase and is worth reviewing for the NL layer.

- **Partially:**  
  - The rest implement “degraded mode” with mocks and keywords. Active code may have resilience elsewhere (e.g. fallback providers, retries) but not necessarily this full fallback-component set.

### 2. Worth salvaging?

- **Worth it:**  
  - **fallback_llm.py** — Integrate negation/action-skip into active prompt/NL handling.  
  - **Concept:** A single `create_fallback_components()` (or similar) for degraded UI/demos when services are down; reimplement using current components rather than copying old code.

- **Reference only:**  
  - Strategy/regime/portfolio/monitor/commentary mocks as specs for what “minimal viable” fallbacks should expose (e.g. mock portfolio shape, regime labels).

### 3. Truly dead?

- **Not entirely.** Fallback_llm has unique logic. The rest are superseded by active implementations and current resilience patterns; only the *design* of a unified fallback layer and negation handling is worth keeping.

---

## _dead_code/interface/

### File list and summaries

| File | Summary |
|------|--------|
| **unified_interface.py** | Production-ready UnifiedInterface: loads config, initializes components with fallback (from fallback package), logging. Imports core.agent_hub, data.live_feed, trading.agents (MarketRegimeAgent, PromptRouterAgent), trading.memory (ModelMonitor, StrategyLogger), strategy_selection_agent, PortfolioManager, ReportExporter, trading.services.quant_gpt, HybridEngine. Uses deprecated paths (e.g. core.agent_hub, data.live_feed, trading.report.export_engine). |

### 1. Unique logic?

- **No.** It’s an integration layer that wires old component names and paths. The *pattern* (single entry point, fallback on import failure, config-driven) is standard; the code is tied to the old layout.

### 2. Worth salvaging?

- Only the **pattern**: one entry point that initializes trading components and falls back gracefully. The current app (Streamlit pages, app.py) already provides that; no need to resurrect this file.

### 3. Truly dead?

- **Yes.** Broken imports and deprecated design. Safe to delete after any one-page “unified interface pattern” note is captured elsewhere if desired.

---

## _dead_code/pages_archive/

### File list and summaries

| File | Summary |
|------|--------|
| **home.py** | Old home page: render_home_page(module_status, agent_hub_available), system status metrics, model trust (ModelMonitor), strategy logger, session_utils. Replaced by pages/0_Home.py (briefing + live events). |
| **forecast.py** | Old forecast page: render_forecast_page(), prompt_agent in session_state, trading.ui.components (create_forecast_chart, create_forecast_metrics, etc.). Replaced by pages/2_Forecasting.py. |
| **settings.py** | Legacy settings page logic. |
| **strategy.py** | Legacy strategy page logic. |
| **risk_dashboard.py** | Legacy risk dashboard. |
| **performance_tracker.py** | Legacy performance tracker. |
| **optimization_dashboard.py** | Legacy optimization dashboard. |
| **nlp_tester.py** | Legacy NLP tester UI. |
| **5_Risk_Analysis.py** | Risk analysis page. |
| **4_Portfolio_Management.py** | Portfolio management page. |
| **2_Backtest_Strategy.py** | Strategy backtest page (e.g. plot_strategy_result_plotly). |
| **HybridModel.py** | HybridModelManager: auto-adjust weights (MSE/Sharpe), performance history, weight history, ensemble composition display. |
| **Strategy_Pipeline_Demo.py** | Minimal stub: title “Strategy Pipeline Demo”, no real implementation. |
| **pre_streamline_20250101/*.py** | Pre-streamline (Jan 2025) copies of many pages: 2_Strategy_Backtest, 4_Portfolio_Management, 5_Risk_Analysis, 5_System_Scorecard, 6_Model_Optimization, 6_Strategy_History, 7_Market_Analysis, 7_Optimizer, 7_Strategy_Performance, 8_Agent_Management, 8_Explainability, 9_System_Monitoring, 10_Strategy_Health_Dashboard, 18_Alerts, 19_Admin_Panel, Forecasting, Forecast_with_AI_Selection, Model_Lab, Model_Performance_Dashboard, Monte_Carlo_Simulation, Reports, Strategy_Combo_Creator, Strategy_Lab, portfolio_dashboard, risk_preview_dashboard. |

### 1. Unique logic?

- **Rare.**  
  - **HybridModel.py** — HybridModelManager with explicit weight_history and “real-time ensemble composition” UI; active Model Lab and hybrid model may cover this; worth a quick check.  
  - **pre_streamline_20250101** — Old Streamlit layouts and widgets; logic has been merged into current numbered pages (0_Home through 12_Memory). No unique algorithms; some old layout/UX ideas might differ.

### 2. Worth salvaging?

- **Worth a look:**  
  - **HybridModel.py** — If active UI doesn’t show weight evolution and ensemble composition the same way, consider porting that UX.  
  - **2_Backtest_Strategy.py** — plot_strategy_result_plotly and similar helpers; only if current strategy testing page doesn’t already have equivalent.

- **Otherwise:**  
  - Use only as reference for “what the old pages did” during support or migration. No need to keep runnable.

### 3. Truly dead?

- **Mostly yes.** Replaced by consolidated pages (0_Home … 12_Memory). Strategy_Pipeline_Demo is a stub. Pre_streamline_20250101 is historical snapshot; safe to delete after archiving HybridModel and any desired widget/plot snippets elsewhere.

---

## Summary table

| Subdirectory    | Unique logic?              | Worth salvaging?                                      | Truly dead?        |
|-----------------|----------------------------|--------------------------------------------------------|--------------------|
| **agents/**     | Yes (multimodal, clarification, negation, meta-learning, trigger-based tuning, strategy research pipeline) | Yes (vision agent, clarification flow, negation, research pipeline, swarm pattern, consensus config) | Partially; many replaced by active agents |
| **archive/**    | No                         | Only as design doc for old modular architecture        | Yes                |
| **examples/**   | No (tutorials only)        | risk_aware_hybrid_model_example + README structure     | Mostly; some broken imports |
| **fallback/**   | Yes (negation in fallback_llm) | Yes (negation/action-skip; concept of unified fallback layer) | No; fallback_llm has value |
| **interface/**  | No                         | Pattern only; already reflected in current app         | Yes                |
| **pages_archive/** | Rare (HybridModelManager UI) | Possibly HybridModel weight/ensemble UX               | Mostly; superseded by current pages |

No deletions were made; this is a read-only review.
