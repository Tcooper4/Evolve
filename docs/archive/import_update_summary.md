# Import Update Summary

## Files Updated with Try/Except Blocks

### ✅ Completed Files

1. **agents/prompt_agent.py**
   - Updated HuggingFace imports (torch, transformers, sentence_transformers)
   - Added proper warning messages and fallback variables

2. **trading/models/lstm_model.py**
   - Updated PyTorch imports (torch, nn, optim, etc.)
   - Updated scikit-learn imports (StandardScaler)
   - Added availability checks in model classes

3. **trading/models/base_model.py**
   - Updated PyTorch imports (torch, nn, optim, etc.)
   - Updated scikit-learn imports (StandardScaler)
   - Added availability checks in TimeSeriesDataset and BaseModel classes

4. **trading/data/providers/yfinance_provider.py**
   - Updated yfinance imports
   - Updated tenacity imports
   - Added availability checks in fetch methods

5. **trading/data/data_loader.py**
   - Updated yfinance imports
   - Added availability checks in load_market_data and get_latest_price methods

6. **features/sentiment_features.py**
   - Updated vaderSentiment imports
   - Updated HuggingFace imports (torch, transformers)
   - Added proper warning messages and fallback variables

7. **trading/signals/sentiment_signals.py**
   - Updated vaderSentiment imports
   - Added proper warning messages and fallback variables

8. **trading/agents/execution/execution_providers.py**
   - Updated alpaca imports in connect and execute_trade methods
   - Added proper warning messages and fallback handling

9. **execution/broker_adapter.py**
   - Updated alpaca imports in connect and submit_order methods
   - Added proper warning messages and fallback handling

10. **trading/models/dataset.py**
    - Updated PyTorch imports (torch, Dataset)
    - Updated scikit-learn imports (StandardScaler)
    - Added availability checks in TimeSeriesDataset class

11. **trading/models/tcn_model.py**
    - Updated PyTorch imports (torch, nn)
    - Added availability checks in TemporalBlock and TCNModel classes

12. **trading/models/transformer_wrapper.py**
    - Updated PyTorch imports (torch, nn, F)
    - Added availability checks in all transformer classes

13. **trading/models/autoformer_model.py**
    - Updated PyTorch imports (torch)
    - Updated autoformer-pytorch imports
    - Added availability checks in AutoformerModel class

14. **trading/models/model_utils.py**
    - Updated PyTorch imports (torch)
    - Added availability checks in utility functions

15. **trading/models/ridge_model.py**
    - Updated scikit-learn imports (Ridge, metrics, preprocessing)
    - Added proper warning messages and fallback variables

16. **trading/models/xgboost_model.py**
    - Updated scikit-learn imports for fallback model
    - Added availability checks in XGBoostModel and FallbackXGBoostModel classes

17. **trading/nlp/sentiment_classifier.py**
    - Updated PyTorch imports (torch, nn, Dataset, DataLoader)
    - Updated transformers imports (AutoTokenizer, AutoModelForSequenceClassification, etc.)
    - Updated scikit-learn imports (TfidfVectorizer, LogisticRegression, metrics)
    - Added availability checks in all classifier classes

18. **trading/market/market_data.py**
    - Updated yfinance imports
    - Updated alpha_vantage imports
    - Added availability checks in fetch_data method

19. **trading/models/advanced/ensemble/ensemble_model.py**
    - Updated PyTorch imports (torch)
    - Updated scipy imports (norm)
    - Added availability checks in EnsembleForecaster class

20. **trading/models/advanced/gnn/gnn_model.py**
    - Updated PyTorch imports (torch, nn, F)
    - Added availability checks in GNNLayer and GNNForecaster classes

21. **trading/models/advanced/lstm/lstm_model.py**
    - Updated PyTorch imports (torch, nn)
    - Added availability checks in LSTMForecaster class

22. **trading/models/advanced/rl/strategy_optimizer.py**
    - Updated PyTorch imports (torch, nn, optim)
    - Added availability checks in StrategyOptimizer class

23. **trading/models/advanced/tcn/tcn_model.py**
    - Updated PyTorch imports (torch, nn, F)
    - Added availability checks in TemporalBlock, TemporalConvNet, and TCNModel classes

24. **trading/models/advanced/transformer/time_series_transformer.py**
    - Updated PyTorch imports (torch, nn)
    - Updated scikit-learn imports (Ridge)
    - Updated Prophet imports (ProphetModel)
    - Added availability checks in PositionalEncoding and TransformerForecaster classes

25. **trading/nlp/sentiment_bridge.py**
    - Updated sentence_transformers imports (SentenceTransformer)
    - Updated scikit-learn imports (cosine_similarity)
    - Added availability checks and proper fallback handling

26. **trading/nlp/sentiment_processor.py**
    - Updated sentence_transformers imports (SentenceTransformer)
    - Updated scikit-learn imports (cosine_similarity)
    - Added availability checks and proper fallback handling

27. **trading/llm/parser_engine.py**
    - Updated PyTorch imports (torch)
    - Updated transformers imports (AutoModelForCausalLM, AutoTokenizer, pipeline)
    - Added availability checks and proper fallback handling

28. **nlp/natural_language_insights.py**
    - Updated transformers imports (pipeline)
    - Updated NLTK imports (nltk, stopwords, WordNetLemmatizer, etc.)
    - Updated spaCy imports (spacy)
    - Added availability checks and proper fallback handling

29. **trading/agents/enhanced_prompt_router.py**
    - Updated OpenAI imports (openai)
    - Updated PyTorch imports (torch)
    - Updated transformers imports (AutoModelForCausalLM, AutoTokenizer, pipeline)
    - Added availability checks and proper fallback handling

30. **trading/agents/meta_research_agent.py**
    - Updated aiohttp imports (aiohttp)
    - Updated BeautifulSoup imports (BeautifulSoup)
    - Added availability checks and proper fallback handling

31. **trading/agents/model_creator_agent.py**
    - Updated scikit-learn imports (various sklearn modules)
    - Updated XGBoost imports (xgb)
    - Updated LightGBM imports (lgb)
    - Updated PyTorch imports (torch.nn)
    - Added availability checks and proper fallback handling

32. **trading/agents/model_discovery_agent.py**
    - Updated arxiv imports (arxiv)
    - Updated huggingface_hub imports (HfApi)
    - Updated PyTorch imports (torch.nn)
    - Added availability checks and proper fallback handling

33. **trading/agents/multimodal_agent.py**
    - Updated plotly imports (go)
    - Updated OpenAI imports (openai)
    - Updated PIL imports (Image)
    - Updated transformers imports (BlipForConditionalGeneration, BlipProcessor)
    - Updated audio processing imports (librosa, speech_recognition)
    - Updated document processing imports (docx, pandas)
    - Added availability checks and proper fallback handling

34. **trading/agents/nlp_agent.py**
    - Updated spaCy imports (spacy)
    - Updated transformers imports (AutoTokenizer, pipeline)
    - Updated TextBlob imports (TextBlob)
    - Added availability checks and proper fallback handling

35. **trading/utils/auto_repair.py**
    - Updated importlib.metadata imports (PackageNotFoundError, version)
    - Added availability checks in check_packages method

36. **trading/utils/diagnostics.py**
    - Updated psutil imports (psutil)
    - Updated PyTorch imports (torch)
    - Added availability checks in check_forecasting_models and check_system_resources methods

37. **trading/utils/data_transformer.py**
    - Updated scikit-learn imports (PCA, MinMaxScaler, RobustScaler, StandardScaler)
    - Added availability checks in DataTransformer class

38. **trading/utils/data_utils.py**
    - Updated scikit-learn imports (SimpleImputer, StandardScaler)
    - Added availability checks in DataPreprocessor class

39. **trading/utils/feature_engineering.py**
    - Updated scikit-learn imports (SelectKBest, f_regression, mutual_info_regression)
    - Added availability checks in select_features method

40. **trading/utils/model_evaluation.py**
    - Updated scikit-learn imports (mean_absolute_error, mean_squared_error, r2_score, cross_val_score)
    - Added availability checks in evaluate_regression and cross_validate methods

41. **scripts/manage_ml.py**
    - Updated MLflow imports (mlflow and all submodules)
    - Updated PyTorch imports (torch, torch.nn, torch.optim)
    - Updated scikit-learn imports (accuracy_score, f1_score, precision_score, recall_score, GridSearchCV, train_test_split)
    - Updated Optuna imports (optuna)
    - Updated Ray imports (ray, ray.serve)
    - Updated deployment library imports (bentoml, kserve, seldon_core, torchserve, triton)
    - Added availability checks in MLManager class

42. **scripts/manage_model.py**
    - Updated PyTorch imports (torch, torch.nn, torch.optim, DataLoader, TensorDataset)
    - Updated scikit-learn imports (accuracy_score, f1_score, precision_score, recall_score, GridSearchCV, train_test_split, StandardScaler)
    - Added availability checks in ModelManager class

43. **trading/risk/risk_analyzer.py**
    - Updated OpenAI imports (openai)
    - Added availability checks in RiskAnalyzer class

44. **trading/analytics/alpha_attribution_engine.py**
    - Updated scikit-learn imports (LinearRegression)
    - Added availability checks in _decompose_by_factors method

45. **trading/optimization/optimization_visualizer.py**
    - Updated optuna imports (optuna)
    - Updated plotly imports (plotly.graph_objects, plotly.subplots)
    - Added availability checks in plot_parameter_importance method

46. **trading/market/market_indicators.py**
    - Updated PyTorch imports (torch)
    - Added availability checks in MarketIndicators class

47. **fallback/data_feed.py**
    - Updated yfinance imports (yfinance)
    - Added availability checks in FallbackDataFeed class

48. **trading/feature_engineering/feature_engineer.py**
    - Updated scikit-learn imports (PCA, RandomForestRegressor, RFE, SelectKBest, VarianceThreshold, f_regression, StandardScaler)
    - Added availability checks in FeatureEngineer.__init__ method

49. **src/features/feature_selector.py**
    - Updated scikit-learn imports (SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif, RFE, SelectFromModel, RandomForestRegressor, RandomForestClassifier, Lasso, Ridge, StandardScaler)
    - Added availability checks in FeatureSelector.__init__ method

50. **trading/optimization/backtest_optimizer.py**
    - Updated matplotlib imports (matplotlib.pyplot)
    - Updated scikit-learn imports (KMeans, StandardScaler)
    - Added availability checks in RegimeDetector.__init__ and plot_results methods

51. **trading/strategies/ensemble_methods.py**
    - Updated scikit-learn imports (VotingRegressor, mean_squared_error)
    - Added availability checks in update_model_performance and combine_ensemble_model methods

52. **trading/strategies/multi_strategy_hybrid_engine.py**
    - Updated scikit-learn imports (VotingRegressor, LinearRegression, StandardScaler)
    - Added availability checks in MultiStrategyHybridEngine.__init__ method

## Remaining Files to Update

### High Priority (Core System Files)

~~1. **trading/models/dataset.py** - torch, sklearn imports~~ ✅ COMPLETED
~~2. **trading/models/tcn_model.py** - torch imports~~ ✅ COMPLETED
~~3. **trading/models/transformer_wrapper.py** - torch imports~~ ✅ COMPLETED
~~4. **trading/models/autoformer_model.py** - torch imports~~ ✅ COMPLETED
~~5. **trading/models/model_utils.py** - torch imports~~ ✅ COMPLETED
~~6. **trading/models/ridge_model.py** - sklearn imports~~ ✅ COMPLETED
~~7. **trading/models/xgboost_model.py** - sklearn imports~~ ✅ COMPLETED

### Medium Priority (Advanced Models)

~~8. **trading/models/advanced/ensemble/ensemble_model.py** - torch imports~~ ✅ COMPLETED
~~9. **trading/models/advanced/gnn/gnn_model.py** - torch imports~~ ✅ COMPLETED
~~10. **trading/models/advanced/lstm/lstm_model.py** - torch imports~~ ✅ COMPLETED
~~11. **trading/models/advanced/rl/strategy_optimizer.py** - torch imports~~ ✅ COMPLETED
~~12. **trading/models/advanced/tcn/tcn_model.py** - torch imports~~ ✅ COMPLETED
~~13. **trading/models/advanced/transformer/time_series_transformer.py** - torch, sklearn imports~~ ✅ COMPLETED

### Medium Priority (NLP and Sentiment)

~~14. **trading/nlp/sentiment_classifier.py** - torch, transformers, sklearn imports~~ ✅ COMPLETED
~~15. **trading/nlp/sentiment_bridge.py** - sklearn, sentence_transformers imports~~ ✅ COMPLETED
~~16. **trading/nlp/sentiment_processor.py** - sklearn, sentence_transformers imports~~ ✅ COMPLETED
~~17. **trading/llm/parser_engine.py** - torch, transformers imports~~ ✅ COMPLETED
~~18. **nlp/natural_language_insights.py** - transformers imports~~ ✅ COMPLETED

### Medium Priority (Agents)

~~19. **trading/agents/enhanced_prompt_router.py** - torch, transformers imports~~ ✅ COMPLETED
~~20. **trading/agents/meta_research_agent.py** - torch, sklearn imports~~ ✅ COMPLETED
~~21. **trading/agents/model_creator_agent.py** - torch, sklearn imports~~ ✅ COMPLETED
~~22. **trading/agents/model_discovery_agent.py** - torch, sklearn imports~~ ✅ COMPLETED
~~23. **trading/agents/multimodal_agent.py** - transformers imports~~ ✅ COMPLETED
~~24. **trading/agents/nlp_agent.py** - transformers imports~~ ✅ COMPLETED

### Lower Priority (Utilities and Scripts)

~~25. **trading/utils/auto_repair.py** - torch, transformers imports~~ ✅ COMPLETED
~~26. **trading/utils/diagnostics.py** - torch imports~~ ✅ COMPLETED
~~27. **trading/utils/data_transformer.py** - sklearn imports~~ ✅ COMPLETED
~~28. **trading/utils/data_utils.py** - sklearn imports~~ ✅ COMPLETED
~~29. **trading/utils/feature_engineering.py** - sklearn imports~~ ✅ COMPLETED
~~30. **trading/utils/model_evaluation.py** - sklearn imports~~ ✅ COMPLETED
~~31. **scripts/manage_ml.py** - torch, sklearn imports~~ ✅ COMPLETED
~~32. **scripts/manage_model.py** - torch, sklearn imports~~ ✅ COMPLETED

### Lower Priority (Data and Market)

~~33. **trading/market/market_data.py** - yfinance imports~~ ✅ COMPLETED
~~34. **trading/market/market_indicators.py** - torch imports~~ ✅ COMPLETED
~~35. **trading/options/options_forecaster.py** - yfinance imports~~ ✅ COMPLETED
~~36. **data/live_data_feed.py** - yfinance, alpaca imports~~ ✅ COMPLETED
~~37. **data/streaming_pipeline.py** - yfinance imports~~ ✅ COMPLETED
~~38. **fallback/data_feed.py** - yfinance imports~~ ✅ COMPLETED

### Lower Priority (Execution)

~~39. **execution/live_trading_interface.py** - alpaca imports~~ ✅ COMPLETED

### Lower Priority (Feature Engineering)

~~40. **trading/feature_engineering/feature_engineer.py** - sklearn imports~~ ✅ COMPLETED
~~41. **trading/feature_engineering/macro_feature_engineering.py** - sklearn imports~~ ✅ COMPLETED
~~42. **src/features/feature_selector.py** - sklearn imports~~ ✅ COMPLETED

### Lower Priority (Analytics and Optimization)

~~43. **trading/analytics/alpha_attribution_engine.py** - sklearn imports~~ ✅ COMPLETED
~~44. **trading/analytics/forecast_explainability.py** - sklearn imports~~ ✅ COMPLETED
~~45. **trading/optimization/backtest_optimizer.py** - sklearn imports~~ ✅ COMPLETED
46. **trading/optimization/grid_search_optimizer.py** - sklearn imports
~~47. **trading/optimization/optuna_optimizer.py** - sklearn imports~~ ✅ COMPLETED

### Lower Priority (Strategies and Backtesting)

~~48. **trading/strategies/ensemble_methods.py** - sklearn imports~~ ✅ COMPLETED
~~49. **trading/strategies/multi_strategy_hybrid_engine.py** - sklearn imports~~ ✅ COMPLETED
50. **trading/backtesting/position_sizing.py** - sklearn imports

### Lower Priority (Memory and Evaluation)

51. **trading/memory/model_monitor.py** - sklearn imports
52. **trading/memory/persistent_memory.py** - sentence_transformers imports
53. **trading/evaluation/metrics.py** - sklearn imports
54. **trading/evaluation/model_evaluator.py** - sklearn imports

### Lower Priority (Training and Models)

55. **trading/training/loop.py** - torch imports
56. **models/forecast_engine.py** - sklearn imports
57. **models/tft_model.py** - torch, sklearn imports

### Lower Priority (Causal Analysis)

58. **causal/causal_model.py** - sklearn imports

### Lower Priority (Agents Implementation)

59. **agents/implementations/implementation_generator.py** - torch, sklearn imports
60. **agents/implementations/model_benchmarker.py** - sklearn imports

### Lower Priority (LLM and Memory)

61. **agents/llm/agent.py** - sentence_transformers imports
62. **agents/llm/memory.py** - sentence_transformers imports
63. **agents/llm/model_loader.py** - torch, transformers imports

### Lower Priority (Agent Models)

64. **agents/model_generator.py** - torch imports
65. **agents/model_generator_agent.py** - sklearn imports
66. **agents/model_innovation_agent.py** - torch, sklearn imports

### Lower Priority (Other Agents)

67. **trading/agents/market_regime_agent.py** - sklearn, yfinance imports
68. **trading/agents/model_synthesizer_agent.py** - sklearn imports
69. **trading/agents/rolling_retraining_agent.py** - sklearn imports
70. **trading/agents/updater/utils.py** - sklearn imports

### Lower Priority (Scripts and Tests)

71. **scripts/manage_data_quality.py** - sklearn imports
72. **scripts/system_check.py** - yfinance imports
73. **tests/comprehensive_codebase_review.py** - torch, sklearn imports
74. **tests/test_model_innovation_agent.py** - sklearn imports
75. **tests/test_system_status.py** - yfinance imports
76. **tests/test_forecasting/test_hybrid.py** - sklearn imports
77. **tests/test_forecasting/test_models.py** - sklearn imports
78. **tests/unit/test_alpaca_migration.py** - alpaca imports
79. **tests/unit/test_tcn_model.py** - torch imports
80. **test_lstm_no_tensorflow.py** - torch imports
81. **test_alpaca_simple.py** - alpaca imports

### Lower Priority (Utilities)

82. **utils/model_utils.py** - torch imports

## Template for Updating Remaining Files

For each file, follow this pattern:

```python
# Try to import [MODULE_NAME]
try:
    import [module_name]
    from [module_name] import [specific_imports]
    [MODULE_NAME.upper().replace('-', '_')]_AVAILABLE = True
except ImportError as e:
    print("⚠️ [MODULE_NAME] not available. Disabling [FEATURE_DESCRIPTION].")
    print(f"   Missing: {e}")
    [module_name] = None
    [specific_imports] = None
    [MODULE_NAME.upper().replace('-', '_')]_AVAILABLE = False
```

Then add availability checks in the relevant methods:

```python
def some_method(self):
    if not [MODULE_NAME.upper().replace('-', '_')]_AVAILABLE:
        raise ImportError("[MODULE_NAME] is not available. Cannot perform [operation].")
    # ... rest of method
```

## Notes

- The most critical files have been updated (core models, data providers, execution)
- Focus on the "High Priority" files next as they are most likely to cause import errors
- Many files already have some try/except blocks but need to be updated with proper warnings
- Test each file after updating to ensure it doesn't break existing functionality 