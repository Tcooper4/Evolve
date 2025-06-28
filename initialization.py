import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'trading'))

from trading.strategies import StrategyManager
from trading.backtesting import BacktestEngine
from trading.risk import RiskManager
from trading.portfolio import PortfolioManager
from trading.analysis import MarketAnalyzer
from trading.optimization import Optimizer
from trading.llm import LLMInterface
from trading.feature_engineering import FeatureEngineer
from trading.evaluation import ModelEvaluator
from trading.knowledge_base.trading_rules import TradingRules
from trading.market.market_data import MarketData
from trading.execution.execution_engine import ExecutionEngine
from trading.market.market_indicators import MarketIndicators
from trading.models.lstm_model import LSTMForecaster
from trading.nlp.llm_processor import LLMProcessor


def init_session() -> None:
    """Initialize Streamlit session configuration."""
    st.set_page_config(
        page_title="Evolve Clean Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def init_api() -> None:
    """Initialize API-related session state variables."""
    if 'use_api' not in st.session_state:
        st.session_state.use_api = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""


def init_portfolio() -> None:
    """Initialize portfolio and risk management components."""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = PortfolioManager()
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager(pd.Series(dtype=float))
    if 'strategy_manager' not in st.session_state:
        st.session_state.strategy_manager = StrategyManager()
    if 'backtest_engine' not in st.session_state:
        st.session_state.backtest_engine = BacktestEngine(pd.DataFrame())


def init_analysis() -> None:
    """Initialize analysis and optimization components."""
    if 'market_analyzer' not in st.session_state:
        st.session_state.market_analyzer = MarketAnalyzer()
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = Optimizer(state_dim=10, action_dim=5)
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = FeatureEngineer()
    if 'model_evaluator' not in st.session_state:
        st.session_state.model_evaluator = ModelEvaluator()


def init_llm() -> None:
    """Initialize LLM interface and processor."""
    if 'llm' not in st.session_state:
        api_key = st.session_state.api_key if st.session_state.use_api else None
        st.session_state.llm = LLMInterface(api_key=api_key)
    
    if 'llm_processor' not in st.session_state:
        st.session_state.llm_processor = LLMProcessor()
        st.session_state.news_cache = {}
        st.session_state.event_cache = {}
        st.session_state.entity_cache = {}
        st.session_state.backtest_results = {}


def init_trading_rules() -> None:
    """Initialize trading rules and knowledge base."""
    if 'trading_rules' not in st.session_state:
        st.session_state.trading_rules = TradingRules()
        st.session_state.trading_rules.load_rules_from_json(
            'trading/knowledge_base/trading_rules.json'
        )


def init_market_components() -> None:
    """Initialize market data and execution components."""
    if 'market_data' not in st.session_state:
        st.session_state.market_data = MarketData()
    if 'execution_engine' not in st.session_state:
        st.session_state.execution_engine = ExecutionEngine()
    if 'market_indicators' not in st.session_state:
        st.session_state.market_indicators = MarketIndicators()


def init_models() -> None:
    """Initialize ML models and their configurations."""
    if 'lstm_model' not in st.session_state:
        lstm_config = {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 10,
            'feature_columns': ['open', 'high', 'low', 'close', 'volume'],
            'target_column': 'close',
            'use_attention': True,
            'use_batch_norm': True,
            'max_sequence_length': 100,
            'max_batch_size': 32,
        }
        st.session_state.lstm_model = LSTMForecaster(lstm_config)
        st.session_state.model_metrics = {}


def init_session_state() -> None:
    """Initialize Streamlit session state with core components."""
    init_session()
    init_api()
    init_portfolio()
    init_analysis()
    init_llm()
    init_trading_rules()
    init_market_components()
    init_models()

