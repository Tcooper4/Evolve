"""
Evolve Trading Platform - Production Ready Streamlit App

A clean, ChatGPT-like interface with a single prompt box that routes to:
- Forecasting
- Strategy Selection
- Signal Generation
- Trade Reports

All triggered from a single input with professional styling.
"""

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add project root to Python path for imports
# This ensures modules can be imported correctly in Streamlit
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure advanced logging first
try:
    from config.logging_config import setup_logging, get_logger
    
    # Setup advanced logging
    logging_config = setup_logging(
        config={
            'level': logging.INFO,
            'file': 'logs/trading_system.log',
            'max_size': 10*1024*1024,  # 10 MB
            'backup_count': 5,
            'console': True,
            'log_rotation': True
        }
    )
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("Trading System Starting Up")
    logger.info("="*80)
except ImportError:
    # Fallback to basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("Advanced logging not available, using basic logging")
except Exception as e:
    # Fallback to basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Error setting up advanced logging: {e}, using basic logging")

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress Streamlit warnings when importing
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    logger.info("✅ Environment variables loaded from .env file")
except ImportError:
    logger.warning("⚠️ python-dotenv not available, using system environment variables")
except Exception as e:
    logger.error(f"⚠️ Error loading .env file: {e}")

# Debug: Check if API keys are loaded
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
finnhub_key = os.getenv("FINNHUB_API_KEY")
polygon_key = os.getenv("POLYGON_API_KEY")

if alpha_vantage_key:
    logger.info("✅ ALPHA_VANTAGE_API_KEY loaded successfully")
else:
    logger.warning("❌ ALPHA_VANTAGE_API_KEY not found")

if finnhub_key:
    logger.info("✅ FINNHUB_API_KEY loaded successfully")
else:
    logger.warning("❌ FINNHUB_API_KEY not found")

if polygon_key:
    logger.info("✅ POLYGON_API_KEY loaded successfully")
else:
    logger.warning("❌ POLYGON_API_KEY not found")

# Configure page
st.set_page_config(
    page_title="Evolve AI Trading Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import core components with error handling
try:
    from ui.page_renderer import (
        render_agent_logs,
        render_conversation_history,
        render_footer,
        render_forecasting_page,
        render_home_page,
        render_model_page,
        render_orchestrator_page,
        render_performance_analytics_page,
        render_prompt_interface,
        render_prompt_result,
        render_reports_page,
        render_risk_management_page,
        render_settings_page,
        render_sidebar,
        render_strategy_page,
        render_system_monitor_page,
        render_top_navigation,
        render_voice_input,
    )

    CORE_COMPONENTS_AVAILABLE = True
    logger.info("✅ UI components loaded successfully")
except ImportError as e:
    logger.warning(f"UI components import error: {e}")
    CORE_COMPONENTS_AVAILABLE = False
    # Create fallback functions
    def render_sidebar():
        return "🏠 Home & Chat", None
    
    def render_top_navigation():
        st.markdown("## 🚀 Evolve AI Trading Platform")
    
    def render_voice_input():
        pass
    
    def render_prompt_interface():
        prompt = st.text_input("Enter your prompt:")
        submit = st.button("Submit")
        return prompt, submit
    
    def render_prompt_result(result):
        st.write("Result:", result)
    
    def render_conversation_history():
        pass
    
    def render_agent_logs():
        pass
    
    def render_home_page():
        st.write("Welcome to Evolve AI Trading Platform")
    
    def render_forecasting_page():
        st.write("Forecasting page - not fully implemented")
    
    def render_strategy_page():
        st.write("Strategy page - not fully implemented")
    
    def render_model_page():
        st.write("Model page - not fully implemented")
    
    def render_reports_page():
        st.write("Reports page - not fully implemented")
    
    def render_settings_page():
        st.write("Settings page - not fully implemented")
    
    def render_system_monitor_page():
        st.write("System monitor page - not fully implemented")
    
    def render_performance_analytics_page():
        st.write("Performance analytics page - not fully implemented")
    
    def render_risk_management_page():
        st.write("Risk management page - not fully implemented")
    
    def render_orchestrator_page():
        st.write("Orchestrator page - not fully implemented")
    
    def render_footer():
        st.markdown("---")
        st.markdown("Evolve AI Trading Platform")
except Exception as e:
    import traceback

    traceback.print_exc()
    logger.warning(f"Some modules not available: {e}")
    CORE_COMPONENTS_AVAILABLE = False

# Add Task Orchestrator integration
try:
    from core.orchestrator.task_orchestrator import TaskOrchestrator
    from core.orchestrator.task_scheduler import TaskScheduler
    from core.orchestrator.task_monitor import TaskMonitor
    
    # Initialize orchestrator components
    task_orchestrator = TaskOrchestrator()
    task_scheduler = TaskScheduler(orchestrator=task_orchestrator)
    task_monitor = TaskMonitor(orchestrator=task_orchestrator)
    
    # Store in session state for access across pages
    if 'task_orchestrator' not in st.session_state:
        st.session_state.task_orchestrator = task_orchestrator
    if 'task_scheduler' not in st.session_state:
        st.session_state.task_scheduler = task_scheduler
    if 'task_monitor' not in st.session_state:
        st.session_state.task_monitor = task_monitor
    
    ORCHESTRATOR_AVAILABLE = True
    logger.info("✅ Task Orchestrator initialized successfully")
    
except ImportError as e:
    logger.warning(f"Task Orchestrator not available: {e}")
    ORCHESTRATOR_AVAILABLE = False
except Exception as e:
    logger.error(f"Error initializing Task Orchestrator: {e}")
    ORCHESTRATOR_AVAILABLE = False

# Add Agent Controller integration
try:
    from agents.agent_controller import AgentController
    from agents.task_router import TaskRouter
    from agents.registry import AgentRegistry
    
    # Initialize agent infrastructure
    agent_registry = AgentRegistry()
    task_router = TaskRouter(registry=agent_registry)
    agent_controller = AgentController(
        registry=agent_registry,
        router=task_router
    )
    
    # Store in session state
    if 'agent_controller' not in st.session_state:
        st.session_state.agent_controller = agent_controller
    if 'agent_registry' not in st.session_state:
        st.session_state.agent_registry = agent_registry
    if 'task_router' not in st.session_state:
        st.session_state.task_router = task_router
    
    AGENT_CONTROLLER_AVAILABLE = True
    logger.info("✅ Agent Controller initialized successfully")
    
except ImportError as e:
    logger.warning(f"Agent Controller not available: {e}")
    AGENT_CONTROLLER_AVAILABLE = False
except Exception as e:
    logger.error(f"Error initializing Agent Controller: {e}")
    AGENT_CONTROLLER_AVAILABLE = False

# Add Notification Service
try:
    from system.infra.agents.notifications.notification_service import NotificationService
    
    # Initialize notification system
    notification_service = NotificationService()
    
    if 'notification_service' not in st.session_state:
        st.session_state.notification_service = notification_service
    
    NOTIFICATION_SERVICE_AVAILABLE = True
    logger.info("✅ Notification Service initialized successfully")
    
except ImportError as e:
    logger.warning(f"Notification Service not available: {e}")
    NOTIFICATION_SERVICE_AVAILABLE = False
except Exception as e:
    logger.error(f"Error initializing Notification Service: {e}")
    NOTIFICATION_SERVICE_AVAILABLE = False

# Add Audit Logger
try:
    from trading.logs.audit_logger import AuditLogger
    
    # Initialize audit logger
    if 'audit_logger' not in st.session_state:
        st.session_state.audit_logger = AuditLogger()
    
    AUDIT_LOGGER_AVAILABLE = True
    logger.info("✅ Audit Logger initialized successfully")
    
except ImportError as e:
    logger.warning(f"Audit Logger not available: {e}")
    AUDIT_LOGGER_AVAILABLE = False
except Exception as e:
    logger.error(f"Error initializing Audit Logger: {e}")
    AUDIT_LOGGER_AVAILABLE = False

# Add System Health Monitoring
try:
    from system.infra.agents.services.automation_health import HealthMonitor
    from system.infra.agents.services.automation_monitoring import SystemMonitor
    
    # Initialize monitoring
    health_monitor = HealthMonitor()
    system_monitor = SystemMonitor()
    
    if 'health_monitor' not in st.session_state:
        st.session_state.health_monitor = health_monitor
    if 'system_monitor' not in st.session_state:
        st.session_state.system_monitor = system_monitor
    
    MONITORING_AVAILABLE = True
    logger.info("✅ System monitoring initialized")
    
except ImportError as e:
    logger.warning(f"Monitoring not available: {e}")
    MONITORING_AVAILABLE = False
except Exception as e:
    logger.error(f"Error initializing monitoring: {e}")
    MONITORING_AVAILABLE = False

# Add Automation Core & Workflows
try:
    from system.infra.agents.services.automation_core import AutomationCore
    from system.infra.agents.services.automation_workflows import WorkflowManager
    from system.infra.agents.services.automation_config import AutomationConfig
    
    # Initialize automation system
    automation_config = AutomationConfig()
    automation_core = AutomationCore(config=automation_config)
    workflow_manager = WorkflowManager(automation_core=automation_core)
    
    # Store in session state
    if 'automation_core' not in st.session_state:
        st.session_state.automation_core = automation_core
    if 'workflow_manager' not in st.session_state:
        st.session_state.workflow_manager = workflow_manager
    if 'automation_config' not in st.session_state:
        st.session_state.automation_config = automation_config
    
    AUTOMATION_AVAILABLE = True
    logger.info("✅ Automation system initialized successfully")
    
except ImportError as e:
    logger.warning(f"Automation system not available: {e}")
    AUTOMATION_AVAILABLE = False
except Exception as e:
    logger.error(f"Error initializing automation: {e}")
    AUTOMATION_AVAILABLE = False

# Add Model Performance Monitoring
try:
    from trading.memory.model_log import ModelLog
    from trading.memory.performance_logger import PerformanceLogger
    from trading.memory.model_monitor import ModelMonitor
    
    model_log = ModelLog()
    perf_logger = PerformanceLogger()
    model_monitor = ModelMonitor()
    
    if 'model_log' not in st.session_state:
        st.session_state.model_log = model_log
    if 'perf_logger' not in st.session_state:
        st.session_state.perf_logger = perf_logger
    if 'model_monitor' not in st.session_state:
        st.session_state.model_monitor = model_monitor
    
    MODEL_MONITORING_AVAILABLE = True
    logger.info("✅ Model monitoring initialized")
    
except ImportError:
    MODEL_MONITORING_AVAILABLE = False
    logger.warning("Model monitoring not available")
except Exception as e:
    MODEL_MONITORING_AVAILABLE = False
    logger.warning(f"Error initializing model monitoring: {e}")

# Add Commentary Service
try:
    from trading.services.commentary_service import CommentaryService
    
    # Initialize commentary system
    # CommentaryService creates its own CommentaryEngine internally
    commentary_service = CommentaryService()
    
    if 'commentary_service' not in st.session_state:
        st.session_state.commentary_service = commentary_service
    
    COMMENTARY_SERVICE_AVAILABLE = True
    logger.info("✅ Commentary service initialized")
    
except ImportError as e:
    logger.warning(f"Commentary service not available: {e}")
    COMMENTARY_SERVICE_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error initializing commentary service: {e}")
    COMMENTARY_SERVICE_AVAILABLE = False

# ============================================================================
# PHASE 1-8: COMPREHENSIVE FEATURE INITIALIZATION
# ============================================================================

logger.info("="*80)
logger.info("Initializing Phase 1-8 features...")
logger.info("="*80)

# ----------------------------------------------------------------------------
# PHASE 1: DATA VALIDATION & PIPELINE
# ----------------------------------------------------------------------------

try:
    from src.utils.data_validation import DataValidator
    from src.utils.data_pipeline import DataPipeline
    
    if 'data_validator' not in st.session_state:
        st.session_state.data_validator = DataValidator()
        st.session_state.data_pipeline = DataPipeline()
    
    logger.info("✅ Data validation & pipeline initialized")
except Exception as e:
    logger.warning(f"⚠️  Data validation: {e}")

# ----------------------------------------------------------------------------
# PHASE 2: ADVANCED MODELS (Ready for registration)
# ----------------------------------------------------------------------------

try:
    from trading.models.advanced.transformer.time_series_transformer import TimeSeriesTransformer
    from trading.forecasting.hybrid_model import HybridModel
    from trading.forecasting.hybrid_model_selector import HybridModelSelector
    from trading.forecasting.forecast_postprocessor import ForecastPostprocessor
    from trading.feature_engineering.macro_feature_engineering import MacroFeatureEngineer
    
    # Store for later use
    if 'hybrid_model_selector' not in st.session_state:
        st.session_state.hybrid_model_selector = HybridModelSelector()
    if 'forecast_postprocessor' not in st.session_state:
        st.session_state.forecast_postprocessor = ForecastPostprocessor()
    if 'macro_feature_engineer' not in st.session_state:
        st.session_state.macro_feature_engineer = MacroFeatureEngineer()
    
    logger.info("✅ Phase 2 models loaded (ready for registration)")
except Exception as e:
    logger.warning(f"⚠️  Phase 2 models: {e}")

# ----------------------------------------------------------------------------
# PHASE 3: VISUALIZATION
# ----------------------------------------------------------------------------

try:
    from trading.visualization.visualizer import PerformanceVisualizer
    
    if 'performance_visualizer' not in st.session_state:
        st.session_state.performance_visualizer = PerformanceVisualizer()
    
    logger.info("✅ Visualization tools initialized")
except Exception as e:
    logger.warning(f"⚠️  Visualization: {e}")

# ----------------------------------------------------------------------------
# PHASE 4: NLP & SENTIMENT
# ----------------------------------------------------------------------------

try:
    from trading.nlp.sentiment_classifier import SentimentClassifier
    from trading.signals.sentiment_signals import SentimentSignals
    
    if 'sentiment_classifier' not in st.session_state:
        st.session_state.sentiment_classifier = SentimentClassifier()
        st.session_state.sentiment_signals = SentimentSignals()
    
    logger.info("✅ NLP & sentiment initialized")
except Exception as e:
    logger.warning(f"⚠️  NLP & sentiment: {e}")

# ----------------------------------------------------------------------------
# PHASE 5: SERVICES (already initialized above, but ensuring completeness)
# ----------------------------------------------------------------------------

# Commentary service already initialized above

# ----------------------------------------------------------------------------
# PHASE 6: ANALYTICS
# ----------------------------------------------------------------------------

try:
    from trading.utils.metrics.scorer import ModelScorer
    from trading.validation.walk_forward_utils import WalkForwardValidator
    from trading.optimization.utils.consolidator import PositionConsolidator
    
    if 'analytics_tools' not in st.session_state:
        st.session_state.model_scorer = ModelScorer()
        st.session_state.walk_forward_validator = WalkForwardValidator()
        st.session_state.position_consolidator = PositionConsolidator()
    
    logger.info("✅ Analytics initialized")
except Exception as e:
    logger.warning(f"⚠️  Analytics: {e}")

# Try to import causal analysis (may not be available)
try:
    from causal.causal_model import CausalModel
    from causal.driver_analysis import DriverAnalysis
    
    if 'causal_model' not in st.session_state:
        st.session_state.causal_model = CausalModel()
        st.session_state.driver_analysis = DriverAnalysis()
    
    logger.info("✅ Causal analysis initialized")
except Exception as e:
    logger.debug(f"Causal analysis not available: {e}")

# ----------------------------------------------------------------------------
# PHASE 7: ADVANCED EXECUTION
# ----------------------------------------------------------------------------

try:
    from trading.execution.execution_engine import AdvancedExecutionEngine
    from trading.execution.execution_replay import ExecutionReplay
    from trading.nlp.llm_processor import LLMProcessor
    
    if 'execution_tools' not in st.session_state:
        st.session_state.execution_engine = AdvancedExecutionEngine()
        st.session_state.execution_replay = ExecutionReplay()
        st.session_state.llm_processor = LLMProcessor()
    
    logger.info("✅ Execution tools initialized")
except Exception as e:
    logger.warning(f"⚠️  Execution tools: {e}")

logger.info("="*80)
logger.info("Phase 1-8 initialization complete")
logger.info("="*80)

# Try to initialize (but don't fail if it doesn't work)
try:
    init_task_orchestrator()
except Exception as e:
    logger.debug(f"Task Orchestrator initialization deferred: {e}")

try:
    init_agent_controller()
except Exception as e:
    logger.debug(f"Agent Controller initialization deferred: {e}")

# Enhanced ChatGPT-like styling
st.markdown(
    """
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 0;
        font-weight: 300;
    }

    /* Prompt input styling */
    .prompt-container {
        background: white;
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        margin: 2rem 0;
        border: 2px solid #f8f9fa;
        transition: all 0.3s ease;
    }

    .prompt-container:hover {
        box-shadow: 0 16px 50px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }

    .stTextInput > div > div > input {
        font-size: 1.4rem !important;
        padding: 1.2rem 1.8rem !important;
        border-radius: 20px !important;
        border: 2px solid #e9ecef !important;
        background: #fafbfc !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
        background: white !important;
        transform: scale(1.02);
    }

    /* Button styling */
    .stButton > button {
        font-size: 1.3rem !important;
        padding: 1.2rem 2.5rem !important;
        border-radius: 20px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5) !important;
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    }

    /* Result card styling */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }

    .result-card h3 {
        color: #2c3e50;
        margin-bottom: 1.2rem;
        font-weight: 700;
        font-size: 1.4rem;
    }

    .result-card p {
        color: #5a6c7d;
        line-height: 1.7;
        font-size: 1.1rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }

    /* Navigation styling */
    .nav-item {
        padding: 1rem 1.2rem;
        margin: 0.3rem 0;
        border-radius: 15px;
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 500;
        border: 2px solid transparent;
    }

    .nav-item:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .status-success {
        background: linear-gradient(45deg, #28a745, #20c997);
        animation: pulse 2s infinite;
    }

    .status-warning {
        background: linear-gradient(45deg, #ffc107, #fd7e14);
    }

    .status-error {
        background: linear-gradient(45deg, #dc3545, #e83e8c);
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    /* Loading animation */
    .loading-spinner {
        text-align: center;
        padding: 3rem;
    }

    /* Conversation styling */
    .conversation-item {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    .conversation-prompt {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    .conversation-response {
        color: #5a6c7d;
        font-style: italic;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem;
        }

        .prompt-container {
            padding: 1.5rem;
        }

        .stTextInput > div > div > input {
            font-size: 1.2rem !important;
            padding: 1rem 1.5rem !important;
        }
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar and Navigation ---
if CORE_COMPONENTS_AVAILABLE:
    primary_nav, advanced_nav = render_sidebar()
else:
    st.error("UI components failed to load. Check logs for details.")
    primary_nav = advanced_nav = None

# --- Top Navigation Bar ---
render_top_navigation()

# --- Voice Input (if enabled) ---
render_voice_input()

# --- Prompt Input and Result ---
prompt, submit = render_prompt_interface()

if submit and prompt:
    with st.spinner("Processing your request..."):
        try:
            import asyncio

            from routing.prompt_router import route_prompt

            # Run the async route_prompt function
            result = asyncio.run(route_prompt(prompt, llm_type="default"))
            render_prompt_result(result)
            # Store response in session state
            if "last_response" not in st.session_state:
                st.session_state.last_response = None
            st.session_state.last_response = result
            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []
            st.session_state.conversation_history.append(
                {"prompt": prompt, "response": result, "timestamp": datetime.now()}
            )
            navigation_info = result.get("navigation_info", {})
            if navigation_info.get("main_nav"):
                st.session_state.main_nav = navigation_info["main_nav"]
            st.success(
                "✅ Request processed successfully! Evolve AI has analyzed your query."
            )
        except ImportError as e:
            st.error(f"❌ System configuration error: {str(e)}")
            st.info("Please ensure all required modules are properly installed.")
        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")

# --- Conversation History ---
render_conversation_history()

# --- Agent Logs ---
render_agent_logs()

# --- Main Content ---
if primary_nav == "🏠 Home & Chat":
    render_home_page()
elif primary_nav == "📈 Forecasting":
    render_forecasting_page()
elif primary_nav == "⚡ Strategy Lab":
    render_strategy_page()
elif primary_nav == "🧠 Model Lab":
    render_model_page()
elif primary_nav == "📋 Reports":
    render_reports_page()

# --- Advanced Navigation ---
if advanced_nav == "⚙️ Settings":
    render_settings_page()
elif advanced_nav == "📊 Monitor":
    render_system_monitor_page()
elif advanced_nav == "📈 Analytics":
    render_performance_analytics_page()
elif advanced_nav == "🛡️ Risk":
    render_risk_management_page()
elif advanced_nav == "🤖 Orchestrator":
    render_orchestrator_page()

# --- Footer ---
render_footer()
