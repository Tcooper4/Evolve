#!/usr/bin/env python3
"""
Comprehensive System Check for Evolve Trading Platform
Tests all modules, imports, and functionality
"""

import sys
import os
import importlib
import traceback
from datetime import datetime

def test_import(module_name, description=""):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, f"✅ {module_name} - {description}"
    except Exception as e:
        return None

def test_function_call(module_name, function_name, description=""):
    """Test if a function can be called."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            if callable(func):
                return True, f"✅ {module_name}.{function_name} - {description}"
            else:
                return False, f"❌ {module_name}.{function_name} - {description}: Not callable"
        else:
            return False, f"❌ {module_name}.{function_name} - {description}: Function not found"
    except Exception as e:
        return False, f"❌ {module_name}.{function_name} - {description}: {str(e)}"

def main():
    print("🚀 Evolve Trading Platform - Comprehensive System Check")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Test results
    results = []
    successes = 0
    failures = 0
    warnings = 0
    
    # Core modules
    print("📦 Testing Core Modules...")
    core_modules = [
        ("trading", "Core trading package"),
        ("trading.config", "Configuration management"),
        ("trading.utils", "Utility functions"),
        ("trading.memory", "Memory management"),
        ("trading.market", "Market data"),
        ("trading.evaluation", "Evaluation metrics"),
        ("trading.execution", "Trade execution"),
        ("trading.llm", "LLM integration"),
        ("trading.risk", "Risk management"),
        ("trading.portfolio", "Portfolio management"),
        ("trading.strategies", "Trading strategies"),
        ("trading.models", "ML models"),
        ("trading.optimization", "Optimization"),
        ("trading.report", "Reporting"),
        ("trading.services", "Services"),
        ("trading.ui", "User interface"),
        ("trading.visualization", "Visualization"),
        ("trading.nlp", "Natural language processing"),
        ("trading.feature_engineering", "Feature engineering"),
        ("trading.backtesting", "Backtesting"),
        ("trading.data", "Data management"),
        ("trading.analytics", "Analytics"),
        ("trading.knowledge_base", "Knowledge base"),
        ("trading.logs", "Logging"),
        ("trading.meta_agents", "Meta agents"),
        ("trading.agents", "Trading agents"),
        ("trading.agents.updater", "Agent updater"),
        ("trading.agents.upgrader", "Agent upgrader"),
    ]
    
    for module_name, description in core_modules:
        success, message = test_import(module_name, description)
        results.append((success, message))
        if success:
            successes += 1
        else:
            failures += 1
    
    # Advanced modules
    print("\n🔬 Testing Advanced Modules...")
    advanced_modules = [
        ("rl", "Reinforcement learning"),
        ("rl.rl_trader", "RL trader"),
        ("causal", "Causal inference"),
        ("causal.causal_model", "Causal model"),
        ("causal.driver_analysis", "Driver analysis"),
        ("agents", "Agent system"),
        ("agents.model_generator_agent", "Model generator agent"),
        ("agents.model_generator", "Model generator"),
        ("llm", "LLM utilities"),
        ("llm.llm_summary", "LLM summary"),
        ("risk", "Risk analytics"),
        ("risk.advanced_risk", "Advanced risk"),
        ("risk.tail_risk", "Tail risk"),
        ("execution", "Execution interface"),
        ("execution.live_trading_interface", "Live trading interface"),
        ("market_analysis", "Market analysis"),
        ("memory", "Memory system"),
        ("memory.model_monitor", "Model monitor"),
        ("memory.goals", "Goals management"),
        ("models", "Models"),
        ("models.forecast_router", "Forecast router"),
        ("optimization", "Optimization"),
        ("optimization.bayesian_optimizer", "Bayesian optimizer"),
        ("optimization.genetic_optimizer", "Genetic optimizer"),
        ("optimization.strategies", "Optimization strategies"),
        ("optimization.visualization", "Optimization visualization"),
        ("optimization.utils", "Optimization utilities"),
        ("optimization.utils.consolidator", "Optimization consolidator"),
        ("strategies", "Strategies"),
        ("strategies.gatekeeper", "Strategy gatekeeper"),
        ("reporting", "Reporting"),
        ("reporting.pnl_attribution", "PnL attribution"),
        ("data", "Data management"),
        ("data.live_feed", "Live data feed"),
        ("utils", "Utilities"),
        ("utils.config_loader", "Config loader"),
        ("utils.runner", "Runner"),
        ("visualization", "Visualization"),
        ("tools", "Tools"),
        ("tools.encoding_utils", "Encoding utilities"),
        ("core", "Core system"),
        ("core.agent_hub", "Agent hub"),
        ("core.capability_router", "Capability router"),
        ("core.agents", "Core agents"),
        ("core.agents.base_agent", "Base agent"),
        ("core.agents.goal_planner", "Goal planner"),
        ("system", "System infrastructure"),
        ("system.infra", "Infrastructure"),
        ("system.infra.agents", "Infrastructure agents"),
        ("system.infra.agents.alert_manager", "Alert manager"),
        ("system.infra.agents.api", "API agents"),
        ("system.infra.agents.api.metrics_api", "Metrics API"),
        ("system.infra.agents.api.task_api", "Task API"),
        ("system.infra.agents.auth", "Auth agents"),
        ("system.infra.agents.auth.session_manager", "Session manager"),
        ("system.infra.agents.auth.user_manager", "User manager"),
        ("system.infra.agents.config", "Config agents"),
        ("system.infra.agents.config.config_manager", "Config manager"),
        ("system.infra.agents.core", "Core agents"),
        ("system.infra.agents.core.models", "Core models"),
        ("system.infra.agents.logs", "Logging agents"),
        ("system.infra.agents.logs.automation_logging", "Automation logging"),
        ("system.infra.agents.monitoring", "Monitoring agents"),
        ("system.infra.agents.notifications", "Notification agents"),
        ("system.infra.agents.notifications.handlers", "Notification handlers"),
        ("system.infra.agents.notifications.notification_service", "Notification service"),
        ("system.infra.agents.scripts", "Script agents"),
        ("system.infra.agents.scripts.deploy_services", "Deploy services"),
        ("system.infra.agents.scripts.manage_secrets", "Manage secrets"),
        ("system.infra.agents.services", "Service agents"),
        ("system.infra.agents.templates", "Template agents"),
        ("system.infra.agents.web", "Web agents"),
        ("system.infra.agents.web.static", "Static web"),
        ("system.infra.agents.web.templates", "Web templates"),
    ]
    
    for module_name, description in advanced_modules:
        success, message = test_import(module_name, description)
        results.append((success, message))
        if success:
            successes += 1
        else:
            failures += 1
    
    # External dependencies
    print("\n🔧 Testing External Dependencies...")
    external_deps = [
        ("streamlit", "Streamlit web framework"),
        ("pandas", "Pandas data manipulation"),
        ("numpy", "NumPy numerical computing"),
        ("plotly", "Plotly visualization"),
        ("yfinance", "Yahoo Finance data"),
        ("scikit-learn", "Scikit-learn ML"),
        ("torch", "PyTorch ML"),
        ("transformers", "Hugging Face transformers"),
        ("stable_baselines3", "Stable Baselines3 RL"),
        ("gymnasium", "Gymnasium RL environment"),
        ("ccxt", "CCXT crypto trading"),
        ("alpaca_trade_api", "Alpaca trading API"),
        ("speech_recognition", "Speech recognition"),
        ("empyrical", "Empyrical financial metrics"),
        ("scipy", "SciPy scientific computing"),
        ("matplotlib", "Matplotlib plotting"),
        ("seaborn", "Seaborn statistical plotting"),
        ("requests", "HTTP requests"),
        ("aiohttp", "Async HTTP client"),
        ("websockets", "WebSocket client"),
        ("asyncio", "Async I/O"),
        ("logging", "Logging"),
        ("json", "JSON handling"),
        ("yaml", "YAML handling"),
        ("datetime", "Date/time handling"),
        ("typing", "Type hints"),
        ("dataclasses", "Data classes"),
        ("enum", "Enumerations"),
        ("pathlib", "Path handling"),
        ("os", "Operating system"),
        ("sys", "System parameters"),
        ("warnings", "Warning control"),
        ("traceback", "Traceback handling"),
        ("uuid", "UUID generation"),
        ("hashlib", "Hash functions"),
        ("base64", "Base64 encoding"),
        ("pickle", "Object serialization"),
        ("sqlite3", "SQLite database"),
        ("threading", "Threading"),
        ("multiprocessing", "Multiprocessing"),
        ("concurrent.futures", "Concurrent futures"),
        ("queue", "Queue data structure"),
        ("collections", "Collections utilities"),
        ("itertools", "Iterator tools"),
        ("functools", "Function tools"),
        ("operator", "Operator functions"),
        ("math", "Mathematical functions"),
        ("random", "Random number generation"),
        ("statistics", "Statistical functions"),
        ("decimal", "Decimal arithmetic"),
        ("fractions", "Fraction arithmetic"),
        ("time", "Time functions"),
        ("calendar", "Calendar functions"),
        ("locale", "Locale settings"),
        ("re", "Regular expressions"),
        ("string", "String utilities"),
        ("textwrap", "Text wrapping"),
        ("unicodedata", "Unicode data"),
        ("difflib", "Sequence comparison"),
        ("struct", "Binary data structures"),
        ("array", "Array data structure"),
        ("copy", "Object copying"),
        ("pprint", "Pretty printing"),
        ("reprlib", "Alternate repr implementation"),
        ("weakref", "Weak references"),
        ("types", "Dynamic type creation"),
        ("abc", "Abstract base classes"),
        ("inspect", "Inspection utilities"),
        ("ast", "Abstract syntax trees"),
        ("symtable", "Symbol tables"),
        ("code", "Code objects"),
        ("codeop", "Compile code"),
        ("dis", "Disassembler"),
        ("pickletools", "Pickle tools"),
        ("tabnanny", "Tab checker"),
        ("py_compile", "Python compiler"),
        ("compileall", "Compile all Python files"),
        ("pyclbr", "Python class browser"),
        ("keyword", "Python keywords"),
        ("token", "Token constants"),
        ("tokenize", "Tokenizer"),
        ("ast", "Abstract syntax trees"),
        ("symtable", "Symbol tables"),
        ("code", "Code objects"),
        ("codeop", "Compile code"),
        ("dis", "Disassembler"),
        ("pickletools", "Pickle tools"),
        ("tabnanny", "Tab checker"),
        ("py_compile", "Python compiler"),
        ("compileall", "Compile all Python files"),
        ("pyclbr", "Python class browser"),
        ("keyword", "Python keywords"),
        ("token", "Token constants"),
        ("tokenize", "Tokenizer"),
    ]
    
    for module_name, description in external_deps:
        success, message = test_import(module_name, description)
        results.append((success, message))
        if success:
            successes += 1
        else:
            warnings += 1  # External deps are warnings, not failures
    
    # Function tests
    print("\n⚙️ Testing Key Functions...")
    function_tests = [
        ("trading.utils.common", "get_logger", "Logger function"),
        ("trading.config.configuration", "TradingConfig", "Config class"),
        ("trading.memory.agent_memory", "AgentMemory", "Memory class"),
        ("trading.market.market_data", "MarketData", "Market data class"),
        ("trading.evaluation.metrics", "calculate_metrics", "Metrics calculation"),
        ("trading.execution.execution_engine", "ExecutionEngine", "Execution engine"),
        ("trading.llm.agent", "LLMAgent", "LLM agent"),
        ("trading.risk.risk_analyzer", "RiskAnalyzer", "Risk analyzer"),
        ("trading.portfolio.portfolio_manager", "PortfolioManager", "Portfolio manager"),
        ("trading.strategies.gatekeeper", "StrategyGatekeeper", "Strategy gatekeeper"),
        ("trading.models.forecast_router", "ForecastRouter", "Forecast router"),
        ("trading.optimization.bayesian_optimizer", "BayesianOptimizer", "Bayesian optimizer"),
        ("trading.report.report_generator", "ReportGenerator", "Report generator"),
        ("trading.services.base_service", "BaseService", "Base service"),
        ("trading.ui.components", "TradingUI", "Trading UI"),
        ("trading.visualization.plotting", "TradingPlotter", "Trading plotter"),
        ("trading.nlp.nl_interface", "NLInterface", "Natural language interface"),
        ("trading.feature_engineering.feature_engineer", "FeatureEngineer", "Feature engineer"),
        ("trading.backtesting.backtester", "Backtester", "Backtester"),
        ("trading.data.data_loader", "DataLoader", "Data loader"),
        ("trading.analytics.alpha_attribution_engine", "AlphaAttributionEngine", "Alpha attribution"),
        ("trading.knowledge_base.trading_rules", "TradingRules", "Trading rules"),
        ("trading.logs.audit_logger", "AuditLogger", "Audit logger"),
        ("trading.meta_agents.agents.agent_router", "AgentRouter", "Agent router"),
        ("trading.agents.base_agent", "BaseAgent", "Base agent"),
        ("trading.agents.updater.agent", "UpdaterAgent", "Updater agent"),
        ("trading.agents.upgrader.agent", "UpgraderAgent", "Upgrader agent"),
        ("rl.rl_trader", "RLTrader", "RL trader"),
        ("causal.causal_model", "CausalModel", "Causal model"),
        ("causal.driver_analysis", "DriverAnalysis", "Driver analysis"),
        ("agents.model_generator_agent", "ModelGeneratorAgent", "Model generator agent"),
        ("llm.llm_summary", "LLMSummary", "LLM summary"),
        ("risk.advanced_risk", "AdvancedRisk", "Advanced risk"),
        ("risk.tail_risk", "TailRisk", "Tail risk"),
        ("execution.live_trading_interface", "LiveTradingInterface", "Live trading interface"),
        ("market_analysis.market_analyzer", "MarketAnalyzer", "Market analyzer"),
        ("memory.model_monitor", "ModelMonitor", "Model monitor"),
        ("memory.goals.status", "GoalStatus", "Goal status"),
        ("models.forecast_router", "ForecastRouter", "Forecast router"),
        ("optimization.bayesian_optimizer", "BayesianOptimizer", "Bayesian optimizer"),
        ("optimization.genetic_optimizer", "GeneticOptimizer", "Genetic optimizer"),
        ("optimization.strategies.strategy_optimizer", "StrategyOptimizer", "Strategy optimizer"),
        ("optimization.visualization.optimization_visualizer", "OptimizationVisualizer", "Optimization visualizer"),
        ("optimization.utils.consolidator", "Consolidator", "Consolidator"),
        ("strategies.gatekeeper", "StrategyGatekeeper", "Strategy gatekeeper"),
        ("reporting.pnl_attribution", "PnLAttribution", "PnL attribution"),
        ("data.live_feed", "LiveFeed", "Live feed"),
        ("utils.config_loader", "ConfigLoader", "Config loader"),
        ("utils.runner", "Runner", "Runner"),
        ("visualization.plotting", "Plotter", "Plotter"),
        ("tools.encoding_utils", "EncodingUtils", "Encoding utilities"),
        ("core.agent_hub", "AgentHub", "Agent hub"),
        ("core.capability_router", "CapabilityRouter", "Capability router"),
        ("core.agents.base_agent", "BaseAgent", "Base agent"),
        ("core.agents.goal_planner", "GoalPlanner", "Goal planner"),
        ("system.infra.agents.alert_manager", "AlertManager", "Alert manager"),
        ("system.infra.agents.api.metrics_api", "MetricsAPI", "Metrics API"),
        ("system.infra.agents.api.task_api", "TaskAPI", "Task API"),
        ("system.infra.agents.auth.session_manager", "SessionManager", "Session manager"),
        ("system.infra.agents.auth.user_manager", "UserManager", "User manager"),
        ("system.infra.agents.config.config_manager", "ConfigManager", "Config manager"),
        ("system.infra.agents.logs.automation_logging", "AutomationLogging", "Automation logging"),
        ("system.infra.agents.notifications.notification_service", "NotificationService", "Notification service"),
        ("system.infra.agents.scripts.deploy_services", "DeployServices", "Deploy services"),
        ("system.infra.agents.scripts.manage_secrets", "ManageSecrets", "Manage secrets"),
    ]
    
    for module_name, function_name, description in function_tests:
        success, message = test_function_call(module_name, function_name, description)
        results.append((success, message))
        if success:
            successes += 1
        else:
            failures += 1
    
    # Streamlit pages
    print("\n📄 Testing Streamlit Pages...")
    streamlit_pages = [
        ("pages.1_Forecast_Trade", "Forecast Trade page"),
        ("pages.2_Backtest_Strategy", "Backtest Strategy page"),
        ("pages.3_Trade_Execution", "Trade Execution page"),
        ("pages.4_Portfolio_Management", "Portfolio Management page"),
        ("pages.5_Risk_Analysis", "Risk Analysis page"),
        ("pages.6_Model_Optimization", "Model Optimization page"),
        ("pages.7_Market_Analysis", "Market Analysis page"),
        ("pages.8_Agent_Management", "Agent Management page"),
        ("pages.9_System_Monitoring", "System Monitoring page"),
        ("pages.10_Strategy_Health_Dashboard", "Strategy Health Dashboard page"),
        ("streamlit_pages.ModelTrust", "Model Trust page"),
        ("streamlit_pages.WeightDashboard", "Weight Dashboard page"),
    ]
    
    for module_name, description in streamlit_pages:
        success, message = test_import(module_name, description)
        results.append((success, message))
        if success:
            successes += 1
        else:
            failures += 1
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 SYSTEM CHECK RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    success_rate = (successes / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Successes: {successes}")
    print(f"❌ Failures: {failures}")
    print(f"⚠️ Warnings: {warnings}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    print("\n" + "=" * 60)
    print("📋 DETAILED RESULTS")
    print("=" * 60)
    
    for success, message in results:
        print(message)
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    
    if success_rate >= 95:
        print("🎉 EXCELLENT! System is production-ready with minimal issues.")
    elif success_rate >= 85:
        print("✅ GOOD! System is mostly functional with some minor issues.")
    elif success_rate >= 70:
        print("⚠️ FAIR! System has several issues that need attention.")
    else:
        print("❌ POOR! System has significant issues requiring immediate attention.")
    
    if failures > 0:
        print(f"\n🔧 RECOMMENDATIONS:")
        print(f"- Fix {failures} critical failures first")
        if warnings > 0:
            print(f"- Address {warnings} warnings for optimal performance")
        print("- Run integration tests after fixes")
        print("- Consider dependency updates")
    
    print(f"\n⏰ Check completed at: {datetime.now()}")
    
    return success_rate

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 85 else 1) 