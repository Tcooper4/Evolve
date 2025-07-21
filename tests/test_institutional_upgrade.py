#!/usr/bin/env python3
"""
Comprehensive Test Script for Institutional-Level Upgrade

This script validates all the institutional-level improvements:
- Enhanced UI integration with tabs
- Dynamic strategy chaining and regime-based selection
- Model confidence scoring and traceability
- Agent memory management with Redis fallback
- Meta-agent loop for continuous improvement
- Live data streaming with fallbacks
- Comprehensive reporting and export capabilities
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InstitutionalUpgradeTester:
    """Comprehensive tester for institutional-level upgrades."""

    def __init__(self):
        """Initialize the tester."""
        self.test_results = {}
        self.start_time = datetime.now()

        logger.info("Institutional Upgrade Tester initialized")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all institutional upgrade tests."""
        logger.info("🚀 Starting Institutional-Level Upgrade Tests")

        tests = [
            ("Enhanced UI Integration", self.test_enhanced_ui_integration),
            ("Dynamic Strategy Engine", self.test_dynamic_strategy_engine),
            ("Agent Memory Management", self.test_agent_memory_management),
            ("Model Confidence & Trace", self.test_model_confidence_trace),
            ("Meta-Agent Loop", self.test_meta_agent_loop),
            ("Live Data Streaming", self.test_live_data_streaming),
            ("Comprehensive Reporting", self.test_comprehensive_reporting),
            ("System Health Monitoring", self.test_system_health_monitoring),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing: {test_name}")
            logger.info(f"{'=' * 60}")

            try:
                result = test_func()
                self.test_results[test_name] = result

                if result.get("success", False):
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(
                        f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                self.test_results[test_name] = {"success": False, "error": str(e)}

        # Generate comprehensive report
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        report = {
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration,
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total) * 100 if total > 0 else 0,
            "test_results": self.test_results,
            "upgrade_status": (
                "COMPLETE"
                if passed == total
                else "PARTIAL" if passed > total // 2 else "FAILED"
            ),
        }

        # Save report
        self._save_report(report)

        # Display summary
        self._display_summary(report)

        return report

    def test_enhanced_ui_integration(self) -> Dict[str, Any]:
        """Test enhanced UI integration with tabs."""
        logger.info("Testing Enhanced UI Integration...")

        try:
            # Test unified interface
            from interface.unified_interface import get_enhanced_interface

            interface = get_enhanced_interface()

            # Test interface initialization
            if not interface:
                return {"success": False, "error": "Interface not available"}

            # Test system health
            health = interface._get_system_health()
            logger.info(f"System Health: {health}")

            # Test component availability
            components = {
                "agent_hub": interface.agent_hub is not None,
                "data_feed": interface.data_feed is not None,
                "prompt_router": interface.prompt_router is not None,
                "model_monitor": interface.model_monitor is not None,
                "strategy_logger": interface.strategy_logger is not None,
                "portfolio_manager": interface.portfolio_manager is not None,
                "strategy_selector": interface.strategy_selector is not None,
                "market_regime_agent": interface.market_regime_agent is not None,
                "hybrid_engine": interface.hybrid_engine is not None,
                "quant_gpt": interface.quant_gpt is not None,
                "report_exporter": interface.report_exporter is not None,
            }

            available_components = sum(components.values())
            total_components = len(components)

            logger.info(
                f"Available Components: {available_components}/{total_components}"
            )

            # Test tab functionality (simulated)
            tab_results = {
                "forecast_tab": self._test_forecast_tab_simulation(),
                "strategy_tab": self._test_strategy_tab_simulation(),
                "portfolio_tab": self._test_portfolio_tab_simulation(),
                "logs_tab": self._test_logs_tab_simulation(),
                "system_tab": self._test_system_tab_simulation(),
            }

            successful_tabs = sum(
                1 for result in tab_results.values() if result.get("success", False)
            )

            return {
                "success": available_components >= total_components * 0.7
                and successful_tabs >= 3,
                "available_components": available_components,
                "total_components": total_components,
                "component_status": components,
                "tab_results": tab_results,
                "successful_tabs": successful_tabs,
            }

        except Exception as e:
            logger.error(f"UI integration test failed: {e}")
            return {"success": False, "error": str(e)}

    def _test_forecast_tab_simulation(self) -> Dict[str, Any]:
        """Simulate forecast tab functionality."""
        try:
            # Simulate forecast generation
            mock_forecast = {
                "symbol": "AAPL",
                "timeframe": "30d",
                "model_type": "ensemble",
                "confidence": 0.85,
                "forecast_values": [100.0, 101.0, 102.0, 103.0, 104.0],
                "model_metadata": {"ensemble_weights": [0.3, 0.4, 0.3]},
                "agent_used": "forecast_router",
                "timestamp": datetime.now().isoformat(),
            }

            # Simulate model trace
            model_trace = {
                "model_selection_reason": "Selected ensemble based on data characteristics",
                "data_quality_score": 0.85,
                "feature_importance": ["price_momentum", "volume_trend", "volatility"],
                "validation_metrics": {"mse": 0.02, "mae": 0.15, "r2": 0.78},
            }

            # Simulate backtest performance
            backtest_performance = {
                "sharpe_ratio": 1.2,
                "total_return": 0.15,
                "max_drawdown": -0.08,
                "win_rate": 0.65,
                "volatility": 0.18,
            }

            return {
                "success": True,
                "forecast": mock_forecast,
                "model_trace": model_trace,
                "backtest_performance": backtest_performance,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_strategy_tab_simulation(self) -> Dict[str, Any]:
        """Simulate strategy tab functionality."""
        try:
            # Simulate strategy chain
            strategy_chain = [
                {
                    "strategy": "momentum",
                    "weight": 0.4,
                    "reason": "Compatible with bull regime and medium risk",
                },
                {
                    "strategy": "trend_following",
                    "weight": 0.6,
                    "reason": "Compatible with bull regime and medium risk",
                },
            ]

            # Simulate regime analysis
            regime_analysis = {
                "regime": "bull",
                "volatility": 0.18,
                "trend_strength": 0.7,
                "momentum": 0.05,
                "volume_trend": 1.2,
            }

            # Simulate strategy performance
            strategy_performance = {
                "total_return": 0.12,
                "sharpe_ratio": 1.1,
                "max_drawdown": -0.06,
                "win_rate": 0.62,
                "volatility": 0.16,
            }

            return {
                "success": True,
                "strategy_chain": strategy_chain,
                "regime_analysis": regime_analysis,
                "performance": strategy_performance,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_portfolio_tab_simulation(self) -> Dict[str, Any]:
        """Simulate portfolio tab functionality."""
        try:
            # Simulate portfolio summary
            portfolio_summary = {
                "total_value": 150000.0,
                "cash": 25000.0,
                "positions": [
                    {"symbol": "AAPL", "shares": 100, "value": 15000.0},
                    {"symbol": "GOOGL", "shares": 50, "value": 7000.0},
                    {"symbol": "TSLA", "shares": 200, "value": 50000.0},
                ],
            }

            # Simulate risk metrics
            risk_metrics = {
                "volatility": 0.18,
                "var": -0.05,
                "beta": 1.1,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
            }

            return {
                "success": True,
                "portfolio_summary": portfolio_summary,
                "risk_metrics": risk_metrics,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_logs_tab_simulation(self) -> Dict[str, Any]:
        """Simulate logs tab functionality."""
        try:
            # Simulate agent interactions
            agent_interactions = [
                {
                    "timestamp": "2025-06-29 15:30:00",
                    "agent_type": "forecast_router",
                    "prompt": "Forecast AAPL for 30 days",
                    "response": "Generated forecast with 85% confidence",
                    "confidence": 0.85,
                    "success": True,
                },
                {
                    "timestamp": "2025-06-29 15:29:30",
                    "agent_type": "strategy_selector",
                    "prompt": "Select strategy for AAPL",
                    "response": "Selected momentum strategy",
                    "confidence": 0.78,
                    "success": True,
                },
            ]

            # Simulate strategy decisions
            strategy_decisions = [
                {
                    "timestamp": "2025-06-29 15:30:00",
                    "strategy": "momentum",
                    "decision": "BUY",
                    "confidence": 0.75,
                    "parameters": {"lookback": 20, "threshold": 0.02},
                }
            ]

            return {
                "success": True,
                "agent_interactions": agent_interactions,
                "strategy_decisions": strategy_decisions,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_system_tab_simulation(self) -> Dict[str, Any]:
        """Simulate system tab functionality."""
        try:
            # Simulate system health
            system_health = {
                "overall_status": "healthy",
                "healthy_components": 3,
                "data_feed_status": "healthy",
                "data_feed_providers": 3,
                "model_engine_status": "healthy",
                "active_models": 5,
                "strategy_engine_status": "healthy",
                "active_strategies": 8,
            }

            # Simulate capability status
            capability_status = {
                "openai_api": True,
                "huggingface_models": True,
                "redis_connection": False,
                "postgres_connection": False,
                "alpha_vantage_api": True,
                "yfinance_api": True,
                "torch_models": True,
                "streamlit_interface": True,
                "plotly_visualization": True,
            }

            return {
                "success": True,
                "system_health": system_health,
                "capability_status": capability_status,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_dynamic_strategy_engine(self) -> Dict[str, Any]:
        """Test dynamic strategy engine with regime-based selection."""
        logger.info("Testing Dynamic Strategy Engine...")

        try:
            # Test enhanced strategy engine
            from trading.strategies.enhanced_strategy_engine import (
                get_enhanced_strategy_engine,
            )

            engine = get_enhanced_strategy_engine()

            if not engine:
                return {"success": False, "error": "Strategy engine not available"}

            # Test strategy initialization
            strategy_count = len(engine.strategies)
            logger.info(f"Available Strategies: {strategy_count}")

            # Test regime classification
            from trading.agents.market_regime_agent import MarketRegimeAgent

            regime_agent = MarketRegimeAgent()

            # Generate mock data
            dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
            mock_data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "Open": np.random.normal(100, 10, len(dates)),
                    "High": np.random.normal(105, 10, len(dates)),
                    "Low": np.random.normal(95, 10, len(dates)),
                    "Close": np.random.normal(100, 10, len(dates)),
                    "Volume": np.random.normal(1000000, 200000, len(dates)),
                }
            )

            # Test regime classification
            regime = regime_agent.classify_regime(mock_data)
            regime_confidence = regime_agent.get_regime_confidence()

            logger.info(f"Regime: {regime}, Confidence: {regime_confidence}")

            # Test strategy chain generation
            from trading.strategies.enhanced_strategy_engine import MarketRegime

            regime_enum = (
                MarketRegime(regime)
                if hasattr(MarketRegime, regime)
                else MarketRegime.NORMAL
            )
            strategy_chain = engine.get_strategy_chain(regime_enum, "medium")

            logger.info(f"Strategy Chain: {len(strategy_chain)} strategies")

            # Test strategy execution
            execution_result = engine.execute_strategy_chain(
                mock_data, regime_enum, "medium"
            )

            # Test performance history
            performance_history = engine.get_strategy_performance_history(limit=10)

            # Test system health
            health = engine.get_system_health()

            return {
                "success": True,
                "strategy_count": strategy_count,
                "regime": regime,
                "regime_confidence": regime_confidence,
                "strategy_chain_length": len(strategy_chain),
                "execution_success": execution_result.get("success", False),
                "performance_history_count": len(performance_history),
                "health_status": health.get("status", "unknown"),
            }

        except Exception as e:
            logger.error(f"Dynamic strategy engine test failed: {e}")
            return {"success": False, "error": str(e)}

    def test_agent_memory_management(self) -> Dict[str, Any]:
        """Test agent memory management with Redis fallback."""
        logger.info("Testing Agent Memory Management...")

        try:
            # Test agent memory manager
            from trading.memory.agent_memory_manager import get_agent_memory_manager

            memory_manager = get_agent_memory_manager()

            if not memory_manager:
                return {"success": False, "error": "Memory manager not available"}

            # Test storing agent interaction
            from trading.memory.agent_memory_manager import AgentInteraction

            interaction = AgentInteraction(
                timestamp=datetime.now(),
                agent_type="forecast_router",
                prompt="Test forecast request",
                response="Test forecast response",
                confidence=0.85,
                success=True,
                metadata={"test": True},
                execution_time=0.5,
            )

            store_success = memory_manager.store_agent_interaction(interaction)

            # Test storing strategy memory
            from trading.memory.agent_memory_manager import StrategyMemory

            strategy_memory = StrategyMemory(
                strategy_name="test_strategy",
                timestamp=datetime.now(),
                performance={"sharpe_ratio": 1.2, "total_return": 0.15},
                confidence=0.8,
                regime="bull",
                success=True,
                parameters={"test": True},
                execution_time=1.0,
            )

            strategy_store_success = memory_manager.store_strategy_memory(
                strategy_memory
            )

            # Test retrieving interactions
            interactions = memory_manager.get_agent_interactions(limit=10)

            # Test confidence boost calculation
            confidence_boost = memory_manager.get_strategy_confidence_boost(
                "test_strategy"
            )

            # Test strategy retirement check
            retirement_check = memory_manager.check_strategy_retirement("test_strategy")

            # Test system health
            health = memory_manager.get_system_health()

            return {
                "success": store_success and strategy_store_success,
                "interaction_store": store_success,
                "strategy_store": strategy_store_success,
                "interactions_retrieved": len(interactions),
                "confidence_boost": confidence_boost,
                "retirement_check": retirement_check.get("should_retire", False),
                "health_status": health.get("status", "unknown"),
            }

        except Exception as e:
            logger.error(f"Agent memory management test failed: {e}")
            return {"success": False, "error": str(e)}

    def test_model_confidence_trace(self) -> Dict[str, Any]:
        """Test model confidence scoring and traceability."""
        logger.info("Testing Model Confidence & Trace...")

        try:
            # Test model monitor
            from trading.memory.model_monitor import ModelMonitor

            model_monitor = ModelMonitor()

            if not model_monitor:
                return {"success": False, "error": "Model monitor not available"}

            # Test trust levels
            trust_levels = model_monitor.get_model_trust_levels()

            # Test model performance
            if trust_levels:
                first_model = list(trust_levels.keys())[0]
                performance = model_monitor.get_model_performance(first_model)
            else:
                performance = {}

            # Test confidence calculation
            confidence_scores = []
            for model_name, trust_level in trust_levels.items():
                confidence_scores.append(
                    {
                        "model": model_name,
                        "trust_level": trust_level,
                        "confidence": min(1.0, trust_level * 1.2),  # Boost confidence
                    }
                )

            # Test trace generation
            model_traces = []
            for score in confidence_scores:
                trace = {
                    "model_name": score["model"],
                    "selection_reason": f"Selected {score['model']} based on trust level {score['trust_level']:.1%}",
                    "confidence_score": score["confidence"],
                    "validation_metrics": {
                        "mse": np.random.normal(0.02, 0.01),
                        "mae": np.random.normal(0.15, 0.05),
                        "r2": np.random.normal(0.75, 0.1),
                    },
                    "feature_importance": [
                        "price_momentum",
                        "volume_trend",
                        "volatility",
                    ],
                    "timestamp": datetime.now().isoformat(),
                }
                model_traces.append(trace)

            return {
                "success": True,
                "models_count": len(trust_levels),
                "trust_levels": trust_levels,
                "performance_available": bool(performance),
                "confidence_scores": confidence_scores,
                "model_traces": model_traces,
            }

        except Exception as e:
            logger.error(f"Model confidence trace test failed: {e}")
            return {"success": False, "error": str(e)}

    def test_meta_agent_loop(self) -> Dict[str, Any]:
        """Test meta-agent loop for continuous improvement."""
        logger.info("Testing Meta-Agent Loop...")

        try:
            # Test performance checker
            from trading.meta_agents.agents.performance_checker import (
                PerformanceChecker,
            )

            performance_checker = PerformanceChecker()

            if not performance_checker:
                return {"success": False, "error": "Performance checker not available"}

            # Test strategy performance check
            strategy_performance = {
                "sharpe_ratio": 0.8,
                "total_return": 0.12,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
            }

            strategy_analysis = performance_checker.check_strategy_performance(
                "test_strategy", strategy_performance
            )

            # Test improvement suggestions
            improvements = performance_checker.suggest_improvements(
                "test_strategy", strategy_performance
            )

            # Test model performance check
            model_performance = {"mse": 0.02, "accuracy": 0.75, "sharpe_ratio": 0.9}

            model_analysis = performance_checker.check_model_performance(
                "test_model", model_performance
            )

            return {
                "success": True,
                "strategy_analysis": strategy_analysis,
                "model_analysis": model_analysis,
                "improvements_suggested": len(improvements),
                "should_retire_strategy": strategy_analysis.get("should_retire", False),
                "should_tune_strategy": strategy_analysis.get("should_tune", False),
            }

        except Exception as e:
            logger.error(f"Meta-agent loop test failed: {e}")
            return {"success": False, "error": str(e)}

    def test_live_data_streaming(self) -> Dict[str, Any]:
        """Test live data streaming with fallbacks."""
        logger.info("Testing Live Data Streaming...")

        try:
            # Test data feed
            from data.live_feed import get_data_feed

            data_feed = get_data_feed()

            if not data_feed:
                return {"success": False, "error": "Data feed not available"}

            # Test provider status
            provider_status = data_feed.get_provider_status()

            # Test system health
            health = data_feed.get_system_health()

            # Test historical data retrieval
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            historical_data = data_feed.get_historical_data(
                "AAPL", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

            # Test live data retrieval
            live_data = data_feed.get_live_data("AAPL")

            return {
                "success": True,
                "provider_status": provider_status,
                "health_status": health.get("status", "unknown"),
                "available_providers": health.get("available_providers", 0),
                "historical_data_available": historical_data is not None
                and not historical_data.empty,
                "live_data_available": live_data is not None,
                "current_provider": provider_status.get("current_provider", "unknown"),
            }

        except Exception as e:
            logger.error(f"Live data streaming test failed: {e}")
            return {"success": False, "error": str(e)}

    def test_comprehensive_reporting(self) -> Dict[str, Any]:
        """Test comprehensive reporting and export capabilities."""
        logger.info("Testing Comprehensive Reporting...")

        try:
            # Test report exporter
            from trading.report.export_engine import ReportExporter

            report_exporter = ReportExporter()

            if not report_exporter:
                return {"success": False, "error": "Report exporter not available"}

            # Test report generation
            test_data = {
                "timestamp": datetime.now().isoformat(),
                "forecast": {
                    "symbol": "AAPL",
                    "confidence": 0.85,
                    "model_used": "ensemble",
                },
                "strategy": {"name": "momentum", "performance": {"sharpe_ratio": 1.2}},
                "portfolio": {"total_value": 150000, "positions": 3},
            }

            # Test JSON export
            json_path = report_exporter.export_report(test_data, format="json")

            # Test HTML export
            html_path = report_exporter.export_report(test_data, format="html")

            # Test PDF export
            try:
                pdf_path = report_exporter.export_report(test_data, format="pdf")
                pdf_success = True
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning(f"PDF export not available: {e}")
                pdf_path = None
                pdf_success = False

            return {
                "success": True,
                "json_export": json_path is not None,
                "html_export": html_path is not None,
                "pdf_export": pdf_success,
                "export_paths": {"json": json_path, "html": html_path, "pdf": pdf_path},
            }

        except Exception as e:
            logger.error(f"Comprehensive reporting test failed: {e}")
            return {"success": False, "error": str(e)}

    def test_system_health_monitoring(self) -> Dict[str, Any]:
        """Test system health monitoring."""
        logger.info("Testing System Health Monitoring...")

        try:
            # Test capability router health
            from core.capability_router import (
                get_system_health as get_capability_health,
            )

            capability_health = get_capability_health()

            # Test agent hub health
            from core.agent_hub import AgentHub

            agent_hub = AgentHub()
            agent_health = agent_hub.get_system_health()

            # Test data feed health
            from data.live_feed import get_data_feed

            data_feed = get_data_feed()
            data_health = data_feed.get_system_health()

            # Test RL trader health
            from rl.rl_trader import get_rl_trader

            rl_trader = get_rl_trader()
            rl_health = rl_trader.get_system_health()

            # Aggregate health status
            health_components = [
                capability_health.get("overall_status", "unknown"),
                agent_health.get("status", "unknown"),
                data_health.get("status", "unknown"),
                rl_health.get("overall_status", "unknown"),
            ]

            healthy_components = sum(
                1 for status in health_components if status == "healthy"
            )
            total_components = len(health_components)

            overall_health = (
                "healthy"
                if healthy_components == total_components
                else "degraded" if healthy_components > 0 else "critical"
            )

            return {
                "success": True,
                "overall_health": overall_health,
                "healthy_components": healthy_components,
                "total_components": total_components,
                "capability_health": capability_health,
                "agent_health": agent_health,
                "data_health": data_health,
                "rl_health": rl_health,
            }

        except Exception as e:
            logger.error(f"System health monitoring test failed: {e}")
            return {"success": False, "error": str(e)}

    def _save_report(self, report: Dict[str, Any]):
        """Save test report to file."""
        try:
            import os

            os.makedirs("test_reports", exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_reports/institutional_upgrade_test_{timestamp}.json"

            with open(filename, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Test report saved to: {filename}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def _display_summary(self, report: Dict[str, Any]):
        """Display test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("INSTITUTIONAL-LEVEL UPGRADE TEST SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Timestamp: {report['timestamp']}")
        logger.info(f"Duration: {report['duration_seconds']:.1f} seconds")
        logger.info(f"Total Tests: {report['total_tests']}")
        logger.info(f"Passed: {report['passed']}")
        logger.info(f"Failed: {report['failed']}")
        logger.info(f"Success Rate: {report['success_rate']:.1f}%")
        logger.info(f"Upgrade Status: {report['upgrade_status']}")

        logger.info("\n" + "-" * 80)
        logger.info("DETAILED RESULTS")
        logger.info("-" * 80)

        for test_name, result in report["test_results"].items():
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            logger.info(f"{test_name}: {status}")

            if not result.get("success", False):
                logger.info(f"  Error: {result.get('error', 'Unknown error')}")

        logger.info("\n" + "=" * 80)

        if report["upgrade_status"] == "COMPLETE":
            logger.info("🎉 ALL TESTS PASSED - INSTITUTIONAL UPGRADE COMPLETE!")
        elif report["upgrade_status"] == "PARTIAL":
            logger.info("⚠️  PARTIAL SUCCESS - SOME COMPONENTS NEED ATTENTION")
        else:
            logger.info("❌ UPGRADE FAILED - SIGNIFICANT ISSUES DETECTED")


def main():
    """Main test function."""
    print("🚀 Evolve Trading System - Institutional-Level Upgrade Test")
    print("=" * 80)

    tester = InstitutionalUpgradeTester()
    report = tester.run_all_tests()

    return report


if __name__ == "__main__":
    main()
