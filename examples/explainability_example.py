"""
Explainability and Audit Trail Example

This example demonstrates the comprehensive explainability and audit trail system:
- Complete audit logging of all trading decisions
- Detailed explanations for forecasts and trades
- LLM-powered explanations (when enabled)
- Performance analysis and reporting
- Compliance and transparency features
"""

import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

# Import audit and explainability modules
from reporting.audit_logger import (
    create_audit_logger,
)
from reporting.explainer_agent import (
    create_explainer_agent,
)

# Import utility functions
from utils.common_helpers import safe_json_save


class ExplainabilityExample:
    """
    Comprehensive explainability and audit trail example
    """

    def __init__(self):
        """Initialize the example"""
        # Create audit logger and explainer agent
        self.audit_logger = create_audit_logger()
        self.explainer_agent = create_explainer_agent()

        # Trading parameters
        self.tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
        self.session_id = f"example_session_{int(time.time())}"

        # Results storage
        self.audit_events = []
        self.explanations = []
        self.performance_metrics = {}

        # Example data
        self.market_data = {
            "AAPL": {"price": 150.0, "volume": 1000000, "volatility": 0.02},
            "TSLA": {"price": 200.0, "volume": 800000, "volatility": 0.03},
            "NVDA": {"price": 400.0, "volume": 600000, "volatility": 0.04},
            "MSFT": {"price": 300.0, "volume": 900000, "volatility": 0.025},
            "GOOGL": {"price": 120.0, "volume": 700000, "volatility": 0.022},
        }

    def run_complete_example(self):
        """Run complete explainability and audit trail example"""
        print("=" * 80)
        print("EXPLAINABILITY AND AUDIT TRAIL EXAMPLE")
        print("=" * 80)

        # Run all examples
        self.run_signal_generation_example()
        self.run_model_selection_example()
        self.run_forecast_generation_example()
        self.run_trade_decision_example()
        self.run_risk_management_example()
        self.run_performance_analysis_example()

        # Generate comprehensive reports
        self.generate_comprehensive_reports()

        print("\n" + "=" * 80)
        print("EXAMPLE COMPLETE")
        print("=" * 80)

    def run_signal_generation_example(self):
        """Example of signal generation with audit and explanation"""
        print("\n" + "-" * 60)
        print("SIGNAL GENERATION EXAMPLE")
        print("-" * 60)

        for ticker in self.tickers:
            print(f"\nGenerating signals for {ticker}...")

            # Generate technical signals
            signals = self._generate_technical_signals(ticker)

            # Log signal generation
            signal_event_id = self.audit_logger.log_signal_generated(
                ticker=ticker,
                signal_type="technical_momentum",
                signal_value=signals["momentum_value"],
                signal_strength=signals["signal_strength"],
                features_used=signals["features_used"],
                feature_importance=signals["feature_importance"],
                confidence_score=signals["confidence"],
                metadata={
                    "market_conditions": self.market_data[ticker],
                    "signal_generation_time": datetime.now().isoformat(),
                },
            )

            # Explain feature importance
            explanation_id = self.explainer_agent.explain_feature_importance(
                ticker=ticker,
                features_analyzed=signals["features_used"],
                feature_importance=signals["feature_importance"],
                feature_contributions=signals["feature_contributions"],
                feature_correlations=signals["feature_correlations"],
                feature_trends=signals["feature_trends"],
                market_context=self.market_data[ticker],
            )

            # Store results
            self.audit_events.append(
                {
                    "type": "signal_generated",
                    "ticker": ticker,
                    "event_id": signal_event_id,
                    "explanation_id": explanation_id,
                    "signal_value": signals["momentum_value"],
                    "confidence": signals["confidence"],
                }
            )

            print(
                f"  Signal: {signals['momentum_value']:.3f} (confidence: {signals['confidence']:.1%})"
            )
            print(f"  Top features: {', '.join(signals['features_used'][:3])}")

    def run_model_selection_example(self):
        """Example of model selection with audit and explanation"""
        print("\n" + "-" * 60)
        print("MODEL SELECTION EXAMPLE")
        print("-" * 60)

        for ticker in self.tickers:
            print(f"\nSelecting model for {ticker}...")

            # Simulate model selection process
            models = self._generate_model_candidates(ticker)
            selected_model = self._select_best_model(models)

            # Log model selection
            model_event_id = self.audit_logger.log_model_selected(
                ticker=ticker,
                model_name=selected_model["name"],
                model_version=selected_model["version"],
                model_type=selected_model["type"],
                model_parameters=selected_model["parameters"],
                model_performance=selected_model["performance"],
                selection_criteria=selected_model["selection_criteria"],
                alternative_models=[
                    m["name"] for m in models if m["name"] != selected_model["name"]
                ],
                model_limitations=selected_model["limitations"],
                confidence_score=selected_model["confidence"],
            )

            # Explain model selection
            explanation_id = self.explainer_agent.explain_model_selection(
                ticker=ticker,
                model_name=selected_model["name"],
                model_type=selected_model["type"],
                model_version=selected_model["version"],
                selection_criteria=selected_model["selection_criteria"],
                model_performance=selected_model["performance"],
                alternative_models=[
                    m["name"] for m in models if m["name"] != selected_model["name"]
                ],
                model_limitations=selected_model["limitations"],
                confidence_score=selected_model["confidence"],
            )

            # Store results
            self.audit_events.append(
                {
                    "type": "model_selected",
                    "ticker": ticker,
                    "event_id": model_event_id,
                    "explanation_id": explanation_id,
                    "model_name": selected_model["name"],
                    "performance": selected_model["performance"]["accuracy"],
                }
            )

            print(
                f"  Selected: {selected_model['name']} (accuracy: {selected_model['performance']['accuracy']:.1%})"
            )
            print(f"  Reason: {selected_model['selection_criteria'][0]}")

    def run_forecast_generation_example(self):
        """Example of forecast generation with audit and explanation"""
        print("\n" + "-" * 60)
        print("FORECAST GENERATION EXAMPLE")
        print("-" * 60)

        for ticker in self.tickers:
            print(f"\nGenerating forecast for {ticker}...")

            # Generate forecast
            forecast = self._generate_forecast(ticker)

            # Log forecast generation
            forecast_event_id = self.audit_logger.log_forecast_made(
                ticker=ticker,
                forecast_horizon=forecast["horizon"],
                forecast_value=forecast["value"],
                forecast_confidence=forecast["confidence"],
                model_used=forecast["model_used"],
                features_contributing=forecast["features_contributing"],
                market_conditions=forecast["market_conditions"],
                risk_factors=forecast["risk_factors"],
                assumptions=forecast["assumptions"],
                scenario_analysis=forecast["scenario_analysis"],
            )

            # Explain forecast
            explanation_id = self.explainer_agent.explain_forecast(
                ticker=ticker,
                forecast_value=forecast["value"],
                forecast_horizon=forecast["horizon"],
                forecast_confidence=forecast["confidence"],
                model_used=forecast["model_used"],
                key_factors=forecast["key_factors"],
                market_conditions=forecast["market_conditions"],
                risk_factors=forecast["risk_factors"],
                assumptions=forecast["assumptions"],
                scenario_analysis=forecast["scenario_analysis"],
            )

            # Store results
            self.audit_events.append(
                {
                    "type": "forecast_made",
                    "ticker": ticker,
                    "event_id": forecast_event_id,
                    "explanation_id": explanation_id,
                    "forecast_value": forecast["value"],
                    "confidence": forecast["confidence"],
                }
            )

            print(
                f"  Forecast: {forecast['value']:.2f} (confidence: {forecast['confidence']:.1%})"
            )
            print(f"  Key factors: {', '.join(forecast['key_factors'][:2])}")

    def run_trade_decision_example(self):
        """Example of trade decision with audit and explanation"""
        print("\n" + "-" * 60)
        print("TRADE DECISION EXAMPLE")
        print("-" * 60)

        for ticker in self.tickers:
            print(f"\nMaking trade decision for {ticker}...")

            # Generate trade decision
            trade_decision = self._generate_trade_decision(ticker)

            # Log trade decision
            trade_event_id = self.audit_logger.log_trade_decision(
                ticker=ticker,
                trade_type=trade_decision["type"],
                trade_reason=trade_decision["reason"],
                trade_confidence=trade_decision["confidence"],
                expected_return=trade_decision["expected_return"],
                expected_risk=trade_decision["expected_risk"],
                position_size=trade_decision["position_size"],
                stop_loss=trade_decision["stop_loss"],
                take_profit=trade_decision["take_profit"],
                risk_metrics=trade_decision["risk_metrics"],
            )

            # Explain trade decision
            explanation_id = self.explainer_agent.explain_trade_decision(
                ticker=ticker,
                trade_type=trade_decision["type"],
                trade_reason=trade_decision["reason"],
                expected_return=trade_decision["expected_return"],
                expected_risk=trade_decision["expected_risk"],
                position_size=trade_decision["position_size"],
                entry_price=trade_decision["entry_price"],
                stop_loss=trade_decision["stop_loss"],
                take_profit=trade_decision["take_profit"],
                technical_signals=trade_decision["technical_signals"],
                fundamental_factors=trade_decision["fundamental_factors"],
                market_timing=trade_decision["market_timing"],
                risk_reward_ratio=trade_decision["risk_reward_ratio"],
            )

            # Store results
            self.audit_events.append(
                {
                    "type": "trade_decision",
                    "ticker": ticker,
                    "event_id": trade_event_id,
                    "explanation_id": explanation_id,
                    "trade_type": trade_decision["type"],
                    "expected_return": trade_decision["expected_return"],
                }
            )

            print(f"  Decision: {trade_decision['type'].upper()} {ticker}")
            print(f"  Expected return: {trade_decision['expected_return']:.1%}")
            print(f"  Position size: ${trade_decision['position_size']:,.0f}")

    def run_risk_management_example(self):
        """Example of risk management with audit and explanation"""
        print("\n" + "-" * 60)
        print("RISK MANAGEMENT EXAMPLE")
        print("-" * 60)

        # Portfolio-level risk checks
        risk_checks = [
            {
                "type": "portfolio_concentration",
                "level": "medium",
                "value": 0.25,
                "threshold": 0.3,
                "action": "monitor",
                "mitigation": "Consider diversification",
            },
            {
                "type": "volatility_exposure",
                "level": "low",
                "value": 0.15,
                "threshold": 0.2,
                "action": "acceptable",
                "mitigation": "Continue monitoring",
            },
            {
                "type": "correlation_risk",
                "level": "high",
                "value": 0.8,
                "threshold": 0.7,
                "action": "reduce",
                "mitigation": "Reduce correlated positions",
            },
        ]

        for check in risk_checks:
            print(f"\nRisk check: {check['type']}")

            # Log risk check
            risk_event_id = self.audit_logger.log_risk_check(
                risk_type=check["type"],
                risk_level=check["level"],
                risk_value=check["value"],
                risk_threshold=check["threshold"],
                risk_action=check["action"],
                risk_mitigation=check["mitigation"],
            )

            # noqa: F841 - risk_event_id used for logging but not needed in this example
            print(f"  Level: {check['level']} (value: {check['value']:.2f})")
            print(f"  Action: {check['action']}")
            print(f"  Mitigation: {check['mitigation']}")

    def run_performance_analysis_example(self):
        """Example of performance analysis with audit and explanation"""
        print("\n" + "-" * 60)
        print("PERFORMANCE ANALYSIS EXAMPLE")
        print("-" * 60)

        # Analyze performance metrics
        performance_metrics = {
            "total_trades": len(
                [e for e in self.audit_events if e["type"] == "trade_decision"]
            ),
            "successful_trades": len(
                [
                    e
                    for e in self.audit_events
                    if e["type"] == "trade_decision" and e["expected_return"] > 0
                ]
            ),
            "average_confidence": np.mean(
                [e["confidence"] for e in self.audit_events if "confidence" in e]
            ),
            "total_expected_return": sum(
                [
                    e["expected_return"]
                    for e in self.audit_events
                    if e["type"] == "trade_decision"
                ]
            ),
        }

        print("\nPerformance Summary:")
        print(f"  Total trades: {performance_metrics['total_trades']}")
        print(f"  Successful trades: {performance_metrics['successful_trades']}")
        print(
            f"  Success rate: {performance_metrics['successful_trades'] / performance_metrics['total_trades']:.1%}"
        )
        print(f"  Average confidence: {performance_metrics['average_confidence']:.1%}")
        print(
            f"  Total expected return: {performance_metrics['total_expected_return']:.1%}"
        )

        # Store performance metrics
        self.performance_metrics = performance_metrics

    def generate_comprehensive_reports(self):
        """Generate comprehensive audit and explanation reports"""
        print("\n" + "-" * 60)
        print("GENERATING COMPREHENSIVE REPORTS")
        print("-" * 60)

        # Generate audit summary
        audit_summary = self.audit_logger.get_performance_summary()
        print("\nAudit Summary:")
        print(f"  Total events: {audit_summary['total_events']}")
        print(f"  Session duration: {audit_summary['session_duration']:.1f} seconds")
        print(f"  Events by type: {dict(audit_summary['events_by_type'])}")

        # Generate explanation summary
        explanation_summary = self.explainer_agent.generate_summary_report()
        print("\nExplanation Summary:")
        print(
            f"  Total explanations: {explanation_summary['summary']['total_explanations']}"
        )
        print(f"  Tickers covered: {explanation_summary['summary']['tickers_covered']}")
        print(
            f"  Explanations by type: {explanation_summary['metrics']['explanations_by_type']}"
        )

        # Export reports
        audit_report_path = self.audit_logger.export_session_report()
        explanation_report_path = self.explainer_agent.export_explanations()

        print("\nReports exported:")
        print(f"  Audit report: {audit_report_path}")
        print(f"  Explanation report: {explanation_report_path}")

        # Generate integrated report
        self._generate_integrated_report(audit_summary, explanation_summary)

        # Close audit session
        self.audit_logger.close_session()

    def _generate_integrated_report(
        self, audit_summary: Dict, explanation_summary: Dict
    ):
        """Generate integrated report combining audit and explanations"""
        integrated_report = {
            "session_metadata": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "total_duration": audit_summary["session_duration"],
            },
            "audit_summary": audit_summary,
            "explanation_summary": explanation_summary,
            "performance_metrics": self.performance_metrics,
            "trading_analysis": {
                "tickers_traded": list(
                    set(e["ticker"] for e in self.audit_events if "ticker" in e)
                ),
                "decision_types": list(set(e["type"] for e in self.audit_events)),
                "average_confidence": self.performance_metrics.get(
                    "average_confidence", 0
                ),
                "total_expected_return": self.performance_metrics.get(
                    "total_expected_return", 0
                ),
            },
            "compliance_checklist": {
                "audit_trail_complete": True,
                "explanations_generated": True,
                "risk_checks_performed": True,
                "performance_tracked": True,
                "reports_exported": True,
            },
        }

        # Save integrated report
        report_path = f"logs/integrated_report_{self.session_id}.json"
        safe_json_save(report_path, integrated_report)
        print(f"  Integrated report: {report_path}")

    # Helper methods for generating example data
    def _generate_technical_signals(self, ticker: str) -> Dict[str, Any]:
        """Generate technical signals for a ticker"""
        # noqa: F841 - base_price used for context but not directly referenced
        _ = self.market_data[ticker]["price"]

        # Simulate technical indicators
        rsi = np.random.uniform(30, 70)
        macd = np.random.uniform(-2, 2)
        volume_ratio = np.random.uniform(0.8, 1.2)

        # Calculate momentum value
        momentum_value = (rsi - 50) / 50 + (macd / 2) + (volume_ratio - 1)
        momentum_value = np.clip(momentum_value, -1, 1)

        # Calculate signal strength
        signal_strength = abs(momentum_value)

        # Calculate confidence
        confidence = 0.5 + 0.3 * signal_strength + 0.2 * np.random.random()
        confidence = min(confidence, 0.95)

        return {
            "momentum_value": momentum_value,
            "signal_strength": signal_strength,
            "features_used": ["rsi", "macd", "volume", "price_momentum", "volatility"],
            "feature_importance": {
                "rsi": 0.3,
                "macd": 0.25,
                "volume": 0.2,
                "price_momentum": 0.15,
                "volatility": 0.1,
            },
            "feature_contributions": {
                "rsi": (rsi - 50) / 50 * 0.3,
                "macd": macd / 2 * 0.25,
                "volume": (volume_ratio - 1) * 0.2,
                "price_momentum": np.random.uniform(-0.1, 0.1) * 0.15,
                "volatility": self.market_data[ticker]["volatility"] * 0.1,
            },
            "feature_correlations": {
                "rsi_macd": 0.6,
                "volume_price": 0.4,
                "rsi_volatility": -0.3,
            },
            "feature_trends": {
                "rsi": "increasing" if rsi > 50 else "decreasing",
                "macd": "positive" if macd > 0 else "negative",
                "volume": "above_average" if volume_ratio > 1 else "below_average",
            },
            "confidence": confidence,
        }

    def _generate_model_candidates(self, ticker: str) -> List[Dict[str, Any]]:
        """Generate model candidates for selection"""
        models = [
            {
                "name": "LSTM_Ensemble",
                "type": "Neural Network",
                "version": "v2.1",
                "parameters": {"layers": 3, "units": 64, "dropout": 0.2},
                "performance": {
                    "accuracy": 0.85,
                    "sharpe_ratio": 1.2,
                    "stability": 0.9,
                },
                "selection_criteria": ["accuracy", "stability", "interpretability"],
                "limitations": ["Requires large dataset", "Black box model"],
                "confidence": 0.85,
            },
            {
                "name": "Random_Forest",
                "type": "Ensemble",
                "version": "v1.5",
                "parameters": {"n_estimators": 100, "max_depth": 10},
                "performance": {
                    "accuracy": 0.82,
                    "sharpe_ratio": 1.1,
                    "stability": 0.85,
                },
                "selection_criteria": ["interpretability", "robustness"],
                "limitations": ["Limited to tabular data", "No temporal modeling"],
                "confidence": 0.80,
            },
            {
                "name": "XGBoost",
                "type": "Gradient Boosting",
                "version": "v1.8",
                "parameters": {"n_estimators": 200, "learning_rate": 0.1},
                "performance": {
                    "accuracy": 0.84,
                    "sharpe_ratio": 1.15,
                    "stability": 0.88,
                },
                "selection_criteria": ["accuracy", "speed"],
                "limitations": ["Overfitting risk", "Feature engineering required"],
                "confidence": 0.82,
            },
        ]

        return models

    def _select_best_model(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best model based on performance"""
        # Simple selection based on accuracy
        best_model = max(models, key=lambda x: x["performance"]["accuracy"])
        return best_model

    def _generate_forecast(self, ticker: str) -> Dict[str, Any]:
        """Generate forecast for a ticker"""
        base_price = self.market_data[ticker]["price"]
        volatility = self.market_data[ticker]["volatility"]

        # Generate forecast value
        forecast_change = np.random.normal(0.02, volatility)  # 2% expected return
        forecast_value = base_price * (1 + forecast_change)

        # Calculate confidence based on market conditions
        confidence = 0.6 + 0.3 * np.random.random()

        # Generate scenarios
        scenarios = {
            "bullish": forecast_value * 1.05,
            "bearish": forecast_value * 0.95,
            "neutral": forecast_value,
        }

        return {
            "value": forecast_value,
            "horizon": 5,
            "confidence": confidence,
            "model_used": "LSTM_Ensemble",
            "features_contributing": ["price_momentum", "volume_trend", "sentiment"],
            "key_factors": [
                "Technical momentum",
                "Earnings growth",
                "Market sentiment",
            ],
            "market_conditions": {
                "volatility": volatility,
                "trend": "bullish" if forecast_change > 0 else "bearish",
                "volume": "normal",
            },
            "risk_factors": [
                "Market volatility",
                "Earnings uncertainty",
                "Macro risks",
            ],
            "assumptions": ["Stable market conditions", "No major news events"],
            "scenario_analysis": scenarios,
        }

    def _generate_trade_decision(self, ticker: str) -> Dict[str, Any]:
        """Generate trade decision for a ticker"""
        base_price = self.market_data[ticker]["price"]

        # Determine trade type based on signals
        signals = self._generate_technical_signals(ticker)
        momentum = signals["momentum_value"]

        if momentum > 0.3:
            trade_type = "buy"
            expected_return = 0.05
            position_size = 10000
        elif momentum < -0.3:
            trade_type = "sell"
            expected_return = 0.03
            position_size = 8000
        else:
            trade_type = "hold"
            expected_return = 0.0
            position_size = 0

        # Calculate risk metrics
        expected_risk = abs(expected_return) * 0.4  # 40% of expected return as risk
        risk_reward_ratio = expected_return / expected_risk if expected_risk > 0 else 0

        # Calculate entry/exit prices
        entry_price = base_price
        stop_loss = (
            entry_price * (1 - expected_risk)
            if trade_type == "buy"
            else entry_price * (1 + expected_risk)
        )
        take_profit = (
            entry_price * (1 + expected_return)
            if trade_type == "buy"
            else entry_price * (1 - expected_return)
        )

        return {
            "type": trade_type,
            "reason": f"Strong {trade_type} signal based on technical momentum",
            "confidence": signals["confidence"],
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "position_size": position_size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward_ratio,
            "technical_signals": [
                "RSI momentum",
                "MACD crossover",
                "Volume confirmation",
            ],
            "fundamental_factors": ["Strong earnings", "Market leadership"],
            "market_timing": "Early morning",
            "risk_metrics": {
                "var_95": expected_risk * 1.65,
                "max_drawdown": expected_risk * 2,
                "correlation": 0.3,
            },
        }


def main():
    """Main function to run the explainability example"""
    print("Explainability and Audit Trail Example")
    print("=" * 80)

    # Create example instance
    example = ExplainabilityExample()

    # Run complete example
    example.run_complete_example()

    return example


if __name__ == "__main__":
    main()
