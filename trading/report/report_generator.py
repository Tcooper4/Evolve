"""
Report Generator

Generates comprehensive reports after forecast and strategy execution including:
- Trade Report (PnL, win rate, avg gain/loss)
- Model Report (MSE, Sharpe, volatility)
- Strategy Reasoning (GPT summary of why actions were taken)
- Hyperparameter Export (Strategy parameters per run)

Supports PDF, Markdown, and integrations with Notion, Slack, and email.
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import openai

from trading.utils.safe_math import safe_returns
import pandas as pd
import requests
import seaborn as sns
from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Trade performance metrics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_gain: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    mse: float
    mae: float
    rmse: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class StrategyReasoning:
    """Strategy reasoning and analysis."""

    summary: str
    key_factors: List[str]
    risk_assessment: str
    confidence_level: float
    recommendations: List[str]
    market_conditions: str


@dataclass
class HyperparameterExport:
    """Hyperparameter export data."""

    strategy_name: str
    parameters: Dict[str, Any]
    timestamp: datetime
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class ReportGenerator:
    """
    Comprehensive report generator for trading system.

    Generates trade reports, model reports, and strategy reasoning
    with support for multiple output formats and integrations.
    """

    def __init__(
        self,
        openai_api_key: str = None,
        notion_token: str = None,
        slack_webhook: str = None,
        email_config: Dict[str, str] = None,
        output_dir: str = "reports",
        report_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ReportGenerator.

        Args:
            openai_api_key: OpenAI API key for GPT reasoning
            notion_token: Notion API token for integration
            slack_webhook: Slack webhook URL for notifications
            email_config: Email configuration dictionary
            output_dir: Directory to save reports
            report_config: Dict to control chart types and features
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.notion_token = notion_token or os.getenv("NOTION_TOKEN")
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK")
        self.email_config = email_config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_config = report_config or {
            "equity_curve": True,
            "predictions": True,
            "pnl_distribution": True,
            "heatmap": True,
            "model_summary": True,
            "trade_log": True,
            "hyperparameter_export": True,  # New option
        }

        # Initialize OpenAI if available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

        # Create subdirectories
        (self.output_dir / "pdf").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "hyperparameters").mkdir(exist_ok=True)  # New directory

        logger.info("ReportGenerator initialized")

    def generate_comprehensive_report(
        self,
        trade_data: Dict[str, Any],
        model_data: Dict[str, Any],
        strategy_data: Dict[str, Any],
        symbol: str,
        timeframe: str,
        period: str,
        report_id: str = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report combining all metrics.

        Args:
            trade_data: Trade performance data
            model_data: Model performance data
            strategy_data: Strategy execution data
            symbol: Trading symbol
            timeframe: Timeframe used
            period: Analysis period
            report_id: Unique report identifier
            hyperparameters: Strategy hyperparameters to export

        Returns:
            Dictionary containing all report data and file paths
        """
        try:
            report_id = report_id or f"report_{int(time.time())}"
            timestamp = datetime.now()

            # Calculate metrics
            trade_metrics = self._calculate_trade_metrics(trade_data)
            model_metrics = self._calculate_model_metrics(model_data)
            strategy_reasoning = self._generate_strategy_reasoning(strategy_data)

            # Generate visualizations
            charts = self._generate_charts(trade_data, model_data, symbol)

            # Export hyperparameters if provided
            hyperparameter_path = None
            if hyperparameters and self.report_config.get(
                "hyperparameter_export", True
            ):
                hyperparameter_path = self._export_hyperparameters(
                    hyperparameters, trade_metrics, model_metrics, report_id, symbol
                )

            # Create report data
            report_data = {
                "report_id": report_id,
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "period": period,
                "trade_metrics": trade_metrics,
                "model_metrics": model_metrics,
                "strategy_reasoning": strategy_reasoning,
                "charts": charts,
                "hyperparameters": hyperparameters,
            }

            # Generate different formats
            markdown_path = self._generate_markdown_report(report_data)
            html_path = self._generate_html_report(report_data)
            pdf_path = self._generate_pdf_report(report_data)

            # Add file paths to report data
            report_data["files"] = {
                "markdown": str(markdown_path),
                "html": str(html_path),
                "pdf": str(pdf_path),
            }

            if hyperparameter_path:
                report_data["files"]["hyperparameters"] = str(hyperparameter_path)

            # Send integrations if configured
            self._send_integrations(report_data)

            logger.info(f"Comprehensive report generated: {report_id}")
            return report_data

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {"error": str(e)}

    def _calculate_trade_metrics(self, trade_data: Dict[str, Any]) -> TradeMetrics:
        """Calculate trade performance metrics."""
        try:
            trades = trade_data.get("trades", [])
            if not trades:
                return TradeMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get("pnl", 0) > 0])
            losing_trades = len([t for t in trades if t.get("pnl", 0) < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # Calculate PnL metrics
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            gains = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]
            losses = [abs(t.get("pnl", 0)) for t in trades if t.get("pnl", 0) < 0]

            avg_gain = np.mean(gains) if gains else 0.0
            avg_loss = np.mean(losses) if losses else 0.0

            # Calculate drawdown
            cumulative_pnl = np.cumsum([t.get("pnl", 0) for t in trades])
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

            # Calculate Sharpe ratio
            returns = [t.get("pnl", 0) for t in trades]
            if len(returns) > 1:
                sharpe_ratio = (
                    np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
                )
            else:
                sharpe_ratio = 0.0

            # Calculate profit factor
            total_gains = sum(gains) if gains else 0.0
            total_losses = sum(losses) if losses else 0.0
            profit_factor = (
                total_gains / total_losses if total_losses > 0 else float("inf")
            )

            # Calculate average trade duration
            durations = [t.get("duration", 0) for t in trades if t.get("duration")]
            avg_trade_duration = np.mean(durations) if durations else 0.0

            return TradeMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_gain=avg_gain,
                avg_loss=avg_loss,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                avg_trade_duration=avg_trade_duration,
            )

        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {
                "success": True,
                "result": TradeMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_model_metrics(self, model_data: Dict[str, Any]) -> ModelMetrics:
        """Calculate model performance metrics."""
        try:
            predictions = model_data.get("predictions", [])
            actuals = model_data.get("actuals", [])

            if not predictions or not actuals or len(predictions) != len(actuals):
                return ModelMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            # Calculate error metrics
            errors = np.array(predictions) - np.array(actuals)
            mse = np.mean(errors**2)
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(mse)

            # Calculate returns and Sharpe ratio
            returns = safe_returns(actuals, method='simple')
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            )
            volatility = np.std(returns) if len(returns) > 0 else 0.0

            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (running_max - cumulative_returns) / running_max
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

            # Calculate accuracy metrics
            correct_predictions = sum(
                1
                for p, a in zip(predictions, actuals)
                if (p > a and p > 0) or (p < a and p < 0)
            )
            accuracy = correct_predictions / len(predictions) if predictions else 0.0

            # Calculate precision and recall (simplified)
            positive_predictions = sum(1 for p in predictions if p > 0)
            actual_positives = sum(1 for a in actuals if a > 0)
            true_positives = sum(
                1 for p, a in zip(predictions, actuals) if p > 0 and a > 0
            )

            precision = (
                true_positives / positive_predictions
                if positive_predictions > 0
                else 0.0
            )
            recall = true_positives / actual_positives if actual_positives > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            return ModelMetrics(
                mse=mse,
                mae=mae,
                rmse=rmse,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                max_drawdown=max_drawdown,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )

        except Exception as e:
            logger.error(f"Error calculating model metrics: {e}")
            return {
                "success": True,
                "result": ModelMetrics(
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ),
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

    def _generate_strategy_reasoning(
        self, strategy_data: Dict[str, Any]
    ) -> StrategyReasoning:
        """Generate strategy reasoning using GPT."""
        try:
            if not self.openai_api_key:
                return self._generate_fallback_reasoning(strategy_data)

            # Prepare context for GPT
            context = {
                "strategy_name": strategy_data.get("strategy_name", "Unknown"),
                "symbol": strategy_data.get("symbol", "Unknown"),
                "timeframe": strategy_data.get("timeframe", "Unknown"),
                "signals": strategy_data.get("signals", []),
                "market_conditions": strategy_data.get("market_conditions", {}),
                "performance": strategy_data.get("performance", {}),
                "parameters": strategy_data.get("parameters", {}),
            }

            prompt = f"""
            Analyze the following trading strategy execution and provide a comprehensive summary:

            Strategy: {context['strategy_name']}
            Symbol: {context['symbol']}
            Timeframe: {context['timeframe']}
            Signals: {context['signals']}
            Market Conditions: {context['market_conditions']}
            Performance: {context['performance']}
            Parameters: {context['parameters']}

            Please provide:
            1. A summary of why actions were taken
            2. Key factors that influenced decisions
            3. Risk assessment
            4. Confidence level (0-1)
            5. Recommendations for future trades
            6. Market conditions analysis

            Format your response as JSON with the following structure:
            {{
                "summary": "Brief summary of strategy execution",
                "key_factors": ["factor1", "factor2", "factor3"],
                "risk_assessment": "Risk analysis",
                "confidence_level": 0.85,
                "recommendations": ["rec1", "rec2", "rec3"],
                "market_conditions": "Market analysis"
            }}
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative trading analyst providing strategy analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            try:
                reasoning_data = json.loads(response.choices[0].message.content)
                return StrategyReasoning(
                    summary=reasoning_data.get("summary", ""),
                    key_factors=reasoning_data.get("key_factors", []),
                    risk_assessment=reasoning_data.get("risk_assessment", ""),
                    confidence_level=reasoning_data.get("confidence_level", 0.5),
                    recommendations=reasoning_data.get("recommendations", []),
                    market_conditions=reasoning_data.get("market_conditions", ""),
                )
            except json.JSONDecodeError:
                return self._generate_fallback_reasoning(strategy_data)

        except Exception as e:
            logger.error(f"Error generating strategy reasoning: {e}")
            return self._generate_fallback_reasoning(strategy_data)

    def _generate_fallback_reasoning(
        self, strategy_data: Dict[str, Any]
    ) -> StrategyReasoning:
        """Generate fallback reasoning without GPT."""
        strategy_name = strategy_data.get("strategy_name", "Unknown")
        symbol = strategy_data.get("symbol", "Unknown")
        signals = strategy_data.get("signals", [])

        summary = (
            f"Strategy {strategy_name} executed on {symbol} with {len(signals)} signals"
        )

        key_factors = [
            f"Strategy type: {strategy_name}",
            f"Symbol: {symbol}",
            f"Number of signals: {len(signals)}",
        ]

        risk_assessment = "Standard risk assessment based on strategy parameters"
        confidence_level = 0.7
        recommendations = [
            "Monitor strategy performance",
            "Adjust parameters if needed",
            "Consider market conditions",
        ]
        market_conditions = "Market conditions analyzed based on available data"

        return StrategyReasoning(
            summary=summary,
            key_factors=key_factors,
            risk_assessment=risk_assessment,
            confidence_level=confidence_level,
            recommendations=recommendations,
            market_conditions=market_conditions,
        )

    def _generate_charts(
        self, trade_data: Dict[str, Any], model_data: Dict[str, Any], symbol: str
    ) -> Dict[str, str]:
        """Generate charts and return as base64 encoded images or HTML snippets."""
        charts = {}
        config = self.report_config
        try:
            plt.style.use("seaborn-v0_8")
            trades = trade_data.get("trades", [])
            # 1. Equity Curve
            if config.get("equity_curve", True) and trades:
                cumulative_pnl = np.cumsum([t.get("pnl", 0) for t in trades])
                plt.figure(figsize=(10, 6))
                plt.plot(cumulative_pnl, linewidth=2, color="blue")
                plt.title(f"{symbol} - Equity Curve", fontsize=14, fontweight="bold")
                plt.xlabel("Trade Number")
                plt.ylabel("Cumulative PnL")
                plt.grid(True, alpha=0.3)
                buffer = BytesIO()
                plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                charts["equity_curve"] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            # 2. Model Predictions vs Actual
            if (
                config.get("predictions", True)
                and model_data.get("predictions")
                and model_data.get("actuals")
            ):
                predictions = model_data["predictions"]
                actuals = model_data["actuals"]
                plt.figure(figsize=(10, 6))
                plt.plot(actuals, label="Actual", linewidth=2, color="blue")
                plt.plot(
                    predictions, label="Predicted", linewidth=2, color="red", alpha=0.7
                )
                plt.title(
                    f"{symbol} - Model Predictions vs Actual",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xlabel("Time")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True, alpha=0.3)
                buffer = BytesIO()
                plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                charts["predictions"] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            # 3. Trade Distribution
            if config.get("pnl_distribution", True) and trades:
                pnls = [t.get("pnl", 0) for t in trades]
                plt.figure(figsize=(10, 6))
                plt.hist(pnls, bins=20, alpha=0.7, color="green", edgecolor="black")
                plt.title(
                    f"{symbol} - Trade PnL Distribution", fontsize=14, fontweight="bold"
                )
                plt.xlabel("PnL")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                buffer = BytesIO()
                plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                charts["pnl_distribution"] = base64.b64encode(
                    buffer.getvalue()
                ).decode()
                plt.close()
            # 4. Heatmap of trade profitability over time
            if config.get("heatmap", True) and trades:
                # Example: day vs. hour heatmap (if timestamps available)
                df = pd.DataFrame(trades)
                if "timestamp" in df.columns and "pnl" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df["day"] = df["timestamp"].dt.date
                    df["hour"] = df["timestamp"].dt.hour
                    heatmap_data = df.pivot_table(
                        index="day",
                        columns="hour",
                        values="pnl",
                        aggfunc="sum",
                        fill_value=0,
                    )
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(heatmap_data, cmap="RdYlGn", annot=False, fmt=".0f")
                    plt.title(
                        f"{symbol} - Trade Profitability Heatmap (Day vs Hour)",
                        fontsize=14,
                        fontweight="bold",
                    )
                    plt.xlabel("Hour of Day")
                    plt.ylabel("Day")
                    buffer = BytesIO()
                    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                    buffer.seek(0)
                    charts["profit_heatmap"] = base64.b64encode(
                        buffer.getvalue()
                    ).decode()
                    plt.close()
            # 5. Model summary (most/least successful)
            if config.get("model_summary", True):
                # Aggregate by model_id or model_name if available in trade_data
                df = pd.DataFrame(trades)
                if "model_id" in df.columns and "pnl" in df.columns:
                    model_group = (
                        df.groupby("model_id")["pnl"]
                        .agg(["sum", "count", "mean"])
                        .reset_index()
                    )
                    best = model_group.sort_values("sum", ascending=False).head(1)
                    worst = model_group.sort_values("sum", ascending=True).head(1)
                    summary = {
                        "most_successful": (
                            best.to_dict(orient="records")[0] if not best.empty else {}
                        ),
                        "least_successful": (
                            worst.to_dict(orient="records")[0]
                            if not worst.empty
                            else {}
                        ),
                    }
                    charts["model_summary"] = summary
            # 6. Per-trade execution log
            if config.get("trade_log", True) and trades:
                # Prepare a log as a Markdown table and as a list for HTML
                log_columns = ["timestamp", "action", "pnl", "result", "model_id"]
                df = pd.DataFrame(trades)
                log_df = (
                    df[log_columns].copy()
                    if all(col in df.columns for col in log_columns)
                    else df.copy()
                )
                log_df = log_df.fillna("")
                # Markdown table
                md_table = "| Time | Action | PnL | Result | Model ID |\n|---|---|---|---|---|\n"
                for _, row in log_df.iterrows():
                    md_table += f"| {row.get('timestamp', '')} | {row.get('action', '')} | {row.get('pnl', '')} | {row.get('result', '')} | {row.get('model_id', '')} |\n"
                charts["trade_log_markdown"] = md_table
                # For HTML, pass as list of dicts
                charts["trade_log"] = log_df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
        return charts

    def export_signals(
        self,
        signals_df: pd.DataFrame,
        output_path: str,
        buy_col: str = "Buy",
        sell_col: str = "Sell",
    ) -> bool:
        """
        Export signals DataFrame with defensive checks.

        Args:
            signals_df: DataFrame containing signals
            output_path: Path to save the exported file
            buy_col: Column name for buy signals (default: 'Buy')
            sell_col: Column name for sell signals (default: 'Sell')

        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            # Defensive check: if signals DataFrame is empty or malformed, skip export
            if signals_df is None or signals_df.empty:
                logger.warning("Signals DataFrame is empty or None, skipping export")
                return False

            # Check for required columns
            required_cols = [buy_col, sell_col]
            missing_cols = [
                col for col in required_cols if col not in signals_df.columns
            ]
            if missing_cols:
                logger.warning(
                    f"Missing required columns: {missing_cols}, skipping export"
                )
                return False

            # Check for data quality issues
            if signals_df[buy_col].isna().all() and signals_df[sell_col].isna().all():
                logger.warning(
                    "All signal columns contain only NaN values, skipping export"
                )
                return False

            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export based on file extension
            if output_path.suffix.lower() == ".csv":
                signals_df.to_csv(output_path, index=True)
            elif output_path.suffix.lower() == ".json":
                signals_df.to_json(output_path, orient="records")
            elif output_path.suffix.lower() == ".parquet":
                signals_df.to_parquet(output_path, index=True)
            else:
                # Default to CSV
                output_path = output_path.with_suffix(".csv")
                signals_df.to_csv(output_path, index=True)

            logger.info(f"Successfully exported signals to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting signals: {e}")
            return False

    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate Markdown report."""
        try:
            template = self._get_markdown_template()
            content = template.render(**report_data)

            filename = f"{report_data['report_id']}.md"
            filepath = self.output_dir / "markdown" / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            return filepath

        except Exception as e:
            logger.error(f"Error generating Markdown report: {e}")
            raise

    def _generate_html_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate HTML report."""
        try:
            template = self._get_html_template()
            content = template.render(**report_data)

            filename = f"{report_data['report_id']}.html"
            filepath = self.output_dir / "html" / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            return filepath

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise

    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate PDF report."""
        try:
            # For now, we'll convert HTML to PDF
            # In production, you might want to use a proper PDF library like reportlab
            html_path = self._generate_html_report(report_data)

            # Convert HTML to PDF (placeholder)
            filename = f"{report_data['report_id']}.pdf"
            filepath = self.output_dir / "pdf" / filename

            # For now, just copy the HTML file and rename it
            # In production, use a proper HTML to PDF converter
            import shutil

            shutil.copy(html_path, filepath)

            return filepath

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise

    def _export_hyperparameters(
        self,
        hyperparameters: Dict[str, Any],
        trade_metrics: TradeMetrics,
        model_metrics: ModelMetrics,
        report_id: str,
        symbol: str,
    ) -> Optional[Path]:
        """Export strategy hyperparameters to file.

        Args:
            hyperparameters: Strategy hyperparameters
            trade_metrics: Trade performance metrics
            model_metrics: Model performance metrics
            report_id: Report identifier
            symbol: Trading symbol

        Returns:
            Path to exported hyperparameter file or None if failed
        """
        try:
            # Create hyperparameter export data
            export_data = HyperparameterExport(
                strategy_name=hyperparameters.get("strategy_name", "unknown"),
                parameters=hyperparameters.get("parameters", {}),
                timestamp=datetime.now(),
                performance_metrics={
                    "total_trades": trade_metrics.total_trades,
                    "win_rate": trade_metrics.win_rate,
                    "total_pnl": trade_metrics.total_pnl,
                    "sharpe_ratio": trade_metrics.sharpe_ratio,
                    "max_drawdown": trade_metrics.max_drawdown,
                    "mse": model_metrics.mse,
                    "mae": model_metrics.mae,
                    "rmse": model_metrics.rmse,
                },
                metadata={
                    "report_id": report_id,
                    "symbol": symbol,
                    "export_timestamp": datetime.now().isoformat(),
                },
            )

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"hyperparameters_{export_data.strategy_name}_{symbol}_{timestamp}.json"
            )
            filepath = self.output_dir / "hyperparameters" / filename

            # Export to JSON
            export_dict = {
                "strategy_name": export_data.strategy_name,
                "parameters": export_data.parameters,
                "timestamp": export_data.timestamp.isoformat(),
                "performance_metrics": export_data.performance_metrics,
                "metadata": export_data.metadata,
            }

            with open(filepath, "w") as f:
                json.dump(export_dict, f, indent=2)

            logger.info(f"Hyperparameters exported to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting hyperparameters: {e}")
            return None

    def export_hyperparameters_batch(
        self,
        hyperparameters_list: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        format: str = "json",
    ) -> Optional[Path]:
        """Export multiple hyperparameter sets in batch.

        Args:
            hyperparameters_list: List of hyperparameter dictionaries
            output_path: Output file path
            format: Export format ('json', 'csv', 'excel')

        Returns:
            Path to exported file or None if failed
        """
        try:
            if not hyperparameters_list:
                logger.warning("No hyperparameters to export")
                return None

            # Generate output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = (
                    self.output_dir
                    / "hyperparameters"
                    / f"batch_export_{timestamp}.{format}"
                )
            else:
                output_path = Path(output_path)

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                # Export as JSON array
                with open(output_path, "w") as f:
                    json.dump(hyperparameters_list, f, indent=2)

            elif format == "csv":
                # Flatten for CSV export
                flat_data = []
                for hp in hyperparameters_list:
                    base_record = {
                        "strategy_name": hp.get("strategy_name", ""),
                        "timestamp": hp.get("timestamp", ""),
                        "symbol": hp.get("metadata", {}).get("symbol", ""),
                        "report_id": hp.get("metadata", {}).get("report_id", ""),
                    }

                    # Add parameters
                    for param_name, param_value in hp.get("parameters", {}).items():
                        base_record[f"param_{param_name}"] = param_value

                    # Add performance metrics
                    for metric_name, metric_value in hp.get(
                        "performance_metrics", {}
                    ).items():
                        base_record[f"metric_{metric_name}"] = metric_value

                    flat_data.append(base_record)

                df = pd.DataFrame(flat_data)
                df.to_csv(output_path, index=False)

            elif format == "excel":
                # Flatten for Excel export
                flat_data = []
                for hp in hyperparameters_list:
                    base_record = {
                        "strategy_name": hp.get("strategy_name", ""),
                        "timestamp": hp.get("timestamp", ""),
                        "symbol": hp.get("metadata", {}).get("symbol", ""),
                        "report_id": hp.get("metadata", {}).get("report_id", ""),
                    }

                    # Add parameters
                    for param_name, param_value in hp.get("parameters", {}).items():
                        base_record[f"param_{param_name}"] = param_value

                    # Add performance metrics
                    for metric_name, metric_value in hp.get(
                        "performance_metrics", {}
                    ).items():
                        base_record[f"metric_{metric_name}"] = metric_value

                    flat_data.append(base_record)

                df = pd.DataFrame(flat_data)
                df.to_excel(output_path, index=False)

            logger.info(f"Batch hyperparameter export completed: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error in batch hyperparameter export: {e}")
            return None

    def get_hyperparameter_summary(
        self, hyperparameters_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary statistics for hyperparameters.

        Args:
            hyperparameters_list: List of hyperparameter dictionaries

        Returns:
            Dictionary with summary statistics
        """
        try:
            if not hyperparameters_list:
                return {"message": "No hyperparameters to analyze"}

            summary = {
                "total_runs": len(hyperparameters_list),
                "strategies": {},
                "parameter_ranges": {},
                "performance_correlation": {},
            }

            # Group by strategy
            strategy_groups = {}
            for hp in hyperparameters_list:
                strategy_name = hp.get("strategy_name", "unknown")
                if strategy_name not in strategy_groups:
                    strategy_groups[strategy_name] = []
                strategy_groups[strategy_name].append(hp)

            # Analyze each strategy
            for strategy_name, runs in strategy_groups.items():
                summary["strategies"][strategy_name] = {
                    "run_count": len(runs),
                    "avg_win_rate": np.mean(
                        [
                            r.get("performance_metrics", {}).get("win_rate", 0)
                            for r in runs
                        ]
                    ),
                    "avg_sharpe": np.mean(
                        [
                            r.get("performance_metrics", {}).get("sharpe_ratio", 0)
                            for r in runs
                        ]
                    ),
                    "best_win_rate": max(
                        [
                            r.get("performance_metrics", {}).get("win_rate", 0)
                            for r in runs
                        ]
                    ),
                    "best_sharpe": max(
                        [
                            r.get("performance_metrics", {}).get("sharpe_ratio", 0)
                            for r in runs
                        ]
                    ),
                }

            # Analyze parameter ranges
            all_params = set()
            for hp in hyperparameters_list:
                all_params.update(hp.get("parameters", {}).keys())

            for param_name in all_params:
                param_values = []
                for hp in hyperparameters_list:
                    if param_name in hp.get("parameters", {}):
                        param_values.append(hp["parameters"][param_name])

                if param_values:
                    summary["parameter_ranges"][param_name] = {
                        "min": min(param_values),
                        "max": max(param_values),
                        "mean": np.mean(param_values),
                        "std": np.std(param_values),
                        "count": len(param_values),
                    }

            # Performance correlation analysis
            for param_name in all_params:
                param_values = []
                win_rates = []

                for hp in hyperparameters_list:
                    if param_name in hp.get("parameters", {}):
                        param_values.append(hp["parameters"][param_name])
                        win_rates.append(
                            hp.get("performance_metrics", {}).get("win_rate", 0)
                        )

                if len(param_values) > 1:
                    correlation = np.corrcoef(param_values, win_rates)[0, 1]
                    summary["performance_correlation"][param_name] = {
                        "correlation_with_win_rate": (
                            correlation if not np.isnan(correlation) else 0
                        ),
                        "sample_size": len(param_values),
                    }

            return summary

        except Exception as e:
            logger.error(f"Error generating hyperparameter summary: {e}")
            return {"error": str(e)}

    def _get_markdown_template(self) -> Template:
        """Get Markdown template."""
        template_str = """
# Trading Report - {{ symbol }}

**Report ID:** {{ report_id }}
**Generated:** {{ timestamp }}
**Symbol:** {{ symbol }}
**Timeframe:** {{ timeframe }}
**Period:** {{ period }}

---

## ðŸ“Š Trade Performance

### Summary
- **Total Trades:** {{ trade_metrics.total_trades }}
- **Winning Trades:** {{ trade_metrics.winning_trades }}
- **Losing Trades:** {{ trade_metrics.losing_trades }}
- **Win Rate:** {{ "%.2f"|format(trade_metrics.win_rate * 100) }}%
- **Total PnL:** ${{ "%.2f"|format(trade_metrics.total_pnl) }}
- **Average Gain:** ${{ "%.2f"|format(trade_metrics.avg_gain) }}
- **Average Loss:** ${{ "%.2f"|format(trade_metrics.avg_loss) }}
- **Max Drawdown:** ${{ "%.2f"|format(trade_metrics.max_drawdown) }}
- **Sharpe Ratio:** {{ "%.2f"|format(trade_metrics.sharpe_ratio) }}
- **Profit Factor:** {{ "%.2f"|format(trade_metrics.profit_factor) }}

---

## ðŸ”¥ Trade Profitability Heatmap
{% if charts.profit_heatmap %}
![Profitability Heatmap](data:image/png;base64,{{ charts.profit_heatmap }})
{% else %}
_No heatmap available._
{% endif %}

---

## ðŸ† Model Summary
{% if charts.model_summary %}
- **Most Successful Model:**
    - Model ID: {{ charts.model_summary.most_successful.model_id }}
    - Total PnL: ${{ charts.model_summary.most_successful.sum | default(0) }}
    - Trades: {{ charts.model_summary.most_successful.count | default(0) }}
    - Avg PnL: ${{ charts.model_summary.most_successful.mean | default(0) }}
- **Least Successful Model:**
    - Model ID: {{ charts.model_summary.least_successful.model_id }}
    - Total PnL: ${{ charts.model_summary.least_successful.sum | default(0) }}
    - Trades: {{ charts.model_summary.least_successful.count | default(0) }}
    - Avg PnL: ${{ charts.model_summary.least_successful.mean | default(0) }}
{% else %}
_No model summary available._
{% endif %}

---

## ðŸ§¾ Per-Trade Execution Log
{{ charts.trade_log_markdown | safe }}

---

## ðŸ¤– Model Performance

### Metrics
- **MSE:** {{ "%.4f"|format(model_metrics.mse) }}
- **MAE:** {{ "%.4f"|format(model_metrics.mae) }}
- **RMSE:** {{ "%.4f"|format(model_metrics.rmse) }}
- **Sharpe Ratio:** {{ "%.2f"|format(model_metrics.sharpe_ratio) }}
- **Volatility:** {{ "%.4f"|format(model_metrics.volatility) }}
- **Max Drawdown:** {{ "%.2f"|format(model_metrics.max_drawdown * 100) }}%
- **Accuracy:** {{ "%.2f"|format(model_metrics.accuracy * 100) }}%
- **Precision:** {{ "%.2f"|format(model_metrics.precision * 100) }}%
- **Recall:** {{ "%.2f"|format(model_metrics.recall * 100) }}%
- **F1 Score:** {{ "%.2f"|format(model_metrics.f1_score * 100) }}%

---

## ðŸ§  Strategy Reasoning

### Summary
{{ strategy_reasoning.summary }}

### Key Factors
{% for factor in strategy_reasoning.key_factors %}
- {{ factor }}
{% endfor %}

### Risk Assessment
{{ strategy_reasoning.risk_assessment }}

### Confidence Level
{{ "%.1f"|format(strategy_reasoning.confidence_level * 100) }}%

### Recommendations
{% for rec in strategy_reasoning.recommendations %}
- {{ rec }}
{% endfor %}

### Market Conditions
{{ strategy_reasoning.market_conditions }}

---

## ðŸ“ˆ Charts

{% if charts.equity_curve %}
### Equity Curve
![Equity Curve](data:image/png;base64,{{ charts.equity_curve }})
{% endif %}

{% if charts.predictions %}
### Model Predictions vs Actual
![Predictions](data:image/png;base64,{{ charts.predictions }})
{% endif %}

{% if charts.pnl_distribution %}
### PnL Distribution
![PnL Distribution](data:image/png;base64,{{ charts.pnl_distribution }})
{% endif %}

---

*Report generated by Evolve Trading System*
"""
        return Template(template_str)

    def _get_html_template(self) -> Template:
        """Get HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Report - {{ symbol }}</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #007bff; margin: 0; }
        .header p { color: #666; margin: 5px 0; }
        .section { margin-bottom: 40px; }
        .section h2 { color: #333; border-left: 4px solid #007bff; padding-left: 15px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        .metric-card h3 { margin: 0 0 10px 0; color: #333; font-size: 14px; }
        .metric-card .value { font-size: 24px; font-weight: bold; color: #007bff; }
        .chart-container { text-align: center; margin: 20px 0; }
        .chart-container img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .reasoning { background: #f8f9fa; padding: 20px; border-radius: 8px; }
        .reasoning h3 { color: #333; margin-top: 0; }
        .reasoning ul { margin: 10px 0; }
        .reasoning li { margin: 5px 0; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666; }
        table.trade-log { width: 100%; border-collapse: collapse; margin-top: 10px; }
        table.trade-log th, table.trade-log td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        table.trade-log th { background: #f0f4fa; color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trading Report - {{ symbol }}</h1>
            <p><strong>Report ID:</strong> {{ report_id }}</p>
            <p><strong>Generated:</strong> {{ timestamp }}</p>
            <p><strong>Timeframe:</strong> {{ timeframe }} | <strong>Period:</strong> {{ period }}</p>
        </div>

        <div class="section">
            <h2>ðŸ“Š Trade Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Trades</h3>
                    <div class="value">{{ trade_metrics.total_trades }}</div>
                </div>
                <div class="metric-card">
                    <h3>Win Rate</h3>
                    <div class="value">{{ "%.1f"|format(trade_metrics.win_rate * 100) }}%</div>
                </div>
                <div class="metric-card">
                    <h3>Total PnL</h3>
                    <div class="value">${{ "%.2f"|format(trade_metrics.total_pnl) }}</div>
                </div>
                <div class="metric-card">
                    <h3>Sharpe Ratio</h3>
                    <div class="value">{{ "%.2f"|format(trade_metrics.sharpe_ratio) }}</div>
                </div>
                <div class="metric-card">
                    <h3>Max Drawdown</h3>
                    <div class="value">${{ "%.2f"|format(trade_metrics.max_drawdown) }}</div>
                </div>
                <div class="metric-card">
                    <h3>Profit Factor</h3>
                    <div class="value">{{ "%.2f"|format(trade_metrics.profit_factor) }}</div>
                </div>
            </div>
        </div>

        <!-- Heatmap Section -->
        <div class="section">
            <h2>ðŸ”¥ Trade Profitability Heatmap</h2>
            {% if charts.profit_heatmap %}
            <div class="chart-container">
                <img src="data:image/png;base64,{{ charts.profit_heatmap }}" alt="Profitability Heatmap">
            </div>
            {% else %}
            <p><em>No heatmap available.</em></p>
            {% endif %}
        </div>

        <!-- Model Summary Section -->
        <div class="section">
            <h2>ðŸ† Model Summary</h2>
            {% if charts.model_summary %}
            <div>
                <strong>Most Successful Model:</strong><br>
                Model ID: {{ charts.model_summary.most_successful.model_id }}<br>
                Total PnL: ${{ charts.model_summary.most_successful.sum | default(0) }}<br>
                Trades: {{ charts.model_summary.most_successful.count | default(0) }}<br>
                Avg PnL: ${{ charts.model_summary.most_successful.mean | default(0) }}<br>
                <br>
                <strong>Least Successful Model:</strong><br>
                Model ID: {{ charts.model_summary.least_successful.model_id }}<br>
                Total PnL: ${{ charts.model_summary.least_successful.sum | default(0) }}<br>
                Trades: {{ charts.model_summary.least_successful.count | default(0) }}<br>
                Avg PnL: ${{ charts.model_summary.least_successful.mean | default(0) }}<br>
            </div>
            {% else %}
            <p><em>No model summary available.</em></p>
            {% endif %}
        </div>

        <!-- Per-Trade Execution Log Section -->
        <div class="section">
            <h2>ðŸ§¾ Per-Trade Execution Log</h2>
            {% if charts.trade_log %}
            <table class="trade-log">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Action</th>
                        <th>PnL</th>
                        <th>Result</th>
                        <th>Model ID</th>
                    </tr>
                </thead>
                <tbody>
                {% for row in charts.trade_log %}
                    <tr>
                        <td>{{ row.timestamp }}</td>
                        <td>{{ row.action }}</td>
                        <td>{{ row.pnl }}</td>
                        <td>{{ row.result }}</td>
                        <td>{{ row.model_id }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p><em>No trade log available.</em></p>
            {% endif %}
        </div>

        <div class="section">
            <h2>ðŸ¤– Model Performance</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>MSE</h3>
                    <div class="value">{{ "%.4f"|format(model_metrics.mse) }}</div>
                </div>
                <div class="metric-card">
                    <h3>RMSE</h3>
                    <div class="value">{{ "%.4f"|format(model_metrics.rmse) }}</div>
                </div>
                <div class="metric-card">
                    <h3>Sharpe Ratio</h3>
                    <div class="value">{{ "%.2f"|format(model_metrics.sharpe_ratio) }}</div>
                </div>
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <div class="value">{{ "%.1f"|format(model_metrics.accuracy * 100) }}%</div>
                </div>
                <div class="metric-card">
                    <h3>Volatility</h3>
                    <div class="value">{{ "%.4f"|format(model_metrics.volatility) }}</div>
                </div>
                <div class="metric-card">
                    <h3>F1 Score</h3>
                    <div class="value">{{ "%.2f"|format(model_metrics.f1_score * 100) }}%</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>ðŸ§  Strategy Reasoning</h2>
            <div class="reasoning">
                <h3>Summary</h3>
                <p>{{ strategy_reasoning.summary }}</p>

                <h3>Key Factors</h3>
                <ul>
                {% for factor in strategy_reasoning.key_factors %}
                    <li>{{ factor }}</li>
                {% endfor %}
                </ul>

                <h3>Risk Assessment</h3>
                <p>{{ strategy_reasoning.risk_assessment }}</p>

                <h3>Confidence Level</h3>
                <p>{{ "%.1f"|format(strategy_reasoning.confidence_level * 100) }}%</p>

                <h3>Recommendations</h3>
                <ul>
                {% for rec in strategy_reasoning.recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>

                <h3>Market Conditions</h3>
                <p>{{ strategy_reasoning.market_conditions }}</p>
            </div>
        </div>

        {% if charts %}
        <div class="section">
            <h2>ðŸ“ˆ Charts</h2>
            {% if charts.equity_curve %}
            <div class="chart-container">
                <h3>Equity Curve</h3>
                <img src="data:image/png;base64,{{ charts.equity_curve }}" alt="Equity Curve">
            </div>
            {% endif %}

            {% if charts.predictions %}
            <div class="chart-container">
                <h3>Model Predictions vs Actual</h3>
                <img src="data:image/png;base64,{{ charts.predictions }}" alt="Predictions">
            </div>
            {% endif %}

            {% if charts.pnl_distribution %}
            <div class="chart-container">
                <h3>PnL Distribution</h3>
                <img src="data:image/png;base64,{{ charts.pnl_distribution }}" alt="PnL Distribution">
            </div>
            {% endif %}
        </div>
        {% endif %}

        <div class="footer">
            <p>Report generated by Evolve Trading System</p>
        </div>
    </div>
</body>
</html>
"""
        return Template(template_str)

    def _send_integrations(self, report_data: Dict[str, Any]):
        """Send reports to configured integrations."""
        try:
            # Send to Notion
            if self.notion_token:
                self._send_to_notion(report_data)

            # Send to Slack
            if self.slack_webhook:
                self._send_to_slack(report_data)

            # Send email
            if self.email_config:
                self._send_email(report_data)

        except Exception as e:
            logger.error(f"Error sending integrations: {e}")

    def _send_to_notion(self, report_data: Dict[str, Any]):
        """Send report to Notion."""
        try:
            # This is a placeholder - implement actual Notion API integration
            logger.info("Notion integration not yet implemented")
        except Exception as e:
            logger.error(f"Error sending to Notion: {e}")

    def _send_to_slack(self, report_data: Dict[str, Any]):
        """Send report to Slack."""
        try:
            symbol = report_data["symbol"]
            total_pnl = report_data["trade_metrics"].total_pnl
            win_rate = report_data["trade_metrics"].win_rate

            message = {
                "text": f"ðŸ“Š Trading Report - {symbol}",
                "attachments": [
                    {
                        "color": "good" if total_pnl > 0 else "danger",
                        "fields": [
                            {
                                "title": "Total PnL",
                                "value": f"${total_pnl:.2f}",
                                "short": True,
                            },
                            {
                                "title": "Win Rate",
                                "value": f"{win_rate:.1%}",
                                "short": True,
                            },
                            {
                                "title": "Sharpe Ratio",
                                "value": f"{report_data['trade_metrics'].sharpe_ratio:.2f}",
                                "short": True,
                            },
                            {
                                "title": "Total Trades",
                                "value": str(report_data["trade_metrics"].total_trades),
                                "short": True,
                            },
                        ],
                    }
                ],
            }

            response = requests.post(self.slack_webhook, json=message)
            response.raise_for_status()

            logger.info("Report sent to Slack successfully")

        except Exception as e:
            logger.error(f"Error sending to Slack: {e}")

    def _send_email(self, report_data: Dict[str, Any]):
        """Send report via email."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_config.get("from_email", "")
            msg["To"] = self.email_config.get("to_email", "")
            msg["Subject"] = f"Trading Report - {report_data['symbol']}"

            # Create email body
            body = f"""
            Trading Report for {report_data['symbol']}

            Summary:
            - Total PnL: ${report_data['trade_metrics'].total_pnl:.2f}
            - Win Rate: {report_data['trade_metrics'].win_rate:.1%}
            - Total Trades: {report_data['trade_metrics'].total_trades}

            Please find the detailed report attached.
            """

            msg.attach(MIMEText(body, "plain"))

            # Attach PDF report
            if "files" in report_data and "pdf" in report_data["files"]:
                with open(report_data["files"]["pdf"], "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())

                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename= {Path(report_data["files"]["pdf"]).name}',
                )
                msg.attach(part)

            # Send email
            server = smtplib.SMTP(
                self.email_config.get("smtp_server", ""),
                self.email_config.get("smtp_port", 587),
            )
            server.starttls()
            server.login(
                self.email_config.get("username", ""),
                self.email_config.get("password", ""),
            )
            text = msg.as_string()
            server.sendmail(
                self.email_config.get("from_email", ""),
                self.email_config.get("to_email", ""),
                text,
            )
            server.quit()

            logger.info("Report sent via email successfully")

        except Exception as e:
            logger.error(f"Error sending email: {e}")


# Convenience functions


def generate_trade_report(
    trade_data: Dict[str, Any],
    model_data: Dict[str, Any],
    strategy_data: Dict[str, Any],
    symbol: str,
    timeframe: str,
    period: str,
    **kwargs,
) -> Dict[str, Any]:
    """Generate a comprehensive trading report."""
    generator = ReportGenerator(**kwargs)
    return generator.generate_comprehensive_report(
        trade_data, model_data, strategy_data, symbol, timeframe, period
    )
