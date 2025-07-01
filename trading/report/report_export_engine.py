"""
Report & Export Engine

Auto-generates markdown/PDF reports with strategy logic, performance, backtest graphs, and regime analysis.
Provides comprehensive reporting and export capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced reporting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available, plots will be skipped")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("ReportLab not available, PDF export will be skipped")

logger = logging.getLogger(__name__)

class ReportFormat(Enum):
    """Report export formats."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"
    HTML = "html"

@dataclass
class ReportSection:
    """Report section with content and metadata."""
    title: str
    content: str
    section_type: str
    data: Optional[Dict[str, Any]] = None
    charts: Optional[List[str]] = None

@dataclass
class ReportConfig:
    """Report configuration."""
    title: str
    author: str
    date: datetime
    format: ReportFormat
    include_charts: bool = True
    include_tables: bool = True
    include_metrics: bool = True
    custom_sections: List[str] = None

class ReportExportEngine:
    """Advanced report generation and export engine."""
    
    def __init__(self, 
                 output_dir: str = "reports",
                 template_dir: str = "templates",
                 chart_dir: str = "charts"):
        """Initialize the report export engine.
        
        Args:
            output_dir: Directory for generated reports
            template_dir: Directory for report templates
            chart_dir: Directory for generated charts
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        self.chart_dir = chart_dir
        
        # Create directories with safety guards
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
        
        try:
            os.makedirs(self.template_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create template directory {self.template_dir}: {e}")
        
        try:
            os.makedirs(self.chart_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create chart directory {self.chart_dir}: {e}")
        
        # Initialize components
        self.report_history = []
        self.templates = self._initialize_templates()
        
        logger.info("Report Export Engine initialized successfully")def _initialize_templates(self) -> Dict[str, str]:
        """Initialize report templates."""
        return {
            'executive_summary': """
# Executive Summary

## Key Findings
{key_findings}

## Performance Overview
- **Total Return**: {total_return:.2%}
- **Sharpe Ratio**: {sharpe_ratio:.2f}
- **Max Drawdown**: {max_drawdown:.2%}
- **Win Rate**: {win_rate:.2%}

## Risk Metrics
- **Volatility**: {volatility:.2%}
- **VaR (95%)**: {var_95:.2%}
- **CVaR (95%)**: {cvar_95:.2%}
- **Beta**: {beta:.2f}

## Recommendations
{recommendations}
""",
            'strategy_analysis': """
# Strategy Analysis

## Strategy Overview
{strategy_overview}

## Performance Metrics
{performance_metrics}

## Risk Analysis
{risk_analysis}

## Factor Attribution
{factor_attribution}
""",
            'backtest_results': """
# Backtest Results

## Performance Summary
{performance_summary}

## Equity Curve
{equity_curve_chart}

## Drawdown Analysis
{drawdown_analysis}

## Trade Analysis
{trade_analysis}
""",
            'market_regime': """
# Market Regime Analysis

## Current Regime
{current_regime}

## Regime History
{regime_history}

## Regime Performance
{regime_performance}

## Regime Transitions
{regime_transitions}
""",
            'risk_management': """
# Risk Management

## Position Sizing
{position_sizing}

## Risk Limits
{risk_limits}

## Stress Testing
{stress_testing}

## Risk Metrics
{risk_metrics}
"""
        }
    
    def generate_comprehensive_report(self,
                                    config: ReportConfig,
                                    data: Dict[str, Any]) -> str:
        """Generate comprehensive trading report."""
        try:
            logger.info(f"Generating comprehensive report: {config.title}")
            
            # Generate report sections
            sections = []
            
            # Executive Summary
            if config.include_metrics:
                sections.append(self._generate_executive_summary(data))
            
            # Strategy Analysis
            if 'strategy_data' in data:
                sections.append(self._generate_strategy_analysis(data['strategy_data']))
            
            # Backtest Results
            if 'backtest_data' in data:
                sections.append(self._generate_backtest_results(data['backtest_data']))
            
            # Market Regime Analysis
            if 'regime_data' in data:
                sections.append(self._generate_market_regime_analysis(data['regime_data']))
            
            # Risk Management
            if 'risk_data' in data:
                sections.append(self._generate_risk_management_analysis(data['risk_data']))
            
            # Custom sections
            if config.custom_sections:
                for section_name in config.custom_sections:
                    if section_name in data:
                        sections.append(self._generate_custom_section(section_name, data[section_name]))
            
            # Generate charts if requested
            if config.include_charts and PLOTTING_AVAILABLE:
                self._generate_report_charts(data)
            
            # Export report
            report_path = self._export_report(config, sections, data)
            
            # Store report history
            self._store_report_history(config, report_path)
            
            logger.info(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return ""
    
    def _generate_executive_summary(self, data: Dict[str, Any]) -> ReportSection:
        """Generate executive summary section."""
        try:
            # Extract key metrics
            metrics = data.get('metrics', {})
            
            key_findings = self._format_key_findings(data)
            recommendations = self._format_recommendations(data)
            
            content = self.templates['executive_summary'].format(
                key_findings=key_findings,
                total_return=metrics.get('total_return', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0),
                win_rate=metrics.get('win_rate', 0.0),
                volatility=metrics.get('volatility', 0.0),
                var_95=metrics.get('var_95', 0.0),
                cvar_95=metrics.get('cvar_95', 0.0),
                beta=metrics.get('beta', 1.0),
                recommendations=recommendations
            )
            
            return ReportSection(
                title="Executive Summary",
                content=content,
                section_type="summary",
                data=metrics
            )
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return ReportSection(
                title="Executive Summary",
                content="Error generating executive summary",
                section_type="summary"
            )
    
    def _generate_strategy_analysis(self, strategy_data: Dict[str, Any]) -> ReportSection:
        """Generate strategy analysis section."""
        try:
            strategy_overview = self._format_strategy_overview(strategy_data)
            performance_metrics = self._format_performance_metrics(strategy_data)
            risk_analysis = self._format_risk_analysis(strategy_data)
            factor_attribution = self._format_factor_attribution(strategy_data)
            
            content = self.templates['strategy_analysis'].format(
                strategy_overview=strategy_overview,
                performance_metrics=performance_metrics,
                risk_analysis=risk_analysis,
                factor_attribution=factor_attribution
            )
            
            return ReportSection(
                title="Strategy Analysis",
                content=content,
                section_type="strategy",
                data=strategy_data
            )
            
        except Exception as e:
            logger.error(f"Error generating strategy analysis: {e}")
            return ReportSection(
                title="Strategy Analysis",
                content="Error generating strategy analysis",
                section_type="strategy"
            )
    
    def _generate_backtest_results(self, backtest_data: Dict[str, Any]) -> ReportSection:
        """Generate backtest results section."""
        try:
            performance_summary = self._format_performance_summary(backtest_data)
            equity_curve_chart = self._generate_equity_curve_chart(backtest_data)
            drawdown_analysis = self._format_drawdown_analysis(backtest_data)
            trade_analysis = self._format_trade_analysis(backtest_data)
            
            content = self.templates['backtest_results'].format(
                performance_summary=performance_summary,
                equity_curve_chart=equity_curve_chart,
                drawdown_analysis=drawdown_analysis,
                trade_analysis=trade_analysis
            )
            
            return ReportSection(
                title="Backtest Results",
                content=content,
                section_type="backtest",
                data=backtest_data
            )
            
        except Exception as e:
            logger.error(f"Error generating backtest results: {e}")
            return ReportSection(
                title="Backtest Results",
                content="Error generating backtest results",
                section_type="backtest"
            )
    
    def _generate_market_regime_analysis(self, regime_data: Dict[str, Any]) -> ReportSection:
        """Generate market regime analysis section."""
        try:
            current_regime = self._format_current_regime(regime_data)
            regime_history = self._format_regime_history(regime_data)
            regime_performance = self._format_regime_performance(regime_data)
            regime_transitions = self._format_regime_transitions(regime_data)
            
            content = self.templates['market_regime'].format(
                current_regime=current_regime,
                regime_history=regime_history,
                regime_performance=regime_performance,
                regime_transitions=regime_transitions
            )
            
            return ReportSection(
                title="Market Regime Analysis",
                content=content,
                section_type="regime",
                data=regime_data
            )
            
        except Exception as e:
            logger.error(f"Error generating market regime analysis: {e}")
            return ReportSection(
                title="Market Regime Analysis",
                content="Error generating market regime analysis",
                section_type="regime"
            )
    
    def _generate_risk_management_analysis(self, risk_data: Dict[str, Any]) -> ReportSection:
        """Generate risk management analysis section."""
        try:
            position_sizing = self._format_position_sizing(risk_data)
            risk_limits = self._format_risk_limits(risk_data)
            stress_testing = self._format_stress_testing(risk_data)
            risk_metrics = self._format_risk_metrics(risk_data)
            
            content = self.templates['risk_management'].format(
                position_sizing=position_sizing,
                risk_limits=risk_limits,
                stress_testing=stress_testing,
                risk_metrics=risk_metrics
            )
            
            return ReportSection(
                title="Risk Management",
                content=content,
                section_type="risk",
                data=risk_data
            )
            
        except Exception as e:
            logger.error(f"Error generating risk management analysis: {e}")
            return ReportSection(
                title="Risk Management",
                content="Error generating risk management analysis",
                section_type="risk"
            )
    
    def _generate_custom_section(self, section_name: str, section_data: Dict[str, Any]) -> ReportSection:
        """Generate custom report section."""
        try:
            content = f"# {section_name.title()}\n\n"
            
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    content += f"## {key.title()}\n{value}\n\n"
            else:
                content += str(section_data)
            
            return ReportSection(
                title=section_name.title(),
                content=content,
                section_type="custom",
                data=section_data
            )
            
        except Exception as e:
            logger.error(f"Error generating custom section {section_name}: {e}")
            return ReportSection(
                title=section_name.title(),
                content=f"Error generating {section_name} section",
                section_type="custom"
            )
    
    def _format_key_findings(self, data: Dict[str, Any]) -> str:
        """Format key findings for executive summary."""
        try:
            findings = []
            
            metrics = data.get('metrics', {})
            
            # Performance findings
            total_return = metrics.get('total_return', 0.0)
            if total_return > 0.1:
                findings.append(f"Strong positive performance with {total_return:.1%} total return")
            elif total_return > 0:
                findings.append(f"Moderate positive performance with {total_return:.1%} total return")
            else:
                findings.append(f"Negative performance with {total_return:.1%} total return")
            
            # Risk findings
            sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
            if sharpe_ratio > 1.0:
                findings.append(f"Excellent risk-adjusted returns with Sharpe ratio of {sharpe_ratio:.2f}")
            elif sharpe_ratio > 0.5:
                findings.append(f"Good risk-adjusted returns with Sharpe ratio of {sharpe_ratio:.2f}")
            else:
                findings.append(f"Poor risk-adjusted returns with Sharpe ratio of {sharpe_ratio:.2f}")
            
            # Drawdown findings
            max_drawdown = metrics.get('max_drawdown', 0.0)
            if max_drawdown < -0.2:
                findings.append(f"Significant drawdown of {max_drawdown:.1%} indicates high risk")
            elif max_drawdown < -0.1:
                findings.append(f"Moderate drawdown of {max_drawdown:.1%} within acceptable range")
            else:
                findings.append(f"Low drawdown of {max_drawdown:.1%} indicates good risk management")
            
            return "\n".join([f"- {finding}" for finding in findings])
            
        except Exception as e:
            logger.error(f"Error formatting key findings: {e}")
            return "- Unable to generate key findings"
    
    def _format_recommendations(self, data: Dict[str, Any]) -> str:
        """Format recommendations for executive summary."""
        try:
            recommendations = []
            
            metrics = data.get('metrics', {})
            
            # Performance-based recommendations
            total_return = metrics.get('total_return', 0.0)
            if total_return < 0:
                recommendations.append("Consider strategy optimization or risk reduction")
            
            sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
            if sharpe_ratio < 0.5:
                recommendations.append("Focus on improving risk-adjusted returns")
            
            max_drawdown = metrics.get('max_drawdown', 0.0)
            if max_drawdown < -0.15:
                recommendations.append("Implement stricter risk management controls")
            
            # Market regime recommendations
            regime_data = data.get('regime_data', {})
            if regime_data:
                current_regime = regime_data.get('current_regime', 'unknown')
                if current_regime == 'bear':
                    recommendations.append("Consider defensive positioning in bear market")
                elif current_regime == 'volatile':
                    recommendations.append("Reduce position sizes in volatile market")
            
            if not recommendations:
                recommendations.append("Continue current strategy with monitoring")
            
            return "\n".join([f"- {rec}" for rec in recommendations])
            
        except Exception as e:
            logger.error(f"Error formatting recommendations: {e}")
            return "- Continue monitoring strategy performance"
    
    def _format_strategy_overview(self, strategy_data: Dict[str, Any]) -> str:
        """Format strategy overview."""
        try:
            overview = []
            
            strategy_name = strategy_data.get('name', 'Unknown Strategy')
            overview.append(f"**Strategy Name**: {strategy_name}")
            
            description = strategy_data.get('description', 'No description available')
            overview.append(f"**Description**: {description}")
            
            parameters = strategy_data.get('parameters', {})
            if parameters:
                overview.append("**Parameters**:")
                for param, value in parameters.items():
                    overview.append(f"  - {param}: {value}")
            
            return "\n".join(overview)
            
        except Exception as e:
            logger.error(f"Error formatting strategy overview: {e}")
            return "Unable to format strategy overview"
    
    def _format_performance_metrics(self, strategy_data: Dict[str, Any]) -> str:
        """Format performance metrics."""
        try:
            metrics = strategy_data.get('performance', {})
            
            formatted = []
            formatted.append("| Metric | Value |")
            formatted.append("|--------|-------|")
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    formatted.append(f"| {metric} | {value:.4f} |")
                else:
                    formatted.append(f"| {metric} | {value} |")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting performance metrics: {e}")
            return "Unable to format performance metrics"
    
    def _format_risk_analysis(self, strategy_data: Dict[str, Any]) -> str:
        """Format risk analysis."""
        try:
            risk_data = strategy_data.get('risk', {})
            
            analysis = []
            analysis.append("## Risk Analysis")
            
            for risk_metric, value in risk_data.items():
                analysis.append(f"**{risk_metric}**: {value}")
            
            return "\n".join(analysis)
            
        except Exception as e:
            logger.error(f"Error formatting risk analysis: {e}")
            return "Unable to format risk analysis"
    
    def _format_factor_attribution(self, strategy_data: Dict[str, Any]) -> str:
        """Format factor attribution."""
        try:
            factors = strategy_data.get('factor_attribution', [])
            
            if not factors:
                return "No factor attribution data available"
            
            formatted = []
            formatted.append("| Factor | Contribution | Importance |")
            formatted.append("|--------|--------------|------------|")
            
            for factor in factors:
                name = factor.get('name', 'Unknown')
                contribution = factor.get('contribution', 0.0)
                importance = factor.get('importance', 0.0)
                formatted.append(f"| {name} | {contribution:.4f} | {importance:.4f} |")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting factor attribution: {e}")
            return "Unable to format factor attribution"
    
    def _format_performance_summary(self, backtest_data: Dict[str, Any]) -> str:
        """Format performance summary."""
        try:
            summary = backtest_data.get('summary', {})
            
            formatted = []
            formatted.append("## Performance Summary")
            
            for metric, value in summary.items():
                if isinstance(value, float):
                    formatted.append(f"**{metric}**: {value:.4f}")
                else:
                    formatted.append(f"**{metric}**: {value}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting performance summary: {e}")
            return "Unable to format performance summary"
    
    def _generate_equity_curve_chart(self, backtest_data: Dict[str, Any]) -> str:
        """Generate equity curve chart."""
        try:
            if not PLOTTING_AVAILABLE:
                return "Chart generation not available"
            
            equity_curve = backtest_data.get('equity_curve', pd.Series())
            
            if equity_curve.empty:
                return "No equity curve data available"
            
            # Create chart
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve.index, equity_curve.values)
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.grid(True, alpha=0.3)
            
            # Save chart
            chart_path = os.path.join(self.chart_dir, f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"![Equity Curve]({chart_path})"
            
        except Exception as e:
            logger.error(f"Error generating equity curve chart: {e}")
            return "Error generating equity curve chart"
    
    def _format_drawdown_analysis(self, backtest_data: Dict[str, Any]) -> str:
        """Format drawdown analysis."""
        try:
            drawdown_data = backtest_data.get('drawdown', {})
            
            formatted = []
            formatted.append("## Drawdown Analysis")
            
            for metric, value in drawdown_data.items():
                if isinstance(value, float):
                    formatted.append(f"**{metric}**: {value:.4f}")
                else:
                    formatted.append(f"**{metric}**: {value}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting drawdown analysis: {e}")
            return "Unable to format drawdown analysis"
    
    def _format_trade_analysis(self, backtest_data: Dict[str, Any]) -> str:
        """Format trade analysis."""
        try:
            trade_data = backtest_data.get('trades', {})
            
            formatted = []
            formatted.append("## Trade Analysis")
            
            for metric, value in trade_data.items():
                if isinstance(value, float):
                    formatted.append(f"**{metric}**: {value:.4f}")
                else:
                    formatted.append(f"**{metric}**: {value}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting trade analysis: {e}")
            return "Unable to format trade analysis"
    
    def _format_current_regime(self, regime_data: Dict[str, Any]) -> str:
        """Format current regime information."""
        try:
            current = regime_data.get('current_regime', {})
            
            formatted = []
            formatted.append("## Current Market Regime")
            
            regime_type = current.get('regime_type', 'Unknown')
            confidence = current.get('confidence', 0.0)
            
            formatted.append(f"**Regime**: {regime_type}")
            formatted.append(f"**Confidence**: {confidence:.2f}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting current regime: {e}")
            return "Unable to format current regime"
    
    def _format_regime_history(self, regime_data: Dict[str, Any]) -> str:
        """Format regime history."""
        try:
            history = regime_data.get('history', [])
            
            if not history:
                return "No regime history available"
            
            formatted = []
            formatted.append("## Regime History")
            formatted.append("| Date | Regime | Duration |")
            formatted.append("|------|--------|----------|")
            
            for entry in history[-10:]:  # Last 10 entries
                date = entry.get('date', 'Unknown')
                regime = entry.get('regime', 'Unknown')
                duration = entry.get('duration', 0)
                formatted.append(f"| {date} | {regime} | {duration} |")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting regime history: {e}")
            return "Unable to format regime history"
    
    def _format_regime_performance(self, regime_data: Dict[str, Any]) -> str:
        """Format regime performance."""
        try:
            performance = regime_data.get('performance', {})
            
            formatted = []
            formatted.append("## Regime Performance")
            
            for regime, metrics in performance.items():
                formatted.append(f"### {regime}")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        formatted.append(f"**{metric}**: {value:.4f}")
                    else:
                        formatted.append(f"**{metric}**: {value}")
                formatted.append("")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting regime performance: {e}")
            return "Unable to format regime performance"
    
    def _format_regime_transitions(self, regime_data: Dict[str, Any]) -> str:
        """Format regime transitions."""
        try:
            transitions = regime_data.get('transitions', [])
            
            if not transitions:
                return "No regime transitions available"
            
            formatted = []
            formatted.append("## Regime Transitions")
            formatted.append("| From | To | Date | Probability |")
            formatted.append("|------|----|------|-------------|")
            
            for transition in transitions:
                from_regime = transition.get('from', 'Unknown')
                to_regime = transition.get('to', 'Unknown')
                date = transition.get('date', 'Unknown')
                probability = transition.get('probability', 0.0)
                formatted.append(f"| {from_regime} | {to_regime} | {date} | {probability:.2f} |")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting regime transitions: {e}")
            return "Unable to format regime transitions"
    
    def _format_position_sizing(self, risk_data: Dict[str, Any]) -> str:
        """Format position sizing information."""
        try:
            sizing = risk_data.get('position_sizing', {})
            
            formatted = []
            formatted.append("## Position Sizing")
            
            for metric, value in sizing.items():
                if isinstance(value, float):
                    formatted.append(f"**{metric}**: {value:.4f}")
                else:
                    formatted.append(f"**{metric}**: {value}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting position sizing: {e}")
            return "Unable to format position sizing"
    
    def _format_risk_limits(self, risk_data: Dict[str, Any]) -> str:
        """Format risk limits."""
        try:
            limits = risk_data.get('risk_limits', {})
            
            formatted = []
            formatted.append("## Risk Limits")
            
            for limit, value in limits.items():
                if isinstance(value, float):
                    formatted.append(f"**{limit}**: {value:.4f}")
                else:
                    formatted.append(f"**{limit}**: {value}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting risk limits: {e}")
            return "Unable to format risk limits"
    
    def _format_stress_testing(self, risk_data: Dict[str, Any]) -> str:
        """Format stress testing results."""
        try:
            stress = risk_data.get('stress_testing', {})
            
            formatted = []
            formatted.append("## Stress Testing")
            
            for scenario, results in stress.items():
                formatted.append(f"### {scenario}")
                for metric, value in results.items():
                    if isinstance(value, float):
                        formatted.append(f"**{metric}**: {value:.4f}")
                    else:
                        formatted.append(f"**{metric}**: {value}")
                formatted.append("")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting stress testing: {e}")
            return "Unable to format stress testing"
    
    def _format_risk_metrics(self, risk_data: Dict[str, Any]) -> str:
        """Format risk metrics."""
        try:
            metrics = risk_data.get('risk_metrics', {})
            
            formatted = []
            formatted.append("## Risk Metrics")
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    formatted.append(f"**{metric}**: {value:.4f}")
                else:
                    formatted.append(f"**{metric}**: {value}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error formatting risk metrics: {e}")
            return "Unable to format risk metrics"
    
    def _generate_report_charts(self, data: Dict[str, Any]):
        """Generate charts for the report."""
        try:
            if not PLOTTING_AVAILABLE:

            # Generate various charts based on available data
            if 'backtest_data' in data:
                self._generate_backtest_charts(data['backtest_data'])
            
            if 'regime_data' in data:
                self._generate_regime_charts(data['regime_data'])
            
            if 'risk_data' in data:
                self._generate_risk_charts(data['risk_data'])
            
        except Exception as e:
            logger.error(f"Error generating report charts: {e}")
    
    def _generate_backtest_charts(self, backtest_data: Dict[str, Any]):
        """Generate backtest-specific charts."""
        try:
            # Equity curve
            equity_curve = backtest_data.get('equity_curve', pd.Series())
            if not equity_curve.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(equity_curve.index, equity_curve.values)
                plt.title('Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value')
                plt.grid(True, alpha=0.3)
                
                chart_path = os.path.join(self.chart_dir, f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Drawdown
            drawdown = backtest_data.get('drawdown_series', pd.Series())
            if not drawdown.empty:
                plt.figure(figsize=(12, 6))
                plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
                plt.title('Drawdown')
                plt.xlabel('Date')
                plt.ylabel('Drawdown')
                plt.grid(True, alpha=0.3)
                
                chart_path = os.path.join(self.chart_dir, f"drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"Error generating backtest charts: {e}")

    def _generate_regime_charts(self, regime_data: Dict[str, Any]):
        """Generate regime-specific charts."""
        try:
            # Regime transitions
            transitions = regime_data.get('transitions', [])
            if transitions:
                # Create transition matrix visualization
                pass
            
        except Exception as e:
            logger.error(f"Error generating regime charts: {e}")

    def _generate_risk_charts(self, risk_data: Dict[str, Any]):
        """Generate risk-specific charts."""
        try:
            # Risk metrics over time
            risk_metrics = risk_data.get('risk_metrics_over_time', pd.DataFrame())
            if not risk_metrics.empty:
                plt.figure(figsize=(12, 6))
                risk_metrics.plot()
                plt.title('Risk Metrics Over Time')
                plt.xlabel('Date')
                plt.ylabel('Risk Metric')
                plt.grid(True, alpha=0.3)
                
                chart_path = os.path.join(self.chart_dir, f"risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"Error generating risk charts: {e}")

    def _export_report(self, 
                      config: ReportConfig, 
                      sections: List[ReportSection], 
                      data: Dict[str, Any]) -> str:
        """Export report in specified format."""
        try:
            if config.format == ReportFormat.MARKDOWN:
                return self._export_markdown(config, sections, data)
            elif config.format == ReportFormat.PDF and PDF_AVAILABLE:
                return self._export_pdf(config, sections, data)
            elif config.format == ReportFormat.JSON:
                return self._export_json(config, sections, data)
            elif config.format == ReportFormat.HTML:
                return self._export_html(config, sections, data)
            else:
                logger.warning(f"Format {config.format.value} not supported, falling back to markdown")
                return {'success': True, 'result': self._export_markdown(config, sections, data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return ""
    
    def _export_markdown(self, 
                        config: ReportConfig, 
                        sections: List[ReportSection], 
                        data: Dict[str, Any]) -> str:
        """Export report as markdown."""
        try:
            # Create report header
            content = []
            content.append(f"# {config.title}")
            content.append(f"**Author**: {config.author}")
            content.append(f"**Date**: {config.date.strftime('%Y-%m-%d %H:%M:%S')}")
            content.append("")
            
            # Add sections
            for section in sections:
                content.append(section.content)
                content.append("")
            
            # Write to file
            filename = f"report_{config.date.strftime('%Y%m%d_%H%M%S')}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("\n".join(content))
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting markdown: {e}")
            return ""
    
    def _export_pdf(self, 
                   config: ReportConfig, 
                   sections: List[ReportSection], 
                   data: Dict[str, Any]) -> str:
        """Export report as PDF."""
        try:
            filename = f"report_{config.date.strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30
            )
            story.append(Paragraph(config.title, title_style))
            
            # Author and date
            story.append(Paragraph(f"Author: {config.author}", styles['Normal']))
            story.append(Paragraph(f"Date: {config.date.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Sections
            for section in sections:
                story.append(Paragraph(section.title, styles['Heading2']))
                story.append(Paragraph(section.content, styles['Normal']))
                story.append(Spacer(1, 12))
            
            doc.build(story)
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
            return ""
    
    def _export_json(self, 
                    config: ReportConfig, 
                    sections: List[ReportSection], 
                    data: Dict[str, Any]) -> str:
        """Export report as JSON."""
        try:
            report_data = {
                'title': config.title,
                'author': config.author,
                'date': config.date.isoformat(),
                'format': config.format.value,
                'sections': [
                    {
                        'title': section.title,
                        'content': section.content,
                        'section_type': section.section_type,
                        'data': section.data
                    }
                    for section in sections
                ],
                'data': data
            }
            
            filename = f"report_{config.date.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            return ""
    
    def _export_html(self, 
                    config: ReportConfig, 
                    sections: List[ReportSection], 
                    data: Dict[str, Any]) -> str:
        """Export report as HTML."""
        try:
            content = []
            content.append("<!DOCTYPE html>")
            content.append("<html>")
            content.append("<head>")
            content.append(f"<title>{config.title}</title>")
            content.append("<style>")
            content.append("body { font-family: Arial, sans-serif; margin: 40px; }")
            content.append("h1 { color: #333; }")
            content.append("h2 { color: #666; }")
            content.append("table { border-collapse: collapse; width: 100%; }")
            content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            content.append("th { background-color: #f2f2f2; }")
            content.append("</style>")
            content.append("</head>")
            content.append("<body>")
            
            content.append(f"<h1>{config.title}</h1>")
            content.append(f"<p><strong>Author:</strong> {config.author}</p>")
            content.append(f"<p><strong>Date:</strong> {config.date.strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            for section in sections:
                content.append(f"<h2>{section.title}</h2>")
                content.append(f"<div>{section.content}</div>")
            
            content.append("</body>")
            content.append("</html>")
            
            filename = f"report_{config.date.strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("\n".join(content))
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting HTML: {e}")
            return ""
    
    def _store_report_history(self, config: ReportConfig, filepath: str):
        """Store report in history."""
        try:
            report_record = {
                'title': config.title,
                'author': config.author,
                'date': config.date.isoformat(),
                'format': config.format.value,
                'filepath': filepath,
                'timestamp': datetime.now().isoformat()
            }
            
            self.report_history.append(report_record)
            
            # Keep only last 100 reports
            if len(self.report_history) > 100:
                self.report_history = self.report_history[-100:]
            
        except Exception as e:
            logger.error(f"Error storing report history: {e}")

    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary of report generation."""
        try:
            return {
                'total_reports': len(self.report_history),
                'recent_reports': len([r for r in self.report_history 
                                     if datetime.fromisoformat(r['timestamp']) > 
                                     datetime.now() - timedelta(days=7)]),
                'formats_used': list(set(r['format'] for r in self.report_history)),
                'latest_report': self.report_history[-1] if self.report_history else None
            }
            
        except Exception as e:
            logger.error(f"Error getting report summary: {e}")
            return {'error': str(e)}
    
    def export_report_history(self, filepath: str = "logs/report_history.json"):
        """Export report history to file."""
        try:
            export_data = {
                'report_history': self.report_history,
                'summary': self.get_report_summary(),
                'export_date': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Report history exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting report history: {e}") 
