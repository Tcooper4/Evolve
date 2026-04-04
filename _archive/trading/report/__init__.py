"""
Reporting Module

This module provides comprehensive reporting capabilities for the trading system:
- Trade performance reporting
- Model performance analysis
- Strategy reasoning and analysis
- Multiple export formats (PDF, HTML, CSV, JSON)
- Unified trade reporting engine
- Enhanced metrics and visualizations
- Integration with external services (Notion, Slack, Email)
"""

from .report_client import ReportClient
from .report_export_engine import ReportExportEngine
from .report_generator import (
    ModelMetrics,
    ReportGenerator,
    StrategyReasoning,
    TradeMetrics,
    generate_trade_report,
)
from .report_service import ReportService
from .unified_trade_reporter import (
    EnhancedTradeMetrics,
    EquityCurveData,
    TradeAnalysis,
    UnifiedTradeReporter,
    export_trade_report,
    generate_unified_report,
)

__all__ = [
    "ReportGenerator",
    "TradeMetrics",
    "ModelMetrics",
    "StrategyReasoning",
    "generate_trade_report",
    "ReportService",
    "ReportClient",
    "ReportExportEngine",
    "UnifiedTradeReporter",
    "EnhancedTradeMetrics",
    "EquityCurveData",
    "TradeAnalysis",
    "generate_unified_report",
    "export_trade_report",
]

__version__ = "2.0.0"
__author__ = "Evolve Trading System"
__description__ = "Comprehensive Reporting Framework with Enhanced Trade Analysis"
