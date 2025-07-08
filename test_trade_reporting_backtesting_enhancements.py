"""
Test Script for Trade Reporting and Backtesting Enhancements

This script tests the enhanced trade reporting and backtesting capabilities:
- Unified trade reporting engine
- Enhanced backtesting with automatic signal integration
- Multiple export formats
- Comprehensive metrics calculation
- Chart generation
- Performance analysis
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_unified_trade_reporter():
    """Test the unified trade reporting engine."""
    logger.info("Testing Unified Trade Reporter...")
    
    try:
        from trading.report.unified_trade_reporter import (
            UnifiedTradeReporter, 
            EnhancedTradeMetrics, 
            EquityCurveData, 
            TradeAnalysis,
            generate_unified_report
        )
        
        # Create sample trade data
        sample_trades = [
            {
                'timestamp': datetime.now() - timedelta(days=30),
                'pnl': 100.0,
                'duration': 3600,
                'type': 'buy',
                'strategy': 'RSI',
                'confidence': 0.8
            },
            {
                'timestamp': datetime.now() - timedelta(days=25),
                'pnl': -50.0,
                'duration': 1800,
                'type': 'sell',
                'strategy': 'MACD',
                'confidence': 0.7
            },
            {
                'timestamp': datetime.now() - timedelta(days=20),
                'pnl': 200.0,
                'duration': 7200,
                'type': 'buy',
                'strategy': 'RSI',
                'confidence': 0.9
            },
            {
                'timestamp': datetime.now() - timedelta(days=15),
                'pnl': 75.0,
                'duration': 5400,
                'type': 'sell',
                'strategy': 'Bollinger',
                'confidence': 0.6
            },
            {
                'timestamp': datetime.now() - timedelta(days=10),
                'pnl': -30.0,
                'duration': 2700,
                'type': 'buy',
                'strategy': 'MACD',
                'confidence': 0.5
            }
        ]
        
        trade_data = {'trades': sample_trades}
        
        # Test unified reporter
        reporter = UnifiedTradeReporter(output_dir="test_reports")
        
        # Generate comprehensive report
        report = reporter.generate_comprehensive_report(
            trade_data=trade_data,
            symbol="AAPL",
            timeframe="1h",
            period="30 days"
        )
        
        logger.info("✓ Unified Trade Reporter test passed")
        logger.info(f"  - Report ID: {report.get('report_id')}")
        logger.info(f"  - Export paths: {list(report.get('export_paths', {}).keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Unified Trade Reporter test failed: {e}")
        return False

def test_enhanced_metrics_calculation():
    """Test enhanced metrics calculation."""
    logger.info("Testing Enhanced Metrics Calculation...")
    
    try:
        from trading.report.unified_trade_reporter import EnhancedTradeMetrics
        
        # Create sample trades with more realistic data
        sample_trades = []
        base_time = datetime.now() - timedelta(days=100)
        
        for i in range(50):
            # Generate varied PnL values
            if i % 3 == 0:  # 33% losing trades
                pnl = -np.random.uniform(10, 100)
            else:  # 67% winning trades
                pnl = np.random.uniform(10, 150)
            
            sample_trades.append({
                'timestamp': base_time + timedelta(days=i),
                'pnl': pnl,
                'duration': np.random.uniform(1800, 7200),
                'type': 'buy' if pnl > 0 else 'sell',
                'strategy': np.random.choice(['RSI', 'MACD', 'Bollinger']),
                'confidence': np.random.uniform(0.5, 0.95)
            })
        
        # Test metrics calculation
        from trading.report.unified_trade_reporter import UnifiedTradeReporter
        reporter = UnifiedTradeReporter()
        
        metrics = reporter._calculate_enhanced_metrics(sample_trades)
        
        # Verify metrics
        assert metrics.total_trades == 50
        assert metrics.winning_trades > 0
        assert metrics.losing_trades > 0
        assert 0 <= metrics.win_rate <= 1
        assert metrics.sharpe_ratio != 0
        assert metrics.max_drawdown >= 0
        
        logger.info("✓ Enhanced Metrics Calculation test passed")
        logger.info(f"  - Total trades: {metrics.total_trades}")
        logger.info(f"  - Win rate: {metrics.win_rate:.2%}")
        logger.info(f"  - Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  - Max drawdown: {metrics.max_drawdown:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced Metrics Calculation test failed: {e}")
        return False

def test_equity_curve_generation():
    """Test equity curve generation."""
    logger.info("Testing Equity Curve Generation...")
    
    try:
        from trading.report.unified_trade_reporter import UnifiedTradeReporter, EquityCurveData
        
        # Create sample trades
        sample_trades = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(20):
            pnl = np.random.uniform(-50, 100)
            sample_trades.append({
                'timestamp': base_time + timedelta(days=i),
                'pnl': pnl,
                'duration': np.random.uniform(1800, 7200),
                'type': 'buy' if pnl > 0 else 'sell',
                'strategy': 'RSI',
                'confidence': 0.8
            })
        
        # Test equity curve generation
        reporter = UnifiedTradeReporter()
        equity_curve = reporter._generate_equity_curve(sample_trades)
        
        # Verify equity curve
        assert len(equity_curve.dates) == 20
        assert len(equity_curve.equity_values) == 20
        assert len(equity_curve.drawdown) == 20
        assert len(equity_curve.running_max) == 20
        
        # Check that equity values are monotonically increasing or decreasing
        assert all(equity_curve.equity_values[i] >= equity_curve.equity_values[i-1] 
                  for i in range(1, len(equity_curve.equity_values)) if equity_curve.equity_values[i-1] > 0)
        
        logger.info("✓ Equity Curve Generation test passed")
        logger.info(f"  - Data points: {len(equity_curve.dates)}")
        logger.info(f"  - Final equity: ${equity_curve.equity_values[-1]:,.2f}")
        logger.info(f"  - Max drawdown: {max(equity_curve.drawdown):.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Equity Curve Generation test failed: {e}")
        return False

def test_chart_generation():
    """Test chart generation capabilities."""
    logger.info("Testing Chart Generation...")
    
    try:
        from trading.report.unified_trade_reporter import (
            UnifiedTradeReporter, 
            TradeAnalysis, 
            EnhancedTradeMetrics, 
            EquityCurveData
        )
        
        # Create sample trade analysis
        sample_trades = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(25):
            pnl = np.random.uniform(-75, 125)
            sample_trades.append({
                'timestamp': base_time + timedelta(days=i),
                'pnl': pnl,
                'duration': np.random.uniform(1800, 7200),
                'type': 'buy' if pnl > 0 else 'sell',
                'strategy': np.random.choice(['RSI', 'MACD', 'Bollinger']),
                'confidence': np.random.uniform(0.5, 0.95)
            })
        
        # Create trade analysis
        reporter = UnifiedTradeReporter(output_dir="test_charts")
        metrics = reporter._calculate_enhanced_metrics(sample_trades)
        equity_curve = reporter._generate_equity_curve(sample_trades)
        risk_metrics = reporter._calculate_risk_metrics(equity_curve)
        performance_attribution = reporter._calculate_performance_attribution(sample_trades, equity_curve)
        
        trade_analysis = TradeAnalysis(
            trade_log=sample_trades,
            equity_curve=equity_curve,
            metrics=metrics,
            risk_metrics=risk_metrics,
            performance_attribution=performance_attribution
        )
        
        # Generate charts
        charts = reporter._generate_charts(trade_analysis, "AAPL")
        
        # Verify charts were generated
        assert len(charts) > 0
        assert any('equity_curve' in chart_name for chart_name in charts.keys())
        
        logger.info("✓ Chart Generation test passed")
        logger.info(f"  - Charts generated: {len(charts)}")
        logger.info(f"  - Chart types: {list(charts.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Chart Generation test failed: {e}")
        return False

def test_enhanced_backtester():
    """Test the enhanced backtester functionality."""
    logger.info("Testing Enhanced Backtester...")
    
    try:
        from trading.backtesting.enhanced_backtester import EnhancedBacktester
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        data = pd.DataFrame({'AAPL': prices}, index=dates)
        
        # Create mock model and strategy classes
        class MockModel:
            def __init__(self, name="MockModel"):
                self.name = name
            
            def forecast(self, data, periods=30):
                return np.random.randn(periods) * 10 + 5
        
        class MockStrategy:
            def __init__(self, name="MockStrategy"):
                self.name = name
            
            def generate_signal(self, signal_data):
                forecast_value = signal_data['forecast_value']
                confidence = signal_data['confidence']
                
                if forecast_value > 0 and confidence > 0.6:
                    return {
                        'timestamp': signal_data['timestamp'],
                        'type': 'BUY',
                        'confidence': confidence,
                        'forecast_value': forecast_value
                    }
                elif forecast_value < 0 and confidence > 0.6:
                    return {
                        'timestamp': signal_data['timestamp'],
                        'type': 'SELL',
                        'confidence': confidence,
                        'forecast_value': forecast_value
                    }
                return None
        
        # Test enhanced backtester
        backtester = EnhancedBacktester(data, output_dir="test_backtest")
        
        model = MockModel("TestModel")
        strategy = MockStrategy("TestStrategy")
        
        # Run forecast backtest
        results = backtester.run_forecast_backtest(
            model=model,
            strategy=strategy,
            symbol="AAPL",
            forecast_period=30,
            confidence_threshold=0.6
        )
        
        # Verify results
        assert 'results' in results
        assert 'report' in results
        assert results['model'] == 'MockModel'
        assert results['strategy'] == 'MockStrategy'
        
        logger.info("✓ Enhanced Backtester test passed")
        logger.info(f"  - Model: {results['model']}")
        logger.info(f"  - Strategy: {results['strategy']}")
        logger.info(f"  - Symbol: {results['symbol']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced Backtester test failed: {e}")
        return False

def test_export_functionality():
    """Test export functionality for different formats."""
    logger.info("Testing Export Functionality...")
    
    try:
        from trading.report.unified_trade_reporter import (
            UnifiedTradeReporter, 
            export_trade_report
        )
        
        # Create sample trade data
        sample_trades = [
            {
                'timestamp': datetime.now() - timedelta(days=i),
                'pnl': np.random.uniform(-50, 100),
                'duration': np.random.uniform(1800, 7200),
                'type': 'buy' if np.random.random() > 0.4 else 'sell',
                'strategy': np.random.choice(['RSI', 'MACD', 'Bollinger']),
                'confidence': np.random.uniform(0.5, 0.95)
            }
            for i in range(20)
        ]
        
        trade_data = {'trades': sample_trades}
        
        # Generate report
        reporter = UnifiedTradeReporter(output_dir="test_exports")
        report = reporter.generate_comprehensive_report(
            trade_data=trade_data,
            symbol="TSLA",
            timeframe="1h",
            period="20 days"
        )
        
        # Test export functionality
        export_paths = reporter._export_report(report, report['report_id'])
        
        # Verify exports
        assert len(export_paths) > 0
        assert 'json' in export_paths
        assert 'html' in export_paths
        
        # Check if files exist
        for format_name, filepath in export_paths.items():
            if format_name != 'pdf':  # PDF generation might not be available
                assert os.path.exists(filepath), f"Export file {filepath} does not exist"
        
        logger.info("✓ Export Functionality test passed")
        logger.info(f"  - Export formats: {list(export_paths.keys())}")
        logger.info(f"  - Files created: {len(export_paths)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Export Functionality test failed: {e}")
        return False

def test_performance_analysis():
    """Test comprehensive performance analysis."""
    logger.info("Testing Performance Analysis...")
    
    try:
        from trading.report.unified_trade_reporter import UnifiedTradeReporter
        
        # Create realistic trade data
        sample_trades = []
        base_time = datetime.now() - timedelta(days=100)
        
        # Generate trades with realistic patterns
        for i in range(100):
            # Create some winning streaks and losing streaks
            if i < 20:  # First 20 trades - mostly winning
                pnl = np.random.uniform(20, 80)
            elif i < 40:  # Next 20 trades - mostly losing
                pnl = np.random.uniform(-60, -10)
            elif i < 70:  # Next 30 trades - mixed
                pnl = np.random.uniform(-40, 60)
            else:  # Last 30 trades - mostly winning
                pnl = np.random.uniform(10, 90)
            
            sample_trades.append({
                'timestamp': base_time + timedelta(days=i),
                'pnl': pnl,
                'duration': np.random.uniform(1800, 7200),
                'type': 'buy' if pnl > 0 else 'sell',
                'strategy': np.random.choice(['RSI', 'MACD', 'Bollinger', 'SMA']),
                'confidence': np.random.uniform(0.5, 0.95)
            })
        
        # Test performance analysis
        reporter = UnifiedTradeReporter()
        
        # Calculate metrics
        metrics = reporter._calculate_enhanced_metrics(sample_trades)
        equity_curve = reporter._generate_equity_curve(sample_trades)
        risk_metrics = reporter._calculate_risk_metrics(equity_curve)
        performance_attribution = reporter._calculate_performance_attribution(sample_trades, equity_curve)
        
        # Verify comprehensive analysis
        assert metrics.total_trades == 100
        assert metrics.winning_trades > 0
        assert metrics.losing_trades > 0
        assert metrics.sharpe_ratio != 0
        assert len(risk_metrics) > 0
        assert len(performance_attribution) > 0
        
        logger.info("✓ Performance Analysis test passed")
        logger.info(f"  - Total trades: {metrics.total_trades}")
        logger.info(f"  - Win rate: {metrics.win_rate:.2%}")
        logger.info(f"  - Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  - Sortino ratio: {metrics.sortino_ratio:.2f}")
        logger.info(f"  - Calmar ratio: {metrics.calmar_ratio:.2f}")
        logger.info(f"  - Risk metrics: {len(risk_metrics)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance Analysis test failed: {e}")
        return False

def main():
    """Run all tests for Trade Reporting and Backtesting Enhancements."""
    logger.info("=" * 60)
    logger.info("TRADE REPORTING AND BACKTESTING ENHANCEMENTS TEST SUITE")
    logger.info("=" * 60)
    
    tests = [
        ("Unified Trade Reporter", test_unified_trade_reporter),
        ("Enhanced Metrics Calculation", test_enhanced_metrics_calculation),
        ("Equity Curve Generation", test_equity_curve_generation),
        ("Chart Generation", test_chart_generation),
        ("Enhanced Backtester", test_enhanced_backtester),
        ("Export Functionality", test_export_functionality),
        ("Performance Analysis", test_performance_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Trade Reporting and Backtesting Enhancements are working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 