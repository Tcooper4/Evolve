"""
Report Generation Demo

Demonstrates how to use the report generation system with sample data.
"""

import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from report.report_generator import ReportGenerator
from report.report_client import ReportClient, generate_quick_report

def create_sample_trade_data():
    """Create sample trade data."""
    trades = []
    
    # Generate 20 sample trades
    for i in range(20):
        # Random PnL between -200 and 300
        pnl = np.random.normal(50, 100)
        
        # Random duration between 30 minutes and 4 hours
        duration = np.random.randint(1800, 14400)
        
        # Entry time (spread over 7 days)
        entry_time = datetime.now() - timedelta(days=np.random.randint(0, 7), 
                                               hours=np.random.randint(0, 24))
        
        trade = {
            'pnl': round(pnl, 2),
            'duration': duration,
            'entry_time': entry_time.isoformat(),
            'exit_time': (entry_time + timedelta(seconds=duration)).isoformat(),
            'symbol': 'AAPL',
            'side': 'BUY' if pnl > 0 else 'SELL',
            'quantity': np.random.randint(1, 10),
            'entry_price': 150 + np.random.normal(0, 5),
            'exit_price': 150 + np.random.normal(0, 5)
        }
        trades.append(trade)
    
    return {'trades': trades}

def create_sample_model_data():
    """Create sample model data."""
    # Generate 100 data points
    n_points = 100
    
    # Generate actual prices (random walk)
    actuals = [100]
    for i in range(1, n_points):
        change = np.random.normal(0, 1)
        actuals.append(actuals[-1] + change)
    
    # Generate predictions (actuals with some noise)
    predictions = []
    for actual in actuals:
        noise = np.random.normal(0, 0.5)
        predictions.append(actual + noise)
    
    return {
        'predictions': [round(p, 2) for p in predictions],
        'actuals': [round(a, 2) for a in actuals],
        'model_name': 'LSTM Neural Network',
        'model_params': {
            'layers': 3,
            'units': 50,
            'dropout': 0.2,
            'learning_rate': 0.001
        },
        'training_metrics': {
            'loss': 0.0234,
            'accuracy': 0.876,
            'val_loss': 0.0289,
            'val_accuracy': 0.854
        }
    }

def create_sample_strategy_data():
    """Create sample strategy data."""
    return {
        'strategy_name': 'RSI + MACD Strategy',
        'symbol': 'AAPL',
        'timeframe': '1h',
        'signals': [
            {'timestamp': '2024-01-01T10:00:00', 'action': 'BUY', 'reason': 'RSI oversold + MACD bullish crossover'},
            {'timestamp': '2024-01-01T12:00:00', 'action': 'SELL', 'reason': 'RSI overbought'},
            {'timestamp': '2024-01-01T15:00:00', 'action': 'BUY', 'reason': 'MACD bullish crossover'},
            {'timestamp': '2024-01-01T18:00:00', 'action': 'HOLD', 'reason': 'No clear signal'},
            {'timestamp': '2024-01-01T21:00:00', 'action': 'BUY', 'reason': 'RSI oversold'}
        ],
        'market_conditions': {
            'trend': 'bullish',
            'volatility': 'medium',
            'volume': 'high',
            'support_level': 145.50,
            'resistance_level': 155.75
        },
        'performance': {
            'total_return': 0.15,
            'max_drawdown': 0.05,
            'sharpe_ratio': 1.8,
            'win_rate': 0.65
        },
        'parameters': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
    }

def demo_basic_report():
    """Demonstrate basic report generation."""
    print("=== Basic Report Generation Demo ===")
    
    # Create sample data
    trade_data = create_sample_trade_data()
    model_data = create_sample_model_data()
    strategy_data = create_sample_strategy_data()
    
    # Generate report
    report_data = generate_quick_report(
        trade_data=trade_data,
        model_data=model_data,
        strategy_data=strategy_data,
        symbol='AAPL',
        timeframe='1h',
        period='7d'
    )
    
    print(f"✅ Report generated successfully!")
    print(f"📄 Report ID: {report_data['report_id']}")
    print(f"📊 Trade Metrics:")
    print(f"   - Total Trades: {report_data['trade_metrics'].total_trades}")
    print(f"   - Win Rate: {report_data['trade_metrics'].win_rate:.1%}")
    print(f"   - Total PnL: ${report_data['trade_metrics'].total_pnl:.2f}")
    print(f"   - Sharpe Ratio: {report_data['trade_metrics'].sharpe_ratio:.2f}")
    
    print(f"🤖 Model Metrics:")
    print(f"   - MSE: {report_data['model_metrics'].mse:.4f}")
    print(f"   - Accuracy: {report_data['model_metrics'].accuracy:.1%}")
    print(f"   - Sharpe Ratio: {report_data['model_metrics'].sharpe_ratio:.2f}")
    
    print(f"📁 Files created:")
    for format_type, file_path in report_data['files'].items():
        print(f"   - {format_type.upper()}: {file_path}")
    
    return report_data

def demo_service_integration():
    """Demonstrate service integration."""
    print("\n=== Service Integration Demo ===")
    
    try:
        # Initialize client
        client = ReportClient()
        
        # Check service status
        status = client.get_service_status()
        print(f"🔍 Service Status: {status}")
        
        if status['running']:
            print("✅ Report service is running")
            
            # Create sample data
            trade_data = create_sample_trade_data()
            model_data = create_sample_model_data()
            strategy_data = create_sample_strategy_data()
            
            # Trigger strategy report
            event_id = client.trigger_strategy_report(
                strategy_data=strategy_data,
                trade_data=trade_data,
                model_data=model_data,
                symbol='AAPL',
                timeframe='1h',
                period='7d'
            )
            
            print(f"🚀 Triggered strategy report: {event_id}")
            
            # Wait for completion
            print("⏳ Waiting for report completion...")
            report_result = client.wait_for_report(event_id, timeout=30)
            
            if report_result:
                print(f"✅ Report completed: {report_result['report_id']}")
                print(f"📊 Summary: {report_result['summary']}")
            else:
                print("❌ Report timeout or service not responding")
        else:
            print("⚠️  Report service is not running")
            print("   Start the service with: python launch_report_service.py")
            
    except Exception as e:
        print(f"❌ Service integration error: {e}")

def demo_custom_report():
    """Demonstrate custom report generation with different configurations."""
    print("\n=== Custom Report Demo ===")
    
    # Initialize report generator with custom config
    generator = ReportGenerator(
        output_dir='custom_reports',
        openai_api_key=None  # Disable GPT for demo
    )
    
    # Create sample data
    trade_data = create_sample_trade_data()
    model_data = create_sample_model_data()
    strategy_data = create_sample_strategy_data()
    
    # Generate custom report
    report_data = generator.generate_comprehensive_report(
        trade_data=trade_data,
        model_data=model_data,
        strategy_data=strategy_data,
        symbol='TSLA',  # Different symbol
        timeframe='4h',  # Different timeframe
        period='30d',    # Different period
        report_id='custom_demo_report'
    )
    
    print(f"✅ Custom report generated!")
    print(f"📄 Report ID: {report_data['report_id']}")
    print(f"📁 Files: {list(report_data['files'].keys())}")

def demo_report_management():
    """Demonstrate report management features."""
    print("\n=== Report Management Demo ===")
    
    try:
        client = ReportClient()
        
        # List available reports
        reports = client.list_available_reports()
        print(f"📋 Available reports: {len(reports)}")
        
        for report in reports[:5]:  # Show first 5
            print(f"   - {report['report_id']} ({report['format']}) - {report['created']}")
        
        if reports:
            # Get files for first report
            first_report = reports[0]
            files = client.get_report_files(first_report['report_id'])
            print(f"📁 Files for {first_report['report_id']}: {list(files.keys())}")
            
            # Get recent reports
            recent = client.get_recent_reports(limit=3)
            print(f"🕒 Recent reports: {len(recent)}")
            
    except Exception as e:
        print(f"❌ Report management error: {e}")

def main():
    """Main demo function."""
    print("🚀 Report Generation System Demo")
    print("=" * 50)
    
    # Demo 1: Basic report generation
    demo_basic_report()
    
    # Demo 2: Service integration
    demo_service_integration()
    
    # Demo 3: Custom report
    demo_custom_report()
    
    # Demo 4: Report management
    demo_report_management()
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\n📚 Next steps:")
    print("1. Start the report service: python launch_report_service.py")
    print("2. Configure integrations (Slack, Notion, Email)")
    print("3. Integrate with your trading system")
    print("4. Customize report templates and styling")

if __name__ == '__main__':
    main() 