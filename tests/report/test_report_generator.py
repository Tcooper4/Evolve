"""Test script for report generator."""

import os
import json
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading.report.report_generator import ReportGenerator, ReportConfig

class TestReportGenerator(unittest.TestCase):
    """Test cases for report generator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create sample performance data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        equity = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
        returns = np.diff(equity) / equity[:-1]
        
        cls.performance_data = {
            'equity_curve': dict(zip(dates, equity)),
            'returns': dict(zip(dates[1:], returns)),
            'trades': [
                {
                    'entry_time': '2023-01-01',
                    'exit_time': '2023-01-10',
                    'symbol': 'AAPL',
                    'direction': 'long',
                    'entry_price': 100.0,
                    'exit_price': 110.0,
                    'size': 1.0,
                    'pnl': 10.0
                }
            ],
            'metrics': {
                'sharpe_ratio': 1.5,
                'win_rate': 0.6,
                'max_drawdown': -0.1,
                'total_return': 0.2,
                'avg_trade': 0.01,
                'profit_factor': 1.8
            }
        }
        
        # Create test config
        cls.config = ReportConfig(
            theme='light',
            include_sections=['equity_curve', 'drawdown', 'returns_distribution'],
            export_formats=['html', 'pdf'],
            strategy_params={'param1': 'value1'},
            model_config={'version': '1.0.0'},
            run_metadata={
                'user': 'test_user',
                'run_time': datetime.utcnow().isoformat(),
                'model_version': '1.0.0',
                'strategy_name': 'test_strategy'
            }
        )
        
        # Create output directory
        cls.output_dir = 'tests/report/output'
        os.makedirs(cls.output_dir, exist_ok=True)
    
    def test_report_generation(self):
        """Test report generation."""
        # Initialize generator
        generator = ReportGenerator(self.config)
        
        # Generate report
        output_path = os.path.join(self.output_dir, 'test_report.pdf')
        report_path = generator.generate_report(self.performance_data, output_path)
        
        # Check report was created
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(os.path.exists(report_path.replace('.pdf', '.html')))
    
    def test_data_validation(self):
        """Test data validation."""
        generator = ReportGenerator(self.config)
        
        # Test with missing required field
        invalid_data = self.performance_data.copy()
        del invalid_data['equity_curve']
        
        with self.assertRaises(ValueError):
            generator.generate_report(invalid_data, 'test.pdf')
    
    def test_theme_config(self):
        """Test theme configuration."""
        # Test light theme
        light_config = ReportConfig(theme='light')
        generator = ReportGenerator(light_config)
        self.assertEqual(generator.theme_config['background_color'], '#ffffff')
        
        # Test dark theme
        dark_config = ReportConfig(theme='dark')
        generator = ReportGenerator(dark_config)
        self.assertEqual(generator.theme_config['background_color'], '#1e1e1e')
    
    def test_chart_generation(self):
        """Test chart generation."""
        generator = ReportGenerator(self.config)
        
        # Test equity curve chart
        equity_chart = generator._create_equity_curve_chart(self.performance_data)
        self.assertIsNotNone(equity_chart)
        
        # Test drawdown chart
        drawdown_chart = generator._create_drawdown_chart(self.performance_data)
        self.assertIsNotNone(drawdown_chart)
        
        # Test returns distribution chart
        returns_chart = generator._create_returns_distribution_chart(self.performance_data)
        self.assertIsNotNone(returns_chart)
        
        # Test rolling metrics chart
        rolling_chart = generator._create_rolling_metrics_chart(self.performance_data)
        self.assertIsNotNone(rolling_chart)
    
    def test_data_export(self):
        """Test data export."""
        generator = ReportGenerator(self.config)
        
        # Export data
        generator._export_data(self.performance_data, self.output_dir)
        
        # Check files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'metrics.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'equity_curve.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'returns.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'trades.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'full_data.json')))
    
    def test_insights_generation(self):
        """Test insights generation."""
        generator = ReportGenerator(self.config)
        
        # Generate insights
        insights = generator._generate_performance_insights(self.performance_data)
        
        # Check insights were generated
        self.assertIsNotNone(insights)
        self.assertIsInstance(insights, str)
        self.assertTrue(len(insights) > 0)

if __name__ == '__main__':
    unittest.main() 