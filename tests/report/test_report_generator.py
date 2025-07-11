"""Test script for report generator."""

import os
import json
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading.report.report_generator import ReportGenerator

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

    def test_report_export_structure(self):
        """Validate report export structure (columns, rows, file output consistency)."""
        print("\n📊 Testing Report Export Structure Validation")
        
        generator = ReportGenerator(self.config)
        
        # Test multiple export formats
        export_formats = ['csv', 'json', 'html', 'pdf']
        
        for format_type in export_formats:
            print(f"\n  📁 Testing {format_type.upper()} export format...")
            
            # Export data in specific format
            export_result = generator._export_data(self.performance_data, self.output_dir, format=format_type)
            
            # Validate export result
            self.assertIsNotNone(export_result, f"Export result should not be None for {format_type}")
            self.assertIn('files_created', export_result, f"Should have files_created for {format_type}")
            self.assertIn('export_time', export_result, f"Should have export_time for {format_type}")
            self.assertIn('file_sizes', export_result, f"Should have file_sizes for {format_type}")
            
            print(f"    Files created: {len(export_result['files_created'])}")
            print(f"    Export time: {export_result['export_time']:.3f}s")
            
            # Validate each created file
            for file_path in export_result['files_created']:
                self.assertTrue(os.path.exists(file_path), f"File should exist: {file_path}")
                
                # Check file size
                file_size = os.path.getsize(file_path)
                self.assertGreater(file_size, 0, f"File should not be empty: {file_path}")
                
                print(f"    ✓ {os.path.basename(file_path)}: {file_size} bytes")
        
        # Comprehensive CSV validation
        print(f"\n  📋 Testing CSV export validation...")
        
        # Validate metrics.csv structure
        metrics_path = os.path.join(self.output_dir, 'metrics.csv')
        self.assertTrue(os.path.exists(metrics_path), "metrics.csv should exist")
        
        df_metrics = pd.read_csv(metrics_path)
        expected_metric_columns = ['sharpe_ratio', 'win_rate', 'max_drawdown', 'total_return', 'avg_trade', 'profit_factor']
        
        for col in expected_metric_columns:
            self.assertIn(col, df_metrics.columns, f"metrics.csv should contain column: {col}")
        
        self.assertEqual(len(df_metrics), 1, "metrics.csv should have exactly one row")
        
        # Validate data types and ranges
        self.assertIsInstance(df_metrics['sharpe_ratio'].iloc[0], (int, float), "Sharpe ratio should be numeric")
        self.assertIsInstance(df_metrics['win_rate'].iloc[0], (int, float), "Win rate should be numeric")
        self.assertGreaterEqual(df_metrics['win_rate'].iloc[0], 0, "Win rate should be >= 0")
        self.assertLessEqual(df_metrics['win_rate'].iloc[0], 1, "Win rate should be <= 1")
        
        print(f"    ✓ metrics.csv: {len(df_metrics.columns)} columns, {len(df_metrics)} rows")
        
        # Validate equity_curve.csv structure
        eq_path = os.path.join(self.output_dir, 'equity_curve.csv')
        self.assertTrue(os.path.exists(eq_path), "equity_curve.csv should exist")
        
        df_eq = pd.read_csv(eq_path)
        self.assertIn('date', df_eq.columns, "equity_curve.csv should have date column")
        self.assertIn('equity', df_eq.columns, "equity_curve.csv should have equity column")
        
        # Validate date format and sequence
        df_eq['date'] = pd.to_datetime(df_eq['date'])
        self.assertTrue(df_eq['date'].is_monotonic_increasing, "Dates should be in ascending order")
        
        # Validate equity curve properties
        self.assertGreater(len(df_eq), 0, "equity_curve.csv should have data")
        self.assertTrue(df_eq['equity'].is_monotonic_increasing, "Equity curve should be generally increasing")
        
        print(f"    ✓ equity_curve.csv: {len(df_eq.columns)} columns, {len(df_eq)} rows")
        
        # Validate returns.csv structure
        ret_path = os.path.join(self.output_dir, 'returns.csv')
        self.assertTrue(os.path.exists(ret_path), "returns.csv should exist")
        
        df_ret = pd.read_csv(ret_path)
        self.assertIn('date', df_ret.columns, "returns.csv should have date column")
        self.assertIn('returns', df_ret.columns, "returns.csv should have returns column")
        
        # Validate returns data
        self.assertGreater(len(df_ret), 0, "returns.csv should have data")
        self.assertIsInstance(df_ret['returns'].iloc[0], (int, float), "Returns should be numeric")
        
        print(f"    ✓ returns.csv: {len(df_ret.columns)} columns, {len(df_ret)} rows")
        
        # Validate trades.csv structure
        trades_path = os.path.join(self.output_dir, 'trades.csv')
        self.assertTrue(os.path.exists(trades_path), "trades.csv should exist")
        
        df_trades = pd.read_csv(trades_path)
        expected_trade_columns = ['entry_time', 'exit_time', 'symbol', 'direction', 'entry_price', 'exit_price', 'size', 'pnl']
        
        for col in expected_trade_columns:
            self.assertIn(col, df_trades.columns, f"trades.csv should contain column: {col}")
        
        # Validate trade data
        self.assertGreater(len(df_trades), 0, "trades.csv should have data")
        self.assertIn(df_trades['direction'].iloc[0], ['long', 'short'], "Direction should be long or short")
        self.assertGreater(df_trades['entry_price'].iloc[0], 0, "Entry price should be positive")
        self.assertGreater(df_trades['exit_price'].iloc[0], 0, "Exit price should be positive")
        
        print(f"    ✓ trades.csv: {len(df_trades.columns)} columns, {len(df_trades)} rows")
        
        # Comprehensive JSON validation
        print(f"\n  📄 Testing JSON export validation...")
        
        json_path = os.path.join(self.output_dir, 'full_data.json')
        self.assertTrue(os.path.exists(json_path), "full_data.json should exist")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Validate JSON structure
        required_sections = ['equity_curve', 'returns', 'trades', 'metrics', 'metadata']
        for section in required_sections:
            self.assertIn(section, data, f"JSON should contain section: {section}")
        
        # Validate metrics section
        metrics = data['metrics']
        for metric in expected_metric_columns:
            self.assertIn(metric, metrics, f"JSON metrics should contain: {metric}")
            self.assertIsInstance(metrics[metric], (int, float), f"Metric {metric} should be numeric")
        
        # Validate equity curve section
        equity_curve = data['equity_curve']
        self.assertIsInstance(equity_curve, dict, "Equity curve should be a dictionary")
        self.assertGreater(len(equity_curve), 0, "Equity curve should have data")
        
        # Validate trades section
        trades = data['trades']
        self.assertIsInstance(trades, list, "Trades should be a list")
        self.assertGreater(len(trades), 0, "Trades should have data")
        
        for trade in trades:
            for field in expected_trade_columns:
                self.assertIn(field, trade, f"Trade should contain field: {field}")
        
        print(f"    ✓ full_data.json: {len(data)} sections, {len(trades)} trades")
        
        # Cross-format consistency validation
        print(f"\n  🔄 Testing cross-format consistency...")
        
        # Compare CSV and JSON data
        csv_metrics = df_metrics.iloc[0].to_dict()
        json_metrics = data['metrics']
        
        for metric in expected_metric_columns:
            csv_value = csv_metrics[metric]
            json_value = json_metrics[metric]
            
            # Allow for small floating point differences
            self.assertAlmostEqual(csv_value, json_value, places=6, 
                                 msg=f"Metric {metric} should be consistent between CSV and JSON")
        
        print(f"    ✓ Metrics consistency: {len(expected_metric_columns)} metrics validated")
        
        # Validate file output consistency
        print(f"\n  📊 Testing file output consistency...")
        
        # Check that all expected files exist
        expected_files = ['metrics.csv', 'equity_curve.csv', 'returns.csv', 'trades.csv', 'full_data.json']
        
        for filename in expected_files:
            file_path = os.path.join(self.output_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"Expected file should exist: {filename}")
            
            # Check file is readable
            try:
                if filename.endswith('.csv'):
                    pd.read_csv(file_path)
                elif filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        json.load(f)
            except Exception as e:
                self.fail(f"File {filename} should be readable: {str(e)}")
        
        print(f"    ✓ File consistency: {len(expected_files)} files validated")
        
        # Validate data integrity
        print(f"\n  🔍 Testing data integrity...")
        
        # Check that equity curve dates match returns dates
        equity_dates = set(df_eq['date'].dt.strftime('%Y-%m-%d'))
        returns_dates = set(df_ret['date'].dt.strftime('%Y-%m-%d'))
        
        # Returns should be a subset of equity curve dates (missing first day)
        self.assertTrue(returns_dates.issubset(equity_dates), "Returns dates should be subset of equity curve dates")
        
        # Check that trade dates are within equity curve range
        trade_dates = set()
        for _, trade in df_trades.iterrows():
            trade_dates.add(pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d'))
            trade_dates.add(pd.to_datetime(trade['exit_time']).strftime('%Y-%m-%d'))
        
        self.assertTrue(trade_dates.issubset(equity_dates), "Trade dates should be within equity curve range")
        
        print(f"    ✓ Date consistency: {len(equity_dates)} equity dates, {len(returns_dates)} return dates, {len(trade_dates)} trade dates")
        
        # Validate performance calculations
        print(f"\n  🧮 Testing performance calculations...")
        
        # Verify that calculated metrics match expected ranges
        calculated_sharpe = df_metrics['sharpe_ratio'].iloc[0]
        calculated_win_rate = df_metrics['win_rate'].iloc[0]
        calculated_max_dd = df_metrics['max_drawdown'].iloc[0]
        
        # Reasonable ranges for test data
        self.assertGreater(calculated_sharpe, -5, "Sharpe ratio should be reasonable")
        self.assertLess(calculated_sharpe, 5, "Sharpe ratio should be reasonable")
        self.assertGreaterEqual(calculated_win_rate, 0, "Win rate should be >= 0")
        self.assertLessEqual(calculated_win_rate, 1, "Win rate should be <= 1")
        self.assertLessEqual(calculated_max_dd, 0, "Max drawdown should be <= 0")
        self.assertGreater(calculated_max_dd, -1, "Max drawdown should be reasonable")
        
        print(f"    ✓ Performance validation: Sharpe={calculated_sharpe:.3f}, Win Rate={calculated_win_rate:.3f}, Max DD={calculated_max_dd:.3f}")
        
        print("✅ Report export structure validation completed")

if __name__ == '__main__':
    unittest.main() 