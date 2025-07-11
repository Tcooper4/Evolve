"""
Performance benchmarks for MarketAnalyzer.

This module provides comprehensive performance testing for the MarketAnalyzer,
including data fetching, analysis, and batch processing capabilities.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import psutil
import os
from pathlib import Path
import json
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from trading.market.market_analyzer import MarketAnalyzer
import yfinance as yf
import pandas_ta as ta
import finta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class MarketAnalyzerBenchmark(unittest.TestCase):
    """Benchmark tests for MarketAnalyzer."""
    
    def setUp(self):
        """Set up benchmark environment."""
        # Create benchmark directory
        self.benchmark_dir = Path('benchmarks')
        self.benchmark_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.benchmark_dir / 'benchmark.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Test symbols
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
        
        # Initialize analyzer
        self.analyzer = MarketAnalyzer(config={
            'debug_mode': False,
            'skip_pca': False,
            'results_dir': str(self.benchmark_dir / 'results')
        })
        
        # Initialize results storage
        self.results = {
            'data_fetching': [],
            'single_analysis': [],
            'batch_analysis': [],
            'memory_usage': [],
            'cache_performance': [],
            'library_comparison': {
                'technical_indicators': [],
                'data_processing': [],
                'memory_efficiency': []
            },
            'baseline_comparison': []
        }
        
    def tearDown(self):
        """Clean up benchmark environment and save results."""
        # Save results
        results_file = self.benchmark_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Generate plots
        self._generate_plots()
        
    def _measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
        
    def _generate_plots(self):
        """Generate benchmark result plots."""
        # Set style without modifying global rcParams
        sns.set_style('whitegrid')
        
        # Create interactive HTML reports
        self._generate_interactive_report()
        
        # Data fetching performance
        self._plot_data_fetching()
        
        # Single analysis performance
        self._plot_single_analysis()
        
        # Batch analysis performance
        self._plot_batch_analysis()
        
        # Memory usage
        self._plot_memory_usage()
        
        # Cache performance
        self._plot_cache_performance()
        
        # Library comparison
        self._plot_library_comparison()
        
        # Baseline metrics comparison
        self._plot_baseline_comparison()
        
    def _generate_interactive_report(self):
        """Generate interactive HTML report using Plotly."""
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Data Fetching Performance',
                'Single Analysis Performance',
                'Batch Analysis Performance',
                'Memory Usage',
                'Cache Performance',
                'Library Comparison'
            )
        )
        
        # Add traces for each subplot
        self._add_interactive_traces(fig)
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text='Market Analyzer Performance Report',
            showlegend=True
        )
        
        # Save as HTML
        fig.write_html(self.benchmark_dir / 'interactive_report.html')
        
    def _add_interactive_traces(self, fig):
        """Add interactive traces to the report."""
        # Data fetching
        df_fetch = pd.DataFrame(self.results['data_fetching'])
        fig.add_trace(
            go.Box(
                x=df_fetch['symbol'],
                y=df_fetch['time'],
                name='Data Fetching'
            ),
            row=1, col=1
        )
        
        # Single analysis
        df_single = pd.DataFrame(self.results['single_analysis'])
        fig.add_trace(
            go.Box(
                x=df_single['symbol'],
                y=df_single['time'],
                name='Single Analysis'
            ),
            row=1, col=2
        )
        
        # Batch analysis
        df_batch = pd.DataFrame(self.results['batch_analysis'])
        fig.add_trace(
            go.Box(
                x=df_batch['batch_size'],
                y=df_batch['time'],
                name='Batch Analysis'
            ),
            row=2, col=1
        )
        
        # Memory usage
        df_memory = pd.DataFrame(self.results['memory_usage'])
        fig.add_trace(
            go.Scatter(
                x=df_memory['symbols'],
                y=df_memory['memory'],
                name='Memory Usage',
                mode='lines+markers'
            ),
            row=2, col=2
        )
        
        # Cache performance
        df_cache = pd.DataFrame(self.results['cache_performance'])
        fig.add_trace(
            go.Box(
                x=df_cache['operation'],
                y=df_cache['time'],
                name='Cache Performance'
            ),
            row=3, col=1
        )
        
        # Library comparison
        df_lib = pd.DataFrame(self.results['library_comparison']['technical_indicators'])
        fig.add_trace(
            go.Bar(
                x=df_lib['library'],
                y=df_lib['time'],
                name='Library Comparison'
            ),
            row=3, col=2
        )
        
    def _plot_data_fetching(self):
        """Plot data fetching performance."""
        plt.figure(figsize=(12, 8))
        df_fetch = pd.DataFrame(self.results['data_fetching'])
        
        # Create box plot
        sns.boxplot(data=df_fetch, x='symbol', y='time')
        plt.title('Data Fetching Performance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'data_fetching.png')
        
        # Create violin plot
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=df_fetch, x='symbol', y='time')
        plt.title('Data Fetching Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'data_fetching_distribution.png')
        
    def _plot_single_analysis(self):
        """Plot single analysis performance."""
        plt.figure(figsize=(12, 8))
        df_single = pd.DataFrame(self.results['single_analysis'])
        
        # Create box plot
        sns.boxplot(data=df_single, x='symbol', y='time')
        plt.title('Single Analysis Performance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'single_analysis.png')
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        pivot = df_single.pivot_table(
            values='time',
            index='symbol',
            columns='timestamp',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Analysis Time Heatmap')
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'analysis_heatmap.png')
        
    def _plot_batch_analysis(self):
        """Plot batch analysis performance."""
        plt.figure(figsize=(12, 8))
        df_batch = pd.DataFrame(self.results['batch_analysis'])
        
        # Create box plot
        sns.boxplot(data=df_batch, x='batch_size', y='time')
        plt.title('Batch Analysis Performance')
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'batch_analysis.png')
        
        # Create regression plot
        plt.figure(figsize=(12, 8))
        sns.regplot(data=df_batch, x='batch_size', y='time')
        plt.title('Batch Size vs Processing Time')
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'batch_regression.png')
        
    def _plot_memory_usage(self):
        """Plot memory usage."""
        plt.figure(figsize=(12, 8))
        df_memory = pd.DataFrame(self.results['memory_usage'])
        
        # Create line plot
        sns.lineplot(data=df_memory, x='symbols', y='memory')
        plt.title('Memory Usage')
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'memory_usage.png')
        
        # Create area plot
        plt.figure()
        sns.lineplot(data=df_memory, x='symbols', y='memory', fill=True)
        plt.title('Memory Usage Area')
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'memory_usage_area.png')
        
    def _plot_cache_performance(self):
        """Plot cache performance."""
        plt.figure()
        df_cache = pd.DataFrame(self.results['cache_performance'])
        
        # Create box plot
        sns.boxplot(data=df_cache, x='operation', y='time')
        plt.title('Cache Performance')
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'cache_performance.png')
        
        # Create violin plot
        plt.figure()
        sns.violinplot(data=df_cache, x='operation', y='time')
        plt.title('Cache Performance Distribution')
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'cache_distribution.png')
        
    def _plot_library_comparison(self):
        """Plot library comparison."""
        plt.figure()
        df_lib = pd.DataFrame(self.results['library_comparison']['technical_indicators'])
        
        # Create bar plot
        sns.barplot(data=df_lib, x='library', y='time')
        plt.title('Library Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'library_comparison.png')
        
        # Create radar plot
        fig = go.Figure()
        for metric in ['time', 'memory', 'accuracy']:
            fig.add_trace(go.Scatterpolar(
                r=df_lib[metric],
                theta=df_lib['library'],
                name=metric
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True
        )
        fig.write_image(self.benchmark_dir / 'library_radar.png')
        
    def test_library_comparison(self):
        """Compare performance with different technical analysis libraries."""
        print("\n📊 Testing Library Comparison")
        
        # Test symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            print(f"\n🔍 Testing {symbol}")
            
            # Fetch data
            data = self.analyzer.fetch_data(symbol, period='1y')
            
            # Test pandas_ta
            start_time = time.time()
            ta_indicators = self.analyzer.calculate_technical_indicators_pandas_ta(data)
            ta_time = time.time() - start_time
            
            # Test finta
            start_time = time.time()
            finta_indicators = self.analyzer.calculate_technical_indicators_finta(data)
            finta_time = time.time() - start_time
            
            # Test custom implementation
            start_time = time.time()
            custom_indicators = self.analyzer.calculate_technical_indicators(data)
            custom_time = time.time() - start_time
            
            # Store results
            self.results['library_comparison']['technical_indicators'].append({
                'symbol': symbol,
                'pandas_ta_time': ta_time,
                'finta_time': finta_time,
                'custom_time': custom_time,
                'pandas_ta_accuracy': self._calculate_accuracy(ta_indicators),
                'finta_accuracy': self._calculate_accuracy(finta_indicators),
                'custom_accuracy': self._calculate_accuracy(custom_indicators)
            })
            
            print(f"✅ pandas_ta: {ta_time:.3f}s, finta: {finta_time:.3f}s, custom: {custom_time:.3f}s")

    def test_baseline_metrics_comparison(self):
        """Compare new analyzers against baseline metrics (Sharpe, volatility)."""
        print("\n📈 Testing Baseline Metrics Comparison")
        
        # Define baseline metrics
        baseline_metrics = {
            'sharpe_ratio': 1.0,  # Target Sharpe ratio
            'volatility': 0.15,   # Target volatility (15%)
            'max_drawdown': 0.10, # Target max drawdown (10%)
            'win_rate': 0.55,     # Target win rate (55%)
            'profit_factor': 1.5  # Target profit factor
        }
        
        # Test symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        for symbol in symbols:
            print(f"\n🔍 Testing baseline metrics for {symbol}")
            
            # Fetch data
            data = self.analyzer.fetch_data(symbol, period='1y')
            
            # Calculate baseline metrics using different analyzers
            analyzers = {
                'standard': self.analyzer,
                'enhanced': self._create_enhanced_analyzer(),
                'ml_based': self._create_ml_analyzer()
            }
            
            analyzer_results = {}
            
            for analyzer_name, analyzer in analyzers.items():
                print(f"  📊 Testing {analyzer_name} analyzer...")
                
                # Calculate performance metrics
                start_time = time.time()
                metrics = analyzer.calculate_performance_metrics(data)
                calc_time = time.time() - start_time
                
                # Store results
                analyzer_results[analyzer_name] = {
                    'metrics': metrics,
                    'calculation_time': calc_time,
                    'baseline_comparison': {}
                }
                
                # Compare against baseline
                for metric_name, baseline_value in baseline_metrics.items():
                    if metric_name in metrics:
                        actual_value = metrics[metric_name]
                        deviation = abs(actual_value - baseline_value) / baseline_value
                        
                        analyzer_results[analyzer_name]['baseline_comparison'][metric_name] = {
                            'actual': actual_value,
                            'baseline': baseline_value,
                            'deviation': deviation,
                            'within_tolerance': deviation <= 0.2  # 20% tolerance
                        }
                        
                        print(f"    {metric_name}: {actual_value:.3f} (baseline: {baseline_value:.3f}, deviation: {deviation:.1%})")
                
                # Store in results
                self.results['baseline_comparison'] = self.results.get('baseline_comparison', [])
                self.results['baseline_comparison'].append({
                    'symbol': symbol,
                    'analyzer': analyzer_name,
                    'results': analyzer_results[analyzer_name]
                })
            
            # Determine best performing analyzer
            best_analyzer = self._find_best_analyzer(analyzer_results, baseline_metrics)
            print(f"  🏆 Best analyzer: {best_analyzer}")
            
        print("\n✅ Baseline metrics comparison completed")

    def _create_enhanced_analyzer(self):
        """Create an enhanced analyzer with additional features."""
        # This would create an enhanced version of the analyzer
        # For now, return the standard analyzer
        return self.analyzer

    def _create_ml_analyzer(self):
        """Create an ML-based analyzer."""
        # This would create an ML-enhanced version of the analyzer
        # For now, return the standard analyzer
        return self.analyzer

    def _find_best_analyzer(self, analyzer_results, baseline_metrics):
        """Find the best performing analyzer based on baseline metrics."""
        best_score = float('inf')
        best_analyzer = None
        
        for analyzer_name, results in analyzer_results.items():
            total_deviation = 0
            valid_metrics = 0
            
            for metric_name, baseline_value in baseline_metrics.items():
                if metric_name in results['baseline_comparison']:
                    deviation = results['baseline_comparison'][metric_name]['deviation']
                    total_deviation += deviation
                    valid_metrics += 1
            
            if valid_metrics > 0:
                avg_deviation = total_deviation / valid_metrics
                if avg_deviation < best_score:
                    best_score = avg_deviation
                    best_analyzer = analyzer_name
        
        return best_analyzer

    def _plot_baseline_comparison(self):
        """Plot baseline metrics comparison results."""
        if not self.results['baseline_comparison']:
            print("No baseline comparison results to plot.")
            return

        plt.figure(figsize=(12, 8))
        sns.set_style('whitegrid')

        # Create a DataFrame for plotting
        df_baseline = pd.DataFrame(self.results['baseline_comparison'])
        df_baseline = df_baseline.explode('results') # Explode the list of results

        # Plot calculation time
        plt.subplot(2, 2, 1)
        sns.barplot(data=df_baseline, x='analyzer', y='results.calculation_time')
        plt.title('Calculation Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)

        # Plot metrics comparison
        plt.subplot(2, 2, 2)
        metrics_to_plot = ['sharpe_ratio', 'volatility', 'max_drawdown', 'win_rate', 'profit_factor']
        for metric in metrics_to_plot:
            plt.subplot(2, 2, 2) # This subplot will be overwritten
            sns.barplot(data=df_baseline, x='analyzer', y=f'results.metrics.{metric}')
            plt.title(f'{metric} Comparison')
            plt.ylabel(metric)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(self.benchmark_dir / 'baseline_metrics_comparison.png')

    def _calculate_accuracy(self, indicators: pd.DataFrame) -> float:
        """Calculate accuracy score for indicators."""
        try:
            # Simple accuracy metric based on data quality
            valid_ratio = indicators.notna().mean().mean()
            return float(valid_ratio)
        except Exception:
            return 0.0
            
    def test_data_fetching_performance(self):
        """Benchmark data fetching performance."""
        self.logger.info("Starting data fetching benchmark")
        
        for symbol in self.symbols:
            for _ in range(3):  # Run each test 3 times
                start_time = time.time()
                try:
                    self.analyzer.fetch_data(symbol, period='1y', interval='1d')
                    end_time = time.time()
                    
                    self.results['data_fetching'].append({
                        'symbol': symbol,
                        'time': end_time - start_time,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    
    def test_single_analysis_performance(self):
        """Benchmark single analysis performance."""
        self.logger.info("Starting single analysis benchmark")
        
        for symbol in self.symbols:
            for _ in range(3):  # Run each test 3 times
                start_time = time.time()
                try:
                    self.analyzer.analyze(symbol, period='1y', interval='1d')
                    end_time = time.time()
                    
                    self.results['single_analysis'].append({
                        'symbol': symbol,
                        'time': end_time - start_time,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {str(e)}")
                    
    def test_batch_analysis_performance(self):
        """Benchmark batch analysis performance."""
        self.logger.info("Starting batch analysis benchmark")
        
        batch_sizes = [2, 4, 6, 8, 10]
        for batch_size in batch_sizes:
            for _ in range(3):  # Run each test 3 times
                start_time = time.time()
                try:
                    self.analyzer.analyze_batch(
                        self.symbols[:batch_size],
                        period='1y',
                        interval='1d',
                        batch_size=batch_size,
                        max_workers=4
                    )
                    end_time = time.time()
                    
                    self.results['batch_analysis'].append({
                        'batch_size': batch_size,
                        'time': end_time - start_time,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.error(f"Error in batch analysis with size {batch_size}: {str(e)}")
                    
    def test_memory_usage(self):
        """Benchmark memory usage."""
        self.logger.info("Starting memory usage benchmark")
        
        for i in range(1, len(self.symbols) + 1):
            symbols = self.symbols[:i]
            try:
                # Measure memory before
                memory_before = self._measure_memory()
                
                # Run analysis
                self.analyzer.analyze_batch(
                    symbols,
                    period='1y',
                    interval='1d',
                    batch_size=i,
                    max_workers=4
                )
                
                # Measure memory after
                memory_after = self._measure_memory()
                
                self.results['memory_usage'].append({
                    'symbols': i,
                    'memory': memory_after - memory_before,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error measuring memory for {i} symbols: {str(e)}")
                
    def test_cache_performance(self):
        """Benchmark cache performance."""
        self.logger.info("Starting cache performance benchmark")
        
        # Test symbol
        symbol = 'AAPL'
        
        # Test cache write
        for _ in range(3):  # Run each test 3 times
            start_time = time.time()
            try:
                data = self.analyzer.fetch_data(symbol, period='1y', interval='1d')
                self.analyzer._set_cached_data(f"test_cache_{symbol}", data)
                end_time = time.time()
                
                self.results['cache_performance'].append({
                    'operation': 'write',
                    'time': end_time - start_time,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error in cache write: {str(e)}")
                
        # Test cache read
        for _ in range(3):  # Run each test 3 times
            start_time = time.time()
            try:
                self.analyzer._get_cached_data(f"test_cache_{symbol}")
                end_time = time.time()
                
                self.results['cache_performance'].append({
                    'operation': 'read',
                    'time': end_time - start_time,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error in cache read: {str(e)}")
                
if __name__ == '__main__':
    unittest.main() 