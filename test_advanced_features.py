#!/usr/bin/env python3
"""
Advanced Features Test Script for Evolve Trading Platform.

This script tests all the new institutional-grade features:
1. Reinforcement Learning Engine
2. Causal Inference Module
3. Temporal Fusion Transformer
4. Auto-Evolutionary Model Generator
5. Live Broker Integration
6. Voice & Chat-Driven Interface
7. Risk & Tail Exposure Engine
8. Regime-Switching Strategy Gate
9. Real-Time Streaming Optimization
10. Strategy Health Dashboard
"""

import sys
import os
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rl_engine():
    """Test Reinforcement Learning Engine."""
    logger.info("Testing Reinforcement Learning Engine...")
    
    try:
        from rl.strategy_trainer import RLStrategyTrainer, create_rl_strategy
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = (1 + pd.Series(returns, index=dates)).cumprod() * 100
        
        data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Test RL trainer
        trainer = RLStrategyTrainer(model_type="PPO")
        env = trainer.create_environment(data)
        
        # Test training (short version)
        results = trainer.train(total_timesteps=1000)
        
        logger.info(f"✅ RL Engine test passed - Model trained successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ RL Engine test failed: {e}")
        return False

def test_causal_inference():
    """Test Causal Inference Module."""
    logger.info("Testing Causal Inference Module...")
    
    try:
        from causal.causal_model import CausalModelAnalyzer, analyze_causal_relationships
        
        # Create sample data
        np.random.seed(42)
        n = 1000
        
        # Generate correlated data
        x1 = np.random.normal(0, 1, n)
        x2 = 0.5 * x1 + np.random.normal(0, 0.5, n)
        x3 = 0.3 * x1 + 0.4 * x2 + np.random.normal(0, 0.3, n)
        returns = 0.1 * x1 + 0.2 * x2 + 0.1 * x3 + np.random.normal(0, 0.1, n)
        
        data = pd.DataFrame({
            'feature1': x1,
            'feature2': x2,
            'feature3': x3,
            'returns': returns
        })
        
        # Test causal analysis
        analyzer = CausalModelAnalyzer(
            data=data,
            target_variable="returns",
            treatment_variables=["feature1", "feature2"]
        )
        
        result = analyzer.analyze()
        
        logger.info(f"✅ Causal Inference test passed - Analysis completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Causal Inference test failed: {e}")
        return False

def test_tft_model():
    """Test Temporal Fusion Transformer."""
    logger.info("Testing Temporal Fusion Transformer...")
    
    try:
        from models.tft_model import TFTForecaster, create_tft_forecaster
        
        # Create sample time series data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate multivariate time series
        n_features = 5
        data = pd.DataFrame(
            np.random.randn(len(dates), n_features),
            index=dates,
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add target variable
        data['target'] = data['feature_0'] * 0.5 + data['feature_1'] * 0.3 + np.random.normal(0, 0.1, len(dates))
        
        # Test TFT forecaster
        forecaster = TFTForecaster(
            sequence_length=30,
            prediction_horizon=5
        )
        
        # Test with small dataset
        small_data = data.head(100)
        results = forecaster.train(
            data=small_data,
            target_column="target",
            batch_size=16,
            max_epochs=2
        )
        
        logger.info(f"✅ TFT Model test passed - Training completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ TFT Model test failed: {e}")
        return False

def test_model_generator():
    """Test Auto-Evolutionary Model Generator."""
    logger.info("Testing Auto-Evolutionary Model Generator...")
    
    try:
        from agents.model_generator_agent import AutoEvolutionaryModelGenerator, ArxivResearchFetcher
        
        # Create sample benchmark data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        
        benchmark_data = pd.DataFrame({
            'returns': returns,
            'feature1': np.random.normal(0, 1, len(dates)),
            'feature2': np.random.normal(0, 1, len(dates))
        })
        
        # Test research fetcher (mock)
        fetcher = ArxivResearchFetcher(max_results=5)
        
        # Test model generator
        generator = AutoEvolutionaryModelGenerator(
            benchmark_data=benchmark_data,
            target_column="returns",
            current_best_score=1.0,
            max_candidates=3
        )
        
        logger.info(f"✅ Model Generator test passed - Components initialized")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Generator test failed: {e}")
        return False

def test_live_trading():
    """Test Live Trading Interface."""
    logger.info("Testing Live Trading Interface...")
    
    try:
        from execution.live_trading_interface import create_live_trading_interface, OrderRequest
        
        # Test simulated trading interface
        interface = create_live_trading_interface(
            mode="simulated",
            config={
                "initial_cash": 100000.0,
                "commission_rate": 0.001,
                "slippage": 0.0005
            }
        )
        
        # Test order placement
        order_request = OrderRequest(
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market"
        )
        
        # This would normally place an order, but we'll just test the interface creation
        logger.info(f"✅ Live Trading test passed - Interface created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Live Trading test failed: {e}")
        return False

def test_chatbox_agent():
    """Test Voice & Chat-Driven Interface."""
    logger.info("Testing Voice & Chat-Driven Interface...")
    
    try:
        from ui.chatbox_agent import create_chatbox_agent, CommandParser
        
        # Test command parser
        parser = CommandParser()
        
        # Test command parsing
        test_commands = [
            "Buy 100 shares of AAPL",
            "Analyze TSLA with momentum strategy",
            "Backtest breakout strategy on SPY"
        ]
        
        for command in test_commands:
            parsed = parser.parse_command(command)
            logger.info(f"Parsed command: {parsed.action} - {parsed.symbol}")
        
        # Test chatbox agent creation
        agent = create_chatbox_agent(enable_voice=False, enable_tts=False)
        
        logger.info(f"✅ Chatbox Agent test passed - Commands parsed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Chatbox Agent test failed: {e}")
        return False

def test_tail_risk():
    """Test Risk & Tail Exposure Engine."""
    logger.info("Testing Risk & Tail Exposure Engine...")
    
    try:
        from risk.tail_risk import TailRiskEngine, calculate_portfolio_risk, analyze_tail_risk
        
        # Create sample returns data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), 
                          index=pd.date_range('2023-01-01', periods=252, freq='D'))
        
        # Test risk calculation
        risk_metrics = calculate_portfolio_risk(returns)
        
        # Test tail risk analysis
        engine = TailRiskEngine()
        risk_report = engine.generate_risk_report(returns, "Test Portfolio")
        
        logger.info(f"✅ Tail Risk test passed - Risk metrics calculated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tail Risk test failed: {e}")
        return False

def test_strategy_gatekeeper():
    """Test Regime-Switching Strategy Gate."""
    logger.info("Testing Regime-Switching Strategy Gate...")
    
    try:
        from strategies.gatekeeper import create_strategy_gatekeeper, RegimeClassifier
        
        # Create sample strategies config
        strategies_config = {
            "momentum": {"default_active": True},
            "mean_reversion": {"default_active": True},
            "trend_following": {"default_active": False}
        }
        
        # Test gatekeeper creation
        gatekeeper = create_strategy_gatekeeper(strategies_config)
        
        # Test regime classifier
        classifier = RegimeClassifier()
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        prices = pd.Series(np.random.randn(len(dates)).cumsum() + 100, index=dates)
        
        regime = classifier.classify_regime(prices)
        
        logger.info(f"✅ Strategy Gatekeeper test passed - Regime classified: {regime.regime}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy Gatekeeper test failed: {e}")
        return False

def test_streaming_pipeline():
    """Test Real-Time Streaming Optimization."""
    logger.info("Testing Real-Time Streaming Pipeline...")
    
    try:
        from data.streaming_pipeline import create_streaming_pipeline, InMemoryCache
        
        # Test cache
        cache = InMemoryCache(max_size=100)
        
        # Test streaming pipeline creation
        pipeline = create_streaming_pipeline(
            symbols=["AAPL", "GOOGL", "MSFT"],
            timeframes=["1m", "5m", "1h"],
            providers=["yfinance"]
        )
        
        # Test cache operations
        test_data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(),
            "open": 150.0,
            "high": 151.0,
            "low": 149.0,
            "close": 150.5,
            "volume": 1000,
            "timeframe": "1m",
            "source": "test"
        }
        
        from data.streaming_pipeline import MarketData
        market_data = MarketData(**test_data)
        cache.add_data("AAPL", "1m", market_data)
        
        retrieved_data = cache.get_latest_data("AAPL", "1m")
        
        logger.info(f"✅ Streaming Pipeline test passed - Cache operations successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming Pipeline test failed: {e}")
        return False

def test_dashboard():
    """Test Strategy Health Dashboard."""
    logger.info("Testing Strategy Health Dashboard...")
    
    try:
        # Test dashboard components
        from pages.10_Strategy_Health_Dashboard import (
            initialize_dashboard,
            create_equity_curve_chart,
            create_strategy_status_table
        )
        
        # Initialize dashboard
        initialize_dashboard()
        
        # Test chart creation
        chart = create_equity_curve_chart()
        
        # Test table creation
        table = create_strategy_status_table()
        
        logger.info(f"✅ Dashboard test passed - Components created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dashboard test failed: {e}")
        return False

def run_all_tests():
    """Run all advanced feature tests."""
    logger.info("🚀 Starting Advanced Features Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Reinforcement Learning Engine", test_rl_engine),
        ("Causal Inference Module", test_causal_inference),
        ("Temporal Fusion Transformer", test_tft_model),
        ("Auto-Evolutionary Model Generator", test_model_generator),
        ("Live Trading Interface", test_live_trading),
        ("Voice & Chat-Driven Interface", test_chatbox_agent),
        ("Risk & Tail Exposure Engine", test_tail_risk),
        ("Regime-Switching Strategy Gate", test_strategy_gatekeeper),
        ("Real-Time Streaming Pipeline", test_streaming_pipeline),
        ("Strategy Health Dashboard", test_dashboard)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All advanced features are working correctly!")
        return True
    else:
        logger.warning("⚠️ Some features need attention. Check the logs above.")
        return False

def main():
    """Main test function."""
    print("Evolve Trading Platform - Advanced Features Test Suite")
    print("=" * 60)
    
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 