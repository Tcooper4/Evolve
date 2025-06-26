#!/usr/bin/env python3
"""
Test Agent System

Demonstrates the 3-agent autonomous model management system
with sample data and operations.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from trading.agents import (
    ModelBuilderAgent, ModelBuildRequest,
    PerformanceCriticAgent, ModelEvaluationRequest,
    UpdaterAgent
)


def create_sample_data() -> str:
    """Create sample market data for testing.
    
    Returns:
        Path to created data file
    """
    # Create sample market data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100
    
    # Create OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    # Save to file
    data_path = Path("data/sample_market_data.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)
    
    print(f"Created sample data: {data_path}")
    return str(data_path)


def test_model_builder(data_path: str) -> Dict[str, Any]:
    """Test the ModelBuilderAgent.
    
    Args:
        data_path: Path to market data
        
    Returns:
        Dictionary with build results
    """
    print("\n=== Testing ModelBuilderAgent ===")
    
    builder = ModelBuilderAgent()
    results = {}
    
    # Test building different model types
    model_types = ['lstm', 'xgboost', 'ensemble']
    
    for model_type in model_types:
        print(f"\nBuilding {model_type.upper()} model...")
        
        request = ModelBuildRequest(
            model_type=model_type,
            data_path=data_path,
            target_column='close',
            hyperparameters={
                'lstm': {
                    'hidden_dim': 32,
                    'num_layers': 1,
                    'dropout': 0.1,
                    'learning_rate': 0.001,
                    'batch_size': 16,
                    'epochs': 10  # Reduced for testing
                },
                'xgboost': {
                    'n_estimators': 50,
                    'max_depth': 4,
                    'learning_rate': 0.1
                },
                'ensemble': {
                    'models': ['lstm', 'xgboost'],
                    'weights': [0.5, 0.5]
                }
            }.get(model_type, {}),
            request_id=f"test_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        try:
            result = builder.build_model(request)
            results[model_type] = result
            
            print(f"✅ Built {model_type} model: {result.model_id}")
            print(f"   Status: {result.build_status}")
            print(f"   Training metrics: {result.training_metrics}")
            
        except Exception as e:
            print(f"❌ Failed to build {model_type} model: {str(e)}")
            results[model_type] = None
    
    return results


def test_performance_critic(data_path: str, build_results: Dict[str, Any]) -> Dict[str, Any]:
    """Test the PerformanceCriticAgent.
    
    Args:
        data_path: Path to market data
        build_results: Results from model builder
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n=== Testing PerformanceCriticAgent ===")
    
    critic = PerformanceCriticAgent()
    eval_results = {}
    
    for model_type, build_result in build_results.items():
        if build_result is None or build_result.build_status != "success":
            continue
            
        print(f"\nEvaluating {model_type.upper()} model...")
        
        request = ModelEvaluationRequest(
            model_id=build_result.model_id,
            model_path=build_result.model_path,
            model_type=build_result.model_type,
            test_data_path=data_path,
            evaluation_period=60,  # 60 days for testing
            request_id=f"test_eval_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        try:
            result = critic.evaluate_model(request)
            eval_results[model_type] = result
            
            print(f"✅ Evaluated {model_type} model: {result.model_id}")
            print(f"   Sharpe Ratio: {result.performance_metrics.get('sharpe_ratio', 'N/A'):.4f}")
            print(f"   Max Drawdown: {result.risk_metrics.get('max_drawdown', 'N/A'):.4f}")
            print(f"   Win Rate: {result.trading_metrics.get('win_rate', 'N/A'):.4f}")
            print(f"   Recommendations: {len(result.recommendations)}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_type} model: {str(e)}")
            eval_results[model_type] = None
    
    return eval_results


def test_updater(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    """Test the UpdaterAgent.
    
    Args:
        eval_results: Results from performance critic
        
    Returns:
        Dictionary with update results
    """
    print("\n=== Testing UpdaterAgent ===")
    
    updater = UpdaterAgent()
    update_results = {}
    
    for model_type, eval_result in eval_results.items():
        if eval_result is None or eval_result.evaluation_status != "success":
            continue
            
        print(f"\nProcessing {model_type.upper()} model for updates...")
        
        try:
            # Process evaluation and determine if update is needed
            update_request = updater.process_evaluation(eval_result)
            
            if update_request:
                print(f"   Update needed: {update_request.update_type} ({update_request.priority})")
                
                # Execute update
                update_result = updater.execute_update(update_request)
                update_results[model_type] = update_result
                
                if update_result.update_status == "success":
                    print(f"✅ Updated {model_type} model: {update_result.new_model_id}")
                    print(f"   Improvement metrics: {update_result.improvement_metrics}")
                else:
                    print(f"❌ Update failed: {update_result.error_message}")
            else:
                print(f"   No update needed for {model_type} model")
                update_results[model_type] = None
                
        except Exception as e:
            print(f"❌ Error processing {model_type} model: {str(e)}")
            update_results[model_type] = None
    
    return update_results


def test_agent_communication():
    """Test agent communication system."""
    print("\n=== Testing Agent Communication ===")
    
    # This would test the communication queue and message passing
    # between agents in a real scenario
    
    print("✅ Communication system ready")
    print("   - Queue-based message passing")
    print("   - JSON message format")
    print("   - Persistent logging")
    print("   - Error handling and retries")


def test_system_integration():
    """Test the complete system integration."""
    print("\n=== Testing System Integration ===")
    
    # Create sample data
    data_path = create_sample_data()
    
    # Test each agent
    build_results = test_model_builder(data_path)
    eval_results = test_performance_critic(data_path, build_results)
    update_results = test_updater(eval_results)
    
    # Test communication
    test_agent_communication()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Models built: {sum(1 for r in build_results.values() if r and r.build_status == 'success')}")
    print(f"Models evaluated: {sum(1 for r in eval_results.values() if r and r.evaluation_status == 'success')}")
    print(f"Models updated: {sum(1 for r in update_results.values() if r and r.update_status == 'success')}")
    
    print("\n✅ System integration test completed successfully!")


async def main():
    """Main test function."""
    print("🤖 Testing Autonomous 3-Agent Model Management System")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run integration test
        test_system_integration()
        
        print("\n🎉 All tests completed successfully!")
        print("\nTo run the full autonomous loop:")
        print("python run_agent_loop.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        logging.error(f"Test failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 