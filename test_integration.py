#!/usr/bin/env python3
"""
Integration Test Script for Agentic Forecasting System

This script tests the integration of all modules to ensure they work together correctly.
"""

import sys
import logging
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_goal_status_integration() -> Dict[str, Any]:
    """Test goal status module integration."""
    logger.info("Testing Goal Status Integration...")
    
    try:
        from trading.memory.goals.status import get_status_summary, update_goal_progress, log_agent_contribution
        
        # Test getting status summary
        status = get_status_summary()
        logger.info(f"✅ Goal status retrieved: {status['current_status']}")
        
        # Test updating progress
        update_goal_progress(0.5, status="on_track", message="Integration test")
        logger.info("✅ Goal progress updated")
        
        # Test logging agent contribution
        log_agent_contribution("TestAgent", "Integration test completed", "high")
        logger.info("✅ Agent contribution logged")
        
        return {"success": True, "message": "Goal status integration working"}
        
    except Exception as e:
        logger.error(f"❌ Goal status integration failed: {str(e)}")
        return {"success": False, "error": str(e)}

def test_optimizer_consolidation() -> Dict[str, Any]:
    """Test optimizer consolidation module."""
    logger.info("Testing Optimizer Consolidation...")
    
    try:
        from optimizers.consolidator import get_optimizer_status, OptimizerConsolidator
        
        # Test getting optimizer status
        status = get_optimizer_status()
        logger.info(f"✅ Optimizer status retrieved: {status['consolidation_needed']}")
        
        # Test creating consolidator instance
        consolidator = OptimizerConsolidator()
        logger.info("✅ Optimizer consolidator created")
        
        return {"success": True, "message": "Optimizer consolidation working"}
        
    except Exception as e:
        logger.error(f"❌ Optimizer consolidation failed: {str(e)}")
        return {"success": False, "error": str(e)}

def test_market_analysis() -> Dict[str, Any]:
    """Test market analysis module."""
    logger.info("Testing Market Analysis...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.analysis.market_analysis import MarketAnalysis
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(200, 300, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure valid price relationships
        sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + 10
        sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - 10
        
        # Test market analysis
        analyzer = MarketAnalysis()
        analysis = analyzer.analyze_market(sample_data)
        
        logger.info(f"✅ Market analysis completed: {analysis['regime'].name}")
        
        return {"success": True, "message": "Market analysis working"}
        
    except Exception as e:
        logger.error(f"❌ Market analysis failed: {str(e)}")
        return {"success": False, "error": str(e)}

def test_data_pipeline() -> Dict[str, Any]:
    """Test data pipeline module."""
    logger.info("Testing Data Pipeline...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.utils.data_pipeline import DataPipeline
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(200, 300, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure valid price relationships
        sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + 10
        sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - 10
        
        # Save sample data
        sample_file = "test_data.csv"
        sample_data.to_csv(sample_file)
        
        # Test data pipeline
        pipeline = DataPipeline()
        success = pipeline.run_pipeline(sample_file)
        
        if success:
            logger.info("✅ Data pipeline completed successfully")
            
            # Clean up
            import os
            if os.path.exists(sample_file):
                os.remove(sample_file)
            
            return {"success": True, "message": "Data pipeline working"}
        else:
            return {"success": False, "error": "Pipeline failed"}
        
    except Exception as e:
        logger.error(f"❌ Data pipeline failed: {str(e)}")
        return {"success": False, "error": str(e)}

def test_data_validation() -> Dict[str, Any]:
    """Test data validation module."""
    logger.info("Testing Data Validation...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.utils.data_validation import DataValidator, validate_data_for_training
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(200, 300, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure valid price relationships
        sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + 10
        sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - 10
        
        # Test data validation
        validator = DataValidator()
        is_valid, error_message = validator.validate_dataframe(sample_data)
        
        if is_valid:
            logger.info("✅ Data validation passed")
            
            # Test specific validation functions
            train_valid, train_summary = validate_data_for_training(sample_data)
            logger.info(f"✅ Training validation: {train_valid}")
            
            return {"success": True, "message": "Data validation working"}
        else:
            return {"success": False, "error": f"Validation failed: {error_message}"}
        
    except Exception as e:
        logger.error(f"❌ Data validation failed: {str(e)}")
        return {"success": False, "error": str(e)}

def run_all_tests() -> Dict[str, Any]:
    """Run all integration tests."""
    logger.info("🚀 Starting Integration Tests...")
    
    tests = [
        ("Goal Status", test_goal_status_integration),
        ("Optimizer Consolidation", test_optimizer_consolidation),
        ("Market Analysis", test_market_analysis),
        ("Data Pipeline", test_data_pipeline),
        ("Data Validation", test_data_validation)
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {test_name}")
        logger.info(f"{'='*50}")
        
        result = test_func()
        results[test_name] = result
        
        if result["success"]:
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            failed += 1
            logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Integration successful.")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed. Please check the errors above.")
    
    return {
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "success_rate": passed/len(tests)*100,
        "results": results
    }

def main():
    """Main function to run integration tests."""
    try:
        summary = run_all_tests()
        
        # Exit with appropriate code
        if summary["failed"] == 0:
            logger.info("✅ Integration tests completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Some integration tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Integration tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error during integration tests: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 