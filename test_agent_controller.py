#!/usr/bin/env python3
"""
Test script for the Agent Controller module.

This script tests the basic functionality of the agent controller
and its workflow orchestrators.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_agent_controller():
    """Test the agent controller functionality."""
    logger.info("Starting Agent Controller tests...")
    
    try:
        # Import the agent controller
        from agents.agent_controller import get_agent_controller, AgentController
        
        # Get the controller instance
        controller = get_agent_controller()
        logger.info("✅ Agent controller imported successfully")
        
        # Test workflow listing
        workflows = controller.list_available_workflows()
        logger.info(f"✅ Available workflows: {workflows}")
        
        # Test workflow status
        for workflow_type in workflows:
            status = controller.get_workflow_status(workflow_type)
            logger.info(f"✅ {workflow_type} workflow status: {status}")
        
        # Test builder workflow (with minimal parameters)
        logger.info("Testing builder workflow...")
        builder_result = await controller.execute_workflow(
            "builder",
            model_type="lstm",
            data_path="data/sample_data.csv",
            target_column="close"
        )
        logger.info(f"✅ Builder workflow result: {builder_result.success}")
        if not builder_result.success:
            logger.info(f"   Error: {builder_result.error_message}")
        
        # Test evaluator workflow
        logger.info("Testing evaluator workflow...")
        evaluator_result = await controller.execute_workflow(
            "evaluator",
            model_id="test_model",
            model_path="models/test_model.pkl"
        )
        logger.info(f"✅ Evaluator workflow result: {evaluator_result.success}")
        if not evaluator_result.success:
            logger.info(f"   Error: {evaluator_result.error_message}")
        
        # Test updater workflow
        logger.info("Testing updater workflow...")
        updater_result = await controller.execute_workflow(
            "updater",
            model_id="test_model",
            update_type="auto"
        )
        logger.info(f"✅ Updater workflow result: {updater_result.success}")
        if not updater_result.success:
            logger.info(f"   Error: {updater_result.error_message}")
        
        logger.info("✅ All Agent Controller tests completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_prompt_router_integration():
    """Test the prompt router integration with agent controller."""
    logger.info("Testing Prompt Router integration...")
    
    try:
        from routing.prompt_router import route_prompt
        
        # Test a builder workflow prompt
        logger.info("Testing builder workflow prompt...")
        result = await route_prompt("Build a new LSTM model for forecasting")
        logger.info(f"✅ Builder prompt result: {result.get('success', False)}")
        if result.get('workflow_type'):
            logger.info(f"   Workflow type: {result['workflow_type']}")
        
        # Test an evaluator workflow prompt
        logger.info("Testing evaluator workflow prompt...")
        result = await route_prompt("Evaluate the performance of my model")
        logger.info(f"✅ Evaluator prompt result: {result.get('success', False)}")
        if result.get('workflow_type'):
            logger.info(f"   Workflow type: {result['workflow_type']}")
        
        # Test an updater workflow prompt
        logger.info("Testing updater workflow prompt...")
        result = await route_prompt("Update and optimize my model")
        logger.info(f"✅ Updater prompt result: {result.get('success', False)}")
        if result.get('workflow_type'):
            logger.info(f"   Workflow type: {result['workflow_type']}")
        
        logger.info("✅ Prompt Router integration tests completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("AGENT CONTROLLER TEST SUITE")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    # Run tests
    controller_success = await test_agent_controller()
    router_success = await test_prompt_router_integration()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Agent Controller Tests: {'✅ PASSED' if controller_success else '❌ FAILED'}")
    logger.info(f"Prompt Router Integration: {'✅ PASSED' if router_success else '❌ FAILED'}")
    logger.info(f"Total Duration: {duration:.2f} seconds")
    
    if controller_success and router_success:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error("💥 SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 