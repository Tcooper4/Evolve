"""
Test Agent Controller

This module tests the agent controller functionality and its integration
with the prompt router system.
"""

import asyncio
import logging
from typing import Dict, Any

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


async def test_workflow_execution():
    """Test workflow execution with different parameters."""
    logger.info("Testing workflow execution...")

    try:
        from agents.agent_controller import get_agent_controller

        controller = get_agent_controller()

        # Test different model types
        model_types = ["lstm", "xgboost", "random_forest"]
        
        for model_type in model_types:
            logger.info(f"Testing {model_type} model workflow...")
            
            result = await controller.execute_workflow(
                "builder",
                model_type=model_type,
                data_path="data/sample_data.csv",
                target_column="close",
                test_mode=True
            )
            
            logger.info(f"✅ {model_type} workflow result: {result.success}")
            if not result.success:
                logger.info(f"   Error: {result.error_message}")

        logger.info("✅ Workflow execution tests completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Workflow execution test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling in agent controller."""
    logger.info("Testing error handling...")

    try:
        from agents.agent_controller import get_agent_controller

        controller = get_agent_controller()

        # Test with invalid workflow type
        logger.info("Testing invalid workflow type...")
        result = await controller.execute_workflow(
            "invalid_workflow",
            model_type="lstm"
        )
        logger.info(f"✅ Invalid workflow handled: {not result.success}")

        # Test with missing parameters
        logger.info("Testing missing parameters...")
        result = await controller.execute_workflow("builder")
        logger.info(f"✅ Missing parameters handled: {not result.success}")

        # Test with invalid data path
        logger.info("Testing invalid data path...")
        result = await controller.execute_workflow(
            "builder",
            model_type="lstm",
            data_path="invalid/path.csv"
        )
        logger.info(f"✅ Invalid data path handled: {not result.success}")

        logger.info("✅ Error handling tests completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting Agent Controller Test Suite")
    logger.info("=" * 50)

    test_results = []

    # Run all tests
    tests = [
        ("Agent Controller", test_agent_controller),
        ("Prompt Router Integration", test_prompt_router_integration),
        ("Workflow Execution", test_workflow_execution),
        ("Error Handling", test_error_handling)
    ]

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            test_results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    asyncio.run(main()) 