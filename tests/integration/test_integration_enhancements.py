"""
Integration Enhancement Tests

Comprehensive tests for the new integration components:
- MetaAgentManager
- ModelRegistry
- ServiceMesh
- DataPipelineOrchestrator
"""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import integration components
try:
    from trading.integration.meta_agent_manager import MetaAgentManager, MetaAgentStatus
    from trading.integration.model_registry import (
        ModelRegistry,
        TaskType,
    )
    from trading.integration.service_mesh import RequestType, ServiceMesh

    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"âš ï¸ Integration components not available: {e}")
    INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class TestIntegrationEnhancements:
    """Test suite for integration enhancements."""

    def __init__(self):
        """Initialize the test suite."""
        self.test_results = []
        self.temp_dir = None
        self.logger = logging.getLogger(f"{__name__}.TestIntegrationEnhancements")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Test environment setup in: {self.temp_dir}")

    def teardown(self):
        """Cleanup test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)
            self.logger.info("Test environment cleaned up")

    async def test_meta_agent_manager(self):
        """Test MetaAgentManager functionality."""
        self.logger.info("ðŸ§ª Testing MetaAgentManager...")

        try:
            # Create MetaAgentManager
            manager = MetaAgentManager()

            # Test agent discovery
            available_agents = manager.get_available_agents()
            self.logger.info(f"Available agents: {available_agents}")

            # Test agent status
            statuses = manager.get_all_agent_statuses()
            self.logger.info(f"Agent statuses: {statuses}")

            # Test health check
            health = await manager.health_check()
            self.logger.info(f"Health status: {health['overall_health']}")

            # Test running all agents (if any available)
            if available_agents:
                results = await manager.run_all_meta_agents()
                self.logger.info(f"Executed {len(results)} agents")

                for result in results:
                    self.logger.info(f"Agent {result.agent_name}: {result.status}")

            self.logger.info("âœ… MetaAgentManager tests completed")
            return True

        except Exception as e:
            self.logger.error(f"âŒ MetaAgentManager test failed: {e}")
            return False

    async def test_model_registry(self):
        """Test ModelRegistry functionality."""
        self.logger.info("ðŸ§ª Testing ModelRegistry...")

        try:
            # Create ModelRegistry with temp directory
            registry_path = os.path.join(self.temp_dir, "model_registry")
            registry = ModelRegistry(registry_path)

            # Test model registration
            class TestModel:
                def __init__(self, **kwargs):
                    self.name = "TestModel"

            success = registry.register_model(
                name="test_lstm",
                model_class=TestModel,
                task_type=TaskType.FORECASTING,
                description="Test LSTM model",
                tags=["test", "lstm"],
            )

            assert success, "Model registration failed"
            self.logger.info("âœ… Model registration successful")

            # Test performance tracking
            success = registry.track_performance(
                "test_lstm",
                TaskType.FORECASTING,
                sharpe_ratio=1.2,
                max_drawdown=0.15,
                win_rate=0.65,
            )

            assert success, "Performance tracking failed"
            self.logger.info("âœ… Performance tracking successful")

            # Test getting best model
            best_model = registry.get_best_model(TaskType.FORECASTING)
            self.logger.info(f"Best forecasting model: {best_model}")

            # Test model info
            model_info = registry.get_model_info("test_lstm")
            assert model_info is not None, "Model info retrieval failed"
            self.logger.info("âœ… Model info retrieval successful")

            # Test registry summary
            summary = registry.get_registry_summary()
            self.logger.info(f"Registry summary: {summary}")

            # Test health check
            health = await registry.health_check()
            self.logger.info(f"Registry health: {health['status']}")

            self.logger.info("âœ… ModelRegistry tests completed")
            return True

        except Exception as e:
            self.logger.error(f"âŒ ModelRegistry test failed: {e}")
            return False

    async def test_service_mesh(self):
        """Test ServiceMesh functionality."""
        self.logger.info("ðŸ§ª Testing ServiceMesh...")

        try:
            # Create ServiceMesh
            mesh = ServiceMesh()

            # Test service registration
            success = await mesh.register_service(
                service_name="test_forecast_service",
                service_type="forecasting",
                endpoint="http://localhost:8001",
                capabilities=["forecast", "model"],
            )

            assert success, "Service registration failed"
            self.logger.info("âœ… Service registration successful")

            # Test service listing
            services = mesh.list_services()
            self.logger.info(f"Registered services: {services}")

            # Test service info
            service_info = mesh.get_service_info("test_forecast_service")
            assert service_info is not None, "Service info retrieval failed"
            self.logger.info("âœ… Service info retrieval successful")

            # Test health status
            health = await mesh.get_service_health()
            self.logger.info(f"Service health: {health['overall_health']}")

            # Test request routing (will fail since no real service, but tests the flow)
            try:
                response = await mesh.route_request(
                    RequestType.FORECAST, {"symbol": "AAPL", "horizon": 7}
                )
                self.logger.info(f"Request routing result: {response.status}")
            except Exception as e:
                self.logger.info(f"Request routing failed as expected: {e}")

            # Test service unregistration
            success = await mesh.unregister_service("test_forecast_service")
            assert success, "Service unregistration failed"
            self.logger.info("âœ… Service unregistration successful")

            self.logger.info("âœ… ServiceMesh tests completed")
            return True

        except Exception as e:
            self.logger.error(f"âŒ ServiceMesh test failed: {e}")
            return False

    async def test_integration_workflow(self):
        """Test integration workflow between components."""
        self.logger.info("ðŸ§ª Testing Integration Workflow...")

        try:
            # Create all components
            meta_manager = MetaAgentManager()
            registry = ModelRegistry(os.path.join(self.temp_dir, "workflow_registry"))
            mesh = ServiceMesh()

            # Test component interaction
            self.logger.info("Testing component interaction...")

            # 1. Register models in registry
            class WorkflowModel:
                def __init__(self, **kwargs):
                    self.name = "WorkflowModel"

            registry.register_model(
                name="workflow_lstm",
                model_class=WorkflowModel,
                task_type=TaskType.FORECASTING,
                description="Workflow test model",
            )

            # 2. Track performance
            registry.track_performance(
                "workflow_lstm",
                TaskType.FORECASTING,
                sharpe_ratio=1.5,
                max_drawdown=0.10,
            )

            # 3. Get best model
            best_model = registry.get_best_model(TaskType.FORECASTING)
            self.logger.info(f"Best model from registry: {best_model}")

            # 4. Register service in mesh
            await mesh.register_service(
                service_name="workflow_service",
                service_type="forecasting",
                endpoint="http://localhost:8002",
                capabilities=["forecast", "model"],
            )

            # 5. Check service health
            health = await mesh.get_service_health()
            self.logger.info(f"Service health: {health['overall_health']}")

            # 6. Run meta agents
            if meta_manager.get_available_agents():
                results = await meta_manager.run_all_meta_agents()
                self.logger.info(f"Meta agents executed: {len(results)}")

            self.logger.info("âœ… Integration workflow tests completed")
            return True

        except Exception as e:
            self.logger.error(f"âŒ Integration workflow test failed: {e}")
            return False

    async def test_error_handling(self):
        """Test error handling and resilience."""
        self.logger.info("ðŸ§ª Testing Error Handling...")

        try:
            # Test MetaAgentManager with non-existent agent
            manager = MetaAgentManager()
            result = await manager.execute_agent("non_existent_agent")
            assert (
                result.status == MetaAgentStatus.ERROR
            ), "Should handle non-existent agent"
            self.logger.info("âœ… MetaAgentManager error handling works")

            # Test ModelRegistry with non-existent model
            registry = ModelRegistry()
            model_info = registry.get_model_info("non_existent_model")
            assert model_info is None, "Should handle non-existent model"
            self.logger.info("âœ… ModelRegistry error handling works")

            # Test ServiceMesh with non-existent service
            mesh = ServiceMesh()
            service_info = mesh.get_service_info("non_existent_service")
            assert service_info is None, "Should handle non-existent service"
            self.logger.info("âœ… ServiceMesh error handling works")

            self.logger.info("âœ… Error handling tests completed")
            return True

        except Exception as e:
            self.logger.error(f"âŒ Error handling test failed: {e}")
            return False

    async def test_performance_metrics(self):
        """Test performance metrics and monitoring."""
        self.logger.info("ðŸ§ª Testing Performance Metrics...")

        try:
            # Test ModelRegistry performance tracking
            registry = ModelRegistry(os.path.join(self.temp_dir, "perf_registry"))

            class PerfModel:
                def __init__(self, **kwargs):
                    self.name = "PerfModel"

            # Register model
            registry.register_model(
                name="perf_model", model_class=PerfModel, task_type=TaskType.FORECASTING
            )

            # Track multiple performance entries
            for i in range(5):
                registry.track_performance(
                    "perf_model",
                    TaskType.FORECASTING,
                    sharpe_ratio=1.0 + i * 0.1,
                    max_drawdown=0.15 - i * 0.01,
                    win_rate=0.60 + i * 0.02,
                )

            # Get performance history
            history = registry.get_performance_history("perf_model")
            assert (
                len(history) == 5
            ), f"Expected 5 performance entries, got {len(history)}"
            self.logger.info(
                f"âœ… Performance history tracking works: {len(history)} entries"
            )

            # Test health checks
            meta_health = await MetaAgentManager().health_check()
            registry_health = await registry.health_check()

            self.logger.info(f"MetaAgent health: {meta_health['status']}")
            self.logger.info(f"Registry health: {registry_health['status']}")

            self.logger.info("âœ… Performance metrics tests completed")
            return True

        except Exception as e:
            self.logger.error(f"âŒ Performance metrics test failed: {e}")
            return False

    async def run_all_tests(self):
        """Run all integration tests."""
        self.logger.info("ðŸš€ Starting Integration Enhancement Tests")
        self.logger.info("=" * 60)

        if not INTEGRATION_AVAILABLE:
            self.logger.error(
                "âŒ Integration components not available - skipping tests"
            )
            return False

        self.setup()

        try:
            test_methods = [
                self.test_meta_agent_manager,
                self.test_model_registry,
                self.test_service_mesh,
                self.test_integration_workflow,
                self.test_error_handling,
                self.test_performance_metrics,
            ]

            passed = 0
            total = len(test_methods)

            for test_method in test_methods:
                self.logger.info(f"\n{'=' * 20} {test_method.__name__} {'=' * 20}")
                try:
                    result = await test_method()
                    if result:
                        passed += 1
                        self.logger.info(f"âœ… {test_method.__name__} PASSED")
                    else:
                        self.logger.error(f"âŒ {test_method.__name__} FAILED")
                except Exception as e:
                    self.logger.error(f"âŒ {test_method.__name__} ERROR: {e}")

            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"ðŸ“Š Test Results: {passed}/{total} tests passed")

            if passed == total:
                self.logger.info("ðŸŽ‰ All integration enhancement tests passed!")
            else:
                self.logger.warning(f"âš ï¸ {total - passed} tests failed")

            return passed == total

        finally:
            self.teardown()


async def main():
    """Main test runner."""
    tester = TestIntegrationEnhancements()
    success = await tester.run_all_tests()

    if success:
        print("\nðŸŽ‰ Integration Enhancement Tests: ALL PASSED")
        return 0
    else:
        print("\nâŒ Integration Enhancement Tests: SOME FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
