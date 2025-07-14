"""
Test Enhancements

Comprehensive tests for the enhanced components:
- StateManager (version headers, thread safety, memory management)
- DashboardRunner (port selection, error handling, refresh control)
- StrategyExecutor (queue management, timeouts, logging)
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced components
try:
    from trading.memory.state_manager import StateManager, StateVersion, get_state_manager
    from scripts.run_live_dashboard import DashboardRunner
    from trading.signals.strategy_executor import StrategyExecutor, TaskStatus, submit_strategy_task
    ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced components not available: {e}")
    ENHANCEMENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TestEnhancements:
    """Test suite for enhanced components."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.test_results = []
        self.temp_dir = None
        self.logger = logging.getLogger(f"{__name__}.TestEnhancements")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    
    async def test_state_manager_enhancements(self):
        """Test StateManager enhancements."""
        self.logger.info("🧪 Testing StateManager Enhancements...")
        
        try:
            # Create state manager with temp file
            state_file = os.path.join(self.temp_dir, "test_state.pkl")
            state_manager = StateManager(state_file)
            
            # Test version header
            version = state_manager.get_version()
            self.logger.info(f"State version: {version.version}")
            assert version.version == StateVersion.CURRENT_VERSION, "Version mismatch"
            
            # Test thread-safe operations
            def worker_thread(thread_id: int):
                for i in range(10):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"value_{thread_id}_{i}_{time.time()}"
                    state_manager.set(key, value)
                    time.sleep(0.01)
            
            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for threads to complete
            for thread in threads:
                thread.join()
            
            # Verify data integrity
            total_keys = len(state_manager.keys())
            self.logger.info(f"Total keys after concurrent writes: {total_keys}")
            assert total_keys == 30, f"Expected 30 keys, got {total_keys}"
            
            # Test memory management
            # Add large data to trigger cleanup
            large_data = {"data": "x" * 10000, "timestamp": time.time()}
            for i in range(100):
                state_manager.set(f"large_key_{i}", large_data)
            
            # Get memory usage
            memory_usage = state_manager.get_memory_usage()
            self.logger.info(f"Memory usage: {memory_usage}")
            
            # Test manual compression
            state_manager.compress_state()
            
            # Test cleanup
            removed_count = state_manager.cleanup_old_data(max_age_hours=0)
            self.logger.info(f"Removed {removed_count} old data items")
            
            # Test save and reload
            state_manager.save()
            
            # Create new instance and load
            new_state_manager = StateManager(state_file)
            new_keys = len(new_state_manager.keys())
            self.logger.info(f"Keys after reload: {new_keys}")
            
            # Test stats
            stats = state_manager.get_stats()
            self.logger.info(f"State stats: {stats}")
            
            self.logger.info("✅ StateManager enhancements tests completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ StateManager enhancements test failed: {e}")
            return False
    
    async def test_dashboard_runner_enhancements(self):
        """Test DashboardRunner enhancements."""
        self.logger.info("🧪 Testing DashboardRunner Enhancements...")
        
        try:
            # Test dependency checking
            runner = DashboardRunner(port=8502)
            
            # Test port validation
            valid_ports = [8501, 8080, 3000]
            invalid_ports = [0, 1023, 65536, 99999]
            
            for port in valid_ports:
                runner.port = port
                assert runner._validate_port(), f"Port {port} should be valid"
            
            for port in invalid_ports:
                runner.port = port
                assert not runner._validate_port(), f"Port {port} should be invalid"
            
            # Test app file checking
            # Create a temporary app.py file
            app_file = Path("app.py")
            if not app_file.exists():
                with open(app_file, 'w') as f:
                    f.write("import streamlit as st\nst.write('Test app')")
            
            assert runner._check_app_file(), "App file check should pass"
            
            # Test config file creation
            config_file = os.path.join(self.temp_dir, "streamlit_config.toml")
            runner.config_file = config_file
            runner._create_config_file()
            
            assert os.path.exists(config_file), "Config file should be created"
            
            # Test command building
            cmd = runner._get_streamlit_command()
            self.logger.info(f"Streamlit command: {' '.join(cmd)}")
            assert "--server.port" in cmd, "Command should include port"
            assert "--server.address" in cmd, "Command should include address"
            
            # Test status
            status = runner.get_status()
            self.logger.info(f"Runner status: {status}")
            assert status['running'] == False, "Should not be running initially"
            
            # Clean up test app file
            if app_file.exists() and app_file.stat().st_size < 100:
                app_file.unlink()
            
            self.logger.info("✅ DashboardRunner enhancements tests completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ DashboardRunner enhancements test failed: {e}")
            return False
    
    async def test_strategy_executor_enhancements(self):
        """Test StrategyExecutor enhancements."""
        self.logger.info("🧪 Testing StrategyExecutor Enhancements...")
        
        try:
            # Create executor with small limits for testing
            executor = StrategyExecutor(
                max_queue_size=5,
                max_concurrent_tasks=2,
                default_timeout=5.0
            )
            
            # Define test strategies
            async def successful_strategy():
                await asyncio.sleep(1)
                return "success"
            
            async def failing_strategy():
                await asyncio.sleep(0.5)
                raise ValueError("Intentional failure")
            
            async def timeout_strategy():
                await asyncio.sleep(10)  # Longer than timeout
                return "should timeout"
            
            # Start executor
            executor_task = asyncio.create_task(executor.start())
            
            # Submit tasks
            task_ids = []
            
            # Submit successful tasks
            for i in range(3):
                task_id = await executor.submit_task(f"success_{i}", successful_strategy)
                if task_id:
                    task_ids.append(task_id)
            
            # Submit failing task
            fail_task_id = await executor.submit_task("fail", failing_strategy)
            if fail_task_id:
                task_ids.append(fail_task_id)
            
            # Submit timeout task
            timeout_task_id = await executor.submit_task("timeout", timeout_strategy)
            if timeout_task_id:
                task_ids.append(timeout_task_id)
            
            # Test queue overflow
            for i in range(10):
                task_id = await executor.submit_task(f"overflow_{i}", successful_strategy)
                if task_id is None:
                    self.logger.info(f"Queue overflow detected at task {i}")
                    break
            
            # Wait for execution
            await asyncio.sleep(8)
            
            # Check task statuses
            for task_id in task_ids:
                status = executor.get_task_status(task_id)
                if status:
                    self.logger.info(f"Task {task_id}: {status.status}")
            
            # Get metrics
            metrics = executor.get_metrics()
            self.logger.info(f"Executor metrics: {metrics}")
            
            # Get failed tasks summary
            failed_summary = executor.get_failed_tasks_summary()
            self.logger.info(f"Failed tasks summary: {failed_summary}")
            
            # Test memory cleanup
            executor.clear_completed_tasks(max_age_hours=0)
            
            # Stop executor
            await executor.stop()
            executor_task.cancel()
            
            self.logger.info("✅ StrategyExecutor enhancements tests completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ StrategyExecutor enhancements test failed: {e}")
            return False
    
    async def test_integration_workflow(self):
        """Test integration between enhanced components."""
        self.logger.info("🧪 Testing Integration Workflow...")
        
        try:
            # Create components
            state_manager = StateManager(os.path.join(self.temp_dir, "integration_state.pkl"))
            executor = StrategyExecutor(max_queue_size=3, max_concurrent_tasks=1)
            
            # Start executor
            executor_task = asyncio.create_task(executor.start())
            
            # Define strategy that uses state manager
            async def state_aware_strategy():
                # Read from state
                value = state_manager.get("test_key", "default")
                
                # Process
                result = f"processed_{value}_{time.time()}"
                
                # Write to state
                state_manager.set("result_key", result)
                
                return result
            
            # Submit task
            task_id = await executor.submit_task("state_aware", state_aware_strategy)
            
            # Set initial state
            state_manager.set("test_key", "initial_value")
            
            # Wait for execution
            await asyncio.sleep(3)
            
            # Check results
            result = state_manager.get("result_key")
            self.logger.info(f"Strategy result: {result}")
            
            # Check task status
            status = executor.get_task_status(task_id)
            self.logger.info(f"Task status: {status.status if status else 'unknown'}")
            
            # Stop executor
            await executor.stop()
            executor_task.cancel()
            
            self.logger.info("✅ Integration workflow tests completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Integration workflow test failed: {e}")
            return False
    
    async def test_error_handling(self):
        """Test error handling and resilience."""
        self.logger.info("🧪 Testing Error Handling...")
        
        try:
            # Test StateManager with invalid file
            try:
                invalid_state = StateManager("/invalid/path/state.pkl")
                # Should handle gracefully
                invalid_state.set("test", "value")
                self.logger.info("✅ StateManager handles invalid paths gracefully")
            except Exception as e:
                self.logger.warning(f"StateManager error handling: {e}")
            
            # Test StrategyExecutor with invalid strategy
            executor = StrategyExecutor(max_queue_size=2, max_concurrent_tasks=1)
            executor_task = asyncio.create_task(executor.start())
            
            async def invalid_strategy():
                # This will cause an error
                return 1 / 0
            
            task_id = await executor.submit_task("invalid", invalid_strategy)
            await asyncio.sleep(2)
            
            # Check that error was handled
            status = executor.get_task_status(task_id)
            if status and status.status == TaskStatus.FAILED:
                self.logger.info("✅ StrategyExecutor handles errors gracefully")
            
            await executor.stop()
            executor_task.cancel()
            
            self.logger.info("✅ Error handling tests completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error handling test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all enhancement tests."""
        self.logger.info("🚀 Starting Enhancement Tests")
        self.logger.info("=" * 60)
        
        if not ENHANCEMENTS_AVAILABLE:
            self.logger.error("❌ Enhanced components not available - skipping tests")
            return False
        
        self.setup()
        
        try:
            test_methods = [
                self.test_state_manager_enhancements,
                self.test_dashboard_runner_enhancements,
                self.test_strategy_executor_enhancements,
                self.test_integration_workflow,
                self.test_error_handling
            ]
            
            passed = 0
            total = len(test_methods)
            
            for test_method in test_methods:
                self.logger.info(f"\n{'='*20} {test_method.__name__} {'='*20}")
                try:
                    result = await test_method()
                    if result:
                        passed += 1
                        self.logger.info(f"✅ {test_method.__name__} PASSED")
                    else:
                        self.logger.error(f"❌ {test_method.__name__} FAILED")
                except Exception as e:
                    self.logger.error(f"❌ {test_method.__name__} ERROR: {e}")
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📊 Test Results: {passed}/{total} tests passed")
            
            if passed == total:
                self.logger.info("🎉 All enhancement tests passed!")
            else:
                self.logger.warning(f"⚠️ {total - passed} tests failed")
            
            return passed == total
            
        finally:
            self.teardown()


async def main():
    """Main test runner."""
    tester = TestEnhancements()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Enhancement Tests: ALL PASSED")
        return 0
    else:
        print("\n❌ Enhancement Tests: SOME FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 