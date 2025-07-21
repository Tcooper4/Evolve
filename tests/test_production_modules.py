"""
Comprehensive tests for productionized modules.

This script tests the hardened and productionized versions of:
- notification_service.py
- agent_memory_manager.py
- lstm_model.py
- prophet_model.py
- agent.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestNotificationService:
    """Test the productionized notification service."""

    def test_message_sending_consolidation(self):
        """Test the consolidated send_message function."""
        from system.infra.agents.services.notification_service import (
            MessageChannel,
            MessageConfig,
            NotificationService,
            SendMessageResult,
        )

        # Mock services for testing
        class MockNotificationManager:
            async def create_notification(self, **kwargs):
                return type("MockNotification", (), {"id": "test-123"})()

        class MockRateLimitingService:
            async def check_rate_limit(self, *args):
                return True

        class MockTemplateService:
            async def get_template(self, template_id):
                return type("MockTemplate", (), {"id": template_id})()

        class MockCacheService:
            async def get(self, key):
                return None

        class MockMetricsService:
            def record_notification_sent(self, notification_id):
                pass

        class MockRetryService:
            async def execute_with_retry(self, func, **kwargs):
                return await func()

        class MockErrorHandlingService:
            async def handle_error(self, error):
                pass

        class MockConfigService:
            def get_config(self, key):
                return {}

        class MockAuditService:
            async def record_audit(self, *args):
                pass

        class MockHealthService:
            async def check_health(self):
                return {"status": "healthy"}

        class MockTransactionService:
            async def begin_transaction(self):
                return type("MockTransaction", (), {"commit": lambda: None})()

        class MockValidationService:
            async def validate(self, data):
                return True

        class MockSecurityService:
            async def validate_security(self, data):
                return True

        class MockLoggingService:
            def log(self, message):
                pass

        class MockMonitoringService:
            def record_metric(self, name, value):
                pass

        class MockSchedulerService:
            def schedule_task(self, task, run_at):
                pass

        class MockQueueService:
            async def enqueue(self, task):
                pass

        class MockWorkerService:
            async def process_task(self, task):
                pass

        class MockPersistenceService:
            async def save_delivery(self, delivery):
                pass

        class MockDependencyService:
            async def check_dependencies(self):
                return True

        class MockRecoveryService:
            async def recover(self):
                pass

        class MockStateService:
            async def get_state(self):
                return {}

        # Create notification service with mocked dependencies
        notification_service = NotificationService(
            notification_manager=MockNotificationManager(),
            rate_limiting_service=MockRateLimitingService(),
            template_service=MockTemplateService(),
            cache_service=MockCacheService(),
            metrics_service=MockMetricsService(),
            retry_service=MockRetryService(),
            error_handling_service=MockErrorHandlingService(),
            config_service=MockConfigService(),
            audit_service=MockAuditService(),
            health_service=MockHealthService(),
            transaction_service=MockTransactionService(),
            validation_service=MockValidationService(),
            security_service=MockSecurityService(),
            logging_service=MockLoggingService(),
            monitoring_service=MockMonitoringService(),
            scheduler_service=MockSchedulerService(),
            queue_service=MockQueueService(),
            worker_service=MockWorkerService(),
            persistence_service=MockPersistenceService(),
            dependency_service=MockDependencyService(),
            recovery_service=MockRecoveryService(),
            state_service=MockStateService(),
        )

        # Test message sending
        async def test_send_message():
            config = MessageConfig(
                channel=MessageChannel.EMAIL, timeout=5, max_retries=2, retry_delay=1.0
            )

            result = await notification_service.send_message(
                message="Test message",
                title="Test title",
                channel=MessageChannel.EMAIL,
                recipients=["test@example.com"],
                config=config,
                template_vars={"name": "Test User"},
            )

            assert isinstance(result, SendMessageResult)
            assert result.message_id is not None
            assert result.channel == MessageChannel.EMAIL

        # Run the test
        asyncio.run(test_send_message())
        logger.info("âœ… Notification service consolidation test passed")


class TestAgentMemoryManager:
    """Test the productionized agent memory manager."""

    def test_lru_cache_eviction(self):
        """Test LRU cache eviction logic."""
        from trading.memory.agent_memory_manager import ThreadSafeCache

        # Create cache with small size to trigger eviction
        cache = ThreadSafeCache(max_size=3, default_ttl=3600)

        # Add items
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add another item to trigger eviction
        cache.set("key4", "value4")

        # key2 should be evicted (least recently used)
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # New item

        logger.info("âœ… LRU cache eviction test passed")

    def test_memory_validation(self):
        """Test memory data validation."""
        from trading.memory.agent_memory_manager import AgentMemoryManager

        # Create memory manager
        memory_manager = AgentMemoryManager(fallback_storage="local")

        # Test valid data
        valid_data = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "agent_type": "test_agent",
                "prompt": "test prompt",
                "response": "test response",
                "confidence": 0.8,
                "success": True,
            }
        ]

        # Test invalid data (missing required field)
        invalid_data = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "agent_type": "test_agent",
                # Missing required fields
            }
        ]

        # Test validation
        assert (
            memory_manager._validate_memory_data(valid_data, "interaction", "test.json")

        )
        assert (
            memory_manager._validate_memory_data(
                invalid_data, "interaction", "test.json"
            )
            == False
        )

        logger.info("âœ… Memory validation test passed")

    def test_corrupted_file_handling(self):
        """Test handling of corrupted memory files."""
        import tempfile

        from trading.memory.agent_memory_manager import AgentMemoryManager

        memory_manager = AgentMemoryManager(fallback_storage="local")

        # Create a temporary corrupted file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            corrupted_file = f.name

        try:
            # Test corrupted file handling
            memory_manager._handle_corrupted_file(corrupted_file)

            # Check if backup was created
            backup_dir = Path(corrupted_file).parent / "corrupted_backups"
            assert backup_dir.exists()

            # Check if original file was moved
            assert not Path(corrupted_file).exists()

        finally:
            # Cleanup
            if Path(corrupted_file).exists():
                Path(corrupted_file).unlink()
            if backup_dir.exists():
                import shutil

                shutil.rmtree(backup_dir)

        logger.info("âœ… Corrupted file handling test passed")


class TestLSTMForecaster:
    """Test the productionized LSTM forecaster."""

    def test_data_normalization_check(self):
        """Test data normalization checking."""
        from trading.models.lstm_model import LSTMForecaster

        # Create LSTM forecaster
        config = {
            "input_size": 5,
            "hidden_size": 50,
            "num_layers": 2,
            "dropout": 0.1,
            "sequence_length": 10,
            "use_batch_norm": True,
            "use_layer_norm": False,
            "additional_dropout": 0.1,
        }

        forecaster = LSTMForecaster(config)

        # Test normalized data
        normalized_data = pd.DataFrame(np.random.randn(100, 5))  # Mean ~0, std ~1
        assert forecaster._check_data_normalization(normalized_data)

        # Test unnormalized data
        unnormalized_data = pd.DataFrame(np.random.rand(100, 5) * 1000)  # Large values
        assert forecaster._check_data_normalization(unnormalized_data) == False

        logger.info("âœ… Data normalization check test passed")

    def test_input_validation(self):
        """Test input validation with NaN handling."""
        from trading.models.lstm_model import LSTMForecaster

        config = {
            "input_size": 3,
            "hidden_size": 20,
            "num_layers": 1,
            "dropout": 0.1,
            "sequence_length": 5,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "additional_dropout": 0.0,
        }

        forecaster = LSTMForecaster(config)

        # Create data with NaNs
        data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [1, 2, 3, np.nan, 5],
                "feature3": [1, 2, 3, 4, 5],
            }
        )
        target = pd.Series([1, 2, 3, 4, 5])

        # Test that validation raises error for insufficient data after cleaning
        with pytest.raises(ValueError, match="At least 20 data points are required"):
            forecaster.fit(data, target, epochs=1)

        logger.info("âœ… Input validation test passed")

    def test_batch_wise_prediction(self):
        """Test batch-wise prediction to reduce memory usage."""
        from trading.models.lstm_model import LSTMForecaster

        config = {
            "input_size": 3,
            "hidden_size": 20,
            "num_layers": 1,
            "dropout": 0.1,
            "sequence_length": 5,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "additional_dropout": 0.0,
        }

        forecaster = LSTMForecaster(config)

        # Create larger dataset
        data = pd.DataFrame(np.random.randn(100, 3))

        # Test batch-wise prediction
        predictions = forecaster.predict(data, batch_size=10)
        assert len(predictions) > 0

        logger.info("âœ… Batch-wise prediction test passed")


class TestProphetForecaster:
    """Test the productionized Prophet forecaster."""

    def test_dynamic_forecast_horizon(self):
        """Test dynamic forecast horizon calculation."""
        from utils.time_utils import calculate_forecast_horizon

        # Create test data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = pd.DataFrame({"date": dates, "value": np.random.randn(100)})

        # Test dynamic horizon calculation
        horizon = calculate_forecast_horizon(data, "date")
        assert isinstance(horizon, int)
        assert horizon > 0
        assert horizon <= 365  # Max horizon

        logger.info("âœ… Dynamic forecast horizon test passed")

    def test_time_utilities(self):
        """Test time utility functions."""
        from utils.time_utils import (
            detect_seasonality,
            format_datetime_for_prophet,
            parse_datetime,
            validate_date_column,
        )

        # Test datetime parsing
        dt = parse_datetime("2024-01-01")
        assert isinstance(dt, datetime)

        # Test formatting for Prophet
        formatted = format_datetime_for_prophet(dt)
        assert isinstance(formatted, str)

        # Test date column validation
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "value": np.random.randn(10),
            }
        )
        assert validate_date_column(data, "date")
        assert validate_date_column(data, "nonexistent") == False

        # Test seasonality detection
        seasonality = detect_seasonality(data, "date", "value")
        assert isinstance(seasonality, dict)

        logger.info("âœ… Time utilities test passed")


class TestAgent:
    """Test the productionized agent."""

    def test_token_usage_estimation(self):
        """Test token usage estimation."""
        from agents.llm.agent import PromptAgent

        agent = PromptAgent()

        # Test token estimation
        prompt = "This is a test prompt for token estimation."
        estimate = agent.estimate_token_usage(prompt, model="gpt-4")

        assert "token_count" in estimate
        assert "estimated_cost" in estimate
        assert "model" in estimate
        assert estimate["model"] == "gpt-4"

        logger.info("âœ… Token usage estimation test passed")

    def test_prompt_sanitization(self):
        """Test prompt sanitization."""
        from agents.llm.agent import PromptAgent

        agent = PromptAgent()

        # Test injection pattern removal
        malicious_prompt = "Hello <script>alert('xss')</script> world"
        sanitized = agent.sanitize_prompt(malicious_prompt)

        assert "<script>" not in sanitized
        assert "Hello" in sanitized
        assert "world" in sanitized

        # Test length truncation
        long_prompt = "A" * 5000
        truncated = agent.sanitize_prompt(long_prompt, max_length=100)

        assert len(truncated) <= 103  # 100 + "..."
        assert truncated.endswith("...")

        logger.info("âœ… Prompt sanitization test passed")

    def test_log_batching(self):
        """Test log batching functionality."""
        from agents.llm.agent import PromptAgent

        agent = PromptAgent()

        # Add multiple log messages
        for i in range(5):
            agent.batch_log(f"Test message {i}")

        # Force flush
        agent._flush_log_buffer()

        # Check that buffer is empty
        assert len(agent.log_buffer) == 0

        logger.info("âœ… Log batching test passed")

    def test_token_usage_tracking(self):
        """Test token usage tracking."""
        from agents.llm.agent import PromptAgent

        agent = PromptAgent()

        # Update token usage
        agent.update_token_usage(1000, model="gpt-4")
        agent.update_token_usage(500, model="gpt-3.5-turbo")

        # Get stats
        stats = agent.get_token_usage_stats()

        assert stats["total_tokens"] == 1500
        assert stats["requests_count"] == 2
        assert stats["total_cost"] > 0

        logger.info("âœ… Token usage tracking test passed")


def run_all_tests():
    """Run all production module tests."""
    logger.info("ðŸš€ Starting production module tests...")

    # Test notification service
    logger.info("\nðŸ“§ Testing Notification Service...")
    test_notification = TestNotificationService()
    test_notification.test_message_sending_consolidation()

    # Test agent memory manager
    logger.info("\nðŸ§  Testing Agent Memory Manager...")
    test_memory = TestAgentMemoryManager()
    test_memory.test_lru_cache_eviction()
    test_memory.test_memory_validation()
    test_memory.test_corrupted_file_handling()

    # Test LSTM forecaster
    logger.info("\nðŸ”® Testing LSTM Forecaster...")
    test_lstm = TestLSTMForecaster()
    test_lstm.test_data_normalization_check()
    test_lstm.test_input_validation()
    test_lstm.test_batch_wise_prediction()

    # Test Prophet forecaster
    logger.info("\nðŸ“Š Testing Prophet Forecaster...")
    test_prophet = TestProphetForecaster()
    test_prophet.test_dynamic_forecast_horizon()
    test_prophet.test_time_utilities()

    # Test agent
    logger.info("\nðŸ¤– Testing Agent...")
    test_agent = TestAgent()
    test_agent.test_token_usage_estimation()
    test_agent.test_prompt_sanitization()
    test_agent.test_log_batching()
    test_agent.test_token_usage_tracking()

    logger.info("\nâœ… All production module tests completed successfully!")
    logger.info("\nðŸ“‹ Summary of productionization improvements:")
    logger.info(
        "â€¢ Notification Service: Consolidated send_message with timeout, retry, and error handling"
    )
    logger.info(
        "â€¢ Agent Memory Manager: LRU cache eviction, memory validation, corrupted file handling"
    )
    logger.info(
        "â€¢ LSTM Forecaster: Data normalization checks, input validation, batch-wise prediction"
    )
    logger.info(
        "â€¢ Prophet Forecaster: Dynamic forecast horizon, time utilities extraction"
    )
    logger.info("â€¢ Agent: Token usage estimation, prompt sanitization, log batching")


if __name__ == "__main__":
    run_all_tests()
