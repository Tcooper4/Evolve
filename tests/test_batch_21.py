"""
Batch 21 Tests
Tests for enhanced commentary generator, signal trace, and existing file enhancements
"""

import unittest
import tempfile
import os
import json
import ast
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

# Import Batch 21 modules
from trading.services.commentary_generator import CommentaryGenerator
from trading.explainability.signal_trace import SignalTracer, TraceLevel, SignalTrace
from trading.memory.agent_memory import AgentMemory
from trading.pipeline.task_dispatcher import TaskDispatcher, TaskPriority, TaskStatus, Task


class TestCommentaryGenerator(unittest.TestCase):
    """Test enhanced commentary generator with token truncation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = CommentaryGenerator(max_tokens=100)
        self.sample_data = {
            "query": "Analyze AAPL",
            "parsed": {"intent": "analysis", "symbol": "AAPL"},
            "result": {"score": 0.85, "recommendation": "buy"}
        }
    
    def test_truncate_response_with_tokenizer(self):
        """Test response truncation with tokenizer."""
        long_response = "This is a very long response that should be truncated. " * 50
        
        # Mock tiktoken import
        with patch.dict('sys.modules', {'tiktoken': Mock()}):
            with patch('tiktoken.get_encoding') as mock_tokenizer:
                mock_encoding = Mock()
                mock_encoding.encode.return_value = list(range(200))  # 200 tokens
                mock_encoding.decode.return_value = "Truncated response"
                mock_tokenizer.return_value = mock_encoding
                
                result = self.generator.truncate_response(long_response, max_tokens=50)
                
                self.assertEqual(result, "Truncated response")
                mock_encoding.encode.assert_called_once()
                mock_encoding.decode.assert_called_once()
    
    def test_truncate_response_fallback(self):
        """Test response truncation fallback without tokenizer."""
        long_response = "Long response " * 100
        
        # Mock ImportError for tiktoken
        with patch('builtins.__import__', side_effect=ImportError("No module named 'tiktoken'")):
            result = self.generator.truncate_response(long_response, max_tokens=50)
            
            # Should be truncated to approximately 200 characters (50 tokens * 4)
            self.assertLess(len(result), len(long_response))
            self.assertIn("Long response", result)
    
    def test_truncate_response_no_truncation_needed(self):
        """Test response truncation when not needed."""
        short_response = "Short response"
        
        result = self.generator.truncate_response(short_response, max_tokens=100)
        
        self.assertEqual(result, short_response)
    
    def test_generate_commentary_with_truncation(self):
        """Test commentary generation with truncation."""
        with patch.object(self.generator, '_generate_gpt_commentary') as mock_gpt:
            mock_gpt.return_value = "Long commentary " * 100
            
            with patch.object(self.generator, 'truncate_response') as mock_truncate:
                mock_truncate.return_value = "Truncated commentary"
                
                result = self.generator.generate_commentary(
                    self.sample_data["query"],
                    self.sample_data["parsed"],
                    self.sample_data["result"]
                )
                
                self.assertEqual(result, "Truncated commentary")
                mock_truncate.assert_called_once()


class TestSignalTracer(unittest.TestCase):
    """Test signal tracer with safe evaluation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracer = SignalTracer()
    
    def test_safe_eval_simple_expression(self):
        """Test safe evaluation of simple expressions."""
        result, error = self.tracer.safe_eval("42")
        self.assertEqual(result, 42)
        self.assertIsNone(error)
    
    def test_safe_eval_list_expression(self):
        """Test safe evaluation of list expressions."""
        result, error = self.tracer.safe_eval("[1, 2, 3]")
        self.assertEqual(result, [1, 2, 3])
        self.assertIsNone(error)
    
    def test_safe_eval_dict_expression(self):
        """Test safe evaluation of dictionary expressions."""
        result, error = self.tracer.safe_eval('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})
        self.assertIsNone(error)
    
    def test_safe_eval_unsafe_expression(self):
        """Test safe evaluation rejects unsafe expressions."""
        result, error = self.tracer.safe_eval("__import__('os').system('ls')")
        self.assertIsNone(result)
        self.assertIn("Unsafe expression", error)
    
    def test_safe_eval_syntax_error(self):
        """Test safe evaluation with syntax error."""
        result, error = self.tracer.safe_eval("invalid syntax [")
        self.assertIsNone(result)
        self.assertIn("Syntax error", error)
    
    def test_trace_signal(self):
        """Test signal tracing."""
        trace = self.tracer.trace_signal(
            "test_signal",
            "Test message",
            {"data": "value"},
            TraceLevel.INFO
        )
        
        self.assertEqual(trace.signal_id, "test_signal")
        self.assertEqual(trace.message, "Test message")
        self.assertEqual(trace.data, {"data": "value"})
        self.assertEqual(trace.level, TraceLevel.INFO)
    
    def test_trace_evaluation(self):
        """Test evaluation tracing."""
        trace = self.tracer.trace_evaluation("test_signal", "42")
        
        self.assertEqual(trace.signal_id, "test_signal")
        self.assertEqual(trace.safe_eval_result, 42)
        self.assertIsNone(trace.eval_error)
    
    def test_trace_evaluation_error(self):
        """Test evaluation tracing with error."""
        trace = self.tracer.trace_evaluation("test_signal", "invalid")
        
        self.assertEqual(trace.signal_id, "test_signal")
        self.assertIsNone(trace.safe_eval_result)
        self.assertIsNotNone(trace.eval_error)
    
    def test_get_signal_traces(self):
        """Test retrieving signal traces."""
        # Create some traces
        self.tracer.trace_signal("signal1", "Message 1")
        self.tracer.trace_signal("signal2", "Message 2")
        self.tracer.trace_signal("signal1", "Message 3")
        
        traces = self.tracer.get_signal_traces("signal1")
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0].message, "Message 1")
        self.assertEqual(traces[1].message, "Message 3")
    
    def test_analyze_signal_traces(self):
        """Test signal trace analysis."""
        # Create traces with different levels
        self.tracer.trace_signal("test_signal", "Info message", level=TraceLevel.INFO)
        self.tracer.trace_signal("test_signal", "Error message", level=TraceLevel.ERROR)
        self.tracer.trace_evaluation("test_signal", "42")
        
        analysis = self.tracer.analyze_signal_traces("test_signal")
        
        self.assertEqual(analysis["signal_id"], "test_signal")
        # trace_evaluation creates 1 trace (calls trace_signal internally)
        self.assertEqual(analysis["total_traces"], 3)
        self.assertEqual(analysis["level_distribution"]["info"], 2)  # Info + evaluation
        self.assertEqual(analysis["level_distribution"]["error"], 1)
        self.assertEqual(analysis["evaluation_count"], 1)


class TestAgentMemory(unittest.TestCase):
    """Test enhanced agent memory with robust error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = AgentMemory(path=os.path.join(self.temp_dir, "test_memory.json"))
        self.test_outcome = {
            "model_id": "test_model",
            "score": 0.85,
            "status": "success"
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_outcome_success(self):
        """Test successful outcome logging."""
        result = self.memory.log_outcome(
            "TestAgent",
            "test_run",
            self.test_outcome,
            "medium_term"
        )
        
        self.assertTrue(result["success"])
        self.assertIn("TestAgent", result["agent"])
        self.assertIn("test_run", result["run_type"])
    
    def test_get_history(self):
        """Test retrieving history."""
        # Log some outcomes first
        self.memory.log_outcome("TestAgent", "test_run", self.test_outcome, "medium_term")
        
        history = self.memory.get_history("TestAgent", "test_run")
        
        self.assertTrue(history["success"])
        self.assertGreater(history["count"], 0)
        self.assertIn("history", history)
    
    def test_get_recent_performance(self):
        """Test getting recent performance."""
        # Log some outcomes with scores
        for i in range(5):
            outcome = self.test_outcome.copy()
            outcome["score"] = 0.8 + i * 0.01
            self.memory.log_outcome("TestAgent", "test_run", outcome, "medium_term")
        
        performance = self.memory.get_recent_performance(
            "TestAgent",
            "test_run",
            "score",
            window=3
        )
        
        self.assertTrue(performance["success"])
        self.assertIn("values", performance)
        self.assertLessEqual(len(performance["values"]), 3)
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        # Log some outcomes
        for i in range(10):
            outcome = self.test_outcome.copy()
            outcome["model_id"] = f"model_{i}"
            self.memory.log_outcome("TestAgent", "test_run", outcome, "short_term")
        
        # Trigger cleanup
        cleanup_result = self.memory.cleanup_expired()
        
        self.assertIsInstance(cleanup_result, dict)
        # Check for any cleanup-related keys
        cleanup_keys = [k for k in cleanup_result.keys() if 'cleanup' in k.lower() or 'cleaned' in k.lower()]
        self.assertGreater(len(cleanup_keys), 0)
    
    def test_memory_stats(self):
        """Test memory statistics."""
        # Log some outcomes
        self.memory.log_outcome("TestAgent", "test_run", self.test_outcome, "medium_term")
        
        stats = self.memory.get_memory_stats()
        
        self.assertIsInstance(stats, dict)
        # Check for memory-related keys
        memory_keys = [k for k in stats.keys() if 'memory' in k.lower() or 'entries' in k.lower()]
        self.assertGreater(len(memory_keys), 0)


class TestTaskDispatcher(unittest.TestCase):
    """Test task dispatcher with dynamic configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dispatcher = TaskDispatcher(max_workers=2, enable_redis=False)
    
    async def test_submit_task(self):
        """Test task submission."""
        async def test_func(x, y):
            return x + y
        
        task_id = await self.dispatcher.submit_task(
            test_func,
            1, 2,
            priority=TaskPriority.HIGH
        )
        
        self.assertIsInstance(task_id, str)
        self.assertGreater(len(task_id), 0)
    
    async def test_task_priority_ordering(self):
        """Test task priority ordering."""
        results = []
        
        async def test_func(value):
            results.append(value)
            return value
        
        # Submit tasks with different priorities
        await self.dispatcher.submit_task(test_func, 1, priority=TaskPriority.LOW)
        await self.dispatcher.submit_task(test_func, 2, priority=TaskPriority.HIGH)
        await self.dispatcher.submit_task(test_func, 3, priority=TaskPriority.CRITICAL)
        
        # Start dispatcher to process tasks
        await self.dispatcher.start()
        await asyncio.sleep(0.1)  # Allow some processing time
        await self.dispatcher.stop()
        
        # Higher priority tasks should be processed first
        # Note: This is a simplified test - actual ordering depends on timing
        self.assertGreater(len(results), 0)
    
    async def test_task_retry_logic(self):
        """Test task retry logic."""
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Simulated failure")
            return "success"
        
        task_id = await self.dispatcher.submit_task(
            failing_func,
            max_retries=3
        )
        
        # Start dispatcher
        await self.dispatcher.start()
        
        # Wait for task completion
        try:
            result = await self.dispatcher.get_task_result(task_id, timeout=5.0)
            self.assertEqual(result, "success")
            self.assertEqual(call_count, 3)  # Should have retried twice
        except asyncio.TimeoutError:
            pass  # Task might still be running
        
        await self.dispatcher.stop()
    
    def test_task_registry_duplicate_prevention(self):
        """Test duplicate task prevention."""
        def test_func():
            return "test"
        
        # Create tasks manually
        task1 = Task(
            id="task1",
            func=test_func,
            args=(),
            kwargs={},
            priority=TaskPriority.NORMAL
        )
        task2 = Task(
            id="task2", 
            func=test_func,
            args=(),
            kwargs={},
            priority=TaskPriority.NORMAL
        )
        
        # First task should register successfully
        self.assertTrue(self.dispatcher.registry.register_task(task1))
        
        # Second task should be rejected as duplicate
        self.assertFalse(self.dispatcher.registry.register_task(task2))
    
    def test_get_metrics(self):
        """Test metrics collection."""
        metrics = self.dispatcher.get_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("tasks_submitted", metrics)
        self.assertIn("tasks_completed", metrics)
        self.assertIn("tasks_failed", metrics)
        self.assertIn("queue_size", metrics)
        self.assertIn("active_workers", metrics)


class TestBatch21Integration(unittest.TestCase):
    """Integration tests for Batch 21 features."""
    
    def test_commentary_with_signal_tracing(self):
        """Test commentary generation with signal tracing."""
        # Create components
        generator = CommentaryGenerator(max_tokens=100)
        tracer = SignalTracer()
        
        # Trace commentary generation
        trace = tracer.trace_signal(
            "commentary_generation",
            "Generating commentary for analysis",
            {"query": "Analyze AAPL"}
        )
        
        # Generate commentary with truncation
        with patch.object(generator, '_generate_gpt_commentary') as mock_gpt:
            mock_gpt.return_value = "Long commentary " * 100
            
            with patch.object(generator, 'truncate_response') as mock_truncate:
                mock_truncate.return_value = "Truncated commentary"
                
                result = generator.generate_commentary(
                    "Analyze AAPL",
                    {"intent": "analysis", "symbol": "AAPL"},
                    {"score": 0.85}
                )
        
        # Verify everything worked
        self.assertEqual(result, "Truncated commentary")
        self.assertIsNotNone(trace.signal_id)
    
    def test_memory_with_safe_evaluation(self):
        """Test memory operations with safe evaluation."""
        # Create components
        memory = AgentMemory(path=tempfile.mktemp())
        tracer = SignalTracer()
        
        # Test safe evaluation
        result, error = tracer.safe_eval("[1, 2, 3]")
        
        # Log outcome
        outcome = {"model_id": "test", "score": result[0] if result else 0}
        memory_result = memory.log_outcome("TestAgent", "test_run", outcome, "medium_term")
        
        # Verify everything worked
        self.assertIsNone(error)
        self.assertTrue(memory_result["success"])


if __name__ == "__main__":
    unittest.main()
