"""Tests for the performance memory system."""

import pytest
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from trading.memory.performance_memory import PerformanceMemory

@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create a temporary directory for memory files."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    yield memory_dir
    shutil.rmtree(memory_dir)

@pytest.fixture
def memory(temp_memory_dir):
    """Create a PerformanceMemory instance with temporary storage."""
    return PerformanceMemory(metrics_file=str(temp_memory_dir / "metrics.json"))

def test_initialization(memory):
    """Test memory initialization."""
    assert memory.metrics_file is not None
    assert isinstance(memory.get_all_metrics(), dict)
    assert len(memory.get_all_metrics()) == 0

def test_update_metrics(memory):
    """Test metric updates."""
    # Test single metric update
    memory.update_metrics("AAPL", {"accuracy": 0.95, "sharpe": 1.5})
    metrics = memory.get_metrics("AAPL")
    assert metrics["accuracy"] == 0.95
    assert metrics["sharpe"] == 1.5
    
    # Test metric update with new values
    memory.update_metrics("AAPL", {"accuracy": 0.97})
    metrics = memory.get_metrics("AAPL")
    assert metrics["accuracy"] == 0.97
    assert metrics["sharpe"] == 1.5  # Unchanged

def test_get_metrics(memory):
    """Test metric retrieval."""
    # Test non-existent ticker
    assert memory.get_metrics("NONEXISTENT") == {}
    
    # Test existing ticker
    memory.update_metrics("AAPL", {"accuracy": 0.95})
    metrics = memory.get_metrics("AAPL")
    assert metrics["accuracy"] == 0.95

def test_get_all_metrics(memory):
    """Test retrieval of all metrics."""
    # Add multiple tickers
    memory.update_metrics("AAPL", {"accuracy": 0.95})
    memory.update_metrics("MSFT", {"accuracy": 0.92})
    
    all_metrics = memory.get_all_metrics()
    assert "AAPL" in all_metrics
    assert "MSFT" in all_metrics
    assert all_metrics["AAPL"]["accuracy"] == 0.95
    assert all_metrics["MSFT"]["accuracy"] == 0.92

def test_clear_metrics(memory):
    """Test clearing metrics."""
    # Add metrics
    memory.update_metrics("AAPL", {"accuracy": 0.95})
    memory.update_metrics("MSFT", {"accuracy": 0.92})
    
    # Clear all metrics
    memory.clear_metrics()
    assert len(memory.get_all_metrics()) == 0

def test_backup_restore(memory, temp_memory_dir):
    """Test backup and restore functionality."""
    # Add metrics
    memory.update_metrics("AAPL", {"accuracy": 0.95})
    memory.update_metrics("MSFT", {"accuracy": 0.92})
    
    # Create backup
    backup_file = temp_memory_dir / "backup.json"
    memory.backup(str(backup_file))
    assert backup_file.exists()
    
    # Clear metrics
    memory.clear_metrics()
    assert len(memory.get_all_metrics()) == 0
    
    # Restore from backup
    memory.restore(str(backup_file))
    all_metrics = memory.get_all_metrics()
    assert "AAPL" in all_metrics
    assert "MSFT" in all_metrics
    assert all_metrics["AAPL"]["accuracy"] == 0.95
    assert all_metrics["MSFT"]["accuracy"] == 0.92

def test_edge_cases(memory):
    """Test edge cases and error handling."""
    # Test empty metrics
    memory.update_metrics("AAPL", {})
    assert memory.get_metrics("AAPL") == {}
    
    # Test invalid metric values
    memory.update_metrics("AAPL", {"accuracy": float('inf')})
    memory.update_metrics("MSFT", {"accuracy": float('nan')})
    assert memory.get_metrics("AAPL")["accuracy"] == float('inf')
    assert np.isnan(memory.get_metrics("MSFT")["accuracy"])
    
    # Test very long ticker names
    long_ticker = "A" * 100
    memory.update_metrics(long_ticker, {"accuracy": 0.95})
    assert memory.get_metrics(long_ticker)["accuracy"] == 0.95
    
    # Test special characters in ticker names
    special_ticker = "AAPL-USD"
    memory.update_metrics(special_ticker, {"accuracy": 0.95})
    assert memory.get_metrics(special_ticker)["accuracy"] == 0.95

def test_concurrent_access(memory):
    """Test concurrent access to memory."""
    import threading
    
    def update_metrics(ticker):
        for _ in range(100):
            memory.update_metrics(ticker, {"accuracy": 0.95})
    
    # Create multiple threads
    threads = [
        threading.Thread(target=update_metrics, args=(f"TICKER_{i}",))
        for i in range(5)
    ]
    
    # Start threads
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify all updates were processed
    all_metrics = memory.get_all_metrics()
    assert len(all_metrics) == 5
    for i in range(5):
        assert f"TICKER_{i}" in all_metrics

def test_persistence(memory, temp_memory_dir):
    """Test data persistence across instances."""
    # Add metrics
    memory.update_metrics("AAPL", {"accuracy": 0.95})
    memory.update_metrics("MSFT", {"accuracy": 0.92})
    
    # Create new instance
    new_memory = PerformanceMemory(metrics_file=str(temp_memory_dir / "metrics.json"))
    
    # Verify metrics are preserved
    assert new_memory.get_metrics("AAPL")["accuracy"] == 0.95
    assert new_memory.get_metrics("MSFT")["accuracy"] == 0.92

def test_cleanup(memory):
    """Test cleanup of old metrics."""
    # Add metrics with timestamps
    memory.update_metrics("AAPL", {"accuracy": 0.95, "timestamp": datetime.now().isoformat()})
    memory.update_metrics("MSFT", {
        "accuracy": 0.92,
        "timestamp": (datetime.now() - timedelta(days=31)).isoformat()
    })
    
    # Clean up old metrics
    memory.cleanup(max_age_days=30)
    
    # Verify only recent metrics remain
    assert "AAPL" in memory.get_all_metrics()
    assert "MSFT" not in memory.get_all_metrics() 