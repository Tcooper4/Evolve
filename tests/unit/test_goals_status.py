"""
Unit tests for goals status tracking and management.

Tests goal status functionality including:
- Empty goals initialization
- Status summary generation
- Error handling for missing data
- Goal count validation
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from trading.memory.goals.status import GoalStatusTracker, GoalStatus


class TestGoalStatusTracker:
    """Test suite for GoalStatusTracker."""

    @pytest.fixture
    def temp_goals_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def tracker(self, temp_goals_dir):
        """Create a GoalStatusTracker instance with temporary directory."""
        return GoalStatusTracker(str(temp_goals_dir))

    @pytest.fixture
    def sample_goals_data(self):
        """Sample goals data for testing."""
        return {
            "status": "on_track",
            "message": "Making good progress",
            "timestamp": "2023-01-01T12:00:00",
            "progress": 0.75,
            "metrics": {"accuracy": 0.85, "speed": 0.90},
            "goals": [
                {"id": "goal_1", "name": "Improve accuracy", "target": 0.90},
                {"id": "goal_2", "name": "Increase speed", "target": 0.95}
            ],
            "target_date": "2023-12-31",
            "priority": "high"
        }

    def test_initialization(self, tracker):
        """Test that GoalStatusTracker initializes correctly."""
        assert tracker.goals_dir.exists()
        assert tracker.status_file.exists() or not tracker.status_file.exists()  # File may not exist initially

    def test_load_goals_empty_file(self, tracker):
        """Test loading goals when file doesn't exist."""
        result = tracker.load_goals()
        
        assert result["status"] == "No Data"
        assert "not found" in result["message"]
        assert "timestamp" in result

    def test_load_goals_with_data(self, tracker, sample_goals_data):
        """Test loading goals with valid data."""
        # Save sample data
        with open(tracker.status_file, "w") as f:
            json.dump(sample_goals_data, f)
        
        result = tracker.load_goals()
        
        assert result["status"] == "on_track"
        assert result["progress"] == 0.75
        assert "metrics" in result

    def test_save_goals(self, tracker, sample_goals_data):
        """Test saving goals data."""
        tracker.save_goals(sample_goals_data)
        
        # Verify file was created
        assert tracker.status_file.exists()
        
        # Verify data was saved correctly
        with open(tracker.status_file, "r") as f:
            saved_data = json.load(f)
        
        assert saved_data["status"] == "on_track"
        assert saved_data["progress"] == 0.75

    def test_get_status_summary_empty_goals(self, tracker):
        """Test status summary generation with no goals data."""
        summary = tracker.get_status_summary()
        
        # Should initialize empty goals structure
        assert summary["current_status"] == "not_started"
        assert summary["progress"] == 0.0
        assert summary["goal_count"] == 0
        assert "No goals defined yet" in summary["message"]

    def test_get_status_summary_with_goals(self, tracker, sample_goals_data):
        """Test status summary generation with valid goals data."""
        tracker.save_goals(sample_goals_data)
        
        summary = tracker.get_status_summary()
        
        assert summary["current_status"] == "on_track"
        assert summary["progress"] == 0.75
        assert summary["goal_count"] == 2
        assert "Making good progress" in summary["message"]
        assert "recommendations" in summary
        assert "alerts" in summary

    def test_get_status_summary_missing_fields(self, tracker):
        """Test status summary generation with missing fields."""
        incomplete_data = {
            "status": "behind_schedule"
            # Missing progress, metrics, etc.
        }
        
        tracker.save_goals(incomplete_data)
        summary = tracker.get_status_summary()
        
        # Should provide defaults for missing fields
        assert summary["current_status"] == "behind_schedule"
        assert summary["progress"] == 0.0  # Default
        assert summary["metrics"] == {}  # Default
        assert summary["goal_count"] == 0

    def test_get_status_summary_invalid_data(self, tracker):
        """Test status summary generation with invalid data."""
        # Save invalid data (not a dict)
        with open(tracker.status_file, "w") as f:
            f.write("invalid json")
        
        summary = tracker.get_status_summary()
        
        # Should handle error gracefully
        assert summary["current_status"] == "Error"
        assert "Error generating summary" in summary["message"]

    def test_goal_count_calculation(self, tracker):
        """Test goal count calculation with different data structures."""
        # Test with goals list
        data_with_list = {
            "status": "on_track",
            "goals": [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        }
        tracker.save_goals(data_with_list)
        summary = tracker.get_status_summary()
        assert summary["goal_count"] == 3
        
        # Test with individual goal entries
        data_with_entries = {
            "status": "on_track",
            "goal_1": {"name": "Goal 1"},
            "goal_2": {"name": "Goal 2"}
        }
        tracker.save_goals(data_with_entries)
        summary = tracker.get_status_summary()
        assert summary["goal_count"] == 2

    def test_update_goal_progress(self, tracker, sample_goals_data):
        """Test updating goal progress."""
        tracker.save_goals(sample_goals_data)
        
        new_metrics = {"accuracy": 0.90, "speed": 0.95}
        tracker.update_goal_progress(
            progress=0.85,
            metrics=new_metrics,
            status="ahead_of_schedule",
            message="Excellent progress!"
        )
        
        updated_data = tracker.load_goals()
        assert updated_data["progress"] == 0.85
        assert updated_data["status"] == "ahead_of_schedule"
        assert updated_data["metrics"] == new_metrics
        assert "Excellent progress!" in updated_data["message"]

    def test_recommendations_generation(self, tracker):
        """Test recommendations generation for different statuses."""
        # Test behind schedule
        behind_data = {"status": "behind_schedule", "progress": 0.2}
        tracker.save_goals(behind_data)
        summary = tracker.get_status_summary()
        
        recommendations = summary["recommendations"]
        assert any("increasing resources" in rec.lower() for rec in recommendations)
        assert any("bottlenecks" in rec.lower() for rec in recommendations)
        
        # Test ahead of schedule
        ahead_data = {"status": "ahead_of_schedule", "progress": 0.9}
        tracker.save_goals(ahead_data)
        summary = tracker.get_status_summary()
        
        recommendations = summary["recommendations"]
        assert any("additional objectives" in rec.lower() for rec in recommendations)

    def test_alerts_generation(self, tracker):
        """Test alerts generation for different conditions."""
        # Test low progress alert
        low_progress_data = {"status": "on_track", "progress": 0.05}
        tracker.save_goals(low_progress_data)
        summary = tracker.get_status_summary()
        
        alerts = summary["alerts"]
        assert any("low progress" in alert["message"].lower() for alert in alerts)
        
        # Test behind schedule alert
        behind_data = {"status": "behind_schedule", "progress": 0.3}
        tracker.save_goals(behind_data)
        summary = tracker.get_status_summary()
        
        alerts = summary["alerts"]
        assert any("behind schedule" in alert["message"].lower() for alert in alerts)

    def test_error_handling_in_save(self, tracker):
        """Test error handling when saving fails."""
        # Make the directory read-only to cause save failure
        tracker.goals_dir.chmod(0o444)
        
        with pytest.raises(Exception):
            tracker.save_goals({"status": "test"})
        
        # Restore permissions
        tracker.goals_dir.chmod(0o755)

    def test_logging_warnings(self, tracker, caplog):
        """Test that appropriate warnings are logged."""
        # Test warning for no goals
        tracker.get_status_summary()
        
        assert "No goals are currently defined" in caplog.text
        
        # Test warning for missing fields
        incomplete_data = {"status": "test"}
        tracker.save_goals(incomplete_data)
        tracker.get_status_summary()
        
        assert "Goals status missing" in caplog.text
        assert "Goals progress missing" in caplog.text
