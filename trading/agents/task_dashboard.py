"""
Task Dashboard for the financial forecasting system.

This module provides a Streamlit-based dashboard for monitoring tasks.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from .task_memory import TaskMemory, TaskStatus


class TaskDashboard:
    """Dashboard for monitoring and managing tasks."""

    def __init__(self, task_memory: TaskMemory):
        """
        Initialize the task dashboard.

        Args:
            task_memory: TaskMemory instance to monitor
        """
        self.task_memory = task_memory
        self._setup_session_state()

        return {
            "success": True,
            "message": "Task dashboard initialized",
            "timestamp": datetime.now().isoformat(),
        }

    def _setup_session_state(self):
        """Setup Streamlit session state variables."""
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if "auto_refresh" not in st.session_state:
            st.session_state.auto_refresh = True
        if "selected_task" not in st.session_state:
            st.session_state.selected_task = None

    def render(self):
        """Render the task dashboard."""
        st.title("Task Dashboard")

        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Auto-refresh", value=st.session_state.auto_refresh
        )

        # Refresh interval
        refresh_interval = st.sidebar.slider(
            "Refresh interval (seconds)", min_value=5, max_value=60, value=10, step=5
        )

        # Manual refresh button
        if st.sidebar.button("Refresh Now"):
            self._refresh_data()

        # Auto-refresh logic
        if st.session_state.auto_refresh:
            current_time = datetime.now()
            if (
                current_time - st.session_state.last_refresh
            ).total_seconds() >= refresh_interval:
                self._refresh_data()

        # Display task statistics
        self._show_statistics()

        # Display task timeline
        self._show_task_timeline()

        # Display task list
        self._show_task_list()

        # Display task details
        self._show_task_details()

        return {
            "success": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _refresh_data(self):
        """Refresh the dashboard data."""
        st.session_state.last_refresh = datetime.now()
        st.experimental_rerun()

        return {
            "success": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _show_statistics(self):
        """Display task statistics."""
        st.subheader("Task Statistics")

        # Calculate statistics
        total_tasks = len(self.task_memory.tasks)
        completed_tasks = len(
            self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
        )
        pending_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.PENDING))
        failed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.FAILED))

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tasks", total_tasks)
        col2.metric("Completed", completed_tasks)
        col3.metric("Pending", pending_tasks)
        col4.metric("Failed", failed_tasks)

        # Show completion rate
        if total_tasks > 0:
            completion_rate = (completed_tasks / total_tasks) * 100
            st.progress(completion_rate / 100)
            st.text(f"Completion Rate: {completion_rate:.1f}%")

            # Display real-time task completion progress with Streamlit
            for task in self.task_memory.tasks.values():
                if hasattr(task, "completion_pct"):
                    st.progress(task.completion_pct)
                    st.text(f"Task {task.task_id}: {task.completion_pct:.1f}% complete")

        return {
            "success": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _show_task_timeline(self):
        """Display task timeline."""
        st.subheader("Task Timeline")

        # Convert tasks to DataFrame
        tasks_data = []
        for task in self.task_memory.tasks.values():
            tasks_data.append(
                {
                    "Task": task.task_id,
                    "Type": task.task_type,
                    "Status": task.status.value,
                    "Agent": task.agent,
                    "Start": task.created_at,
                    "End": task.updated_at
                    if task.status == TaskStatus.COMPLETED
                    else datetime.now(),
                }
            )

        if tasks_data:
            df = pd.DataFrame(tasks_data)

            # Create timeline
            fig = px.timeline(
                df,
                x_start="Start",
                x_end="End",
                y="Task",
                color="Status",
                hover_data=["Type", "Agent"],
            )

            fig.update_layout(
                title="Task Timeline",
                xaxis_title="Time",
                yaxis_title="Task",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tasks available for timeline")

        return {
            "success": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _show_task_list(self):
        """Display the list of tasks."""
        st.subheader("Task List")

        # Convert tasks to DataFrame
        tasks_data = []
        for task in self.task_memory.tasks.values():
            tasks_data.append(
                {
                    "ID": task.task_id,
                    "Type": task.task_type,
                    "Status": task.status.value,
                    "Agent": task.agent,
                    "Created": task.created_at.isoformat(),
                    "Updated": task.updated_at.isoformat(),
                }
            )

        if tasks_data:
            df = pd.DataFrame(tasks_data)

            # Add task selection
            selected_rows = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_click=self._on_task_select,
            )
        else:
            st.info("No tasks found")

        return {
            "success": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _on_task_select(self, selected_rows):
        """Handle task selection.

        Args:
            selected_rows: Selected rows from dataframe
        """
        if selected_rows:
            st.session_state.selected_task = selected_rows[0]["ID"]

        return {
            "success": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _show_task_details(self):
        """Display detailed information for selected tasks."""
        st.subheader("Task Details")

        if st.session_state.selected_task:
            task = self.task_memory.get_task(st.session_state.selected_task)
            if task:
                # Display task information
                col1, col2 = st.columns(2)

                with col1:
                    st.write("### Basic Information")
                    st.json(
                        {
                            "ID": task.task_id,
                            "Type": task.task_type,
                            "Status": task.status.value,
                            "Agent": task.agent,
                            "Created": task.created_at.isoformat(),
                            "Updated": task.updated_at.isoformat(),
                        }
                    )

                with col2:
                    st.write("### Additional Information")
                    st.json({"Notes": task.notes, "Metadata": task.metadata})

                # Add task management controls
                if task.status == TaskStatus.PENDING:
                    st.write("### Task Management")
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Mark as Completed"):
                            task.status = TaskStatus.COMPLETED
                            self.task_memory.update_task(task)
                            st.experimental_rerun()

                    with col2:
                        if st.button("Mark as Failed"):
                            task.status = TaskStatus.FAILED
                            self.task_memory.update_task(task)
                            st.experimental_rerun()
        else:
            st.info("Select a task to view details")

        return {
            "success": True,
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }


def get_active_tasks() -> List[Dict[str, Any]]:
    """Get list of active tasks for frontend integration.

    Returns:
        List of active task dictionaries
    """
    task_memory = TaskMemory()
    active_tasks = task_memory.get_tasks_by_status(
        [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
    )

    return [task.to_dict() for task in active_tasks]


def get_task_log(days: int = 7) -> List[Dict[str, Any]]:
    """Get task log for frontend integration.

    Args:
        days: Number of days of history to retrieve

    Returns:
        List of task dictionaries
    """
    task_memory = TaskMemory()
    cutoff = datetime.now() - timedelta(days=days)

    tasks = [task for task in task_memory.tasks.values() if task.updated_at >= cutoff]

    return [task.to_dict() for task in tasks]


def run_dashboard():
    """Run the task dashboard."""
    # Initialize task memory
    task_memory = TaskMemory()

    # Create and render dashboard
    dashboard = TaskDashboard(task_memory)
    dashboard.render()


if __name__ == "__main__":
    run_dashboard()
