"""
Task Dashboard for the financial forecasting system.

This module provides a Streamlit-based dashboard for monitoring tasks.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
from .task_memory import TaskMemory, Task, TaskStatus

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
        
    def _setup_session_state(self):
        """Setup Streamlit session state variables."""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
            
    def render(self):
        """Render the task dashboard."""
        st.title("Task Dashboard")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Auto-refresh",
            value=st.session_state.auto_refresh
        )
        
        # Refresh interval
        refresh_interval = st.sidebar.slider(
            "Refresh interval (seconds)",
            min_value=5,
            max_value=60,
            value=10,
            step=5
        )
        
        # Manual refresh button
        if st.sidebar.button("Refresh Now"):
            self._refresh_data()
            
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            current_time = datetime.now()
            if (current_time - st.session_state.last_refresh).total_seconds() >= refresh_interval:
                self._refresh_data()
                
        # Display task statistics
        self._show_statistics()
        
        # Display task list
        self._show_task_list()
        
        # Display task details
        self._show_task_details()
        
    def _refresh_data(self):
        """Refresh the dashboard data."""
        st.session_state.last_refresh = datetime.now()
        st.experimental_rerun()
        
    def _show_statistics(self):
        """Display task statistics."""
        st.subheader("Task Statistics")
        
        # Calculate statistics
        total_tasks = len(self.task_memory.tasks)
        completed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED))
        pending_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.PENDING))
        failed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.FAILED))
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tasks", total_tasks)
        col2.metric("Completed", completed_tasks)
        col3.metric("Pending", pending_tasks)
        col4.metric("Failed", failed_tasks)
        
    def _show_task_list(self):
        """Display the list of tasks."""
        st.subheader("Task List")
        
        # Convert tasks to DataFrame
        tasks_data = []
        for task in self.task_memory.tasks.values():
            tasks_data.append({
                'ID': task.task_id,
                'Type': task.task_type,
                'Status': task.status.value,
                'Agent': task.agent_name,
                'Created': task.created_at.isoformat(),
                'Updated': task.updated_at.isoformat()
            })
            
        if tasks_data:
            df = pd.DataFrame(tasks_data)
            st.dataframe(df)
        else:
            st.info("No tasks found")
            
    def _show_task_details(self):
        """Display detailed information for selected tasks."""
        st.subheader("Task Details")
        
        # Get task IDs
        task_ids = list(self.task_memory.tasks.keys())
        
        if task_ids:
            # Task selector
            selected_id = st.selectbox(
                "Select a task to view details",
                task_ids
            )
            
            # Display task details
            task = self.task_memory.tasks[selected_id]
            
            st.write("### Task Information")
            st.json({
                'ID': task.task_id,
                'Type': task.task_type,
                'Status': task.status.value,
                'Agent': task.agent_name,
                'Created': task.created_at.isoformat(),
                'Updated': task.updated_at.isoformat(),
                'Notes': task.notes,
                'Metadata': task.metadata
            })
            
            # Add task management controls
            if task.status == TaskStatus.PENDING:
                if st.button("Mark as Completed"):
                    task.status = TaskStatus.COMPLETED
                    self.task_memory.update_task(task)
                    st.experimental_rerun()
                    
                if st.button("Mark as Failed"):
                    task.status = TaskStatus.FAILED
                    self.task_memory.update_task(task)
                    st.experimental_rerun()
        else:
            st.info("No tasks available for detailed view")
            
def run_dashboard():
    """Run the task dashboard."""
    # Initialize task memory
    task_memory = TaskMemory()
    
    # Create and render dashboard
    dashboard = TaskDashboard(task_memory)
    dashboard.render()

if __name__ == "__main__":
    run_dashboard() 