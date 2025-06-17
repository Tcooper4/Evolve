import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
from task_memory import TaskMemory, Task, TaskStatus

class TaskDashboard:
    def __init__(self, task_memory: TaskMemory):
        """Initialize the task dashboard with task memory instance."""
        self.task_memory = task_memory
        self.setup_page()
        
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Task Management Dashboard",
            page_icon="📊",
            layout="wide"
        )
        st.title("Task Management Dashboard")
        
    def display_task_metrics(self):
        """Display key task metrics and statistics."""
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        total_tasks = len(self.task_memory.tasks)
        completed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED))
        failed_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.FAILED))
        pending_tasks = len(self.task_memory.get_tasks_by_status(TaskStatus.PENDING))
        
        # Display metrics
        col1.metric("Total Tasks", total_tasks)
        col2.metric("Completed", completed_tasks)
        col3.metric("Failed", failed_tasks)
        col4.metric("Pending", pending_tasks)
        
    def create_status_distribution_chart(self):
        """Create a pie chart showing task status distribution."""
        status_counts = {
            status: len(self.task_memory.get_tasks_by_status(status))
            for status in TaskStatus
        }
        
        fig = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Task Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
    def create_timeline_chart(self):
        """Create a timeline chart of task completions."""
        completed_tasks = self.task_memory.get_tasks_by_status(TaskStatus.COMPLETED)
        if not completed_tasks:
            st.info("No completed tasks to display timeline")
            return
            
        # Create timeline data
        timeline_data = []
        for task in completed_tasks:
            if 'completion_time' in task.metadata:
                timeline_data.append({
                    'task_id': task.task_id,
                    'completion_time': task.metadata['completion_time'],
                    'type': task.type
                })
                
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            df['completion_time'] = pd.to_datetime(df['completion_time'])
            
            fig = px.scatter(
                df,
                x='completion_time',
                y='type',
                title="Task Completion Timeline",
                labels={'completion_time': 'Completion Time', 'type': 'Task Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
    def display_task_list(self, status_filter: Optional[TaskStatus] = None):
        """Display a list of tasks with filtering options."""
        st.subheader("Task List")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            selected_status = st.selectbox(
                "Filter by Status",
                ["All"] + [status.name for status in TaskStatus]
            )
            
        with col2:
            search_query = st.text_input("Search tasks", "")
            
        # Get filtered tasks
        tasks = self.task_memory.tasks
        if selected_status != "All":
            tasks = [t for t in tasks if t.status.name == selected_status]
        if search_query:
            tasks = [
                t for t in tasks
                if search_query.lower() in t.type.lower() or
                search_query.lower() in t.task_id.lower()
            ]
            
        # Display tasks in a table
        if tasks:
            task_data = []
            for task in tasks:
                task_data.append({
                    "Task ID": task.task_id,
                    "Type": task.type,
                    "Status": task.status.name,
                    "Created": task.metadata.get('creation_time', 'N/A'),
                    "Completed": task.metadata.get('completion_time', 'N/A'),
                    "Agent": task.metadata.get('agent', 'N/A')
                })
                
            df = pd.DataFrame(task_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No tasks found matching the current filters")
            
    def display_task_details(self, task_id: str):
        """Display detailed information about a specific task."""
        task = self.task_memory.get_task(task_id)
        if not task:
            st.error(f"Task {task_id} not found")
            return
            
        st.subheader(f"Task Details: {task_id}")
        
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Type:**", task.type)
            st.write("**Status:**", task.status.name)
            st.write("**Agent:**", task.metadata.get('agent', 'N/A'))
            
        with col2:
            st.write("**Created:**", task.metadata.get('creation_time', 'N/A'))
            st.write("**Completed:**", task.metadata.get('completion_time', 'N/A'))
            st.write("**Duration:**", task.metadata.get('duration', 'N/A'))
            
        # Metrics
        if task.metadata.get('metrics'):
            st.subheader("Metrics")
            metrics = task.metadata['metrics']
            metric_cols = st.columns(len(metrics))
            for col, (metric_name, value) in zip(metric_cols, metrics.items()):
                col.metric(metric_name, f"{value:.4f}")
                
        # Notes
        if task.notes:
            st.subheader("Notes")
            st.write(task.notes)
            
    def run(self):
        """Run the dashboard."""
        self.display_task_metrics()
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        with col1:
            self.create_status_distribution_chart()
        with col2:
            self.create_timeline_chart()
            
        # Task list and details
        self.display_task_list()
        
        # Task details section
        st.subheader("Task Details")
        task_id = st.text_input("Enter Task ID to view details")
        if task_id:
            self.display_task_details(task_id)

def main():
    """Main function to run the dashboard."""
    # Initialize task memory
    task_memory = TaskMemory()
    
    # Create and run dashboard
    dashboard = TaskDashboard(task_memory)
    dashboard.run()

if __name__ == "__main__":
    main() 