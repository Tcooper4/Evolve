"""
Model Performance Visualization Dashboard

This module provides a Streamlit interface for visualizing model performance metrics
stored in the PerformanceMemory system.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import threading
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from trading.memory.performance_memory import PerformanceMemory

logger = logging.getLogger(__name__)

class TaskProgressTracker:
    """Tracks and displays real-time task progress."""
    
    def __init__(self):
        self.tasks = {}
        self.progress_queue = queue.Queue()
        self.running = True
        
        # Start progress update thread
        self.update_thread = threading.Thread(target=self._update_progress, daemon=True)
        self.update_thread.start()
    
    def add_task(self, task_id: str, task_name: str, total_steps: int = 100) -> None:
        """Add a new task to track.
        
        Args:
            task_id: Unique task identifier
            task_name: Human-readable task name
            total_steps: Total number of steps for the task
        """
        self.tasks[task_id] = {
            'name': task_name,
            'current_step': 0,
            'total_steps': total_steps,
            'status': 'running',
            'start_time': datetime.now(),
            'last_update': datetime.now(),
            'progress': 0.0,
            'message': 'Initializing...'
        }
    
    def update_task_progress(self, task_id: str, current_step: int, 
                           message: str = None, status: str = None) -> None:
        """Update task progress.
        
        Args:
            task_id: Task identifier
            current_step: Current step number
            message: Optional status message
            status: Task status ('running', 'completed', 'failed', 'paused')
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task['current_step'] = current_step
            task['progress'] = (current_step / task['total_steps']) * 100
            task['last_update'] = datetime.now()
            
            if message:
                task['message'] = message
            if status:
                task['status'] = status
            
            # Put update in queue for real-time display
            self.progress_queue.put({
                'task_id': task_id,
                'update': task.copy()
            })
    
    def complete_task(self, task_id: str, message: str = "Task completed successfully") -> None:
        """Mark task as completed.
        
        Args:
            task_id: Task identifier
            message: Completion message
        """
        self.update_task_progress(task_id, self.tasks[task_id]['total_steps'], 
                                message, 'completed')
    
    def fail_task(self, task_id: str, error_message: str) -> None:
        """Mark task as failed.
        
        Args:
            task_id: Task identifier
            error_message: Error description
        """
        self.update_task_progress(task_id, self.tasks[task_id]['current_step'], 
                                f"Failed: {error_message}", 'failed')
    
    def _update_progress(self) -> None:
        """Background thread for updating progress."""
        while self.running:
            try:
                # Process queue updates
                while not self.progress_queue.empty():
                    update = self.progress_queue.get_nowait()
                    task_id = update['task_id']
                    task_update = update['update']
                    self.tasks[task_id] = task_update
                
                time.sleep(0.1)  # Update every 100ms
            except Exception as e:
                logger.error(f"Error in progress update thread: {e}")
                time.sleep(1)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status dictionary or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked tasks.
        
        Returns:
            Dictionary of all tasks
        """
        return self.tasks.copy()
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> None:
        """Remove old completed tasks.
        
        Args:
            max_age_hours: Maximum age in hours for completed tasks
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if (task['status'] in ['completed', 'failed'] and 
                task['last_update'] < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]

# Global progress tracker
progress_tracker = TaskProgressTracker()

def display_task_progress():
    """Display real-time task progress in Streamlit."""
    st.subheader("🔄 Active Tasks")
    
    tasks = progress_tracker.get_all_tasks()
    
    if not tasks:
        st.info("No active tasks")
        return
    
    # Clean up old tasks
    progress_tracker.cleanup_completed_tasks()
    
    for task_id, task in tasks.items():
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{task['name']}**")
                st.write(task['message'])
            
            with col2:
                # Progress bar
                progress_bar = st.progress(task['progress'] / 100)
                
                # Update progress bar based on status
                if task['status'] == 'completed':
                    progress_bar.progress(1.0)
                    st.success("✅")
                elif task['status'] == 'failed':
                    progress_bar.progress(task['progress'] / 100)
                    st.error("❌")
                elif task['status'] == 'paused':
                    progress_bar.progress(task['progress'] / 100)
                    st.warning("⏸️")
                else:
                    progress_bar.progress(task['progress'] / 100)
                    st.info("🔄")
            
            with col3:
                # Show elapsed time
                elapsed = datetime.now() - task['start_time']
                st.write(f"⏱️ {elapsed.total_seconds():.1f}s")
                
                # Show progress percentage
                st.write(f"{task['progress']:.1f}%")
            
            st.divider()

def simulate_task_progress(task_name: str, duration_seconds: int = 10):
    """Simulate a task with progress updates.
    
    Args:
        task_name: Name of the task
        duration_seconds: Duration of the task in seconds
    """
    task_id = f"task_{int(time.time())}"
    progress_tracker.add_task(task_id, task_name, 100)
    
    def run_task():
        for i in range(101):
            if i == 0:
                message = "Initializing..."
            elif i < 25:
                message = "Loading data..."
            elif i < 50:
                message = "Processing..."
            elif i < 75:
                message = "Analyzing results..."
            elif i < 100:
                message = "Finalizing..."
            else:
                message = "Task completed!"
            
            progress_tracker.update_task_progress(task_id, i, message)
            time.sleep(duration_seconds / 100)
        
        progress_tracker.complete_task(task_id)
    
    # Run task in background thread
    thread = threading.Thread(target=run_task, daemon=True)
    thread.start()

def main():
    st.title("📈 Agentic Model Performance Dashboard")
    
    # Add task progress display
    display_task_progress()
    
    # Add task simulation button
    if st.button("🚀 Simulate Task"):
        simulate_task_progress("Model Training", 15)
        st.rerun()
    
    st.divider()
    
    # Initialize memory
    memory = PerformanceMemory()
    
    # Get available tickers
    tickers = memory.get_all_tickers()
    if not tickers:
        st.warning("No performance data available.")
        st.stop()
    
    # Ticker selection
    selected_ticker = st.selectbox("Select Ticker", tickers)
    all_metrics = memory.get_metrics(selected_ticker)
    
    if not all_metrics:
        st.warning("No model metrics found for this ticker.")
        st.stop()
    
    # Flatten metrics into DataFrame
    records = []
    for model, metrics in all_metrics.items():
        record = {
            "model": model,
            "mse": metrics.get("mse"),
            "sharpe": metrics.get("sharpe"),
            "win_rate": metrics.get("win_rate"),
            "timestamp": metrics.get("timestamp"),
            "confidence_low": metrics.get("confidence_intervals", {}).get("low"),
            "confidence_high": metrics.get("confidence_intervals", {}).get("high"),
            "dataset_size": metrics.get("dataset_size", 0)
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Display best model
    best_model = memory.get_best_model(selected_ticker, metric="mse")
    st.subheader(f"✅ Best Model by MSE: `{best_model}`")
    
    # Plot MSE vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        ax.plot(subset["timestamp"], subset["mse"], marker='o', label=model)
    ax.set_title("Model MSE Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Additional metrics visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sharpe Ratio")
        fig_sharpe, ax_sharpe = plt.subplots(figsize=(8, 4))
        for model in df["model"].unique():
            subset = df[df["model"] == model]
            ax_sharpe.plot(subset["timestamp"], subset["sharpe"], marker='o', label=model)
        ax_sharpe.set_title("Sharpe Ratio Over Time")
        ax_sharpe.set_xlabel("Timestamp")
        ax_sharpe.set_ylabel("Sharpe Ratio")
        ax_sharpe.legend()
        ax_sharpe.grid(True)
        st.pyplot(fig_sharpe)
    
    with col2:
        st.subheader("Win Rate")
        fig_win, ax_win = plt.subplots(figsize=(8, 4))
        for model in df["model"].unique():
            subset = df[df["model"] == model]
            ax_win.plot(subset["timestamp"], subset["win_rate"], marker='o', label=model)
        ax_win.set_title("Win Rate Over Time")
        ax_win.set_xlabel("Timestamp")
        ax_win.set_ylabel("Win Rate")
        ax_win.legend()
        ax_win.grid(True)
        st.pyplot(fig_win)
    
    # Show raw metrics table
    st.subheader("📊 Raw Metrics")
    st.dataframe(
        df.style.format({
            'mse': '{:.4f}',
            'sharpe': '{:.2f}',
            'win_rate': '{:.2%}',
            'confidence_low': '{:.2f}',
            'confidence_high': '{:.2f}'
        })
    )
    
    # Show dataset statistics
    st.subheader("📈 Dataset Statistics")
    dataset_stats = df.groupby('model')['dataset_size'].agg(['mean', 'min', 'max']).reset_index()
    st.dataframe(
        dataset_stats.style.format({
            'mean': '{:.0f}',
            'min': '{:.0f}',
            'max': '{:.0f}'
        })
    )

if __name__ == "__main__":
    main() 