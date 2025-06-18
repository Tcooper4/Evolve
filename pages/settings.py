"""Settings page for expert controls and system status."""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import sys
import platform
import psutil
import torch
from typing import Dict, Any

from trading.utils.auto_repair import auto_repair
from trading.utils.error_logger import error_logger
from trading.llm.llm_interface import LLMInterface
from core.agents.router import AgentRouter
from trading.agents.updater import ModelUpdater
from trading.memory.performance_memory import PerformanceMemory

def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'memory_usage': f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB",
        'cpu_usage': f"{psutil.cpu_percent()}%",
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory': f"{torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB" if torch.cuda.is_available() else "N/A",
        'last_refresh': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def get_agent_status() -> Dict[str, Any]:
    """Get status of all agents."""
    return {
        'router': {
            'status': 'active' if 'router' in st.session_state else 'inactive',
            'last_activity': st.session_state.get('router_last_activity', 'N/A')
        },
        'updater': {
            'status': 'active' if 'updater' in st.session_state else 'inactive',
            'last_update': st.session_state.get('updater_last_update', 'N/A')
        },
        'forecaster': {
            'status': 'active' if 'forecaster' in st.session_state else 'inactive',
            'last_forecast': st.session_state.get('forecaster_last_forecast', 'N/A')
        }
    }

def get_memory_status() -> Dict[str, Any]:
    """Get memory usage statistics."""
    if 'memory' not in st.session_state:
        return {'status': 'inactive'}
    
    memory = st.session_state.memory
    return {
        'status': 'active',
        'metrics_count': len(memory.get_all_metrics()),
        'last_backup': memory.last_backup_time,
        'memory_size': f"{Path(memory.metrics_file).stat().st_size / 1024:.1f} KB"
    }

def main():
    """Render the settings page."""
    st.title("Settings & System Status")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["System Status", "Expert Controls", "Error Log"])
    
    with tab1:
        st.header("System Status")
        
        # System Information
        st.subheader("System Information")
        system_status = get_system_status()
        for key, value in system_status.items():
            st.text(f"{key.replace('_', ' ').title()}: {value}")
        
        # Agent Status
        st.subheader("Agent Status")
        agent_status = get_agent_status()
        for agent, status in agent_status.items():
            st.text(f"{agent.title()}: {status['status']}")
            st.text(f"Last Activity: {status.get('last_activity', 'N/A')}")
        
        # Memory Status
        st.subheader("Memory Status")
        memory_status = get_memory_status()
        for key, value in memory_status.items():
            st.text(f"{key.replace('_', ' ').title()}: {value}")
        
        # Auto-Repair Status
        st.subheader("Auto-Repair Status")
        repair_status = auto_repair.get_repair_status()
        for component, status in repair_status['status'].items():
            st.text(f"{component.title()}: {'✅' if status else '❌'}")
        
        # Manual Repair Button
        if st.button("Run System Repair"):
            with st.spinner("Running system repair..."):
                results = auto_repair.run_repair()
                if results['status'] == 'success':
                    st.success("System repair completed successfully!")
                elif results['status'] == 'partial':
                    st.warning("System repair completed with some issues.")
                else:
                    st.error("System repair failed.")
    
    with tab2:
        st.header("Expert Controls")
        
        # Expert Mode Toggle
        expert_mode = st.toggle("Expert Mode", value=st.session_state.get('expert_mode', False))
        st.session_state.expert_mode = expert_mode
        
        if expert_mode:
            # Advanced Strategy Controls
            st.subheader("Strategy Controls")
            st.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.1)
            st.slider("Position Size", 0.0, 1.0, 0.5, 0.1)
            st.slider("Stop Loss", 0.0, 0.1, 0.02, 0.01)
            
            # Model Tuning
            st.subheader("Model Tuning")
            st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
            st.slider("Max Tokens", 100, 2000, 500, 100)
        
        # Auto Mode Toggle
        auto_mode = st.toggle("Auto Mode", value=st.session_state.get('auto_mode', False))
        st.session_state.auto_mode = auto_mode
        
        if auto_mode:
            st.info("Auto mode is enabled. Agents will operate without user input.")
    
    with tab3:
        st.header("Error Log")
        
        # Show last error
        last_error = error_logger.get_last_error()
        if last_error:
            st.error(f"Last Error: {last_error['message']}")
            with st.expander("Error Details"):
                st.json(last_error)
        else:
            st.success("No errors logged.")
        
        # Error count
        st.text(f"Total Errors: {error_logger.get_error_count()}")
        
        # Clear errors button
        if st.button("Clear Error Log"):
            error_logger.clear_errors()
            st.success("Error log cleared.")

if __name__ == "__main__":
    main() 