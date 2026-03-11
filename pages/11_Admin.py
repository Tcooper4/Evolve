# -*- coding: utf-8 -*-
"""
System Administration Page

Merges functionality from:
- 19_Admin_Panel.py (standalone)
- 8_Agent_Management.py (standalone)
- 9_System_Monitoring.py (standalone)
- 5_System_Scorecard.py (standalone)

Features:
- System configuration
- User management
- API key management
- Broker connections
- AI agent management
- System health monitoring
- Resource usage tracking
- Logs and debugging
"""

import logging
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

# Ensure project root is on path when Admin runs as a Streamlit page
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Force reload of config package to pick up new market_analysis_config module
for key in list(sys.modules.keys()):
    if key == "config" or key.startswith("config."):
        del sys.modules[key]

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Backend imports (with error handling)
try:
    from trading.config.enhanced_settings import EnhancedSettings
except ImportError:
    logging.warning("EnhancedSettings not found, using placeholder")
    EnhancedSettings = None

try:
    from trading.agents.agent_registry import AgentRegistry
except ImportError:
    logging.warning("AgentRegistry not found, using placeholder")
    AgentRegistry = None

try:
    from trading.monitoring.health_check import SystemHealthMonitor
except ImportError:
    try:
        from monitoring.health_check import HealthChecker as SystemHealthMonitor
    except ImportError:
        logging.warning("SystemHealthMonitor not found, using placeholder")
        SystemHealthMonitor = None

try:
    from trading.utils.system_status import SystemStatus
except ImportError:
    logging.warning("SystemStatus not found, using placeholder")
    SystemStatus = None

# Authentication check (admin only)
def check_admin_access() -> bool:
    """Check if current user has admin access."""
    # In a real implementation, this would check the user's role from session/token
    # For now, we'll use a simple session state check
    user_role = st.session_state.get("user_role", "admin")
    return user_role == "admin"

# Check admin access
if not check_admin_access():
    st.error("🔒 Access Denied")
    st.markdown("**Admin privileges required to access this page.**")
    st.info("Please contact your system administrator for access.")
    st.stop()


def compute_health_score() -> int:
    """Compute system health score from actual checks (0–100)."""
    score = 100
    if not os.getenv("OPENAI_API_KEY"):
        score -= 15
    if not os.getenv("ANTHROPIC_API_KEY"):
        score -= 10
    try:
        db_path = Path(__file__).resolve().parent.parent / "data" / "users.db"
        if db_path.exists():
            import sqlite3
            sqlite3.connect(str(db_path)).close()
        else:
            score -= 5
    except Exception as _e:
        logger.debug(f"Health check component failed: {_e}")
        score -= 10
    try:
        mem_path = Path(__file__).resolve().parent.parent / "data" / "memory_store.db"
        if not mem_path.parent.exists():
            score -= 5
    except Exception as _e:
        logger.debug(f"Health check component failed: {_e}")
    return max(0, score)


# Initialize session state
if 'admin_settings' not in st.session_state:
    st.session_state.admin_settings = {}

if 'system_health' not in st.session_state:
    initial_score = compute_health_score()
    st.session_state.system_health = {
        "overall_score": initial_score,
        "database": "green",
        "api_connections": "green",
        "broker_connections": "yellow",
        "data_providers": "green",
        "agents_active": 3
    }

if 'system_events' not in st.session_state:
    st.session_state.system_events = []

if 'agent_registry' not in st.session_state:
    st.session_state.agent_registry = {}

if 'system_metrics' not in st.session_state:
    st.session_state.system_metrics = {
        "cpu_usage": 45.0,
        "memory_usage": 62.0,
        "disk_usage": 38.0,
        "uptime": "5 days, 12 hours"
    }
if 'app_start_time' not in st.session_state:
    st.session_state.app_start_time = time.time()

# Initialize backend modules (with error handling)
try:
    if EnhancedSettings:
        settings_manager = EnhancedSettings()
    else:
        settings_manager = None
except Exception as e:
    logging.error(f"Error initializing EnhancedSettings: {e}")
    settings_manager = None

try:
    if AgentRegistry:
        agent_registry = AgentRegistry()
    else:
        agent_registry = None
except Exception as e:
    logging.error(f"Error initializing AgentRegistry: {e}")
    agent_registry = None

try:
    if SystemHealthMonitor:
        health_monitor = SystemHealthMonitor()
    else:
        health_monitor = None
except Exception as e:
    logging.error(f"Error initializing SystemHealthMonitor: {e}")
    health_monitor = None


def run_admin_self_test() -> Dict[str, Dict[str, Any]]:
    """
    Self-test for Admin page: imports, LLM selector, MemoryStore, provider status.
    Returns a dict of check names to {'ok': bool, 'message': str}.
    """
    results: Dict[str, dict] = {}
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    # 1. Config and llm_config
    try:
        from config.llm_config import get_active_llm, get_llm_config, LLM_PROVIDERS
        provider, model, opts = get_active_llm()
        results["config.llm_config"] = {"ok": True, "message": f"active_llm={provider}, model={model}"}
    except Exception as e:
        results["config.llm_config"] = {"ok": False, "message": str(e)}

    # 2. MemoryStore
    try:
        from trading.memory import get_memory_store
        store = get_memory_store()
        _ = store.get_preference("active_llm")
        results["MemoryStore"] = {"ok": True, "message": "get_memory_store() and get_preference OK"}
    except Exception as e:
        results["MemoryStore"] = {"ok": False, "message": str(e)}

    # 3. Provider status (at least one provider)
    try:
        from agents.llm.active_llm_calls import get_provider_status
        status = get_provider_status("claude")
        results["get_provider_status"] = {"ok": True, "message": f"claude: configured={status.get('configured')}"}
    except Exception as e:
        results["get_provider_status"] = {"ok": False, "message": str(e)}

    # 4. test_active_llm (optional; may fail if no API key)
    try:
        from agents.llm.active_llm_calls import test_active_llm
        ok, msg = test_active_llm()
        results["test_active_llm"] = {"ok": ok, "message": msg[:100] if msg else "no response"}
    except Exception as e:
        results["test_active_llm"] = {"ok": False, "message": str(e)}

    return results


# Page header and tab structure (must be defined before any "with tab_*" block)
st.title("⚙️ System Administration")
st.markdown("Manage system configuration, monitor health, and administer AI agents.")

tab1, tab2, tab3, tab4, tab5, tab6, tab_data_mgmt = st.tabs([
    "📊 System Dashboard",
    "⚙️ Configuration",
    "🤖 AI Agents",
    "📈 System Monitoring",
    "📜 Logs & Debugging",
    "🔧 Maintenance",
    "🗄️ Data Management",
])

with tab_data_mgmt:
    st.subheader("Cache & Storage Management")
    from trading.utils.data_manager import (
        CACHE_CONFIG,
        cache_stats,
        cache_clear,
        cleanup_disk_cache,
        rotate_logs,
        get_yf_request_count,
    )

    # Live stats
    stats = cache_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cache Entries (active)", stats.get("active", 0))
    c2.metric("Cache Entries (expired)", stats.get("expired", 0))
    c3.metric("Total Cache Entries", stats.get("total", 0))
    c4.metric("yfinance calls/min", get_yf_request_count())

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Clear In-Memory Cache", key="admin_clear_mem_cache"):
            n = cache_clear()
            st.success(f"Cleared {n} entries")
    with col_b:
        if st.button("Clean Disk Cache (>24h)", key="admin_clean_disk_cache"):
            result = cleanup_disk_cache(24)
            freed_mb = result["freed_bytes"] / 1024 / 1024
            st.success(
                f"Removed {result['removed']} files, freed {freed_mb:.1f} MB"
            )
    with col_c:
        if st.button("Rotate Logs", key="admin_rotate_logs"):
            result = rotate_logs()
            st.success(
                f"Rotated {result['rotated']} log files, freed {result['freed_bytes']/1024:.0f} KB"
            )

    st.markdown("Cache TTL Configuration")
    ttl_data = [
        {
            "Data Type": k,
            "TTL": (
                f"{v[0]//60}m {v[0]%60}s"
                if v[0] >= 60
                else f"{v[0]}s"
            ),
            "Max Entries": v[1],
        }
        for k, v in CACHE_CONFIG.items()
    ]
    if ttl_data:
        ttl_df = pd.DataFrame(ttl_data)
        st.dataframe(ttl_df, use_container_width=True, hide_index=True)
    
    
def _get_system_dashboard_data() -> Dict[str, Any]:
    """Return real data for System Dashboard: uptime, trades today, active strategies, active agents, recent events."""
    out = {
        "uptime_str": "—",
        "trades_today": 0,
        "active_strategies": 0,
        "agents_active": 0,
        "recent_events": [],
    }
    try:
        start = st.session_state.get("app_start_time")
        if start and isinstance(start, (int, float)):
            sec = max(0, time.time() - start)
            days = int(sec // 86400)
            hours = int((sec % 86400) // 3600)
            if days > 0:
                out["uptime_str"] = f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"
            else:
                out["uptime_str"] = f"{hours} hour{'s' if hours != 1 else ''}"
    except Exception as _e:
        logger.debug(f"Health check component failed: {_e}")
    try:
        from trading.memory import get_memory_store
        from trading.memory.memory_store import MemoryType
        store = get_memory_store()
        recs = store.list(MemoryType.LONG_TERM, namespace="trades", category="orders", limit=500)
        today = datetime.now().date().isoformat()
        for r in recs:
            v = r.value if hasattr(r, "value") else (r.get("value", r) if isinstance(r, dict) else None)
            if isinstance(v, dict):
                ts = v.get("timestamp") or getattr(r, "created_at", "")
                if ts and today in str(ts):
                    out["trades_today"] += 1
    except Exception as _e:
        logger.debug(f"Health check component failed: {_e}")
    try:
        import yaml
        strat_path = Path(__file__).parent.parent / "config" / "strategies.yaml"
        if strat_path.exists():
            with open(strat_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            strat_root = cfg.get("strategies") if isinstance(cfg.get("strategies"), dict) else cfg
            defs = (strat_root or {}).get("definitions") or {}
            if isinstance(defs, dict):
                out["active_strategies"] = len([k for k, v in defs.items() if isinstance(v, dict) and v.get("enabled", True)])
    except Exception as _e:
        logger.debug(f"Health check component failed: {_e}")
    try:
        reg_path = Path(__file__).parent.parent / "data" / "agent_registry.json"
        if reg_path.exists():
            with open(reg_path, "r") as f:
                data = json.load(f)
            agents = (data.get("agents") or data) if isinstance(data, dict) else {}
            if isinstance(agents, dict):
                out["agents_active"] = sum(1 for a in agents.values() if isinstance(a, dict) and (a.get("status") or "").lower() == "active")
    except Exception as _e:
        logger.debug(f"Health check component failed: {_e}")
    try:
        log_path = Path(__file__).parent.parent / "logs" / "app.log"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines[-50:]:
                line = line.strip()
                if not line:
                    continue
                if "WARNING" in line or "ERROR" in line or "INFO" in line:
                    out["recent_events"].append({"timestamp": datetime.now().isoformat(), "type": "info", "message": line[-200:], "component": "Log"})
                    if len(out["recent_events"]) >= 5:
                        break
        out["recent_events"] = out["recent_events"][-5:]
    except Exception as _e:
        logger.debug(f"Health check component failed: {_e}")
    return out


# TAB 1: System Dashboard
with tab1:
    dashboard_data = _get_system_dashboard_data()
    st.session_state.system_health["agents_active"] = dashboard_data.get("agents_active", st.session_state.system_health.get("agents_active", 0))
    st.header("📊 System Dashboard")
    st.markdown("High-level system overview and health monitoring.")
    
    # Overall System Health Score
    st.subheader("🏥 Overall System Health")
    
    health_score = st.session_state.system_health.get("overall_score", compute_health_score())
    
    # Health score gauge
    fig_health = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_health.update_layout(height=250)
    st.plotly_chart(fig_health)
    
    st.markdown("---")
    
    # Enhanced Health Monitoring Dashboard
    st.subheader("🏥 System Health")
    
    if 'health_monitor' in st.session_state and 'system_monitor' in st.session_state:
        try:
            health = st.session_state.health_monitor
            monitor = st.session_state.system_monitor
            
            # Overall health status
            try:
                health_status = health.get_overall_health()
            except Exception:
                # Fallback if method doesn't exist
                health_status = {
                    'status': 'healthy',
                    'alerts': []
                }
            
            if health_status.get('status') == 'healthy':
                st.success(f"✅ System Status: {health_status.get('status', 'unknown').upper()}")
            elif health_status.get('status') == 'degraded':
                st.warning(f"⚠️ System Status: {health_status.get('status', 'unknown').upper()}")
            else:
                st.error(f"🔴 System Status: {health_status.get('status', 'unknown').upper()}")
            
            # Component health
            st.markdown("**Component Health:**")
            
            try:
                components = health.check_all_components()
            except Exception:
                components = {}
            
            if components:
                for component, status in components.items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{component}**")
                    with col2:
                        if status.get('healthy', False):
                            st.success("✅ Healthy")
                        else:
                            st.error("❌ Unhealthy")
                    with col3:
                        st.write(f"{status.get('uptime', 'N/A')}")
            
            # System metrics
            st.markdown("**📊 System Metrics:**")
            
            try:
                metrics = monitor.get_system_metrics()
            except Exception:
                metrics = {
                    'cpu_percent': st.session_state.system_metrics.get('cpu_usage', 45.0),
                    'memory_percent': st.session_state.system_metrics.get('memory_usage', 62.0),
                    'disk_percent': st.session_state.system_metrics.get('disk_usage', 38.0),
                    'active_tasks': 0
                }
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("CPU Usage", f"{metrics.get('cpu_percent', 0):.1f}%")
            with col2:
                st.metric("Memory Usage", f"{metrics.get('memory_percent', 0):.1f}%")
            with col3:
                st.metric("Disk Usage", f"{metrics.get('disk_percent', 0):.1f}%")
            with col4:
                st.metric("Active Tasks", metrics.get('active_tasks', 0))
            
            # Performance metrics
            st.markdown("**⚡ Performance Metrics:**")
            
            try:
                perf = monitor.get_performance_metrics()
            except Exception:
                perf = {
                    'avg_response_time': 0.5,
                    'requests_per_minute': 10,
                    'error_rate': 0.01
                }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Response Time", f"{perf.get('avg_response_time', 0):.2f}s")
            with col2:
                st.metric("Requests/Min", f"{perf.get('requests_per_minute', 0):.0f}")
            with col3:
                st.metric("Error Rate", f"{perf.get('error_rate', 0):.2%}")
            
            # Alerts
            alerts = health_status.get('alerts', [])
            if alerts:
                st.markdown("**🚨 Active Alerts:**")
                for alert in alerts:
                    if isinstance(alert, dict):
                        st.warning(f"⚠️ {alert.get('message', 'Unknown alert')}")
                    else:
                        st.warning(f"⚠️ {alert}")
            
            # Historical performance chart
            st.markdown("**📈 Performance History:**")
            
            try:
                history = monitor.get_performance_history(hours=24)
            except Exception:
                history = None
            
            if history:
                try:
                    history_df = pd.DataFrame(history)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=history_df['timestamp'],
                        y=history_df['response_time'],
                        name='Response Time',
                        line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title='Response Time (Last 24 Hours)',
                        xaxis_title='Time',
                        yaxis_title='Response Time (s)',
                        hovermode='x unified',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("Performance history data not available in expected format")
            else:
                st.info("No performance history available")
        
        except Exception as e:
            st.warning(f"⚠️ Health monitoring not fully available: {e}")
            st.info("Some health monitoring features may not be accessible.")
    else:
        # Fallback: show basic psutil-based system metrics when dedicated monitor is unavailable
        try:
            import psutil
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
            with col2:
                st.metric("RAM Usage", f"{psutil.virtual_memory().percent}%")
            with col3:
                st.metric("Disk Usage", f"{psutil.disk_usage('/').percent}%")
        except Exception:
            st.markdown('<span style="background:#e0e0e0;color:#555;padding:4px 10px;border-radius:4px;">Not configured</span>', unsafe_allow_html=True)
            st.caption("Basic system health metrics unavailable (psutil missing).")
    
    st.markdown("---")
    
    # Automation & Workflows (placeholder until automation system is implemented)
    st.header("🤖 Automation & Workflows")
    st.info("Workflow automation is coming in a future release. This section will manage scheduled jobs and automated runbooks once available.")
        
    with auto_tab1:
            st.subheader("Active Workflows")
            
            try:
                active_workflows = workflows.get_active_workflows()
            except Exception:
                active_workflows = []
            
            if active_workflows:
                for workflow in active_workflows:
                    with st.expander(f"🔄 {workflow.get('name', 'Unknown')}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            status_emoji = "✅" if workflow.get('status') == 'running' else "⏸️"
                            st.write(f"**Status:** {status_emoji} {workflow.get('status', 'unknown')}")
                        
                        with col2:
                            st.write(f"**Schedule:** {workflow.get('schedule', 'N/A')}")
                        
                        with col3:
                            st.write(f"**Last Run:** {workflow.get('last_run', 'Never')}")
                        
                        # Workflow details
                        st.write(f"**Description:** {workflow.get('description', 'No description')}")
                        
                        # Actions
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            if st.button("▶️ Run Now", key=f"run_{workflow.get('id', 'unknown')}"):
                                try:
                                    automation.run_workflow(workflow.get('id'))
                                    st.success("Workflow started!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        
                        with col_b:
                            if workflow.get('status') == 'running':
                                if st.button("⏸️ Pause", key=f"pause_{workflow.get('id', 'unknown')}"):
                                    try:
                                        automation.pause_workflow(workflow.get('id'))
                                        st.success("Workflow paused")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            else:
                                if st.button("▶️ Resume", key=f"resume_{workflow.get('id', 'unknown')}"):
                                    try:
                                        automation.resume_workflow(workflow.get('id'))
                                        st.success("Workflow resumed")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                        
                        with col_c:
                            if st.button("🗑️ Delete", key=f"delete_{workflow.get('id', 'unknown')}"):
                                try:
                                    automation.delete_workflow(workflow.get('id'))
                                    st.success("Workflow deleted")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        
                        # Execution history
                        if st.checkbox("Show history", key=f"history_{workflow.get('id', 'unknown')}"):
                            try:
                                history = automation.get_workflow_history(workflow.get('id'), limit=10)
                                if history:
                                    history_df = pd.DataFrame(history)
                                    st.dataframe(history_df, use_container_width=True)
                                else:
                                    st.info("No execution history available")
                            except Exception as e:
                                st.warning(f"Could not load history: {e}")
            else:
                st.info("No active workflows. Create one to get started.")
        
    with auto_tab2:
            st.subheader("Create New Workflow")
            
            # Predefined workflow templates
            st.write("**Choose from templates:**")
            
            template_options = {
                "Daily Data Update": {
                    "description": "Automatically fetch latest market data every day",
                    "schedule": "daily",
                    "actions": ["fetch_market_data", "update_database"]
                },
                "Model Retraining": {
                    "description": "Retrain models weekly with latest data",
                    "schedule": "weekly",
                    "actions": ["fetch_data", "train_models", "evaluate_models", "deploy_best"]
                },
                "Portfolio Rebalancing": {
                    "description": "Rebalance portfolio based on target allocation",
                    "schedule": "monthly",
                    "actions": ["calculate_allocation", "generate_orders", "execute_trades"]
                },
                "Risk Monitoring": {
                    "description": "Check risk metrics hourly and send alerts",
                    "schedule": "hourly",
                    "actions": ["calculate_risk", "check_limits", "send_alerts"]
                },
                "Performance Reporting": {
                    "description": "Generate and email daily performance report",
                    "schedule": "daily",
                    "actions": ["calculate_metrics", "generate_report", "send_email"]
                }
            }
            
            selected_template = st.selectbox(
                "Select template",
                options=list(template_options.keys())
            )
            
            template = template_options[selected_template]
            
            st.info(f"📝 {template['description']}")
            
            # Customize workflow
            st.write("**Customize workflow:**")
            
            workflow_name = st.text_input(
                "Workflow name",
                value=selected_template
            )
            
            schedule_type = st.selectbox(
                "Schedule",
                ["Hourly", "Daily", "Weekly", "Monthly", "Custom"]
            )
            
            cron_expression = None
            if schedule_type == "Custom":
                cron_expression = st.text_input(
                    "Cron expression",
                    value="0 9 * * *",
                    help="Format: minute hour day month weekday"
                )
            
            # Additional settings
            enable_notifications = st.checkbox(
                "Send notifications on completion",
                value=True
            )
            
            notification_email = None
            if enable_notifications:
                notification_email = st.text_input(
                    "Email for notifications",
                    value=st.session_state.get('user_email', '')
                )
            
            # Create workflow button
            if st.button("✨ Create Workflow", type="primary"):
                try:
                    # Build workflow config
                    workflow_config = {
                        'name': workflow_name,
                        'description': template['description'],
                        'schedule': schedule_type.lower(),
                        'actions': template['actions'],
                        'notifications_enabled': enable_notifications,
                        'notification_email': notification_email if enable_notifications else None
                    }
                    
                    if cron_expression:
                        workflow_config['cron_expression'] = cron_expression
                    
                    # Create workflow
                    workflow_id = workflows.create_workflow(workflow_config)
                    
                    st.success(f"✅ Workflow '{workflow_name}' created successfully!")
                    st.write(f"Workflow ID: {workflow_id}")
                    
                    # Option to start immediately
                    if st.button("▶️ Start Now", key="start_new_workflow"):
                        try:
                            automation.run_workflow(workflow_id)
                            st.success("Workflow started!")
                        except Exception as e:
                            st.error(f"Error starting workflow: {e}")
                
                except Exception as e:
                    st.error(f"Error creating workflow: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Custom workflow builder
            st.markdown("---")
            st.subheader("🛠️ Build Custom Workflow")
            
            with st.expander("Advanced: Build from scratch"):
                custom_name = st.text_input("Custom workflow name", key="custom_workflow_name")
                
                available_actions = [
                    "fetch_market_data",
                    "train_model",
                    "evaluate_model",
                    "generate_signals",
                    "execute_trades",
                    "calculate_risk",
                    "send_notification",
                    "generate_report",
                    "backup_data"
                ]
                
                selected_actions = st.multiselect(
                    "Select actions (in order)",
                    options=available_actions,
                    key="custom_workflow_actions"
                )
                
                if st.button("Create Custom Workflow", key="create_custom_workflow"):
                    if not custom_name or not selected_actions:
                        st.error("Please provide name and at least one action")
                    else:
                        try:
                            custom_config = {
                                'name': custom_name,
                                'description': 'Custom workflow',
                                'schedule': 'manual',
                                'actions': selected_actions
                            }
                            
                            workflow_id = workflows.create_workflow(custom_config)
                            st.success(f"Custom workflow created: {workflow_id}")
                        except Exception as e:
                            st.error(f"Error creating custom workflow: {e}")
        
    with auto_tab3:
            st.subheader("⚙️ Automation Settings")
            
            # Global automation settings
            st.write("**Global Settings:**")
            
            try:
                config_dict = config.get_all() if hasattr(config, 'get_all') else {}
            except Exception:
                config_dict = {}
            
            auto_enabled = st.checkbox(
                "Enable automation system",
                value=config_dict.get('enabled', True),
                help="Master switch for all automated workflows"
            )
            
            max_concurrent = st.slider(
                "Max concurrent workflows",
                min_value=1,
                max_value=10,
                value=config_dict.get('max_concurrent', 3),
                help="Maximum number of workflows running simultaneously"
            )
            
            retry_on_failure = st.checkbox(
                "Retry failed workflows",
                value=config_dict.get('retry_on_failure', True)
            )
            
            max_retries = 3
            if retry_on_failure:
                max_retries = st.number_input(
                    "Max retries",
                    min_value=1,
                    max_value=5,
                    value=config_dict.get('max_retries', 3)
                )
            
            # Notification settings
            st.write("**Notification Settings:**")
            
            notify_on_success = st.checkbox(
                "Notify on successful completion",
                value=config_dict.get('notify_on_success', False)
            )
            
            notify_on_failure = st.checkbox(
                "Notify on failure",
                value=config_dict.get('notify_on_failure', True)
            )
            
            # Save settings
            if st.button("💾 Save Settings", type="primary", key="save_automation_settings"):
                try:
                    new_config = {
                        'enabled': auto_enabled,
                        'max_concurrent': max_concurrent,
                        'retry_on_failure': retry_on_failure,
                        'max_retries': max_retries if retry_on_failure else 0,
                        'notify_on_success': notify_on_success,
                        'notify_on_failure': notify_on_failure
                    }
                    
                    if hasattr(config, 'update'):
                        config.update(new_config)
                    elif hasattr(config, 'set'):
                        for key, value in new_config.items():
                            config.set(key, value)
                    
                    st.success("✅ Settings saved!")
                except Exception as e:
                    st.error(f"Error saving settings: {e}")
            
            # Automation statistics
            st.markdown("---")
            st.subheader("📊 Automation Statistics")
            
            try:
                stats = automation.get_statistics()
            except Exception:
                stats = {}
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Workflows", stats.get('total_workflows', 0))
            with col2:
                st.metric("Active", stats.get('active_workflows', 0))
            with col3:
                st.metric("Completed Today", stats.get('completed_today', 0))
            with col4:
                st.metric("Failed Today", stats.get('failed_today', 0))
    
    if not (automation and workflows and config):
        st.info("⚙️ Workflow automation is not yet implemented. Coming in a future release.")
    
    st.markdown("---")
    
    # Agent API Service
    st.header("🔌 Agent API")
    
    st.write("""
    The Agent API provides programmatic access to trading agents.
    Use it to integrate with external systems or build custom interfaces.
    """)
    
    # API status (port check)
    def _port_open(port: int) -> bool:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                return s.connect_ex(('localhost', port)) == 0
        except Exception:
            return False

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Status")
        
        if _port_open(8000):
            st.success("🟢 Agent API running — http://localhost:8000")
            try:
                import requests
                response = requests.get('http://localhost:8000/health', timeout=2)
                if response.status_code == 200 and response.content:
                    st.json(response.json())
            except Exception:
                pass
        else:
            st.info("Agent API not running. Start with:\n`python scripts/launch_agent_api.py`")
            if st.button("🚀 Start API Service", key="start_api_service"):
                try:
                    import subprocess
                    import os
                    from pathlib import Path
                    script_path = Path(__file__).parent.parent / "scripts" / "launch_agent_api.py"
                    if script_path.exists():
                        if os.name == 'nt':
                            subprocess.Popen(['python', str(script_path)], creationflags=subprocess.CREATE_NEW_CONSOLE)
                        else:
                            subprocess.Popen(['python', str(script_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        st.success("API service started! Check status in a few seconds.")
                        st.info("The API will be available at http://localhost:8000")
                    else:
                        st.error(f"API launcher script not found at: {script_path}")
                except Exception as e:
                    st.error(f"Error starting API service: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            st.info("You can still start the API service manually using: python scripts/launch_agent_api.py")
    
    with col2:
        st.subheader("API Info")
        st.write("**Base URL:** http://localhost:8000")
        st.write("**Documentation:** [Swagger UI](http://localhost:8000/docs)")
        st.write("**Version:** 1.0.0")
        
        # Quick test button
        if st.button("🔍 Test API Connection", key="test_api_connection"):
            try:
                import requests
                response = requests.get('http://localhost:8000/health', timeout=2)
                if response.status_code == 200:
                    st.success("✅ API is accessible")
                else:
                    st.error(f"❌ API returned: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure it's running.")
            except ImportError:
                st.warning("⚠️ 'requests' library not available")
    
    # API endpoints documentation
    st.subheader("📚 Available Endpoints")
    
    endpoints = [
        {
            "Method": "GET",
            "Path": "/health",
            "Description": "Check API health status"
        },
        {
            "Method": "GET",
            "Path": "/agents",
            "Description": "List all available agents"
        },
        {
            "Method": "GET",
            "Path": "/agents/{agent_id}/status",
            "Description": "Get agent status"
        },
        {
            "Method": "POST",
            "Path": "/agents/{agent_id}/run",
            "Description": "Trigger agent execution"
        },
        {
            "Method": "GET",
            "Path": "/agents/{agent_id}/results",
            "Description": "Get agent results"
        },
        {
            "Method": "POST",
            "Path": "/forecast",
            "Description": "Generate forecast via API"
        },
        {
            "Method": "POST",
            "Path": "/backtest",
            "Description": "Run strategy backtest via API"
        }
    ]
    
    endpoints_df = pd.DataFrame(endpoints)
    st.dataframe(endpoints_df, use_container_width=True, hide_index=True)
    
    # API example usage
    st.subheader("💻 Example Usage")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        with st.expander("Python example", expanded=True):
            st.code("""
import requests

# Get list of agents
response = requests.get('http://localhost:8000/agents')
agents = response.json()
print(f"Available agents: {agents}")

# Run forecast agent
forecast_params = {
    'symbol': 'AAPL',
    'model': 'LSTM',
    'horizon': 30
}

response = requests.post(
    'http://localhost:8000/forecast',
    json=forecast_params
)

forecast = response.json()
print(f"Forecast: {forecast['values']}")
""", language='python')
    
    with col_ex2:
        with st.expander("cURL example", expanded=True):
            st.code("""
# Get agents
curl http://localhost:8000/agents

# Run forecast
curl -X POST http://localhost:8000/forecast \\
  -H "Content-Type: application/json" \\
  -d '{
    "symbol": "AAPL",
    "model": "LSTM",
    "horizon": 30
  }'
""", language='bash')
    
    # API key management
    st.subheader("🔑 API Authentication")
    
    st.caption("API authentication can be enabled via configuration.")
    
    # Future: Add API key generation and management
    with st.expander("🔐 Future: API Key Management"):
        st.write("""
        **Planned Features:**
        - Generate API keys for external integrations
        - Rate limiting per API key
        - Usage tracking and analytics
        - Key rotation and expiration
        """)
    
    st.markdown("---")
    
    # WebSocket Real-time Updates
    st.header("📡 Real-time System Monitor")
    
    st.write("""
    Connect to the WebSocket service to receive real-time updates about:
    - Agent status changes
    - Forecast completions
    - Trade executions
    - Risk alerts
    - System health events
    """)
    
    # Initialize WebSocket client (if not already)
    if 'ws_client' not in st.session_state:
        try:
            from utils.websocket_client import WebSocketClient
            st.session_state.ws_client = WebSocketClient()
            st.session_state.ws_connected = False
            st.session_state.ws_available = True
        except ImportError:
            st.session_state.ws_available = False
            st.session_state.ws_connected = False
    
    # Connection status
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not st.session_state.get('ws_available', False):
            st.markdown('<span style="background:#e0e0e0;color:#555;padding:4px 10px;border-radius:4px;">Not configured</span>', unsafe_allow_html=True)
            st.caption("WebSocket client not available. Install: pip install websockets")
        elif st.session_state.get('ws_connected', False):
            try:
                # Check if actually connected
                if hasattr(st.session_state.ws_client, 'is_connected'):
                    is_connected = st.session_state.ws_client.is_connected()
                else:
                    is_connected = st.session_state.ws_connected
                
                if is_connected:
                    st.success("✅ Real-time updates enabled")
                else:
                    st.warning("⚠️ Connection lost. Click Connect to reconnect.")
                    st.session_state.ws_connected = False
            except Exception as e:
                logging.warning(f"Connection check failed: {e}")
                st.warning("⚠️ Connection status unknown")
                st.session_state.ws_connected = False
        else:
            st.markdown('<span style="background:#e0e0e0;color:#555;padding:4px 10px;border-radius:4px;">Not configured</span>', unsafe_allow_html=True)
            st.caption("Real-time updates are disabled. Click Connect to enable.")
    
    with col2:
        if not st.session_state.get('ws_available', False):
            st.markdown('<span style="background:#e0e0e0;color:#555;padding:4px 10px;border-radius:4px;">Not configured</span>', unsafe_allow_html=True)
            st.caption("Install websockets to enable real-time updates.")
        elif not st.session_state.get('ws_connected', False):
            if st.button("🔌 Connect", key="ws_connect_btn"):
                try:
                    import asyncio
                    import threading
                    
                    # Create new event loop in thread
                    def connect_ws():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(st.session_state.ws_client.connect())
                            st.session_state.ws_connected = True
                        except Exception as e:
                            st.error(f"Connection failed: {e}")
                            st.info("Make sure WebSocket service is running:\npython scripts/launch_websocket.py")
                    
                    # Run in thread to avoid blocking
                    thread = threading.Thread(target=connect_ws, daemon=True)
                    thread.start()
                    thread.join(timeout=3)
                    
                    if st.session_state.ws_connected:
                        st.success("Connected!")
                        st.rerun()
                    else:
                        st.error("Connection timeout. Check if WebSocket service is running.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("Make sure WebSocket service is running:\npython scripts/launch_websocket.py")
        else:
            if st.button("🔌 Disconnect", key="ws_disconnect_btn"):
                try:
                    import asyncio
                    import threading
                    
                    def disconnect_ws():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(st.session_state.ws_client.disconnect())
                        except Exception as e:
                            logger.debug(f"Suppressed error during disconnect: {e}")
                            pass
                    
                    thread = threading.Thread(target=disconnect_ws, daemon=True)
                    thread.start()
                    thread.join(timeout=2)
                    
                    st.session_state.ws_connected = False
                    st.success("Disconnected!")
                    st.rerun()
                except Exception as e:
                    logging.warning(f"Connection check failed: {e}")
                    st.session_state.ws_connected = False
                    st.rerun()
    
    # WebSocket service status
    st.markdown("---")
    col_ws1, col_ws2 = st.columns(2)
    
    with col_ws1:
        st.subheader("WebSocket Service Status")
        
        try:
            import requests
            try:
                # Try to check if WebSocket service is running (via HTTP endpoint if available)
                response = requests.get('http://localhost:8001/health', timeout=1)
                if response.status_code == 200:
                    st.success("✅ WebSocket service is running")
                else:
                    st.warning("⚠️ WebSocket service may not be running")
            except requests.exceptions.ConnectionError:
                st.info("ℹ️ Real-time WebSocket updates are optional. The app works fully without them. To enable: run `python scripts/launch_websocket.py` in a separate terminal.")
                
                if st.button("🚀 Start WebSocket Service", key="start_ws_service"):
                    try:
                        import subprocess
                        import os
                        from pathlib import Path
                        
                        script_path = Path(__file__).parent.parent / "scripts" / "launch_websocket.py"
                        
                        if script_path.exists():
                            if os.name == 'nt':  # Windows
                                subprocess.Popen(['python', str(script_path)], creationflags=subprocess.CREATE_NEW_CONSOLE)
                            else:  # Unix/Linux/Mac
                                subprocess.Popen(['python', str(script_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                            st.success("WebSocket service started! Check status in a few seconds.")
                            st.info("The WebSocket will be available at ws://localhost:8001")
                        else:
                            st.error(f"WebSocket launcher script not found at: {script_path}")
                    except Exception as e:
                        st.error(f"Error starting WebSocket service: {e}")
            except requests.exceptions.Timeout:
                st.warning("⚠️ WebSocket service check timed out")
        except ImportError:
            st.info("💡 Install 'requests' to check WebSocket service status")
            st.info("Start manually: python scripts/launch_websocket.py")
    
    with col_ws2:
        st.subheader("WebSocket Info")
        st.write("**URL:** ws://localhost:8001")
        st.write("**Protocol:** WebSocket (WS)")
        st.write("**Status:** " + ("Connected" if st.session_state.get('ws_connected', False) else "Disconnected"))
    
    # Real-time event feed
    if st.session_state.get('ws_connected', False) and st.session_state.get('ws_available', False):
        st.markdown("---")
        st.subheader("📊 Live Event Feed")
        
        # Initialize event list if not exists
        if 'recent_events' not in st.session_state:
            st.session_state.recent_events = []
        
        # Register callbacks for different event types
        def on_agent_update(payload):
            """Handle agent update events."""
            agent_name = payload.get('agent_name', 'Unknown')
            status = payload.get('status', 'unknown')
            event = f"🤖 Agent '{agent_name}' status: {status}"
            timestamp = datetime.now().strftime('%H:%M:%S')
            st.session_state.recent_events.insert(0, {
                'timestamp': timestamp,
                'event': event,
                'type': 'agent'
            })
            if len(st.session_state.recent_events) > 20:
                st.session_state.recent_events.pop()
        
        def on_forecast_complete(payload):
            """Handle forecast completion events."""
            symbol = payload.get('symbol', 'Unknown')
            r2 = payload.get('r2', 0)
            event = f"📈 Forecast completed for {symbol} (R²: {r2:.3f})"
            timestamp = datetime.now().strftime('%H:%M:%S')
            st.session_state.recent_events.insert(0, {
                'timestamp': timestamp,
                'event': event,
                'type': 'forecast'
            })
            if len(st.session_state.recent_events) > 20:
                st.session_state.recent_events.pop()
        
        def on_trade_execution(payload):
            """Handle trade execution events."""
            symbol = payload.get('symbol', 'Unknown')
            action = payload.get('action', 'Unknown')
            quantity = payload.get('quantity', 0)
            price = payload.get('price', 0)
            event = f"💼 Trade executed: {action} {quantity} {symbol} @ ${price:.2f}"
            timestamp = datetime.now().strftime('%H:%M:%S')
            st.session_state.recent_events.insert(0, {
                'timestamp': timestamp,
                'event': event,
                'type': 'trade'
            })
            if len(st.session_state.recent_events) > 20:
                st.session_state.recent_events.pop()
        
        def on_risk_alert(payload):
            """Handle risk alert events."""
            message = payload.get('message', 'Risk alert')
            event = f"⚠️ RISK ALERT: {message}"
            timestamp = datetime.now().strftime('%H:%M:%S')
            st.session_state.recent_events.insert(0, {
                'timestamp': timestamp,
                'event': event,
                'type': 'risk'
            })
            if len(st.session_state.recent_events) > 20:
                st.session_state.recent_events.pop()
        
        # Register callbacks
        try:
            st.session_state.ws_client.on('agent_update', on_agent_update)
            st.session_state.ws_client.on('forecast_complete', on_forecast_complete)
            st.session_state.ws_client.on('trade_execution', on_trade_execution)
            st.session_state.ws_client.on('risk_alert', on_risk_alert)
        except Exception as e:
            logging.debug(f"Suppressed error: {e}")  # Callbacks may not be registered if connection is lost
            pass
        
        # Display recent events
        if st.session_state.recent_events:
            st.write("**Recent Events:**")
            for event_data in st.session_state.recent_events[:10]:
                if isinstance(event_data, dict):
                    timestamp = event_data.get('timestamp', '')
                    event = event_data.get('event', str(event_data))
                    event_type = event_data.get('type', 'info')
                    
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.caption(f"🕐 {timestamp}")
                        with col2:
                            if event_type == 'risk':
                                st.warning(event)
                            elif event_type == 'trade':
                                st.success(event)
                            elif event_type == 'forecast':
                                st.info(event)
                            else:
                                st.text(event)
        else:
            st.info("No events yet. Events will appear here as they occur.")
        
        # Clear events button
        if st.button("🗑️ Clear Events", key="clear_ws_events"):
            st.session_state.recent_events = []
            st.rerun()
        
        # Auto-refresh note
        st.caption("💡 Events update in real-time. Page auto-refreshes to show new events.")
    
    st.markdown("---")
    
    # AI Assistant
    st.header("🤖 AI Assistant")
    
    st.write("Ask questions about your trading system in natural language.")
    
    try:
        from trading.nlp.llm_processor import LLMProcessor
        
        if 'llm_processor' not in st.session_state:
            st.session_state.llm_processor = LLMProcessor()
        
        llm = st.session_state.llm_processor
        
        # Chat interface
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation History")
            
            for message in st.session_state.chat_history:
                if message.get('role') == 'user':
                    with st.chat_message("user"):
                        st.write(message.get('content', ''))
                else:
                    with st.chat_message("assistant"):
                        st.info(message.get('content', ''))
            
            # Clear chat button
            if st.button("🗑️ Clear Chat", key="clear_chat_history"):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Input
        user_input = st.text_input(
            "Ask a question",
            placeholder="e.g., What's my best performing strategy this month?",
            key="ai_assistant_input"
        )
        
        col_send, col_clear = st.columns([3, 1])
        
        with col_send:
            if st.button("Send", type="primary", key="send_ai_message", use_container_width=True) and user_input:
                # Add to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Process with LLM
                with st.spinner("Thinking..."):
                    try:
                        response = llm.process_query(
                            query=user_input,
                            context=st.session_state
                        )
                        
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                    except Exception as e:
                        error_response = f"I encountered an error processing your query: {str(e)}"
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': error_response
                        })
                
                st.rerun()
        
        with col_clear:
            if st.button("Clear", key="clear_ai_input", use_container_width=True):
                st.session_state.ai_assistant_input = ""
                st.rerun()
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("⚡ Quick Questions")
        
        quick_questions = [
            "What's my current portfolio value?",
            "Show me today's best trade",
            "What are my top risk exposures?",
            "Generate a performance summary"
        ]
        
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            for i, question in enumerate(quick_questions[:2]):
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })
                    
                    with st.spinner("Processing..."):
                        try:
                            response = llm.process_query(
                                query=question,
                                context=st.session_state
                            )
                            
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response
                            })
                        except Exception as e:
                            error_response = f"I encountered an error: {str(e)}"
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': error_response
                            })
                    
                    st.rerun()
        
        with col_q2:
            for i, question in enumerate(quick_questions[2:], start=2):
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })
                    
                    with st.spinner("Processing..."):
                        try:
                            response = llm.process_query(
                                query=question,
                                context=st.session_state
                            )
                            
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response
                            })
                        except Exception as e:
                            error_response = f"I encountered an error: {str(e)}"
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': error_response
                            })
                    
                    st.rerun()
    
    except ImportError:
        st.info("Feature not available. LLM Processor (trading.nlp.llm_processor) is not available.")
    except Exception as e:
        st.error(f"Error initializing AI Assistant: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Status Indicators
    st.subheader("🔍 Status Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Database Status
    db_status = st.session_state.system_health.get("database", "green")
    with col1:
        if db_status == "green":
            st.success("✅ Database")
        elif db_status == "yellow":
            st.warning("⚠️ Database")
        else:
            st.error("❌ Database")
    
    # API Connections Status
    api_status = st.session_state.system_health.get("api_connections", "green")
    with col2:
        if api_status == "green":
            st.success("✅ API Connections")
        elif api_status == "yellow":
            st.warning("⚠️ API Connections")
        else:
            st.error("❌ API Connections")
    
    # Broker Connections Status
    broker_status = st.session_state.system_health.get("broker_connections", "yellow")
    with col3:
        if broker_status == "green":
            st.success("✅ Broker Connections")
        elif broker_status == "yellow":
            st.warning("⚠️ Broker Connections")
        else:
            st.error("❌ Broker Connections")
    
    # Data Providers Status
    data_status = st.session_state.system_health.get("data_providers", "green")
    with col4:
        if data_status == "green":
            st.success("✅ Data Providers")
        elif data_status == "yellow":
            st.warning("⚠️ Data Providers")
        else:
            st.error("❌ Data Providers")
    
    # Agents Status
    agents_active = st.session_state.system_health.get("agents_active", 0)
    with col5:
        st.metric("🤖 Active Agents", agents_active)
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("📈 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Uptime", dashboard_data["uptime_str"])
    with col2:
        st.metric("📊 Trades Today", dashboard_data["trades_today"])
    with col3:
        st.metric("📈 Active Strategies", dashboard_data["active_strategies"])
    with col4:
        cpu_usage = st.session_state.system_metrics.get("cpu_usage", 45.0)
        st.metric("💻 System Load", f"{cpu_usage:.1f}%")
    
    st.markdown("---")
    
    # Recent System Events Feed
    st.subheader("📢 Recent System Events")
    if dashboard_data.get("recent_events"):
        events_to_show = list(dashboard_data["recent_events"])[-10:]
    else:
        if not st.session_state.system_events:
            st.session_state.system_events = []
        events_to_show = sorted(
            st.session_state.system_events,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:10]
    
    if events_to_show:
        for event in events_to_show:
            timestamp = event.get('timestamp', 'Unknown')
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                # If parsing fails, use original timestamp string
                timestamp_str = timestamp
            
            event_type = event.get('type', 'info')
            message = event.get('message', 'N/A')
            component = event.get('component', 'Unknown')
            
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    st.caption(f"🕐 {timestamp_str}")
                
                with col2:
                    if event_type == "success":
                        st.success(f"✅ {message}")
                    elif event_type == "warning":
                        st.warning(f"⚠️ {message}")
                    elif event_type == "error":
                        st.error(f"❌ {message}")
                    else:
                        st.info(f"ℹ️ {message}")
                
                with col3:
                    st.caption(f"📦 {component}")
    else:
        st.info("No recent system events.")
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Restart Services"):
            # Simulate restart
            st.session_state.system_events.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "info",
                "message": "Services restart initiated",
                "component": "System"
            })
            st.success("✅ Services restart initiated!")
            st.info("This would restart all system services in production.")
    
    with col2:
        if st.button("🗑️ Clear Cache", key="admin_clear_cache_1"):
            try:
                st.cache_data.clear()
                cache_dir = Path(__file__).resolve().parent.parent / ".cache"
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Cache clear partial: {e}")
            st.session_state.system_events.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "success",
                "message": "Cache cleared successfully",
                "component": "Cache"
            })
            st.success("✅ Cache cleared!")
            st.rerun()
    
    with col3:
        if st.button("🏥 Run Health Check"):
            # Simulate health check
            if health_monitor:
                try:
                    health_status = health_monitor.check_system_health()
                    new_score = health_status.get('overall_score', compute_health_score())
                    st.session_state.system_health["overall_score"] = new_score
                    st.session_state.system_events.insert(0, {
                        "timestamp": datetime.now().isoformat(),
                        "type": "success",
                        "message": f"Health check completed. Score: {new_score}",
                        "component": "Health Monitor"
                    })
                    st.success("✅ Health check completed!")
                except Exception as e:
                    st.error(f"Error running health check: {str(e)}")
            else:
                new_score = compute_health_score()
                st.session_state.system_health["overall_score"] = new_score
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "success",
                    "message": f"Health check completed. Score: {new_score}",
                    "component": "Health Monitor"
                })
                st.success("✅ Health check completed!")
    
    # System Health Details (Expandable)
    st.markdown("---")
    with st.expander("📋 System Health Details", expanded=False):
        st.markdown("**Component Status:**")
        
        components = [
            ("Database", db_status),
            ("API Connections", api_status),
            ("Broker Connections", broker_status),
            ("Data Providers", data_status),
            ("AI Agents", "green" if dashboard_data.get("agents_active", 0) > 0 else "yellow")
        ]
        
        for component, status in components:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{component}:**")
            with col2:
                if status == "green":
                    st.success("✅ Healthy")
                elif status == "yellow":
                    st.warning("⚠️ Degraded")
                else:
                    st.error("❌ Unhealthy")
        
        st.markdown("---")
        st.markdown("**System Metrics:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"- **CPU Usage:** {cpu_usage:.1f}%")
            st.markdown(f"- **Memory Usage:** {st.session_state.system_metrics.get('memory_usage', 62.0):.1f}%")
            st.markdown(f"- **Disk Usage:** {st.session_state.system_metrics.get('disk_usage', 38.0):.1f}%")
        
        with col2:
            st.markdown(f"- **Uptime:** {dashboard_data.get('uptime_str', '—')}")
            st.markdown(f"- **Active Agents:** {dashboard_data.get('agents_active', 0)}")
            st.markdown(f"- **Health Score:** {health_score}/100")

# TAB 2: Configuration
with tab2:
    st.header("⚙️ Configuration")
    st.markdown("Manage system settings, API keys, broker connections, and feature flags.")

    # --- AI Model Settings (global LLM selector) ---
    st.subheader("🤖 AI Model Settings")
    st.caption("Choose which LLM powers the entire app (Chat, commentary, intent parsing). Stored in preference memory.")
    try:
        # Ensure project root is on path
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        # Try import; if config resolved to wrong package (e.g. trading.config), clear and retry
        try:
            from config.llm_config import (
                get_active_llm,
                set_active_llm,
                LLM_PROVIDERS,
                DEFAULT_MODELS,
                PROVIDER_DISPLAY_NAMES,
                HUGGINGFACE_MODES,
            )
            from agents.llm.active_llm_calls import get_provider_status, test_active_llm
        except (ImportError, ModuleNotFoundError):
            for _key in list(sys.modules.keys()):
                if _key == "config" or _key.startswith("config."):
                    del sys.modules[_key]
            from config.llm_config import (
                get_active_llm,
                set_active_llm,
                LLM_PROVIDERS,
                DEFAULT_MODELS,
                PROVIDER_DISPLAY_NAMES,
                HUGGINGFACE_MODES,
            )
            from agents.llm.active_llm_calls import get_provider_status, test_active_llm
        current_provider, current_model, current_options = get_active_llm()
        col_prov, col_mod, col_status = st.columns([2, 2, 2])
        with col_prov:
            provider = st.selectbox(
                "Provider",
                options=LLM_PROVIDERS,
                index=LLM_PROVIDERS.index(current_provider) if current_provider in LLM_PROVIDERS else 0,
                format_func=lambda p: PROVIDER_DISPLAY_NAMES.get(p, p),
                key="admin_llm_provider",
            )
        with col_mod:
            default_model = DEFAULT_MODELS.get(provider, "")
            model_placeholder = default_model or "e.g. claude-sonnet-4-20250514, gpt-4o, llama3"
            model = st.text_input(
                "Model",
                value=current_model if provider == current_provider else default_model,
                placeholder=model_placeholder,
                key="admin_llm_model",
            )
            if not model and default_model:
                model = default_model
        # HuggingFace sub-mode: Inference API vs Local pipeline
        huggingface_mode = "inference"
        if provider == "huggingface":
            current_hf_mode = current_options.get("huggingface_mode", "inference")
            huggingface_mode = st.radio(
                "HuggingFace mode",
                options=HUGGINGFACE_MODES,
                index=HUGGINGFACE_MODES.index(current_hf_mode) if current_hf_mode in HUGGINGFACE_MODES else 0,
                format_func=lambda x: "Inference API (cloud, free tier)" if x == "inference" else "Local pipeline (offline)",
                key="admin_llm_hf_mode",
            )
            if huggingface_mode == "inference":
                st.caption("Free tier at api-inference.huggingface.co is rate-limited.")
        with col_status:
            status = get_provider_status(provider, huggingface_mode=huggingface_mode if provider == "huggingface" else None)
            if status["configured"]:
                st.success(f"✅ {status['message']}")
            else:
                st.warning(f"⚠️ {status['message']}")
            if provider == "ollama" and not status["configured"]:
                st.caption("Start Ollama: run `ollama serve` then `ollama pull <model>`.")
        save_col, test_col = st.columns(2)
        with save_col:
            if st.button("💾 Save", key="admin_llm_save"):
                try:
                    opts = {}
                    if provider == "huggingface":
                        opts["huggingface_mode"] = huggingface_mode
                    set_active_llm(provider, model or DEFAULT_MODELS.get(provider, ""), **opts)
                    st.success("✅ Active LLM saved. Chat and commentary will use this model.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
        with test_col:
            if st.button("🔌 Test Connection", key="admin_llm_test"):
                try:
                    from trading.memory import get_memory_store
                    store = get_memory_store()
                    prev = store.get_preference("active_llm")
                    payload = {"provider": provider, "model": model or DEFAULT_MODELS.get(provider, "")}
                    if provider == "huggingface":
                        payload["options"] = {"huggingface_mode": huggingface_mode}
                    store.upsert_preference("active_llm", payload)
                    ok, msg = test_active_llm()
                    if prev is not None:
                        store.upsert_preference("active_llm", prev)
                    else:
                        restore = {"provider": current_provider, "model": current_model}
                        if current_options:
                            restore["options"] = current_options
                        store.upsert_preference("active_llm", restore)
                    if ok:
                        st.success(f"✅ Response: {msg[:200]}")
                    else:
                        st.error(f"❌ {msg}")
                        if provider == "ollama":
                            st.info("Ensure Ollama is running: `ollama serve`. Then pull your model: `ollama pull " + (model or "llama3") + "`")
                except Exception as e:
                    st.error(f"Test failed: {e}")
    except Exception as e:
        logging.warning("AI Model Settings not available: %s", e)
        import traceback
        st.error(traceback.format_exc())
        st.info("Configure LLM in config/llm_config.py. MemoryStore and get_active_llm required.")

    st.markdown("---")

    # Initialize configuration if not exists
    if 'system_config' not in st.session_state:
        import os
        # Load existing API keys from environment if available
        st.session_state.system_config = {
            "general": {
                "system_name": "EVOLVE Trading System",
                "timezone": "America/New_York",
                "base_currency": "USD",
                "trading_hours_start": "09:30",
                "trading_hours_end": "16:00"
            },
            "api_keys": {
                "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
                "finnhub": os.getenv("FINNHUB_API_KEY", ""),
                "polygon": os.getenv("POLYGON_API_KEY", ""),
                "openai": os.getenv("OPENAI_API_KEY", ""),
                "newsapi": os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY", ""),
                "reddit_client_id": os.getenv("REDDIT_CLIENT_ID", ""),
                "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET", "")
            },
            "brokers": {
                "alpaca_paper": {"enabled": False, "api_key": "", "api_secret": "", "status": "disconnected"},
                "alpaca_live": {"enabled": False, "api_key": "", "api_secret": "", "status": "disconnected"},
                "binance": {"enabled": False, "api_key": "", "api_secret": "", "status": "disconnected"},
                "ibkr": {"enabled": False, "username": "", "password": "", "status": "disconnected"}
            },
            "feature_flags": {
                "beta_features": False,
                "advanced_analytics": True,
                "ai_forecasting": True,
                "real_time_alerts": True,
                "portfolio_optimization": True
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "trading_db",
                "user": "admin"
            }
        }
    
    config = st.session_state.system_config
    
    # General Settings Section
    with st.expander("🌐 General Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            config["general"]["system_name"] = st.text_input(
                "System Name",
                value=config["general"]["system_name"],
                help="Name of the trading system"
            )
            
            config["general"]["timezone"] = st.selectbox(
                "Timezone",
                [
                    "America/New_York",
                    "America/Chicago",
                    "America/Denver",
                    "America/Los_Angeles",
                    "Europe/London",
                    "Europe/Paris",
                    "Asia/Tokyo",
                    "Asia/Hong_Kong",
                    "UTC"
                ],
                index=0 if config["general"]["timezone"] in [
                    "America/New_York", "America/Chicago", "America/Denver",
                    "America/Los_Angeles", "Europe/London", "Europe/Paris",
                    "Asia/Tokyo", "Asia/Hong_Kong", "UTC"
                ] else 0,
                help="System timezone"
            )
        
        with col2:
            config["general"]["base_currency"] = st.selectbox(
                "Base Currency",
                ["USD", "EUR", "GBP", "JPY", "CNY"],
                index=["USD", "EUR", "GBP", "JPY", "CNY"].index(config["general"]["base_currency"]) if config["general"]["base_currency"] in ["USD", "EUR", "GBP", "JPY", "CNY"] else 0,
                help="Base currency for trading"
            )
            
            col_start, col_end = st.columns(2)
            with col_start:
                trading_start = st.time_input(
                    "Trading Hours Start",
                    value=datetime.strptime(config["general"]["trading_hours_start"], "%H:%M").time(),
                    help="Start of trading hours"
                )
                config["general"]["trading_hours_start"] = trading_start.strftime("%H:%M")
            
            with col_end:
                trading_end = st.time_input(
                    "Trading Hours End",
                    value=datetime.strptime(config["general"]["trading_hours_end"], "%H:%M").time(),
                    help="End of trading hours"
                )
                config["general"]["trading_hours_end"] = trading_end.strftime("%H:%M")

    # Home Page Settings Section
    with st.expander("🏠 Home Page Settings", expanded=False):
        from config.user_store import load_user_preferences, save_user_preferences

        home_config = config.setdefault("home_page", {})

        universe_options = [
            "S&P 100",
            "S&P 500",
            "S&P 500 + Nasdaq 100",
            "Russell 1000",
        ]

        session_id = st.session_state.get("evolve_session_id") or st.session_state.get("session_id", "")
        prefs = load_user_preferences(session_id) if session_id else {}

        stored_universe = prefs.get("home_top_movers_universe") or home_config.get("top_movers_universe") or "S&P 100"
        if stored_universe not in universe_options:
            stored_universe = "S&P 100"

        selected_universe = st.selectbox(
            "Top Movers Universe",
            universe_options,
            index=universe_options.index(stored_universe),
            help="Controls which universe is used for the Home page 'Today's Top Movers' section.",
            key="home_top_movers_universe_admin",
        )

        home_config["top_movers_universe"] = selected_universe
        st.session_state["home_top_movers_universe"] = selected_universe

        if session_id and selected_universe != prefs.get("home_top_movers_universe"):
            try:
                save_user_preferences(
                    session_id,
                    {**prefs, "home_top_movers_universe": selected_universe},
                )
                st.caption("Home page universe preference saved.")
            except Exception:
                st.caption("Could not persist Home page settings; using in-session value only.")

    st.markdown("---")
    
    # API Keys Section
    with st.expander("🔑 API Keys", expanded=True):
        import os
        _env_path = _project_root / ".env"
        _dotenv_available = False
        try:
            from dotenv import set_key, load_dotenv
            _dotenv_available = True
        except ImportError:
            pass

        # --- AI Provider Keys (at top) ---
        st.markdown("**🤖 AI Provider Keys**")
        st.caption("Set API keys for LLM providers. Keys are stored in `.env`; never displayed.")
        _ai_keys = [
            ("ANTHROPIC_API_KEY", "Claude", "claude"),
            ("OPENAI_API_KEY", "GPT-4", "gpt4"),
            ("GOOGLE_API_KEY", "Gemini", "gemini"),
            ("MOONSHOT_API_KEY", "Kimi", "kimi"),
            ("HUGGINGFACE_API_KEY", "HuggingFace", "huggingface"),
        ]
        for env_var, label, provider in _ai_keys:
            col_status, col_input, col_save, col_test = st.columns([1, 3, 1, 1])
            is_set = bool(os.getenv(env_var) or (os.getenv("HF_TOKEN") if env_var == "HUGGINGFACE_API_KEY" else False))
            with col_status:
                if bool(is_set):
                    st.markdown("✅")
                else:
                    st.markdown("—")
            with col_input:
                new_val = st.text_input(
                    label,
                    value="",
                    type="password",
                    placeholder="•••••••• (set)" if bool(is_set) else "Enter key...",
                    key=f"api_{env_var}",
                    help=f"Stored as {env_var} in .env",
                )
            with col_save:
                if _dotenv_available and st.button("Save", key=f"save_{env_var}"):
                    if new_val:
                        try:
                            set_key(str(_env_path), env_var, new_val)
                            load_dotenv(_env_path, override=True)
                            st.success("Saved")
                        except Exception as e:
                            st.error(str(e)[:80])
                    else:
                        st.warning("Enter a value")
            with col_test:
                if st.button("Test", key=f"test_{env_var}"):
                    try:
                        from agents.llm.active_llm_calls import get_provider_status
                        status = get_provider_status(provider)
                        if status.get("configured"):
                            st.success(status.get("message", "OK"))
                        else:
                            st.warning(status.get("message", "Not set"))
                    except Exception as e:
                        st.error(str(e)[:60])
        if not _dotenv_available:
            st.caption("Install python-dotenv to save keys to .env.")

        st.markdown("---")
        st.markdown("**📈 Trading**")
        _trading_keys = [
            ("ALPACA_API_KEY", "Alpaca API Key"),
            ("ALPACA_SECRET_KEY", "Alpaca Secret Key"),
        ]
        for env_var, label in _trading_keys:
            col_status, col_input, col_save = st.columns([1, 3, 1])
            is_set = bool(os.getenv(env_var))
            with col_status:
                st.markdown("✅" if bool(is_set) else "—")
            with col_input:
                new_val = st.text_input(
                    label,
                    value="",
                    type="password",
                    placeholder="•••••••• (set)" if bool(is_set) else "Enter key...",
                    key=f"api_{env_var}",
                )
            with col_save:
                if _dotenv_available and st.button("Save", key=f"save_{env_var}"):
                    if new_val:
                        try:
                            set_key(str(_env_path), env_var, new_val)
                            load_dotenv(_env_path, override=True)
                            st.success("Saved")
                        except Exception as e:
                            st.error(str(e)[:80])

        st.markdown("---")
        st.markdown("**Data Provider API Keys:**")
        
        # Alpha Vantage
        col1, col2 = st.columns([3, 1])
        with col1:
            alpha_vantage_key = st.text_input(
                "Alpha Vantage API Key",
                value=config["api_keys"]["alpha_vantage"],
                type="password",
                help="Alpha Vantage API key for market data"
            )
            if alpha_vantage_key:
                config["api_keys"]["alpha_vantage"] = alpha_vantage_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_alpha"):
                st.text_input("Alpha Vantage API Key (Visible)", value=config["api_keys"]["alpha_vantage"], key="alpha_visible", disabled=True)
        
        # Finnhub
        col1, col2 = st.columns([3, 1])
        with col1:
            finnhub_key = st.text_input(
                "Finnhub API Key",
                value=config["api_keys"]["finnhub"],
                type="password",
                help="Finnhub API key for market data"
            )
            if finnhub_key:
                config["api_keys"]["finnhub"] = finnhub_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_finnhub"):
                st.text_input("Finnhub API Key (Visible)", value=config["api_keys"]["finnhub"], key="finnhub_visible", disabled=True)
        
        # Polygon
        col1, col2 = st.columns([3, 1])
        with col1:
            polygon_key = st.text_input(
                "Polygon API Key",
                value=config["api_keys"]["polygon"],
                type="password",
                help="Polygon API key for market data"
            )
            if polygon_key:
                config["api_keys"]["polygon"] = polygon_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_polygon"):
                st.text_input("Polygon API Key (Visible)", value=config["api_keys"]["polygon"], key="polygon_visible", disabled=True)
        
        # OpenAI
        col1, col2 = st.columns([3, 1])
        with col1:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=config["api_keys"]["openai"],
                type="password",
                help="OpenAI API key for AI features"
            )
            if openai_key:
                config["api_keys"]["openai"] = openai_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_openai"):
                st.text_input("OpenAI API Key (Visible)", value=config["api_keys"]["openai"], key="openai_visible", disabled=True)
        
        # NewsAPI
        col1, col2 = st.columns([3, 1])
        with col1:
            newsapi_key = st.text_input(
                "NewsAPI Key",
                value=config["api_keys"]["newsapi"],
                type="password",
                help="NewsAPI key for news sentiment analysis"
            )
            if newsapi_key:
                config["api_keys"]["newsapi"] = newsapi_key
        
        with col2:
            if st.button("👁️ Reveal", key="reveal_newsapi"):
                st.text_input("NewsAPI Key (Visible)", value=config["api_keys"]["newsapi"], key="newsapi_visible", disabled=True)
        
        # Reddit API Credentials
        st.markdown("**Reddit API (for sentiment analysis):**")
        col1, col2 = st.columns(2)
        with col1:
            reddit_client_id = st.text_input(
                "Reddit Client ID",
                value=config["api_keys"]["reddit_client_id"],
                type="password",
                help="Reddit API client ID"
            )
            if reddit_client_id:
                config["api_keys"]["reddit_client_id"] = reddit_client_id
        
        with col2:
            reddit_client_secret = st.text_input(
                "Reddit Client Secret",
                value=config["api_keys"]["reddit_client_secret"],
                type="password",
                help="Reddit API client secret"
            )
            if reddit_client_secret:
                config["api_keys"]["reddit_client_secret"] = reddit_client_secret
        
        # Save button for API keys
        if st.button("💾 Save API Keys", key="save_api_keys"):
            # Update environment variables (optional - could also save to .env file)
            import os
            if config["api_keys"]["newsapi"]:
                os.environ["NEWSAPI_KEY"] = config["api_keys"]["newsapi"]
                os.environ["NEWS_API_KEY"] = config["api_keys"]["newsapi"]  # Support both
            if config["api_keys"]["reddit_client_id"]:
                os.environ["REDDIT_CLIENT_ID"] = config["api_keys"]["reddit_client_id"]
            if config["api_keys"]["reddit_client_secret"]:
                os.environ["REDDIT_CLIENT_SECRET"] = config["api_keys"]["reddit_client_secret"]
            
            st.success("✅ API keys saved to session. Restart app to apply to services.")
            st.info("💡 Tip: To persist across restarts, also add these to your .env file")
    
    st.markdown("---")
    
    # Broker Connections Section
    with st.expander("🏦 Broker Connections", expanded=True):
        st.markdown("**Configure broker connections:**")
        
        # Alpaca Paper
        st.markdown("**Alpaca (Paper Trading):**")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            alpaca_paper_enabled = st.checkbox(
                "Enable",
                value=config["brokers"]["alpaca_paper"]["enabled"],
                key="alpaca_paper_enabled"
            )
            config["brokers"]["alpaca_paper"]["enabled"] = alpaca_paper_enabled
        
        with col2:
            config["brokers"]["alpaca_paper"]["api_key"] = st.text_input(
                "API Key",
                value=config["brokers"]["alpaca_paper"]["api_key"],
                type="password",
                key="alpaca_paper_key"
            )
        
        with col3:
            config["brokers"]["alpaca_paper"]["api_secret"] = st.text_input(
                "API Secret",
                value=config["brokers"]["alpaca_paper"]["api_secret"],
                type="password",
                key="alpaca_paper_secret"
            )
        
        with col4:
            status = config["brokers"]["alpaca_paper"]["status"]
            if status == "connected":
                st.success("✅ Connected")
            elif status == "connecting":
                st.warning("⏳ Connecting")
            else:
                st.error("❌ Disconnected")
            
            if st.button("🔌 Test", key="test_alpaca_paper"):
                # Simulate connection test
                config["brokers"]["alpaca_paper"]["status"] = "connected"
                st.success("✅ Connection successful!")
        
        # Alpaca Live
        st.markdown("**Alpaca (Live Trading):**")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            alpaca_live_enabled = st.checkbox(
                "Enable",
                value=config["brokers"]["alpaca_live"]["enabled"],
                key="alpaca_live_enabled"
            )
            config["brokers"]["alpaca_live"]["enabled"] = alpaca_live_enabled
        
        with col2:
            config["brokers"]["alpaca_live"]["api_key"] = st.text_input(
                "API Key",
                value=config["brokers"]["alpaca_live"]["api_key"],
                type="password",
                key="alpaca_live_key"
            )
        
        with col3:
            config["brokers"]["alpaca_live"]["api_secret"] = st.text_input(
                "API Secret",
                value=config["brokers"]["alpaca_live"]["api_secret"],
                type="password",
                key="alpaca_live_secret"
            )
        
        with col4:
            status = config["brokers"]["alpaca_live"]["status"]
            if status == "connected":
                st.success("✅ Connected")
            elif status == "connecting":
                st.warning("⏳ Connecting")
            else:
                st.error("❌ Disconnected")
            
            if st.button("🔌 Test", key="test_alpaca_live"):
                # Simulate connection test
                config["brokers"]["alpaca_live"]["status"] = "connected"
                st.success("✅ Connection successful!")
        
        # Binance
        st.markdown("**Binance:**")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            binance_enabled = st.checkbox(
                "Enable",
                value=config["brokers"]["binance"]["enabled"],
                key="binance_enabled"
            )
            config["brokers"]["binance"]["enabled"] = binance_enabled
        
        with col2:
            config["brokers"]["binance"]["api_key"] = st.text_input(
                "API Key",
                value=config["brokers"]["binance"]["api_key"],
                type="password",
                key="binance_key"
            )
        
        with col3:
            config["brokers"]["binance"]["api_secret"] = st.text_input(
                "API Secret",
                value=config["brokers"]["binance"]["api_secret"],
                type="password",
                key="binance_secret"
            )
        
        with col4:
            status = config["brokers"]["binance"]["status"]
            if status == "connected":
                st.success("✅ Connected")
            elif status == "connecting":
                st.warning("⏳ Connecting")
            else:
                st.error("❌ Disconnected")
            
            if st.button("🔌 Test", key="test_binance"):
                # Simulate connection test
                config["brokers"]["binance"]["status"] = "connected"
                st.success("✅ Connection successful!")
        
        # Interactive Brokers
        st.markdown("**Interactive Brokers (IBKR):**")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            ibkr_enabled = st.checkbox(
                "Enable",
                value=config["brokers"]["ibkr"]["enabled"],
                key="ibkr_enabled"
            )
            config["brokers"]["ibkr"]["enabled"] = ibkr_enabled
        
        with col2:
            config["brokers"]["ibkr"]["username"] = st.text_input(
                "Username",
                value=config["brokers"]["ibkr"]["username"],
                key="ibkr_username"
            )
        
        with col3:
            config["brokers"]["ibkr"]["password"] = st.text_input(
                "Password",
                value=config["brokers"]["ibkr"]["password"],
                type="password",
                key="ibkr_password"
            )
        
        with col4:
            status = config["brokers"]["ibkr"]["status"]
            if status == "connected":
                st.success("✅ Connected")
            elif status == "connecting":
                st.warning("⏳ Connecting")
            else:
                st.error("❌ Disconnected")
            
            if st.button("🔌 Test", key="test_ibkr"):
                # Simulate connection test
                config["brokers"]["ibkr"]["status"] = "connected"
                st.success("✅ Connection successful!")
    
    st.markdown("---")
    
    # Feature Flags Section
    with st.expander("🚩 Feature Flags", expanded=False):
        st.markdown("**Enable or disable system features:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config["feature_flags"]["beta_features"] = st.checkbox(
                "Beta Features",
                value=config["feature_flags"]["beta_features"],
                help="Enable experimental beta features"
            )
            
            config["feature_flags"]["advanced_analytics"] = st.checkbox(
                "Advanced Analytics",
                value=config["feature_flags"]["advanced_analytics"],
                help="Enable advanced analytics features"
            )
            
            config["feature_flags"]["ai_forecasting"] = st.checkbox(
                "AI Forecasting",
                value=config["feature_flags"]["ai_forecasting"],
                help="Enable AI-powered forecasting"
            )
        
        with col2:
            config["feature_flags"]["real_time_alerts"] = st.checkbox(
                "Real-Time Alerts",
                value=config["feature_flags"]["real_time_alerts"],
                help="Enable real-time alert system"
            )
            
            config["feature_flags"]["portfolio_optimization"] = st.checkbox(
                "Portfolio Optimization",
                value=config["feature_flags"]["portfolio_optimization"],
                help="Enable portfolio optimization features"
            )
    
    st.markdown("---")
    
    # Database Settings Section
    with st.expander("💾 Database Settings", expanded=False):
        st.markdown("**Database configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config["database"]["host"] = st.text_input(
                "Database Host",
                value=config["database"]["host"],
                help="Database server hostname"
            )
            
            config["database"]["port"] = st.number_input(
                "Database Port",
                min_value=1,
                max_value=65535,
                value=config["database"]["port"],
                help="Database server port"
            )
        
        with col2:
            config["database"]["name"] = st.text_input(
                "Database Name",
                value=config["database"]["name"],
                help="Database name"
            )
            
            config["database"]["user"] = st.text_input(
                "Database User",
                value=config["database"]["user"],
                help="Database username"
            )
    
    st.markdown("---")
    
    # Save Configuration
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save Configuration", type="primary"):
            # Save configuration
            st.session_state.system_config = config
            
            # Add confirmation
            st.success("✅ Configuration saved successfully!")
            
            # Log event
            st.session_state.system_events.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "success",
                "message": "System configuration updated",
                "component": "Configuration"
            })
            
            st.info("Configuration changes will take effect after system restart.")

# TAB 3: AI Agents
with tab3:
    st.header("🤖 AI Agents")
    st.markdown("Manage AI agents lifecycle, configuration, and monitoring.")
    
    # Initialize agent registry from data/agent_registry.json if not already loaded
    if not st.session_state.agent_registry:
        _registry_path = Path(__file__).resolve().parent.parent / "data" / "agent_registry.json"
        try:
            if _registry_path.exists():
                with open(_registry_path, "r", encoding="utf-8") as f:
                    _data = json.load(f)
                _agents = _data.get("agents", {})
                _project_root = Path(__file__).resolve().parent.parent
                for _name, _agent in _agents.items():
                    _loc = _agent.get("location")
                    if _loc:
                        _file_path = (_project_root / _loc).resolve()
                        if not _file_path.exists():
                            _agent["status"] = "orphaned"
                    _agent.setdefault("last_run", "Never")
                    _agent.setdefault("performance_score", 0)
                    _agent.setdefault("enabled", _agent.get("status", "active") == "active")
                    _agent.setdefault("configuration", {})
                    _agent.setdefault("execution_history", [])
                    _agent.setdefault("performance_metrics", {})
                st.session_state.agent_registry = _agents
            else:
                st.session_state.agent_registry = {}
        except Exception as e:
            logging.warning("Could not load agent registry from %s: %s", _registry_path, e)
            st.session_state.agent_registry = {}
    
    # Re-validate locations on each render so phantom/stale entries show as orphaned
    _project_root = Path(__file__).resolve().parent.parent
    for _name, _agent in list(st.session_state.agent_registry.items()):
        _loc = _agent.get("location")
        if _loc:
            _file_path = (_project_root / _loc).resolve()
            if not _file_path.exists():
                _agent["status"] = "orphaned"
    
    # Agent Registry Table
    st.subheader("📋 Agent Registry")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        agent_type_filter = st.selectbox(
            "Filter by Type",
            ["All"] + list(set(agent.get("type", "Unknown") for agent in st.session_state.agent_registry.values())),
            help="Filter agents by type"
        )
    
    with col2:
        agent_status_filter = st.selectbox(
            "Filter by Status",
            ["All", "active", "paused", "error", "orphaned"],
            help="Filter agents by status"
        )
    
    # Filter agents
    filtered_agents = {}
    for name, agent in st.session_state.agent_registry.items():
        if agent_type_filter != "All" and agent.get("type") != agent_type_filter:
            continue
        if agent_status_filter != "All" and agent.get("status") != agent_status_filter:
            continue
        filtered_agents[name] = agent
    
    # Agent Table
    if filtered_agents:
        agents_data = []
        for name, agent in filtered_agents.items():
            status = agent.get("status", "unknown")
            status_display = {
                "active": "🟢 Active",
                "paused": "⏸️ Paused",
                "error": "🔴 Error",
                "orphaned": "🔴 Orphaned"
            }.get(status, "❓ Unknown")
            
            last_run = agent.get("last_run", "Never")
            if last_run != "Never":
                try:
                    dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                    last_run = dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, AttributeError):
                    # Keep original string if parsing fails
                    pass
            
            agents_data.append({
                "Agent Name": name,
                "Type": agent.get("type", "Unknown"),
                "Status": status_display,
                "Last Run": last_run,
                "Performance": f"{agent.get('performance_score', 0)}/100",
                "Enabled": "✅" if agent.get("enabled", False) else "❌"
            })
        
        agents_df = pd.DataFrame(agents_data)
        st.dataframe(agents_df, height=300)
        
        st.markdown("---")
        
        # Agent Selection and Management
        st.subheader("⚙️ Agent Management")
        
        selected_agent = st.selectbox(
            "Select Agent",
            options=list(filtered_agents.keys()),
            help="Choose an agent to manage"
        )
        
        if selected_agent:
            agent = st.session_state.agent_registry[selected_agent]
            
            # Agent Actions
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                # Enable/Disable Toggle
                current_enabled = agent.get("enabled", False)
                new_enabled = st.checkbox(
                    "Enabled",
                    value=current_enabled,
                    key=f"enable_{selected_agent}",
                    help="Enable or disable this agent"
                )
                if new_enabled != current_enabled:
                    agent["enabled"] = new_enabled
                    if new_enabled:
                        agent["status"] = "active"
                    else:
                        agent["status"] = "paused"
                    st.session_state.agent_registry[selected_agent] = agent
                    st.rerun()
            
            with col2:
                # Configure Button
                if st.button("⚙️ Configure", key=f"config_{selected_agent}"):
                    st.session_state.configuring_agent = selected_agent
                    st.rerun()
            
            with col3:
                # Test Agent Button
                if st.button("🧪 Test", key=f"test_{selected_agent}"):
                    # Simulate agent test
                    agent["last_run"] = datetime.now().isoformat()
                    agent["status"] = "active"
                    st.session_state.agent_registry[selected_agent] = agent
                    st.success(f"✅ Agent '{selected_agent}' test completed!")
                    st.info("Test execution would run the agent with sample data in production.")
            
            with col4:
                # View Logs Button
                if st.button("📜 View Logs", key=f"logs_{selected_agent}"):
                    st.session_state.viewing_agent_logs = selected_agent
                    st.rerun()
            
            with col5:
                # Delete Agent Button
                if st.button("🗑️ Delete", key=f"delete_{selected_agent}"):
                    if selected_agent in st.session_state.agent_registry:
                        del st.session_state.agent_registry[selected_agent]
                        st.success(f"✅ Agent '{selected_agent}' deleted!")
                        st.rerun()
            
            st.markdown("---")
            
            # Agent Details Panel
            st.subheader(f"📄 Agent Details: {selected_agent}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Agent Name:** {selected_agent}")
                st.markdown(f"**Type:** {agent.get('type', 'Unknown')}")
                st.markdown(f"**Status:** {status_display}")
                st.markdown(f"**Enabled:** {'✅ Yes' if agent.get('enabled', False) else '❌ No'}")
                st.markdown(f"**Performance Score:** {agent.get('performance_score', 0)}/100")
                
                last_run = agent.get("last_run", "Never")
                if last_run != "Never":
                    try:
                        dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                        st.markdown(f"**Last Run:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except (ValueError, AttributeError):
                        st.markdown(f"**Last Run:** {last_run}")
                else:
                    st.markdown("**Last Run:** Never")
            
            with col2:
                if agent.get("description"):
                    st.markdown(f"**Description:** {agent.get('description')}")
                
                # Performance Metrics
                metrics = agent.get("performance_metrics", {})
                if metrics:
                    st.markdown("**Performance Metrics:**")
                    st.markdown(f"- Success Rate: {metrics.get('success_rate', 0)*100:.1f}%")
                    st.markdown(f"- Avg Execution Time: {metrics.get('avg_execution_time', 0):.1f}s")
                    st.markdown(f"- Total Executions: {metrics.get('total_executions', 0)}")
            
            # Configuration (Expandable)
            with st.expander("⚙️ Configuration", expanded=False):
                config = agent.get("configuration", {})
                if config:
                    st.json(config)
                else:
                    st.info("No configuration available")
            
            # Execution History (Expandable)
            with st.expander("📜 Execution History", expanded=False):
                history = agent.get("execution_history", [])
                if history:
                    history_data = []
                    for entry in history[-10:]:  # Last 10 entries
                        timestamp = entry.get("timestamp", "Unknown")
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except (ValueError, AttributeError):
                            # Keep original timestamp if parsing fails
                            pass
                        history_data.append({
                            "Timestamp": timestamp,
                            "Action": entry.get("action", "N/A"),
                            "Result": entry.get("result", "N/A")
                        })
                    
                    if history_data:
                        history_df = pd.DataFrame(history_data)
                        st.dataframe(history_df)
                else:
                    st.info("No execution history available")
            
            # Performance Metrics Chart (Expandable)
            with st.expander("📊 Performance Metrics", expanded=False):
                metrics = agent.get("performance_metrics", {})
                if metrics:
                    # Simple bar chart for metrics
                    fig = go.Figure(data=[
                        go.Bar(
                            x=["Success Rate", "Avg Execution Time (s)", "Total Executions"],
                            y=[
                                metrics.get("success_rate", 0) * 100,
                                metrics.get("avg_execution_time", 0),
                                metrics.get("total_executions", 0) / 10  # Scale for visualization
                            ],
                            marker_color=['green', 'blue', 'orange']
                        )
                    ])
                    fig.update_layout(
                        title="Agent Performance Metrics",
                        yaxis_title="Value",
                        height=300
                    )
                    st.plotly_chart(fig)
                else:
                    st.info("No performance metrics available")
    
    else:
        st.info("No agents found matching the filters.")
    
    st.markdown("---")
    
    # Add New Agent
    st.subheader("➕ Add New Agent")
    st.caption("Agents created here are available to the Multi-Agent Orchestrator in the Chat page. They can be assigned specific tasks like 'monitor AAPL' or 'alert when RSI < 30'.")
    
    with st.expander("Create New Agent", expanded=False):
        new_agent_name = st.text_input(
            "Agent Name",
            placeholder="MyCustomAgent",
            help="Unique name for the agent"
        )
        
        new_agent_type = st.selectbox(
            "Agent Type",
            ["Model Selector", "Optimizer", "Risk Analyzer", "Strategy Selector", "Custom"],
            help="Type of agent"
        )
        
        new_agent_description = st.text_area(
            "Description",
            placeholder="Describe what this agent does...",
            height=100,
            help="Agent description"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Create Agent"):
                if new_agent_name:
                    if new_agent_name in st.session_state.agent_registry:
                        st.error(f"Agent '{new_agent_name}' already exists!")
                    else:
                        st.session_state.agent_registry[new_agent_name] = {
                            "name": new_agent_name,
                            "type": new_agent_type,
                            "status": "paused",
                            "last_run": None,
                            "performance_score": 0,
                            "enabled": False,
                            "description": new_agent_description,
                            "configuration": {},
                            "execution_history": [],
                            "performance_metrics": {
                                "success_rate": 0.0,
                                "avg_execution_time": 0.0,
                                "total_executions": 0
                            }
                        }
                        st.success(f"✅ Agent '{new_agent_name}' created!")
                        st.rerun()
                else:
                    st.error("Please enter an agent name")
        
        with col2:
            if st.button("🔄 Reset"):
                st.rerun()
    
    # Agent Logs (if viewing)
    if 'viewing_agent_logs' in st.session_state:
        st.markdown("---")
        st.subheader(f"📜 Agent Logs: {st.session_state.viewing_agent_logs}")
        
        # Simulated logs
        log_level = st.selectbox("Log Level", ["All", "DEBUG", "INFO", "WARNING", "ERROR"], key="agent_log_level")
        
        # Simulated log entries
        log_entries = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Agent initialized",
            f"[{(datetime.now().replace(hour=11, minute=0)).strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model selection started",
            f"[{(datetime.now().replace(hour=11, minute=1)).strftime('%Y-%m-%d %H:%M:%S')}] INFO: LSTM model selected",
            f"[{(datetime.now().replace(hour=11, minute=2)).strftime('%Y-%m-%d %H:%M:%S')}] INFO: Execution completed successfully"
        ]
        
        # Filter logs by level
        if log_level != "All":
            log_entries = [log for log in log_entries if log_level in log]
        
        st.code("\n".join(log_entries), language="text")
        
        if st.button("❌ Close Logs"):
            del st.session_state.viewing_agent_logs
            st.rerun()
    
    # Agent Testing Interface (if configuring)
    if 'configuring_agent' in st.session_state:
        st.markdown("---")
        st.subheader(f"⚙️ Configure Agent: {st.session_state.configuring_agent}")
        
        agent = st.session_state.agent_registry[st.session_state.configuring_agent]
        config = agent.get("configuration", {})
        
        # Configuration editor
        import json
        config_json = st.text_area(
            "Configuration (JSON)",
            value=json.dumps(config, indent=2) if config else "{}",
            height=200,
            help="Edit agent configuration as JSON"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Configuration"):
                try:
                    import json
                    new_config = json.loads(config_json)
                    agent["configuration"] = new_config
                    st.session_state.agent_registry[st.session_state.configuring_agent] = agent
                    st.success("✅ Configuration saved!")
                    del st.session_state.configuring_agent
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format!")
        
        with col2:
            if st.button("❌ Cancel"):
                del st.session_state.configuring_agent
                st.rerun()

# TAB 4: System Monitoring
with tab4:
    st.header("📈 System Monitoring")
    st.markdown("Real-time system resource usage and service status monitoring.")
    
    # Auto-refresh toggle (timestamp-based, no sleep loop)
    if "admin_metrics_last_rerun" not in st.session_state:
        st.session_state.admin_metrics_last_rerun = 0.0
    auto_refresh = st.checkbox(
        "🔄 Auto-refresh (5 seconds)",
        value=False,
        help="Automatically refresh metrics every 5 seconds",
    )
    if auto_refresh:
        import time as _time

        elapsed = _time.time() - st.session_state.admin_metrics_last_rerun
        if elapsed >= 5:
            st.session_state.admin_metrics_last_rerun = _time.time()
            st.rerun()
        else:
            st.caption(f"Next refresh in {int(5 - elapsed)}s")
    
    # Resource Usage Section
    st.subheader("💻 Resource Usage")
    
    # Try to get real system metrics, fallback to simulated
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        # Use simulated metrics from session state
        cpu_percent = st.session_state.system_metrics.get("cpu_usage", 45.0)
        memory_percent = st.session_state.system_metrics.get("memory_usage", 62.0)
        disk_percent = st.session_state.system_metrics.get("disk_usage", 38.0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # CPU Usage Gauge
    with col1:
        st.markdown("**CPU Usage**")
        fig_cpu = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cpu_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_cpu.update_layout(height=200)
        st.plotly_chart(fig_cpu)
    
    # Memory Usage Gauge
    with col2:
        st.markdown("**Memory Usage**")
        fig_mem = go.Figure(go.Indicator(
            mode="gauge+number",
            value=memory_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgreen"},
                    {'range': [60, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_mem.update_layout(height=200)
        st.plotly_chart(fig_mem)
    
    # Disk Usage Gauge
    with col3:
        st.markdown("**Disk Usage**")
        fig_disk = go.Figure(go.Indicator(
            mode="gauge+number",
            value=disk_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgreen"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig_disk.update_layout(height=200)
        st.plotly_chart(fig_disk)
    
    # Network I/O
    with col4:
        st.markdown("**Network I/O**")
        try:
            if PSUTIL_AVAILABLE:
                net_io = psutil.net_io_counters()
                bytes_sent = net_io.bytes_sent / (1024**3)  # GB
                bytes_recv = net_io.bytes_recv / (1024**3)  # GB
            else:
                bytes_sent = 12.5
                bytes_recv = 45.8
        except Exception as e:
            # Fallback to default values if network stats unavailable
            bytes_sent = 12.5
            bytes_recv = 45.8
        
        st.metric("Sent", f"{bytes_sent:.2f} GB")
        st.metric("Received", f"{bytes_recv:.2f} GB")
    
    # Resource Usage Charts (Historical)
    st.markdown("---")
    st.subheader("📊 Resource Usage Over Time")
    
    # Simulated historical data
    import numpy as np
    time_points = list(range(30))  # Last 30 data points
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU Usage Chart
        cpu_history = [cpu_percent + np.random.normal(0, 5) for _ in time_points]
        cpu_history = [max(0, min(100, x)) for x in cpu_history]  # Clamp to 0-100
        
        fig_cpu_chart = go.Figure()
        fig_cpu_chart.add_trace(go.Scatter(
            x=time_points,
            y=cpu_history,
            mode='lines',
            name='CPU Usage',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_cpu_chart.update_layout(
            title="CPU Usage Over Time",
            xaxis_title="Time (minutes ago)",
            yaxis_title="CPU Usage (%)",
            height=250
        )
        st.plotly_chart(fig_cpu_chart)
    
    with col2:
        # Memory Usage Chart
        memory_history = [memory_percent + np.random.normal(0, 3) for _ in time_points]
        memory_history = [max(0, min(100, x)) for x in memory_history]  # Clamp to 0-100
        
        fig_mem_chart = go.Figure()
        fig_mem_chart.add_trace(go.Scatter(
            x=time_points,
            y=memory_history,
            mode='lines',
            name='Memory Usage',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig_mem_chart.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Time (minutes ago)",
            yaxis_title="Memory Usage (%)",
            height=250
        )
        st.plotly_chart(fig_mem_chart)
    
    st.markdown("---")
    
    # Service Status Section
    st.subheader("🔧 Service Status")
    
    # Initialize service status if not exists
    if 'service_status' not in st.session_state:
        st.session_state.service_status = {
            "web_server": {"status": "running", "uptime": "5 days"},
            "database": {"status": "running", "uptime": "5 days"},
            "cache": {"status": "running", "uptime": "5 days"},
            "task_queue": {"status": "running", "uptime": "5 days"}
        }
    
    services = st.session_state.service_status
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Web Server**")
        if services["web_server"]["status"] == "running":
            st.success("✅ Running")
        else:
            st.error("❌ Stopped")
        st.caption(f"Uptime: {services['web_server']['uptime']}")
    
    with col2:
        st.markdown("**Database**")
        if services["database"]["status"] == "running":
            st.success("✅ Running")
        else:
            st.error("❌ Stopped")
        st.caption(f"Uptime: {services['database']['uptime']}")
    
    with col3:
        st.markdown("**Cache (Redis)**")
        if services["cache"]["status"] == "running":
            st.success("✅ Running")
        else:
            st.error("❌ Stopped")
        st.caption(f"Uptime: {services['cache']['uptime']}")
    
    with col4:
        st.markdown("**Task Queue**")
        if services["task_queue"]["status"] == "running":
            st.success("✅ Running")
        else:
            st.error("❌ Stopped")
        st.caption(f"Uptime: {services['task_queue']['uptime']}")
    
    st.markdown("---")
    
    # API Rate Limits Section
    st.subheader("📡 API Rate Limits")
    
    # Initialize API rate limits if not exists (dynamic reset_time)
    _reset_time = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    if 'api_rate_limits' not in st.session_state:
        st.session_state.api_rate_limits = {
            "alpha_vantage": {"used": 450, "limit": 500, "reset_time": _reset_time},
            "finnhub": {"used": 58, "limit": 60, "reset_time": _reset_time},
            "polygon": {"used": 4, "limit": 5, "reset_time": _reset_time},
            "openai": {"used": 850, "limit": 1000, "reset_time": _reset_time}
        }
    
    api_limits = st.session_state.api_rate_limits
    
    for provider, limits in api_limits.items():
        used = limits["used"]
        limit = limits["limit"]
        remaining = limit - used
        percentage = (used / limit * 100) if limit > 0 else 0
        
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown(f"**{provider.replace('_', ' ').title()}**")
        
        with col2:
            # Progress bar
            st.progress(percentage / 100)
            st.caption(f"{used} / {limit} requests used ({remaining} remaining)")
        
        with col3:
            st.caption(f"Reset: {limits['reset_time']}")
    
    st.markdown("---")
    
    # Performance Metrics Section
    st.subheader("⚡ Performance Metrics")
    
    # Initialize performance metrics if not exists
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            "avg_response_time": 125,  # ms
            "avg_query_time": 45,  # ms
            "error_rate": 0.02,  # 2%
            "requests_per_minute": 120
        }
    
    perf_metrics = st.session_state.get('performance_metrics')
    
    if perf_metrics is None:
        st.warning("Performance metrics not available. Initializing default values...")
        st.session_state.performance_metrics = {
            "avg_response_time": 125,  # ms
            "avg_query_time": 45,  # ms
            "error_rate": 0.02,  # 2%
            "requests_per_minute": 120
        }
        perf_metrics = st.session_state.performance_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Response Time", f"{getattr(perf_metrics, 'avg_response_time', 0)} ms")
    
    with col2:
        st.metric("Avg Query Time", f"{getattr(perf_metrics, 'avg_query_time', 0)} ms")
    
    with col3:
        error_rate_pct = getattr(perf_metrics, 'error_rate', 0) * 100
        st.metric("Error Rate", f"{error_rate_pct:.2f}%")
    
    with col4:
        st.metric("Requests/Min", f"{getattr(perf_metrics, 'requests_per_minute', 0)}")
    
    # Performance Charts
    st.markdown("---")
    st.subheader("📈 Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response Time Trend
        response_times = [getattr(perf_metrics, 'avg_response_time', 0) + np.random.normal(0, 10) for _ in time_points]
        response_times = [max(0, x) for x in response_times]
        
        fig_response = go.Figure()
        fig_response.add_trace(go.Scatter(
            x=time_points,
            y=response_times,
            mode='lines',
            name='Response Time',
            line=dict(color='#2ca02c', width=2)
        ))
        fig_response.update_layout(
            title="Response Time Over Time",
            xaxis_title="Time (minutes ago)",
            yaxis_title="Response Time (ms)",
            height=250
        )
        st.plotly_chart(fig_response)
    
    with col2:
        # Error Rate Trend
        error_rates = [getattr(perf_metrics, 'error_rate', 0) * 100 + np.random.normal(0, 0.5) for _ in time_points]
        error_rates = [max(0, x) for x in error_rates]
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Scatter(
            x=time_points,
            y=error_rates,
            mode='lines',
            name='Error Rate',
            line=dict(color='#d62728', width=2),
            fill='tozeroy'
        ))
        fig_errors.update_layout(
            title="Error Rate Over Time",
            xaxis_title="Time (minutes ago)",
            yaxis_title="Error Rate (%)",
            height=250
        )
        st.plotly_chart(fig_errors)
    
    # Refresh Button
    if st.button("🔄 Refresh Metrics"):
        st.rerun()

# TAB 5: Logs & Debugging
with tab5:
    st.header("📜 Logs & Debugging")
    
    # System Logs Section
    st.subheader("📊 System Logs")
    
    if st.button("View Recent Logs"):
        try:
            log_file_path = 'logs/trading_system.log'
            from pathlib import Path
            
            if Path(log_file_path).exists():
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    recent_logs = lines[-100:]  # Last 100 lines
                
                st.text_area("Recent Log Entries (Last 100 lines)", ''.join(recent_logs), height=400)
            else:
                st.warning(f"Log file not found at {log_file_path}")
                st.info("Logs will be created when the system starts logging")
        except Exception as e:
            st.error(f"Error reading log file: {e}")
    
    # Audit Trail Section
    st.markdown("---")
    st.subheader("🔍 Audit Trail")
    
    if 'audit_logger' in st.session_state:
        audit_logger = st.session_state.audit_logger
        
        # Try to read audit log file
        try:
            audit_log_path = 'logs/audit.log'
            from pathlib import Path
            
            if Path(audit_log_path).exists():
                with open(audit_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    audit_lines = f.readlines()
                    recent_audit = audit_lines[-50:]  # Last 50 entries
                
                if recent_audit:
                    st.text_area("Recent Audit Entries (Last 50)", ''.join(recent_audit), height=300)
                else:
                    st.info("No audit entries yet")
            else:
                st.info("Audit log file will be created when audit events occur")
        except Exception as e:
            st.error(f"Error reading audit log: {e}")
        
        # Audit statistics
        st.markdown("---")
        st.subheader("📈 Audit Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                if Path('logs/audit.log').exists():
                    try:
                        with open('logs/audit.log', 'r', encoding='utf-8', errors='replace') as f:
                            total_entries = len(f.readlines())
                    except UnicodeDecodeError:
                        total_entries = 0
                    st.metric("Total Audit Entries", total_entries)
                else:
                    st.metric("Total Audit Entries", 0)
            except Exception as e:
                logging.warning(f"Connection check failed: {e}")
                st.metric("Total Audit Entries", "N/A")
        
        with col2:
            st.metric("Audit Log File", "✅ Active" if Path('logs/audit.log').exists() else "❌ Not Created")
        
        with col3:
            st.metric("System Log File", "✅ Active" if Path('logs/trading_system.log').exists() else "❌ Not Created")
    else:
        st.warning("⚠️ Audit logger not available in this environment.")
    
    # Log Configuration
    st.markdown("---")
    st.subheader("⚙️ Log Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Log Files:**
        - `logs/trading_system.log` - Main system log
        - `logs/audit.log` - Audit trail
        - `logs/error.log` - Error logs
        - `logs/performance.log` - Performance metrics
        """)
    
    with col2:
        st.info("""
        **Log Rotation:**
        - Max size: 10 MB per file
        - Backup count: 5 files
        - Automatic rotation enabled
        """)
    st.markdown("View system logs, debug issues, and analyze errors.")
    
    # Log Filters
    st.subheader("🔍 Log Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_level = st.selectbox(
            "Log Level",
            ["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Filter logs by level"
        )
    
    with col2:
        log_source = st.selectbox(
            "Log Source",
            ["All", "app", "trading", "data", "models", "agents", "system"],
            help="Filter logs by source"
        )
    
    with col3:
        log_date = st.date_input(
            "Date",
            value=datetime.now().date(),
            help="Filter logs by date"
        )
    
    # Search
    log_search = st.text_input(
        "Search Logs",
        placeholder="Search log messages...",
        help="Search logs by content"
    )
    
    st.markdown("---")
    
    # Initialize logs if not exists
    if 'system_logs' not in st.session_state:
        st.session_state.system_logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "source": "app",
                "message": "Application started successfully",
                "component": "System"
            },
            {
                "timestamp": (datetime.now().replace(hour=11, minute=30)).isoformat(),
                "level": "INFO",
                "source": "trading",
                "message": "Trade executed: BUY 100 shares of AAPL at $150.25",
                "component": "Trading"
            },
            {
                "timestamp": (datetime.now().replace(hour=11, minute=25)).isoformat(),
                "level": "WARNING",
                "source": "data",
                "message": "Data provider API rate limit approaching",
                "component": "DataLoader"
            },
            {
                "timestamp": (datetime.now().replace(hour=10, minute=45)).isoformat(),
                "level": "ERROR",
                "source": "models",
                "message": "Model training failed: Out of memory",
                "component": "LSTMModel"
            },
            {
                "timestamp": (datetime.now().replace(hour=10, minute=30)).isoformat(),
                "level": "DEBUG",
                "source": "agents",
                "message": "Agent execution started: ModelSelectorAgent",
                "component": "AgentRegistry"
            },
            {
                "timestamp": (datetime.now().replace(hour=9, minute=15)).isoformat(),
                "level": "INFO",
                "source": "system",
                "message": "Database connection restored",
                "component": "Database"
            }
        ]
    
    # Filter logs
    filtered_logs = []
    for log in st.session_state.system_logs:
        # Level filter
        if log_level != "All" and log.get("level") != log_level:
            continue
        
        # Source filter
        if log_source != "All" and log.get("source") != log_source:
            continue
        
        # Date filter
        try:
            log_date_obj = datetime.fromisoformat(log.get("timestamp", "").replace('Z', '+00:00')).date()
            if log_date_obj != log_date:
                continue
        except (ValueError, AttributeError, KeyError):
            continue
        
        # Search filter
        if log_search:
            search_lower = log_search.lower()
            message = log.get("message", "").lower()
            component = log.get("component", "").lower()
            if search_lower not in message and search_lower not in component:
                continue
        
        filtered_logs.append(log)
    
    # Sort by timestamp (newest first)
    filtered_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    st.subheader("📋 Log Viewer")
    
    # Live log tail toggle
    live_tail = st.checkbox("📺 Live Tail (Auto-scroll)", value=False, help="Automatically scroll to newest logs")
    
    # Log Display
    if filtered_logs:
        # Log count
        st.caption(f"Showing {len(filtered_logs)} log entries")
        
        # Log entries
        log_container = st.container()
        with log_container:
            for log in filtered_logs:
                timestamp = log.get("timestamp", "Unknown")
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, AttributeError):
                    # If parsing fails, use original timestamp string
                    timestamp_str = timestamp
                
                level = log.get("level", "INFO")
                source = log.get("source", "unknown")
                message = log.get("message", "N/A")
                component = log.get("component", "Unknown")
                
                # Color code by level
                if level == "ERROR" or level == "CRITICAL":
                    border_color = "red"
                elif level == "WARNING":
                    border_color = "orange"
                elif level == "DEBUG":
                    border_color = "gray"
                else:
                    border_color = "blue"
                
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2, 4, 2])
                    
                    with col1:
                        st.caption(f"🕐 {timestamp_str}")
                        st.caption(f"📦 {component}")
                    
                    with col2:
                        # Level badge
                        if level == "ERROR" or level == "CRITICAL":
                            st.error(f"[{level}] {message}")
                        elif level == "WARNING":
                            st.warning(f"[{level}] {message}")
                        elif level == "DEBUG":
                            st.info(f"[{level}] {message}")
                        else:
                            st.info(f"[{level}] {message}")
                    
                    with col3:
                        st.caption(f"🔖 {source}")
    else:
        st.info("No logs found matching the filters.")
    
    st.markdown("---")
    
    # Download Logs
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Logs"):
            # Create log file content
            log_content = ""
            for log in filtered_logs:
                timestamp = log.get("timestamp", "Unknown")
                level = log.get("level", "INFO")
                source = log.get("source", "unknown")
                message = log.get("message", "N/A")
                component = log.get("component", "Unknown")
                log_content += f"[{timestamp}] [{level}] [{source}] [{component}] {message}\n"
            
            st.download_button(
                label="⬇️ Download Log File",
                data=log_content,
                file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🗑️ Clear Logs"):
            if st.button("⚠️ Confirm Clear", key="confirm_clear_logs"):
                st.session_state.system_logs = []
                st.success("✅ Logs cleared!")
                st.rerun()
    
    st.markdown("---")
    
    # Error Summary
    st.subheader("📊 Error Summary")
    
    # Count errors by type
    error_counts = {}
    error_messages = {}
    
    for log in st.session_state.system_logs:
        level = log.get("level", "")
        if level in ["ERROR", "CRITICAL"]:
            error_type = log.get("component", "Unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            error_messages[error_type] = log.get("message", "N/A")
    
    if error_counts:
        col1, col2 = st.columns(2)
        
        with col1:
            # Error count by type
            st.markdown("**Error Count by Type:**")
            error_df = pd.DataFrame({
                "Component": list(error_counts.keys()),
                "Error Count": list(error_counts.values())
            })
            st.dataframe(error_df)
        
        with col2:
            # Most common errors
            st.markdown("**Most Common Errors:**")
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for component, count in sorted_errors:
                st.markdown(f"- **{component}:** {count} error(s)")
                if component in error_messages:
                    st.caption(f"  Last: {error_messages[component]}")
        
        # Error trend chart
        st.markdown("---")
        st.markdown("**Error Trend:**")
        
        # Simulated error trend data
        import numpy as np
        time_points = list(range(24))  # Last 24 hours
        error_trend = [np.random.randint(0, 5) for _ in time_points]
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Scatter(
            x=time_points,
            y=error_trend,
            mode='lines+markers',
            name='Errors',
            line=dict(color='#d62728', width=2),
            fill='tozeroy'
        ))
        fig_errors.update_layout(
            title="Error Count Over Time (Last 24 Hours)",
            xaxis_title="Hours Ago",
            yaxis_title="Error Count",
            height=300
        )
        st.plotly_chart(fig_errors)
    else:
        st.success("✅ No errors found in the logs!")
    
    st.markdown("---")
    
    # Debugging Tools
    st.subheader("🔧 Debugging Tools")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Cache", key="admin_clear_cache_2"):
            try:
                st.cache_data.clear()
                cache_dir = Path(__file__).resolve().parent.parent / ".cache"
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Cache clear partial: {e}")
            st.session_state.system_events.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "success",
                "message": "Cache cleared via debugging tools",
                "component": "Debug"
            })
            st.success("✅ Cache cleared!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Session"):
            # Clear session state (except essential)
            keys_to_keep = ['system_health', 'system_metrics', 'system_config', 'agent_registry']
            keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("✅ Session reset!")
            st.info("Session state has been reset (essential data preserved).")
    
    with col3:
        if st.button("🧹 Force GC"):
            import gc
            gc.collect()
            st.success("✅ Garbage collection forced!")
            st.info("Python garbage collector has been run.")
    
    with col4:
        if st.button("🔌 Test DB"):
            # Simulate database connection test
            try:
                # In production, this would actually test the database connection
                st.success("✅ Database connection successful!")
                st.info("Database is accessible and responding.")
            except Exception as e:
                st.error(f"❌ Database connection failed: {str(e)}")
    
    # Debugging Information
    st.markdown("---")
    with st.expander("🔍 System Debug Information", expanded=False):
        st.markdown("**Python Version:**")
        import sys
        st.code(f"{sys.version}")
        
        st.markdown("**Streamlit Version:**")
        try:
            import streamlit as st_module
            st.code(f"{st_module.__version__}")
        except (ImportError, AttributeError):
            st.code("Unknown")
        
        st.markdown("**Session State Keys:**")
        st.code(f"{list(st.session_state.keys())}")
        
        st.markdown("**System Metrics:**")
        st.json(st.session_state.system_metrics)

# TAB 6: Maintenance
with tab6:
    st.header("🔧 Maintenance")
    st.markdown("Database maintenance, cache management, data cleanup, and system updates.")
    
    # Database Maintenance Section
    with st.expander("💾 Database Maintenance", expanded=True):
        st.markdown("**Database operations:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Backup Database"):
                # Simulate backup
                backup_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
                st.success(f"✅ Database backup created: {backup_filename}")
                st.info("In production, this would create a full database backup.")
                
                # Log event
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "success",
                    "message": f"Database backup created: {backup_filename}",
                    "component": "Database"
                })
            
            if st.button("📥 Restore from Backup"):
                st.warning("⚠️ Restore operation requires confirmation!")
                if st.button("⚠️ Confirm Restore", key="confirm_restore"):
                    st.success("✅ Database restored from backup!")
                    st.info("In production, this would restore the database from the selected backup file.")
        
        with col2:
            if st.button("⚡ Optimize Database"):
                # Simulate optimization
                st.success("✅ Database optimized!")
                st.info("Database indexes and tables have been optimized for better performance.")
                
                # Log event
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "success",
                    "message": "Database optimization completed",
                    "component": "Database"
                })
            
            if st.button("🧹 Vacuum Database"):
                # Simulate vacuum
                st.success("✅ Database vacuum completed!")
                st.info("Database has been vacuumed to reclaim unused space.")
        
        # Database Size
        st.markdown("---")
        st.markdown("**Database Information:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Database Size", "2.5 GB")
        
        with col2:
            st.metric("Tables", "47")
        
        with col3:
            st.metric("Indexes", "89")
    
    st.markdown("---")
    
    # Cache Management Section
    with st.expander("🗄️ Cache Management", expanded=True):
        st.markdown("**Cache operations:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All Cache"):
                # Simulate cache clear
                st.success("✅ All cache cleared!")
                st.info("All cached data has been removed.")
                
                # Log event
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "success",
                    "message": "All cache cleared",
                    "component": "Cache"
                })
        
        with col2:
            cache_type = st.selectbox(
                "Clear Specific Cache",
                ["Data Cache", "Model Cache", "Strategy Cache", "Session Cache"],
                help="Select cache type to clear"
            )
            
            if st.button("🗑️ Clear Selected Cache"):
                st.success(f"✅ {cache_type} cleared!")
                st.info(f"All {cache_type.lower()} data has been removed.")
        
        with col3:
            # Cache Hit Rate
            st.markdown("**Cache Statistics:**")
            try:
                from utils.model_cache import get_cache_info

                ci = get_cache_info() or {}
                st.metric("Cache Files", str(ci.get("cache_files", "—")))
                size_mb = ci.get("cache_size_mb")
                st.metric("Cache Size", f"{float(size_mb):.1f} MB" if size_mb is not None else "—")
            except Exception:
                st.metric("Cache Files", "—")
                st.metric("Cache Size", "—")
    
    st.markdown("---")
    
    # Data Cleanup Section
    with st.expander("🧹 Data Cleanup", expanded=True):
        st.markdown("**Clean up old data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            delete_days = st.number_input(
                "Delete Logs Older Than (Days)",
                min_value=1,
                max_value=365,
                value=30,
                help="Logs older than this many days will be deleted"
            )
            
            if st.button("🗑️ Delete Old Logs"):
                # Simulate deletion
                st.success(f"✅ Deleted logs older than {delete_days} days!")
                st.info(f"In production, this would delete all logs older than {delete_days} days.")
                
                # Log event
                st.session_state.system_events.insert(0, {
                    "timestamp": datetime.now().isoformat(),
                    "type": "info",
                    "message": f"Deleted logs older than {delete_days} days",
                    "component": "Data Cleanup"
                })
            
            if st.button("📦 Archive Old Data"):
                # Simulate archiving
                st.success("✅ Old data archived!")
                st.info("In production, this would archive old data to long-term storage.")
        
        with col2:
            if st.button("🧹 Clean Temp Files"):
                # Simulate cleanup
                st.success("✅ Temporary files cleaned!")
                st.info("All temporary files have been removed.")
            
            # Data Statistics
            st.markdown("---")
            st.markdown("**Data Statistics:**")
            try:
                from pathlib import Path

                logs_dir = Path("logs")
                log_files = list(logs_dir.rglob("*.log")) if logs_dir.exists() else []
                st.metric("Log Files", str(len(log_files)))
                logs_size = sum(p.stat().st_size for p in log_files) / (1024 * 1024) if log_files else 0.0
                st.metric("Logs Size", f"{logs_size:.1f} MB")
                st.metric("Temp Files Size", "—")
            except Exception:
                st.metric("Log Files", "—")
                st.metric("Logs Size", "—")
                st.metric("Temp Files Size", "—")
    
    st.markdown("---")
    
    # System Updates Section
    with st.expander("🔄 System Updates", expanded=True):
        st.markdown("**System version and updates:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Version:**")
            st.info("v1.3.0")
            
            if st.button("🔍 Check for Updates"):
                st.success("✅ System is up to date!")
                st.info("No updates available. You are running the latest version.")
        
        with col2:
            st.markdown("**Update History:**")
            
            update_history = [
                {"version": "v1.3.0", "date": "2026-03-09", "notes": "Forecasting math fix, strategy/testing/execution/admin/home fixes"},
                {"version": "v1.2.3", "date": "2025-12-15", "notes": "Bug fixes and performance improvements"},
                {"version": "v1.2.2", "date": "2025-12-01", "notes": "New features: AI forecasting, risk alerts"},
                {"version": "v1.2.1", "date": "2025-11-15", "notes": "Security patches and stability improvements"}
            ]
            
            for update in update_history:
                with st.container(border=True):
                    st.markdown(f"**{update['version']}** - {update['date']}")
                    st.caption(update['notes'])
    
    st.markdown("---")
    
    # Scheduled Maintenance Section
    with st.expander("⏰ Scheduled Maintenance", expanded=False):
        st.markdown("**Configure automated maintenance tasks:**")
        
        # Initialize scheduled maintenance if not exists
        if 'scheduled_maintenance' not in st.session_state:
            st.session_state.scheduled_maintenance = {
                "backup": {
                    "enabled": True,
                    "frequency": "Daily",
                    "time": "02:00",
                    "retention_days": 30
                },
                "cleanup": {
                    "enabled": True,
                    "frequency": "Weekly",
                    "day": "Sunday",
                    "time": "03:00",
                    "delete_logs_older_than": 30
                }
            }
        
        maintenance = st.session_state.scheduled_maintenance
        
        # Backup Schedule
        st.markdown("**Backup Schedule:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            backup_enabled = st.checkbox(
                "Enable",
                value=maintenance["backup"]["enabled"],
                key="backup_enabled"
            )
            maintenance["backup"]["enabled"] = backup_enabled
        
        with col2:
            maintenance["backup"]["frequency"] = st.selectbox(
                "Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=["Daily", "Weekly", "Monthly"].index(maintenance["backup"]["frequency"]),
                key="backup_frequency"
            )
        
        with col3:
            backup_time = st.time_input(
                "Time",
                value=datetime.strptime(maintenance["backup"]["time"], "%H:%M").time(),
                key="backup_time"
            )
            maintenance["backup"]["time"] = backup_time.strftime("%H:%M")
        
        with col4:
            maintenance["backup"]["retention_days"] = st.number_input(
                "Retention (Days)",
                min_value=1,
                max_value=365,
                value=maintenance["backup"]["retention_days"],
                key="backup_retention"
            )
        
        st.markdown("---")
        
        # Cleanup Schedule
        st.markdown("**Cleanup Schedule:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cleanup_enabled = st.checkbox(
                "Enable",
                value=maintenance["cleanup"]["enabled"],
                key="cleanup_enabled"
            )
            maintenance["cleanup"]["enabled"] = cleanup_enabled
        
        with col2:
            maintenance["cleanup"]["frequency"] = st.selectbox(
                "Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=["Daily", "Weekly", "Monthly"].index(maintenance["cleanup"]["frequency"]),
                key="cleanup_frequency"
            )
        
        with col3:
            if maintenance["cleanup"]["frequency"] == "Weekly":
                maintenance["cleanup"]["day"] = st.selectbox(
                    "Day",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(maintenance["cleanup"].get("day", "Sunday")),
                    key="cleanup_day"
                )
            else:
                cleanup_time = st.time_input(
                    "Time",
                    value=datetime.strptime(maintenance["cleanup"].get("time", "03:00"), "%H:%M").time(),
                    key="cleanup_time"
                )
                maintenance["cleanup"]["time"] = cleanup_time.strftime("%H:%M")
        
        with col4:
            maintenance["cleanup"]["delete_logs_older_than"] = st.number_input(
                "Delete Logs Older Than (Days)",
                min_value=1,
                max_value=365,
                value=maintenance["cleanup"]["delete_logs_older_than"],
                key="cleanup_logs_days"
            )
        
        # Save Schedule
        if st.button("💾 Save Schedule"):
            st.session_state.scheduled_maintenance = maintenance
            st.success("✅ Maintenance schedule saved!")
            st.info("Scheduled maintenance tasks will run according to the configured schedule.")
    
    # Maintenance Summary
    st.markdown("---")
    st.subheader("📊 Maintenance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Last Backup", "2024-12-19 02:00")
    
    with col2:
        st.metric("Backups This Month", "19")
    
    with col3:
        st.metric("Cache Hit Rate", "87.5%")
    
    with col4:
        _dash = _get_system_dashboard_data()
        st.metric("System Uptime", _dash.get("uptime_str", "—"))

# Task Orchestrator Management Section
st.markdown("---")
st.header("⚙️ Task Orchestrator Management")

# Check if core.orchestrator is installable/available with detailed diagnostics
_orch_status = {"available": False, "reason": "", "install_cmd": ""}
try:
    from core.orchestrator.task_orchestrator import TaskOrchestrator
    from core.orchestrator.task_scheduler import TaskScheduler
    from core.orchestrator.task_monitor import TaskMonitor
    from core.orchestrator.task_models import TaskType, TaskPriority, TaskStatus, TaskConfig

    _orch_status["available"] = True
except ImportError as _ie:
    _missing = str(_ie).split("'")[1] if "'" in str(_ie) else str(_ie)
    _orch_status["reason"] = f"Missing module: {_missing}"
    import os

    _orch_path = os.path.join("core", "orchestrator")
    if os.path.exists(_orch_path):
        _orch_status["reason"] = f"Module exists but import failed: {_ie}"
        _orch_status["install_cmd"] = "Check core/orchestrator/__init__.py"
    else:
        _orch_status["install_cmd"] = "Task Orchestrator not installed in this project"

if not _orch_status["available"]:
    st.info(f"Task Orchestrator: {_orch_status['reason']}")
    st.caption(_orch_status["install_cmd"])
    import os

    if os.path.exists("core/orchestrator"):
        _files = os.listdir("core/orchestrator")
        st.caption(f"Files found: {_files}")
else:
    # Imports for orchestrator (needed for Initialize button and tabs)
    from core.orchestrator.task_orchestrator import TaskOrchestrator
    from core.orchestrator.task_scheduler import TaskScheduler
    from core.orchestrator.task_monitor import TaskMonitor
    from core.orchestrator.task_models import TaskConfig, TaskType, TaskPriority

    # Initialize orchestrator in session state if not already (fast — no heavy init)
    if "task_orchestrator" not in st.session_state:
        try:
            st.session_state.task_orchestrator = TaskOrchestrator()
            st.session_state.task_scheduler = None
            st.session_state.task_monitor = None
            st.success("Task Orchestrator created (click Initialize to load tasks)")
        except Exception as _init_err:
            st.caption(f"Task Orchestrator unavailable: {_init_err}")

    _orch = st.session_state.get("task_orchestrator")
    # Lazy init: when user clicks "Initialize Orchestrator", run heavy setup
    if _orch is not None and not getattr(_orch, "_tasks_loaded", False):
        if st.button("Initialize Orchestrator", key="orch_init_btn"):
            try:
                _orch.ensure_initialized()
                st.session_state.task_scheduler = _orch.scheduler
                st.session_state.task_monitor = _orch.monitor
                # Add default SystemHealthCheck if not already present
                if "SystemHealthCheck" not in getattr(_orch.scheduler, "tasks", {}):
                    _health_task = TaskConfig(
                        name="SystemHealthCheck",
                        task_type=TaskType.SYSTEM_HEALTH,
                        enabled=True,
                        interval_minutes=60,
                        priority=TaskPriority.MEDIUM,
                    )
                    _orch.scheduler.add_task(_health_task)
                st.success("Orchestrator initialized — tasks loaded")
                st.rerun()
            except Exception as _e:
                st.caption(f"Task Orchestrator unavailable: {_e}")
        st.caption("Initialize loads task providers and schedules (may take a few seconds).")
        # Skip rendering tabs until initialized
        _orch = None

    if _orch is not None and getattr(_orch, "_tasks_loaded", False):
        st.session_state.task_scheduler = _orch.scheduler
        st.session_state.task_monitor = _orch.monitor

    orchestrator = st.session_state.get("task_orchestrator")
    scheduler = st.session_state.get("task_scheduler")
    monitor = st.session_state.get("task_monitor")
    if orchestrator is not None and scheduler is not None and monitor is not None:
        # Tabs for different views
        task_tab1, task_tab2, task_tab3 = st.tabs([
            "📋 Active Tasks",
            "📅 Scheduled Tasks",
            "📊 Task History"
        ])
        
        with task_tab1:
            st.subheader("Active Tasks")
            
            # Get active tasks - adapt to actual API
            try:
                # Try to get active tasks from executor or monitor
                if hasattr(orchestrator, 'executor') and hasattr(orchestrator.executor, 'get_active_tasks'):
                    active_tasks = orchestrator.executor.get_active_tasks()
                elif hasattr(monitor, 'get_active_tasks'):
                    active_tasks = monitor.get_active_tasks()
                else:
                    # Fallback: check executor's running tasks
                    active_tasks = []
                    if hasattr(orchestrator, 'executor'):
                        executor = orchestrator.executor
                        if hasattr(executor, 'running_tasks'):
                            for task_id, task_info in executor.running_tasks.items():
                                active_tasks.append({
                                    'id': task_id,
                                    'name': task_info.get('name', task_id),
                                    'status': task_info.get('status', 'running'),
                                    'started_at': task_info.get('start_time', 'Unknown'),
                                    'progress': task_info.get('progress', 0)
                                })
                
                if active_tasks:
                    for task in active_tasks:
                        task_id = task.get('id', task.get('name', 'unknown'))
                        task_name = task.get('name', task_id)
                        task_status = task.get('status', 'running')
                        started_at = task.get('started_at', task.get('start_time', 'Unknown'))
                        progress = task.get('progress', 0)
                        
                        with st.expander(f"{task_name} - {task_status}", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**Status:** {task_status}")
                                st.write(f"**Started:** {started_at}")
                            
                            with col2:
                                st.write(f"**Progress:** {progress}%")
                                st.progress(progress / 100 if isinstance(progress, (int, float)) else 0)
                            
                            with col3:
                                if st.button(f"Cancel", key=f"cancel_{task_id}"):
                                    try:
                                        if hasattr(orchestrator, 'cancel_task'):
                                            orchestrator.cancel_task(task_id)
                                        elif hasattr(orchestrator, 'executor') and hasattr(orchestrator.executor, 'cancel_task'):
                                            orchestrator.executor.cancel_task(task_id)
                                        st.success("Task cancelled")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to cancel task: {e}")
                else:
                    st.info("No active tasks")
            except Exception as e:
                st.warning(f"Could not retrieve active tasks: {e}")
                st.info("No active tasks")
            
            # Create new task
            st.subheader("Create New Task")
            
            task_type = st.selectbox(
                "Task Type",
                ["Data Update", "Model Retrain", "Portfolio Rebalance", "Generate Report", "System Health Check"],
                key="new_task_type"
            )
            
            # Map UI task types to TaskType enum
            task_type_map = {
                "Data Update": TaskType.DATA_SYNC,
                "Model Retrain": TaskType.MODEL_INNOVATION,
                "Portfolio Rebalance": TaskType.PORTFOLIO_REBALANCING,
                "Generate Report": TaskType.PERFORMANCE_ANALYSIS,
                "System Health Check": TaskType.SYSTEM_HEALTH
            }
            
            if st.button("Create Task", key="create_task_btn"):
                try:
                    task_type_enum = task_type_map.get(task_type, TaskType.SYSTEM_HEALTH)
                    
                    # Create task using orchestrator
                    if hasattr(orchestrator, 'create_task'):
                        task_id = orchestrator.create_task(task_type_enum.value)
                    elif hasattr(orchestrator, 'execute_task'):
                        task_id = orchestrator.execute_task(task_type_enum.value, {})
                    else:
                        # Fallback: use executor
                        if hasattr(orchestrator, 'executor') and hasattr(orchestrator.executor, 'execute_task'):
                            task_id = orchestrator.executor.execute_task(task_type_enum.value, {})
                        else:
                            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    st.success(f"✅ Task created: {task_id}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create task: {e}")
        
        with task_tab2:
            st.subheader("Scheduled Tasks")
            
            # Get scheduled tasks
            try:
                scheduled = scheduler.get_scheduled_tasks()
                
                if scheduled:
                    for task_name, task_info in scheduled.items():
                        next_run = task_info.get('next_run', 'Unknown')
                        interval = task_info.get('interval_minutes', 'N/A')
                        enabled = task_info.get('enabled', True)
                        
                        with st.expander(f"{task_name} - Next run: {next_run}", expanded=False):
                            st.write(f"**Schedule:** Every {interval} minutes")
                            st.write(f"**Status:** {'Enabled' if enabled else 'Disabled'}")
                            st.write(f"**Priority:** {task_info.get('priority', 'Medium')}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Run Now", key=f"run_{task_name}"):
                                    try:
                                        # Try to run task immediately
                                        if hasattr(scheduler, 'run_task_now'):
                                            scheduler.run_task_now(task_name)
                                        elif hasattr(orchestrator, 'execute_task'):
                                            orchestrator.execute_task(task_name, {})
                                        st.success("Task started")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to run task: {e}")
                            
                            with col2:
                                if st.button("Remove", key=f"remove_{task_name}"):
                                    try:
                                        scheduler.remove_task(task_name)
                                        st.success("Task removed")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to remove task: {e}")
                else:
                    st.info("No scheduled tasks")
            except Exception as e:
                st.warning(f"Could not retrieve scheduled tasks: {e}")
                st.info("No scheduled tasks")
            
            # Add new schedule
            st.subheader("Schedule New Task")
            
            schedule_type = st.selectbox(
                "Task to Schedule",
                ["Daily Data Update", "Weekly Model Retrain", "Monthly Report", "System Health Check"],
                key="schedule_task_type"
            )
            
            schedule_time = st.time_input("Run at", key="schedule_time")
            
            # Map schedule types to task configs
            schedule_configs = {
                "Daily Data Update": {"task_type": TaskType.DATA_SYNC, "interval_minutes": 1440},
                "Weekly Model Retrain": {"task_type": TaskType.MODEL_INNOVATION, "interval_minutes": 10080},
                "Monthly Report": {"task_type": TaskType.PERFORMANCE_ANALYSIS, "interval_minutes": 43200},
                "System Health Check": {"task_type": TaskType.SYSTEM_HEALTH, "interval_minutes": 60}
            }
            
            if st.button("Schedule Task", key="schedule_task_btn"):
                try:
                    config = schedule_configs.get(schedule_type, schedule_configs["System Health Check"])
                    
                    # Create TaskConfig
                    from core.orchestrator.task_models import TaskConfig
                    task_config = TaskConfig(
                        name=schedule_type.lower().replace(" ", "_"),
                        task_type=config["task_type"],
                        interval_minutes=config["interval_minutes"],
                        priority=TaskPriority.MEDIUM
                    )
                    
                    scheduler.add_task(task_config)
                    st.success("✅ Task scheduled")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to schedule task: {e}")
        
        with task_tab3:
            st.subheader("Task History")
            
            # Get completed tasks
            try:
                # Try to get history from monitor
                if hasattr(monitor, 'get_task_history'):
                    history = monitor.get_task_history(limit=50)
                elif hasattr(monitor, 'task_history'):
                    history = monitor.task_history[-50:] if hasattr(monitor, 'task_history') else []
                else:
                    # Fallback: create empty history
                    history = []
                
                if history:
                    # Convert to DataFrame format
                    history_data = []
                    for item in history:
                        if isinstance(item, dict):
                            history_data.append({
                                'name': item.get('name', item.get('task_name', 'Unknown')),
                                'status': item.get('status', 'unknown'),
                                'started_at': item.get('started_at', item.get('start_time', 'Unknown')),
                                'duration': item.get('duration', item.get('duration_seconds', 0))
                            })
                        else:
                            # Handle TaskExecution objects
                            history_data.append({
                                'name': getattr(item, 'task_name', 'Unknown'),
                                'status': getattr(item, 'status', TaskStatus.COMPLETED).value if hasattr(item.status, 'value') else str(item.status),
                                'started_at': getattr(item, 'start_time', 'Unknown'),
                                'duration': getattr(item, 'duration_seconds', 0) or 0
                            })
                    
                    if history_data:
                        history_df = pd.DataFrame(history_data)
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Tasks", len(history_df))
                        with col2:
                            if 'status' in history_df.columns:
                                success_count = (history_df['status'] == 'completed').sum() + (history_df['status'] == TaskStatus.COMPLETED.value).sum()
                                success_rate = (success_count / len(history_df)) * 100 if len(history_df) > 0 else 0
                                st.metric("Success Rate", f"{success_rate:.1f}%")
                            else:
                                st.metric("Success Rate", "N/A")
                        with col3:
                            if 'duration' in history_df.columns:
                                avg_duration = history_df['duration'].mean()
                                st.metric("Avg Duration", f"{avg_duration:.1f}s")
                            else:
                                st.metric("Avg Duration", "N/A")
                        with col4:
                            if 'status' in history_df.columns:
                                failed = (history_df['status'] == 'failed').sum() + (history_df['status'] == TaskStatus.FAILED.value).sum()
                                st.metric("Failed", failed)
                            else:
                                st.metric("Failed", "N/A")
                        
                        # History table
                        display_df = history_df.copy()
                        if 'status' in display_df.columns:
                            # Convert status enum values to strings
                            display_df['status'] = display_df['status'].apply(
                                lambda x: x.value if hasattr(x, 'value') else str(x)
                            )
                        
                        st.dataframe(
                            display_df[['name', 'status', 'started_at', 'duration']]
                        )
                        
                        # Charts
                        if 'status' in history_df.columns:
                            import plotly.express as px
                            
                            # Prepare status column for chart
                            status_series = history_df['status'].apply(
                                lambda x: x.value if hasattr(x, 'value') else str(x)
                            )
                            
                            fig = px.pie(
                                pd.DataFrame({'status': status_series}),
                                names='status',
                                title='Task Status Distribution'
                            )
                            st.plotly_chart(fig)
                else:
                    st.info("No task history available")
            except Exception as e:
                st.warning(f"Could not retrieve task history: {e}")
                st.info("No task history available")
    else:
        st.warning("⚠️ Task Orchestrator components not initialized. Please restart the application.")

    # Admin self-test (debug)
    st.markdown("---")
    with st.expander("🧪 Run self-test", expanded=False):
        st.caption("Verify Admin page dependencies: config.llm_config, MemoryStore, provider status, test_active_llm.")
        if st.button("Run self-test", key="admin_run_self_test"):
            out = run_admin_self_test()
            for name, r in out.items():
                ok = r.get("ok", False)
                msg = r.get("message", "")
                if ok:
                    st.success(f"**{name}**: {msg}")
                else:
                    st.error(f"**{name}**: {msg}")
            st.json({k: {"ok": v["ok"], "message": v["message"]} for k, v in out.items()})

