from flask import Flask, render_template, jsonify
import redis
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from enum import Enum
try:
    from system.infra.agents.auth.security import SecurityManager
except ImportError as e:
    logging.warning(f"SecurityManager import failed: {e}")
    SecurityManager = None
try:
    from system.infra.agents.monitoring.metrics_collector import MetricsCollector
except ImportError as e:
    logging.warning(f"MetricsCollector import failed: {e}")
    MetricsCollector = None
try:
    from system.infra.agents.monitoring.alert_manager import AlertManager
except ImportError as e:
    logging.warning(f"AlertManager import failed: {e}")
    AlertManager = None

logger = logging.getLogger(__name__)

app = Flask(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="automation/templates")

class AgentHealthStatus(str, Enum):
    """Agent health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

class AgentHealthIndicator:
    """Agent health indicator with visual status."""
    
    def __init__(self):
        self.health_colors = {
            AgentHealthStatus.HEALTHY: "#28a745",    # Green
            AgentHealthStatus.WARNING: "#ffc107",    # Yellow
            AgentHealthStatus.CRITICAL: "#dc3545",   # Red
            AgentHealthStatus.OFFLINE: "#6c757d",    # Gray
            AgentHealthStatus.UNKNOWN: "#17a2b8"     # Blue
        }
        
        self.health_icons = {
            AgentHealthStatus.HEALTHY: "🟢",
            AgentHealthStatus.WARNING: "🟡",
            AgentHealthStatus.CRITICAL: "🔴",
            AgentHealthStatus.OFFLINE: "⚫",
            AgentHealthStatus.UNKNOWN: "🔵"
        }
        
        self.health_labels = {
            AgentHealthStatus.HEALTHY: "Healthy",
            AgentHealthStatus.WARNING: "Warning",
            AgentHealthStatus.CRITICAL: "Critical",
            AgentHealthStatus.OFFLINE: "Offline",
            AgentHealthStatus.UNKNOWN: "Unknown"
        }
    
    def get_health_status(self, health_score: float, last_seen: datetime, 
                         error_count: int = 0, response_time: float = 0.0) -> AgentHealthStatus:
        """Determine agent health status based on metrics.
        
        Args:
            health_score: Health score (0-100)
            last_seen: Last time agent was seen
            error_count: Number of recent errors
            response_time: Average response time in seconds
            
        Returns:
            Agent health status
        """
        try:
            # Check if agent is offline (not seen in last 5 minutes)
            if datetime.now() - last_seen > timedelta(minutes=5):
                return AgentHealthStatus.OFFLINE
            
            # Check for critical conditions
            if health_score < 30 or error_count > 10 or response_time > 30:
                return AgentHealthStatus.CRITICAL
            
            # Check for warning conditions
            if health_score < 70 or error_count > 5 or response_time > 10:
                return AgentHealthStatus.WARNING
            
            # Check for healthy conditions
            if health_score >= 70 and error_count <= 2 and response_time <= 5:
                return AgentHealthStatus.HEALTHY
            
            # Default to unknown
            return AgentHealthStatus.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error determining health status: {e}")
            return AgentHealthStatus.UNKNOWN
    
    def get_health_color(self, status: AgentHealthStatus) -> str:
        """Get color for health status."""
        return self.health_colors.get(status, self.health_colors[AgentHealthStatus.UNKNOWN])
    
    def get_health_icon(self, status: AgentHealthStatus) -> str:
        """Get icon for health status."""
        return self.health_icons.get(status, self.health_icons[AgentHealthStatus.UNKNOWN])
    
    def get_health_label(self, status: AgentHealthStatus) -> str:
        """Get label for health status."""
        return self.health_labels.get(status, self.health_labels[AgentHealthStatus.UNKNOWN])

# Global health indicator
health_indicator = AgentHealthIndicator()

def get_redis_client():
    """Get Redis client connection."""
    return {'success': True, 'result': redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    ), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def get_agent_health_data():
    """Get agent health data from Redis with visual indicators."""
    try:
        redis_client = get_redis_client()['result']
        
        # Get agent data from Redis
        agent_keys = redis_client.keys("agent:*")
        agents_data = []
        
        for key in agent_keys:
            try:
                agent_data = json.loads(redis_client.get(key))
                agent_id = key.split(":")[1]
                
                # Extract health metrics
                health_score = agent_data.get('health_score', 0.0)
                last_seen = datetime.fromisoformat(agent_data.get('last_seen', datetime.now().isoformat()))
                error_count = agent_data.get('error_count', 0)
                response_time = agent_data.get('avg_response_time', 0.0)
                
                # Determine health status
                health_status = health_indicator.get_health_status(
                    health_score, last_seen, error_count, response_time
                )
                
                # Add visual indicators
                agent_info = {
                    'id': agent_id,
                    'name': agent_data.get('name', f'Agent {agent_id}'),
                    'type': agent_data.get('type', 'unknown'),
                    'health_score': health_score,
                    'health_status': health_status.value,
                    'health_color': health_indicator.get_health_color(health_status),
                    'health_icon': health_indicator.get_health_icon(health_status),
                    'health_label': health_indicator.get_health_label(health_status),
                    'last_seen': last_seen.isoformat(),
                    'error_count': error_count,
                    'response_time': response_time,
                    'status': agent_data.get('status', 'unknown'),
                    'uptime': agent_data.get('uptime', 0),
                    'tasks_completed': agent_data.get('tasks_completed', 0),
                    'tasks_failed': agent_data.get('tasks_failed', 0)
                }
                
                agents_data.append(agent_info)
                
            except Exception as e:
                logger.error(f"Error processing agent data for key {key}: {e}")
                continue
        
        return agents_data
        
    except Exception as e:
        logger.error(f"Error getting agent health data: {e}")
        return []

def create_agent_health_summary(agents_data: List[Dict]) -> Dict:
    """Create agent health summary with visual indicators."""
    try:
        if not agents_data:
            return {
                'total_agents': 0,
                'healthy_count': 0,
                'warning_count': 0,
                'critical_count': 0,
                'offline_count': 0,
                'unknown_count': 0,
                'overall_health': 'unknown',
                'overall_color': health_indicator.get_health_color(AgentHealthStatus.UNKNOWN),
                'overall_icon': health_indicator.get_health_icon(AgentHealthStatus.UNKNOWN)
            }
        
        # Count agents by health status
        status_counts = {}
        for status in AgentHealthStatus:
            status_counts[status.value] = 0
        
        for agent in agents_data:
            status = agent['health_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall health
        total_agents = len(agents_data)
        critical_ratio = status_counts.get('critical', 0) / total_agents
        warning_ratio = status_counts.get('warning', 0) / total_agents
        offline_ratio = status_counts.get('offline', 0) / total_agents
        
        if critical_ratio > 0.2 or offline_ratio > 0.3:
            overall_status = AgentHealthStatus.CRITICAL
        elif warning_ratio > 0.3 or offline_ratio > 0.1:
            overall_status = AgentHealthStatus.WARNING
        elif status_counts.get('healthy', 0) / total_agents > 0.8:
            overall_status = AgentHealthStatus.HEALTHY
        else:
            overall_status = AgentHealthStatus.UNKNOWN
        
        return {
            'total_agents': total_agents,
            'healthy_count': status_counts.get('healthy', 0),
            'warning_count': status_counts.get('warning', 0),
            'critical_count': status_counts.get('critical', 0),
            'offline_count': status_counts.get('offline', 0),
            'unknown_count': status_counts.get('unknown', 0),
            'overall_health': overall_status.value,
            'overall_color': health_indicator.get_health_color(overall_status),
            'overall_icon': health_indicator.get_health_icon(overall_status),
            'overall_label': health_indicator.get_health_label(overall_status)
        }
        
    except Exception as e:
        logger.error(f"Error creating agent health summary: {e}")
        return {}

def create_agent_health_plot(agents_data: List[Dict]):
    """Create agent health visualization plot."""
    try:
        if not agents_data:
            return None
        
        # Prepare data for plotting
        agent_names = [agent['name'] for agent in agents_data]
        health_scores = [agent['health_score'] for agent in agents_data]
        health_colors = [agent['health_color'] for agent in agents_data]
        health_statuses = [agent['health_status'] for agent in agents_data]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=agent_names,
                y=health_scores,
                marker_color=health_colors,
                text=[f"{score:.1f}%" for score in health_scores],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                            'Health Score: %{y:.1f}%<br>' +
                            'Status: %{customdata}<br>' +
                            '<extra></extra>',
                customdata=health_statuses
            )
        ])
        
        fig.update_layout(
            title="Agent Health Status",
            xaxis_title="Agent",
            yaxis_title="Health Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
            showlegend=False
        )
        
        # Add threshold lines
        fig.add_hline(y=70, line_dash="dash", line_color="green", 
                     annotation_text="Healthy Threshold")
        fig.add_hline(y=30, line_dash="dash", line_color="red", 
                     annotation_text="Critical Threshold")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating agent health plot: {e}")
        return None

def get_metrics_history(hours=24):
    """Get metrics history from Redis."""
    redis_client = get_redis_client()['result']
    metrics_data = redis_client.hgetall("metrics_history")
    
    # Convert to DataFrame
    data = []
    for timestamp, metrics in metrics_data.items():
        metrics_dict = json.loads(metrics)
        metrics_dict["timestamp"] = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        data.append(metrics_dict)
    
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp")
    
    # Filter last N hours
    cutoff = datetime.now() - timedelta(hours=hours)
    df = df[df["timestamp"] >= cutoff]
    
    return df

def get_alerts(hours=24):
    """Get alerts from Redis."""
    redis_client = get_redis_client()['result']
    alerts_data = redis_client.hgetall("alerts")
    
    # Convert to list of alerts
    alerts = []
    for key, value in alerts_data.items():
        alert = json.loads(value)
        alert["timestamp"] = datetime.strptime(key.split("_")[-1], "%Y%m%d_%H%M%S")
        alerts.append(alert)
    
    # Sort by timestamp and filter
    alerts.sort(key=lambda x: x["timestamp"], reverse=True)
    cutoff = datetime.now() - timedelta(hours=hours)
    alerts = [alert for alert in alerts if alert["timestamp"] >= cutoff]
    
    return alerts

def create_system_metrics_plot(df):
    """Create system metrics plot."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("CPU Usage", "Memory Usage"))
    
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["system.cpu_usage"], name="CPU Usage"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["system.memory_usage"], name="Memory Usage"),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="System Metrics")
    return fig

def create_task_metrics_plot(df):
    """Create task metrics plot."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Task Success Rate", "Active Tasks"))
    
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["tasks.success_rate"], name="Success Rate"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["tasks.active_tasks"], name="Active Tasks"),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="Task Metrics")
    return fig

def create_model_metrics_plot(df):
    """Create model metrics plot."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Model Accuracy", "Training Progress"))
    
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["models.accuracy"], name="Accuracy"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["models.training_progress"], name="Training Progress"),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="Model Metrics")
    return fig

@app.route("/")
def index():
    """Render dashboard index page."""
    return render_template("dashboard.html")

@app.route("/api/metrics")
def get_metrics():
    """Get metrics data for dashboard."""
    try:
        df = get_metrics_history()
        
        # Create plots
        system_plot = create_system_metrics_plot(df)
        task_plot = create_task_metrics_plot(df)
        agent_plot = create_agent_health_plot(get_agent_health_data()) # Use the new function
        model_plot = create_model_metrics_plot(df)
        
        # Get alerts
        alerts = get_alerts()
        
        return jsonify({
            "system_plot": system_plot.to_json(),
            "task_plot": task_plot.to_json(),
            "agent_plot": agent_plot.to_json(),
            "model_plot": model_plot.to_json(),
            "alerts": alerts
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/agents/health")
def get_agent_health():
    """Get agent health data with visual indicators."""
    try:
        agents_data = get_agent_health_data()
        health_summary = create_agent_health_summary(agents_data)
        
        return jsonify({
            "agents": agents_data,
            "summary": health_summary,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting agent health: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/agents/health/<agent_id>")
def get_agent_health_detail(agent_id):
    """Get detailed health information for a specific agent."""
    try:
        agents_data = get_agent_health_data()
        agent_data = next((agent for agent in agents_data if agent['id'] == agent_id), None)
        
        if not agent_data:
            return jsonify({"error": "Agent not found"}), 404
        
        # Get historical health data for this agent
        redis_client = get_redis_client()['result']
        history_key = f"agent_health_history:{agent_id}"
        health_history = redis_client.lrange(history_key, 0, 99)  # Last 100 records
        
        # Parse history
        parsed_history = []
        for record in health_history:
            try:
                parsed_record = json.loads(record)
                parsed_history.append(parsed_record)
            except Exception as e:
                logger.error(f"Error parsing health history record: {e}")
                continue
        
        return jsonify({
            "agent": agent_data,
            "health_history": parsed_history,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting agent health detail: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/agents/health/summary")
def get_agent_health_summary_endpoint():
    """Get agent health summary with visual indicators."""
    try:
        agents_data = get_agent_health_data()
        health_summary = create_agent_health_summary(agents_data)
        
        return jsonify({
            "summary": health_summary,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting agent health summary: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/agents/health/trends")
def get_agent_health_trends():
    """Get agent health trends over time."""
    try:
        redis_client = get_redis_client()['result']
        
        # Get health trends from Redis
        trends_key = "agent_health_trends"
        trends_data = redis_client.hgetall(trends_key)
        
        # Parse trends data
        trends = []
        for timestamp, data in trends_data.items():
            try:
                trend_record = json.loads(data)
                trend_record['timestamp'] = timestamp
                trends.append(trend_record)
            except Exception as e:
                logger.error(f"Error parsing trend record: {e}")
                continue
        
        # Sort by timestamp
        trends.sort(key=lambda x: x['timestamp'])
        
        return jsonify({
            "trends": trends,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting agent health trends: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/agents/health/status")
def get_agent_health_status():
    """Get current agent health status with color coding."""
    try:
        agents_data = get_agent_health_data()
        
        # Group agents by health status
        status_groups = {}
        for status in AgentHealthStatus:
            status_groups[status.value] = []
        
        for agent in agents_data:
            status = agent['health_status']
            status_groups[status].append(agent)
        
        # Create status summary
        status_summary = {}
        for status, agents in status_groups.items():
            status_summary[status] = {
                'count': len(agents),
                'color': health_indicator.get_health_color(AgentHealthStatus(status)),
                'icon': health_indicator.get_health_icon(AgentHealthStatus(status)),
                'label': health_indicator.get_health_label(AgentHealthStatus(status)),
                'agents': agents
            }
        
        return jsonify({
            "status_groups": status_summary,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting agent health status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/alerts")
def get_alert_data():
    """Get alerts data for dashboard."""
    try:
        alerts = get_alerts()
        return jsonify(alerts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    security: SecurityManager = Depends(get_security_manager)
):
    """Render dashboard page."""
    try:
        # Check permissions
        if not await security.check_permission(request.user.id, "read:metrics"):
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
            
        # Get metrics
        metrics = await get_metrics_summary()
        
        # Get alerts
        alerts = await get_active_alerts()
        
        # Render template
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "metrics": metrics,
                "alerts": alerts
            }
        )
        
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/api/metrics")
async def get_metrics(
    metric_type: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    security: SecurityManager = Depends(get_security_manager)
):
    """Get metrics data."""
    try:
        # Check permissions
        if not await security.check_permission(request.user.id, "read:metrics"):
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
            
        # Get metrics
        metrics = await metrics_collector.get_metrics(
            metric_type,
            start_time,
            end_time,
            limit
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/api/metrics/summary")
async def get_metrics_summary(
    security: SecurityManager = Depends(get_security_manager)
):
    """Get metrics summary."""
    try:
        # Check permissions
        if not await security.check_permission(request.user.id, "read:metrics"):
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
            
        # Get metrics
        metrics = {
            "cpu": await metrics_collector.get_metric_summary("cpu_usage"),
            "memory": await metrics_collector.get_metric_summary("memory_usage"),
            "disk": await metrics_collector.get_metric_summary("disk_usage"),
            "network": await metrics_collector.get_metric_summary("network_io")
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/api/alerts")
async def get_alerts(
    rule_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    security: SecurityManager = Depends(get_security_manager)
):
    """Get alert history."""
    try:
        # Check permissions
        if not await security.check_permission(request.user.id, "read:alerts"):
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
            
        # Get alerts
        alerts = await alert_manager.get_alert_history(
            rule_id,
            start_time,
            end_time,
            limit
        )
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/api/alerts/active")
async def get_active_alerts(
    security: SecurityManager = Depends(get_security_manager)
):
    """Get active alerts."""
    try:
        # Check permissions
        if not await security.check_permission(request.user.id, "read:alerts"):
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
            
        # Get alerts from last hour
        alerts = await alert_manager.get_alert_history(
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.post("/api/alerts/rules")
async def create_alert_rule(
    rule: AlertRule,
    security: SecurityManager = Depends(get_security_manager)
):
    """Create alert rule."""
    try:
        # Check permissions
        if not await security.check_permission(request.user.id, "write:alerts"):
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
            
        # Create rule
        rule = await alert_manager.create_alert_rule(rule)
        
        return rule
        
    except Exception as e:
        logger.error(f"Error creating alert rule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.put("/api/alerts/rules/{rule_id}")
async def update_alert_rule(
    rule_id: str,
    rule: AlertRule,
    security: SecurityManager = Depends(get_security_manager)
):
    """Update alert rule."""
    try:
        # Check permissions
        if not await security.check_permission(request.user.id, "write:alerts"):
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
            
        # Update rule
        rule = await alert_manager.update_alert_rule(rule_id, **rule.dict())
        
        return rule
        
    except Exception as e:
        logger.error(f"Error updating alert rule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.delete("/api/alerts/rules/{rule_id}")
async def delete_alert_rule(
    rule_id: str,
    security: SecurityManager = Depends(get_security_manager)
):
    """Delete alert rule."""
    try:
        # Check permissions
        if not await security.check_permission(request.user.id, "write:alerts"):
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
            
        # Delete rule
        success = await alert_manager.delete_alert_rule(rule_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Alert rule not found"
            )
            
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error deleting alert rule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) 