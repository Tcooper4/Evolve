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
from ..auth.security import SecurityManager
from ..monitoring.metrics_collector import MetricsCollector
from ..monitoring.alert_manager import AlertManager

logger = logging.getLogger(__name__)

app = Flask(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="automation/templates")

def get_redis_client():
    """Get Redis client connection."""
    return redis.Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )

def get_metrics_history(hours=24):
    """Get metrics history from Redis."""
    redis_client = get_redis_client()
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
    redis_client = get_redis_client()
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

def create_agent_metrics_plot(df):
    """Create agent metrics plot."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Active Agents", "Agent Health"))
    
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["agents.active_agents"], name="Active Agents"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["agents.health_score"], name="Health Score"),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="Agent Metrics")
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
        agent_plot = create_agent_metrics_plot(df)
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