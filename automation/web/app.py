from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
import asyncio
import json
from pathlib import Path
import logging
from datetime import datetime
import os

from ..agents.orchestrator import DevelopmentOrchestrator
from ..agents.monitor import SystemMonitor
from ..agents.error_handler import ErrorHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

# Load configuration
config_path = Path("automation/config/config.json")
with open(config_path) as f:
    config = json.load(f)

# Initialize components
orchestrator = DevelopmentOrchestrator(config)
monitor = SystemMonitor(config)
error_handler = ErrorHandler(config)

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/tasks')
def tasks():
    """Get all tasks."""
    tasks = orchestrator.get_all_tasks()
    return jsonify(tasks)

@app.route('/tasks/<task_id>')
def task_details(task_id):
    """Get task details."""
    task = orchestrator.get_task_status(task_id)
    return jsonify(task)

@app.route('/tasks', methods=['POST'])
def create_task():
    """Create a new task."""
    task = request.json
    task_id = asyncio.run(orchestrator.schedule_task(task))
    return jsonify({"task_id": task_id})

@app.route('/tasks/<task_id>/start', methods=['POST'])
def start_task(task_id):
    """Start a task."""
    asyncio.run(orchestrator.coordinate_agents(task_id))
    return jsonify({"status": "started"})

@app.route('/tasks/<task_id>/stop', methods=['POST'])
def stop_task(task_id):
    """Stop a task."""
    # Implement task stopping logic
    return jsonify({"status": "stopped"})

@app.route('/monitoring')
def monitoring():
    """Get monitoring data."""
    metrics = monitor.get_metrics_summary()
    alerts = monitor.get_alerts()
    return jsonify({
        "metrics": metrics,
        "alerts": alerts
    })

@app.route('/errors')
def errors():
    """Get error statistics."""
    stats = error_handler.get_error_statistics()
    return jsonify(stats)

@app.route('/logs')
def logs():
    """Get system logs."""
    log_path = Path("automation/logs")
    logs = []
    for log_file in log_path.glob("*.log"):
        with open(log_file) as f:
            logs.extend(f.readlines())
    return jsonify(logs)

@app.route('/config')
def get_config():
    """Get system configuration."""
    return jsonify(config)

@app.route('/config', methods=['POST'])
def update_config():
    """Update system configuration."""
    new_config = request.json
    with open(config_path, 'w') as f:
        json.dump(new_config, f, indent=4)
    return jsonify({"status": "updated"})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start real-time monitoring."""
    asyncio.run(monitor.start_monitoring())

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop real-time monitoring."""
    monitor.stop_monitoring()

def main():
    """Run the web application."""
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main() 