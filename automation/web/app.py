from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
import asyncio
import json
from pathlib import Path
import logging
from datetime import datetime
import os
import redis
import ray
from fastapi import Request, HTTPException, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ..agents.orchestrator import DevelopmentOrchestrator
from ..agents.monitor import SystemMonitor
from ..agents.error_handler import ErrorHandler
from automation.api.task_api import TaskAPI, TaskCreate, TaskUpdate
from automation.api.metrics_api import MetricsAPI
from automation.core.orchestrator import Orchestrator
from .middleware import login_required, admin_required, inject_user
from ..auth.user_manager import UserManager
from automation.notifications.notification_manager import NotificationManager, NotificationType, NotificationPriority
from automation.web.websocket import WebSocketManager, WebSocketHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation/logs/web.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)
task_api = TaskAPI()
orchestrator = Orchestrator()
metrics_api = MetricsAPI(orchestrator)

# Load configuration
config_path = Path("automation/config/config.json")
with open(config_path) as f:
    config = json.load(f)

# Initialize components
orchestrator = DevelopmentOrchestrator(config)
monitor = SystemMonitor(config)
error_handler = ErrorHandler(config)

# Initialize user manager
user_manager = UserManager(redis_client, app.config['SECRET_KEY'])

# Initialize Redis client
redis_client = redis.Redis(
    host=config.get("redis", {}).get("host", "localhost"),
    port=config.get("redis", {}).get("port", 6379),
    db=config.get("redis", {}).get("db", 0)
)

# Initialize notification manager
notification_manager = NotificationManager(redis_client)

# Initialize WebSocket manager
websocket_manager = WebSocketManager(notification_manager)
websocket_handler = WebSocketHandler(websocket_manager)

# Initialize FastAPI app
app = Flask(__name__)

# Mount static files
app.mount("/static", StaticFiles(directory="automation/web/static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="automation/web/templates")

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

@app.route('/api/tasks', methods=['GET'])
async def get_tasks():
    try:
        status = request.args.get('status')
        task_type = request.args.get('task_type')
        tasks = await task_api.get_tasks(status=status, task_type=task_type)
        return jsonify(tasks)
    except Exception as e:
        logger.error(f"Error getting tasks: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks', methods=['POST'])
async def create_task_api():
    try:
        task_data = request.json
        task = await task_api.create_task(task_data)
        return jsonify(task), 201
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks/<task_id>', methods=['GET'])
async def get_task(task_id):
    try:
        task = await task_api.get_task(task_id)
        if task:
            return jsonify(task)
        return jsonify({"error": "Task not found"}), 404
    except Exception as e:
        logger.error(f"Error getting task: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks/<task_id>', methods=['PATCH'])
async def update_task(task_id):
    try:
        task_data = request.json
        task = await task_api.update_task(task_id, task_data)
        if task:
            return jsonify(task)
        return jsonify({"error": "Task not found"}), 404
    except Exception as e:
        logger.error(f"Error updating task: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
async def delete_task(task_id):
    try:
        success = await task_api.delete_task(task_id)
        if success:
            return jsonify({"message": "Task deleted successfully"})
        return jsonify({"error": "Task not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks/<task_id>/execute', methods=['POST'])
async def execute_task(task_id):
    try:
        success = await task_api.execute_task(task_id)
        if success:
            return jsonify({"message": "Task execution started"})
        return jsonify({"error": "Task not found"}), 404
    except Exception as e:
        logger.error(f"Error executing task: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tasks/<task_id>/metrics', methods=['GET'])
async def get_task_metrics(task_id):
    try:
        metrics = await metrics_api.get_task_metrics(task_id)
        if metrics:
            return jsonify(metrics)
        return jsonify({"error": "Task metrics not found"}), 404
    except Exception as e:
        logger.error(f"Error getting task metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
async def get_metrics():
    try:
        metrics = await metrics_api.get_all_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics/system', methods=['GET'])
async def get_system_metrics():
    try:
        metrics = await metrics_api.get_system_metrics()
        
        # Check for critical conditions
        if metrics["cpu_usage"] > 90:
            await notification_manager.create_notification(
                title="High CPU Usage",
                message=f"CPU usage is at {metrics['cpu_usage']}%",
                notification_type=NotificationType.SYSTEM,
                priority=NotificationPriority.CRITICAL,
                data={"metric": "cpu_usage", "value": metrics["cpu_usage"]}
            )
        
        if metrics["memory_usage"] > 90:
            await notification_manager.create_notification(
                title="High Memory Usage",
                message=f"Memory usage is at {metrics['memory_usage']}%",
                notification_type=NotificationType.SYSTEM,
                priority=NotificationPriority.CRITICAL,
                data={"metric": "memory_usage", "value": metrics["memory_usage"]}
            )
        
        if metrics["disk_usage"] > 90:
            await notification_manager.create_notification(
                title="High Disk Usage",
                message=f"Disk usage is at {metrics['disk_usage']}%",
                notification_type=NotificationType.SYSTEM,
                priority=NotificationPriority.CRITICAL,
                data={"metric": "disk_usage", "value": metrics["disk_usage"]}
            )
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")

@app.route('/api/metrics/agents', methods=['GET'])
async def get_agent_metrics():
    try:
        metrics = await metrics_api.get_agent_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting agent metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics/models', methods=['GET'])
async def get_model_metrics():
    try:
        metrics = await metrics_api.get_model_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics/history', methods=['GET'])
async def get_metrics_history():
    try:
        limit = request.args.get('limit', default=100, type=int)
        history = await metrics_api.get_metrics_history(limit=limit)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error getting metrics history: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Kubernetes liveness probe."""
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check Ray connection
        ray.is_initialized()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/ready')
def ready_check():
    """Readiness check endpoint for Kubernetes readiness probe."""
    try:
        # Check if the application is ready to accept traffic
        if not app.config.get('initialized', False):
            return jsonify({
                'status': 'not ready',
                'message': 'Application is still initializing',
                'timestamp': datetime.now().isoformat()
            }), 503
            
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        app.logger.error(f"Readiness check failed: {str(e)}")
        return jsonify({
            'status': 'not ready',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/login')
def login_page():
    """Render login page."""
    return render_template('login.html')

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Handle user login."""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        token = user_manager.authenticate(username, password)
        if not token:
            return jsonify({'error': 'Invalid username or password'}), 401
        
        return jsonify({'token': token})
        
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/register', methods=['POST'])
@admin_required
def register():
    """Handle user registration (admin only)."""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        role = data.get('role', 'user')
        
        if not all([username, password, email]):
            return jsonify({'error': 'Username, password, and email are required'}), 400
        
        user = user_manager.create_user(username, password, email, role)
        return jsonify(user), 201
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/users', methods=['GET'])
@admin_required
def list_users():
    """List all users (admin only)."""
    try:
        users = user_manager.list_users()
        return jsonify(users)
    except Exception as e:
        app.logger.error(f"Error listing users: {str(e)}")
        return jsonify({'error': 'Failed to list users'}), 500

@app.route('/api/auth/users/<username>', methods=['GET'])
@login_required
def get_user(username):
    """Get user details."""
    try:
        # Allow users to view their own details or admins to view any user
        if request.user['username'] != username and request.user['role'] != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        user = user_manager.get_user(username)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify(user)
    except Exception as e:
        app.logger.error(f"Error getting user: {str(e)}")
        return jsonify({'error': 'Failed to get user'}), 500

@app.route('/api/auth/users/<username>', methods=['PUT'])
@login_required
def update_user(username):
    """Update user details."""
    try:
        # Allow users to update their own details or admins to update any user
        if request.user['username'] != username and request.user['role'] != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        user = user_manager.update_user(username, data)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify(user)
    except Exception as e:
        app.logger.error(f"Error updating user: {str(e)}")
        return jsonify({'error': 'Failed to update user'}), 500

@app.route('/api/auth/users/<username>', methods=['DELETE'])
@admin_required
def delete_user(username):
    """Delete a user (admin only)."""
    try:
        success = user_manager.delete_user(username)
        if not success:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'message': 'User deleted successfully'})
    except Exception as e:
        app.logger.error(f"Error deleting user: {str(e)}")
        return jsonify({'error': 'Failed to delete user'}), 500

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

async def initialize():
    """Initialize the application components"""
    try:
        # Create necessary directories
        Path('automation/logs').mkdir(parents=True, exist_ok=True)
        
        # Initialize orchestrator
        await orchestrator.initialize()
        
        # Start the orchestrator
        await orchestrator.start()
        
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        raise

def run_app():
    """Run the Flask application"""
    try:
        # Initialize the application
        asyncio.run(initialize())
        
        # Run the Flask app
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        raise

# Add notification endpoints
@app.get("/api/notifications")
@login_required
async def get_notifications(
    request: Request,
    limit: int = 10,
    offset: int = 0,
    unread_only: bool = False
):
    try:
        user_id = request.state.user["username"]
        notifications = await notification_manager.get_user_notifications(
            user_id=user_id,
            limit=limit,
            offset=offset,
            unread_only=unread_only
        )
        
        return {
            "notifications": notifications,
            "has_more": len(notifications) == limit
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get notifications")

@app.post("/api/notifications/{notification_id}/read")
@login_required
async def mark_notification_as_read(request: Request, notification_id: str):
    try:
        user_id = request.state.user["username"]
        success = await notification_manager.mark_as_read(notification_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"message": "Notification marked as read"}
    except Exception as e:
        logger.error(f"Error marking notification as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")

@app.post("/api/notifications/read-all")
@login_required
async def mark_all_notifications_as_read(request: Request):
    try:
        user_id = request.state.user["username"]
        notifications = await notification_manager.get_user_notifications(user_id=user_id, unread_only=True)
        
        for notification in notifications:
            await notification_manager.mark_as_read(notification["id"], user_id)
        
        return {"message": "All notifications marked as read"}
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark all notifications as read")

@app.delete("/api/notifications/{notification_id}")
@login_required
async def delete_notification(request: Request, notification_id: str):
    try:
        user_id = request.state.user["username"]
        success = await notification_manager.delete_notification(notification_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"message": "Notification deleted"}
    except Exception as e:
        logger.error(f"Error deleting notification: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete notification")

@app.delete("/api/notifications")
@login_required
async def clear_all_notifications(request: Request):
    try:
        user_id = request.state.user["username"]
        notifications = await notification_manager.get_user_notifications(user_id=user_id)
        
        for notification in notifications:
            await notification_manager.delete_notification(notification["id"], user_id)
        
        return {"message": "All notifications cleared"}
    except Exception as e:
        logger.error(f"Error clearing notifications: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear notifications")

# Update task-related endpoints to send notifications
@app.post("/api/tasks")
@login_required
async def create_task(request: Request, task: TaskCreate):
    try:
        # Create task
        task_id = await task_api.create_task(task)
        
        # Send notification
        await notification_manager.create_notification(
            title="New Task Created",
            message=f"Task '{task.name}' has been created",
            notification_type=NotificationType.TASK,
            priority=NotificationPriority.MEDIUM,
            data={"task_id": task_id},
            user_id=request.state.user["username"]
        )
        
        return {"task_id": task_id}
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create task")

@app.put("/api/tasks/{task_id}")
@login_required
async def update_task(request: Request, task_id: str, task: TaskUpdate):
    try:
        # Update task
        success = await task_api.update_task(task_id, task)
        
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Send notification
        await notification_manager.create_notification(
            title="Task Updated",
            message=f"Task '{task.name}' has been updated",
            notification_type=NotificationType.TASK,
            priority=NotificationPriority.LOW,
            data={"task_id": task_id},
            user_id=request.state.user["username"]
        )
        
        return {"message": "Task updated successfully"}
    except Exception as e:
        logger.error(f"Error updating task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update task")

@app.delete("/api/tasks/{task_id}")
@login_required
async def delete_task(request: Request, task_id: str):
    try:
        # Get task details before deletion
        task = await task_api.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Delete task
        success = await task_api.delete_task(task_id)
        
        # Send notification
        await notification_manager.create_notification(
            title="Task Deleted",
            message=f"Task '{task['name']}' has been deleted",
            notification_type=NotificationType.TASK,
            priority=NotificationPriority.HIGH,
            data={"task_id": task_id},
            user_id=request.state.user["username"]
        )
        
        return {"message": "Task deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete task")

@app.post("/api/tasks/{task_id}/execute")
@login_required
async def execute_task(request: Request, task_id: str):
    try:
        # Get task details
        task = await task_api.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Execute task
        success = await task_api.execute_task(task_id)
        
        # Send notification
        await notification_manager.create_notification(
            title="Task Execution Started",
            message=f"Task '{task['name']}' has started executing",
            notification_type=NotificationType.TASK,
            priority=NotificationPriority.MEDIUM,
            data={"task_id": task_id},
            user_id=request.state.user["username"]
        )
        
        return {"message": "Task execution started"}
    except Exception as e:
        logger.error(f"Error executing task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to execute task")

@app.post("/api/tasks/{task_id}/stop")
@login_required
async def stop_task(request: Request, task_id: str):
    try:
        # Get task details
        task = await task_api.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Stop task
        success = await task_api.stop_task(task_id)
        
        # Send notification
        await notification_manager.create_notification(
            title="Task Execution Stopped",
            message=f"Task '{task['name']}' has been stopped",
            notification_type=NotificationType.TASK,
            priority=NotificationPriority.HIGH,
            data={"task_id": task_id},
            user_id=request.state.user["username"]
        )
        
        return {"message": "Task stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to stop task")

if __name__ == '__main__':
    run_app() 