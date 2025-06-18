from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
import asyncio
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
import os
import redis
import ray
from fastapi import Request, HTTPException, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import jwt
from functools import wraps
import yaml
import uuid

from ..agents.orchestrator import DevelopmentOrchestrator
from ..agents.monitor import SystemMonitor
from ..agents.error_handler import ErrorHandler
from automation.api.task_api import TaskAPI, TaskCreate, TaskUpdate
from automation.api.metrics_api import MetricsAPI
from automation.core.orchestrator import Orchestrator
from trading.middleware import login_required, admin_required, inject_user
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
config_path = Path(__file__).parent.parent.parent / 'config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

# Initialize components
orchestrator = DevelopmentOrchestrator(config)
monitor = SystemMonitor(config)
error_handler = ErrorHandler(config)

# Initialize Redis client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', '6379')),
    db=int(os.getenv('REDIS_DB', '0')),
    password=os.getenv('REDIS_PASSWORD'),
    ssl=os.getenv('REDIS_SSL', 'false').lower() == 'true'
)

# Initialize user manager
user_manager = UserManager(redis_client, os.getenv('JWT_SECRET'))

# Initialize notification manager
notification_manager = NotificationManager(redis_client)

# Initialize WebSocket manager
websocket_manager = WebSocketManager(notification_manager)
websocket_handler = WebSocketHandler(websocket_manager)

# Initialize Flask app
app.config['SECRET_KEY'] = os.getenv('WEB_SECRET_KEY')
app.config['JWT_SECRET'] = os.getenv('JWT_SECRET')

# Initialize CORS
CORS(app, resources={
    r"/*": {
        "origins": os.getenv('CORS_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{os.getenv('RATE_LIMIT', '100')}/hour"]
)

# Setup logging
log_handler = logging.handlers.RotatingFileHandler(
    os.getenv('LOG_FILE', 'trading.log'),
    maxBytes=int(os.getenv('LOG_MAX_SIZE', 10485760)),
    backupCount=int(os.getenv('LOG_BACKUP_COUNT', 5))
)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
app.logger.addHandler(log_handler)
app.logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
            current_user = data['username']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            token = token.split(' ')[1]
            data = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
            if not data.get('is_admin', False):
                return jsonify({'message': 'Admin privileges required'}), 403
            current_user = data['username']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

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
        yaml.dump(new_config, f)
    return jsonify({"status": "updated"})

@app.route('/api/tasks', methods=['GET'])
@token_required
def get_tasks(current_user):
    tasks = redis_client.hgetall(f'tasks:{current_user}')
    return jsonify({k.decode(): v.decode() for k, v in tasks.items()})

@app.route('/api/tasks', methods=['POST'])
@token_required
def create_task(current_user):
    data = request.get_json()
    task_id = str(uuid.uuid4())
    data['created_by'] = current_user
    data['created_at'] = datetime.utcnow().isoformat()
    redis_client.hset(f'tasks:{current_user}', task_id, json.dumps(data))
    return jsonify({'task_id': task_id})

@app.route('/api/tasks/<task_id>', methods=['GET'])
@token_required
def get_task(current_user, task_id):
    task = redis_client.hget(f'tasks:{current_user}', task_id)
    if task:
        return jsonify(json.loads(task))
    return jsonify({"error": "Task not found"}), 404

@app.route('/api/tasks/<task_id>', methods=['PUT'])
@token_required
def update_task(current_user, task_id):
    data = request.get_json()
    task = redis_client.hget(f'tasks:{current_user}', task_id)
    if not task:
        return jsonify({'message': 'Task not found'}), 404
    task_data = json.loads(task)
    task_data.update(data)
    redis_client.hset(f'tasks:{current_user}', task_id, json.dumps(task_data))
    return jsonify({'message': 'Task updated'})

@app.route('/api/tasks/<task_id>', methods=['DELETE'])
@token_required
def delete_task(current_user, task_id):
    if redis_client.hdel(f'tasks:{current_user}', task_id):
        return jsonify({'message': 'Task deleted'})
    return jsonify({'message': 'Task not found'}), 404

@app.route('/api/tasks/<task_id>/execute', methods=['POST'])
@token_required
def execute_task(current_user, task_id):
    task = redis_client.hget(f'tasks:{current_user}', task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    success = asyncio.run(orchestrator.execute_task(task_id))
    if success:
        return jsonify({"message": "Task execution started"})
    return jsonify({"error": "Task execution failed"}), 500

@app.route('/api/tasks/<task_id>/metrics', methods=['GET'])
@token_required
def get_task_metrics(current_user, task_id):
    metrics = redis_client.hget(f'metrics:{current_user}', task_id)
    if metrics:
        return jsonify(json.loads(metrics))
    return jsonify({"error": "Task metrics not found"}), 404

@app.route('/api/metrics', methods=['GET'])
@token_required
def get_metrics(current_user):
    metrics = redis_client.hgetall('metrics')
    return jsonify({k.decode(): v.decode() for k, v in metrics.items()})

@app.route('/api/metrics/system', methods=['GET'])
@token_required
def get_system_metrics(current_user):
    metrics = redis_client.hgetall('system_metrics')
    if metrics:
        return jsonify(json.loads(metrics))
    return jsonify({"error": "System metrics not found"}), 404

@app.route('/api/metrics/agents', methods=['GET'])
@token_required
def get_agent_metrics(current_user):
    metrics = redis_client.hgetall('agent_metrics')
    if metrics:
        return jsonify(json.loads(metrics))
    return jsonify({"error": "Agent metrics not found"}), 404

@app.route('/api/metrics/models', methods=['GET'])
@token_required
def get_model_metrics(current_user):
    metrics = redis_client.hgetall('model_metrics')
    if metrics:
        return jsonify(json.loads(metrics))
    return jsonify({"error": "Model metrics not found"}), 404

@app.route('/api/metrics/history', methods=['GET'])
@token_required
def get_metrics_history(current_user):
    limit = request.args.get('limit', default=100, type=int)
    history = redis_client.hgetall(f'metrics_history:{current_user}')
    return jsonify({k.decode(): v.decode() for k, v in history.items()})

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
@limiter.limit("5/minute")
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'message': 'Missing credentials'}), 400
        
    # Verify credentials (implement your own logic)
    if not verify_credentials(username, password):
        return jsonify({'message': 'Invalid credentials'}), 401
        
    # Generate token
    token = jwt.encode({
        'username': username,
        'is_admin': is_admin(username),
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, app.config['JWT_SECRET'])
    
    return jsonify({'token': token})

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

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_users(current_user):
    users = redis_client.hgetall('users')
    return jsonify({k.decode(): v.decode() for k, v in users.items()})

@app.route('/api/admin/users', methods=['POST'])
@admin_required
def create_user(current_user):
    data = request.get_json()
    username = data.get('username')
    if not username:
        return jsonify({'message': 'Username is required'}), 400
    if redis_client.hexists('users', username):
        return jsonify({'message': 'User already exists'}), 409
    data['created_by'] = current_user
    data['created_at'] = datetime.utcnow().isoformat()
    redis_client.hset('users', username, json.dumps(data))
    return jsonify({'message': 'User created'})

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'message': 'Rate limit exceeded'}), 429

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f'Internal error: {str(e)}')
    return jsonify({'message': 'Internal server error'}), 500

if __name__ == '__main__':
    run_app() 