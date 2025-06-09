from functools import wraps
from flask import request, jsonify, current_app
from ..auth.user_manager import UserManager

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Verify token
            user_manager = UserManager(current_app.redis, current_app.config['SECRET_KEY'])
            user_data = user_manager.verify_token(token)
            
            if not user_data:
                return jsonify({'error': 'Invalid token'}), 401
            
            # Add user data to request context
            request.user = user_data
            return f(*args, **kwargs)
            
        except Exception as e:
            current_app.logger.error(f"Authentication error: {str(e)}")
            return jsonify({'error': 'Authentication failed'}), 401
    
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Verify token
            user_manager = UserManager(current_app.redis, current_app.config['SECRET_KEY'])
            user_data = user_manager.verify_token(token)
            
            if not user_data:
                return jsonify({'error': 'Invalid token'}), 401
            
            # Check if user is admin
            if user_data.get('role') != 'admin':
                return jsonify({'error': 'Admin access required'}), 403
            
            # Add user data to request context
            request.user = user_data
            return f(*args, **kwargs)
            
        except Exception as e:
            current_app.logger.error(f"Authentication error: {str(e)}")
            return jsonify({'error': 'Authentication failed'}), 401
    
    return decorated_function

def inject_user(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if token:
            try:
                # Remove 'Bearer ' prefix if present
                if token.startswith('Bearer '):
                    token = token[7:]
                
                # Verify token
                user_manager = UserManager(current_app.redis, current_app.config['SECRET_KEY'])
                user_data = user_manager.verify_token(token)
                
                if user_data:
                    request.user = user_data
            except Exception as e:
                current_app.logger.error(f"Token verification error: {str(e)}")
        
        return f(*args, **kwargs)
    
    return decorated_function 