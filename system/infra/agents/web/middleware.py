import json
import logging
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

from flask import current_app, jsonify, request

from system.infra.agents.auth.user_manager import UserManager

logger = logging.getLogger(__name__)


class AgentContext:
    """Agent context for request traceability."""

    def __init__(
        self,
        request_id: str,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.request_id = request_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id
        self.start_time = time.time()
        self.timestamp = datetime.utcnow().isoformat()
        self.metadata = {}
        self.trace_path = []

    def add_trace(self, step: str, details: Optional[Dict[str, Any]] = None):
        """Add a trace step to the context."""
        trace_entry = {"step": step, "timestamp": datetime.utcnow().isoformat(), "details": details or {}}
        self.trace_path.append(trace_entry)

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the context."""
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "trace_path": self.trace_path,
            "duration": time.time() - self.start_time,
        }


def extract_agent_context(request) -> AgentContext:
    """Extract agent context from request headers and parameters."""
    try:
        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Extract agent information
        agent_id = request.headers.get("X-Agent-ID")
        user_id = request.headers.get("X-User-ID")
        session_id = request.headers.get("X-Session-ID")

        # Create context
        context = AgentContext(request_id=request_id, agent_id=agent_id, user_id=user_id, session_id=session_id)

        # Add request metadata
        context.add_metadata("method", request.method)
        context.add_metadata("path", request.path)
        context.add_metadata("user_agent", request.headers.get("User-Agent"))
        context.add_metadata("ip_address", request.remote_addr)
        context.add_metadata("content_type", request.headers.get("Content-Type"))

        # Add query parameters
        if request.args:
            context.add_metadata("query_params", dict(request.args))

        # Add request body metadata (without sensitive data)
        if request.is_json:
            body = request.get_json()
            if body:
                # Only include non-sensitive fields
                safe_body = {}
                for key, value in body.items():
                    if key.lower() not in ["password", "token", "secret", "key"]:
                        safe_body[key] = value
                context.add_metadata("request_body_keys", list(safe_body.keys()))

        return context

    except Exception as e:
        logger.error(f"Error extracting agent context: {e}")
        # Return minimal context
        return AgentContext(request_id=str(uuid.uuid4()))


def inject_agent_context(f):
    """Decorator to inject agent context into requests."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Extract agent context
            context = extract_agent_context(request)

            # Add context to request
            request.agent_context = context

            # Add trace entry for request start
            context.add_trace(
                "request_started",
                {
                    "endpoint": f.__name__,
                    "args": list(args),
                    "kwargs": {k: v for k, v in kwargs.items() if not k.startswith("_")},
                },
            )

            # Log request with context
            logger.info(
                f"Request started: {context.request_id} - {request.method} {request.path} - Agent: {context.agent_id}"
            )

            # Execute the function
            try:
                result = f(*args, **kwargs)

                # Add trace entry for successful completion
                context.add_trace("request_completed", {"status": "success", "result_type": type(result).__name__})

                # Log successful completion
                logger.info(f"Request completed: {context.request_id} - Duration: {context.duration:.3f}s")

                return result

            except Exception as e:
                # Add trace entry for error
                context.add_trace("request_error", {"error_type": type(e).__name__, "error_message": str(e)})

                # Log error with context
                logger.error(f"Request failed: {context.request_id} - Error: {str(e)}")

                # Re-raise the exception
                raise

        except Exception as e:
            logger.error(f"Error in agent context injection: {e}")
            # Continue without context if injection fails
            return f(*args, **kwargs)

    return decorated_function


def log_agent_context(f):
    """Decorator to log agent context for debugging."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Get context from request
            context = getattr(request, "agent_context", None)

            if context:
                # Log detailed context for debugging
                logger.debug(f"Agent Context: {json.dumps(context.to_dict(), indent=2)}")

            return f(*args, **kwargs)

        except Exception as e:
            logger.error(f"Error logging agent context: {e}")
            return f(*args, **kwargs)

    return decorated_function


def correlate_agent_requests(f):
    """Decorator to correlate agent requests across services."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Get context from request
            context = getattr(request, "agent_context", None)

            if context:
                # Add correlation headers to outgoing requests
                def add_correlation_headers(headers: Dict[str, str]):
                    headers["X-Request-ID"] = context.request_id
                    if context.agent_id:
                        headers["X-Agent-ID"] = context.agent_id
                    if context.user_id:
                        headers["X-User-ID"] = context.user_id
                    if context.session_id:
                        headers["X-Session-ID"] = context.session_id
                    headers["X-Correlation-ID"] = context.request_id

                # Store correlation function in request context
                request.add_correlation_headers = add_correlation_headers

                # Add trace entry
                context.add_trace("correlation_enabled")

            return f(*args, **kwargs)

        except Exception as e:
            logger.error(f"Error setting up request correlation: {e}")
            return f(*args, **kwargs)

    return decorated_function


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization")

        if not token:
            return jsonify({"error": "No token provided"}), 401

        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith("Bearer "):
                token = token[7:]

            # Verify token
            user_manager = UserManager(current_app.redis, current_app.config["SECRET_KEY"])
            user_data = user_manager.verify_token(token)

            if not user_data:
                return jsonify({"error": "Invalid token"}), 401

            # Add user data to request context
            request.user = user_data

            # Update agent context with user information
            context = getattr(request, "agent_context", None)
            if context:
                context.user_id = user_data.get("user_id")
                context.add_trace(
                    "user_authenticated", {"user_id": user_data.get("user_id"), "role": user_data.get("role")}
                )

            return f(*args, **kwargs)

        except Exception as e:
            current_app.logger.error(f"Authentication error: {str(e)}")
            return jsonify({"error": "Authentication failed"}), 401

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization")

        if not token:
            return jsonify({"error": "No token provided"}), 401

        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith("Bearer "):
                token = token[7:]

            # Verify token
            user_manager = UserManager(current_app.redis, current_app.config["SECRET_KEY"])
            user_data = user_manager.verify_token(token)

            if not user_data:
                return jsonify({"error": "Invalid token"}), 401

            # Check if user is admin
            if user_data.get("role") != "admin":
                return jsonify({"error": "Admin access required"}), 403

            # Add user data to request context
            request.user = user_data

            # Update agent context with admin information
            context = getattr(request, "agent_context", None)
            if context:
                context.user_id = user_data.get("user_id")
                context.add_trace("admin_authenticated", {"user_id": user_data.get("user_id"), "role": "admin"})

            return f(*args, **kwargs)

        except Exception as e:
            current_app.logger.error(f"Authentication error: {str(e)}")
            return jsonify({"error": "Authentication failed"}), 401

    return decorated_function


def inject_user(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization")

        if token:
            try:
                # Remove 'Bearer ' prefix if present
                if token.startswith("Bearer "):
                    token = token[7:]

                # Verify token
                user_manager = UserManager(current_app.redis, current_app.config["SECRET_KEY"])
                user_data = user_manager.verify_token(token)

                if user_data:
                    request.user = user_data

                    # Update agent context with user information
                    context = getattr(request, "agent_context", None)
                    if context:
                        context.user_id = user_data.get("user_id")
                        context.add_trace(
                            "user_injected", {"user_id": user_data.get("user_id"), "role": user_data.get("role")}
                        )

            except Exception as e:
                current_app.logger.error(f"Token verification error: {str(e)}")

        return f(*args, **kwargs)

    return decorated_function
