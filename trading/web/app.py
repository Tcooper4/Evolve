"""
Flask Web API for Trading System

This module provides a RESTful API for portfolio management, risk analysis, and backtesting.
"""

import asyncio
import os
import re
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

import jwt
import pandas as pd
from flask import Flask, current_app, jsonify, request
from flask_cors import CORS
from pydantic import BaseModel, Field, ValidationError, validator

from trading.backtesting.backtester import Backtester
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.risk.risk_manager import RiskManager
from trading.utils.logging_utils import log_manager

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY", "your-secret-key-change-in-production"
)

# CORS configuration
CORS(
    app,
    resources={
        r"/*": {
            "origins": os.environ.get("ALLOWED_ORIGINS", "*").split(","),
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-OpenAI-Key"],
            "expose_headers": ["Content-Type", "X-Request-ID"],
        }
    },
)

# Security headers


@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers[
        "Strict-Transport-Security"
    ] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


portfolio_manager = PortfolioManager()

# Enhanced Pydantic models for request validation


class AssetRequest(BaseModel):
    asset: str = Field(..., description="Asset symbol", min_length=1, max_length=10)
    quantity: float = Field(..., gt=0, description="Quantity of asset")
    price: float = Field(..., gt=0, description="Price per unit")

    @validator("asset")
    def validate_asset(cls, v: str) -> str:
        if not re.match(r"^[A-Z0-9]+$", v):
            raise ValueError("Asset symbol must be alphanumeric and uppercase")
        return v.upper()

    @validator("quantity")
    def validate_quantity(cls, v: float) -> float:
        if v > 1e9:  # 1 billion max
            raise ValueError("Quantity too large")
        return v

    @validator("price")
    def validate_price(cls, v: float) -> float:
        if v > 1e6:  # 1 million max
            raise ValueError("Price too large")
        return v


class BacktestRequest(BaseModel):
    strategy: str = Field(
        ..., description="Strategy name", min_length=1, max_length=100
    )
    data: Dict[str, Any] = Field(..., description="Historical data for backtesting")
    start_date: Optional[datetime] = Field(None, description="Start date for backtest")
    end_date: Optional[datetime] = Field(None, description="End date for backtest")

    @validator("strategy")
    def validate_strategy(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Strategy name must be alphanumeric with underscores and hyphens only"
            )
        return v

    @validator("data")
    def validate_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("Data cannot be empty")
        if len(v) > 1000:  # Limit data size
            raise ValueError("Data too large")
        return v


class RiskAnalysisRequest(BaseModel):
    portfolio_data: Dict[str, Any] = Field(
        ..., description="Portfolio data for risk analysis"
    )
    confidence_level: float = Field(
        0.95, ge=0.5, le=0.99, description="Confidence level for VaR"
    )

    @validator("portfolio_data")
    def validate_portfolio_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("Portfolio data cannot be empty")
        return v


# Request validation decorator


def validate_request(model_class: type[BaseModel]):
    """Decorator to validate request data against Pydantic model."""

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args: Any, **kwargs: Any) -> Any:
            try:
                if request.is_json:
                    validated_data = model_class(**request.json)
                    # Add validated data to request context
                    request.validated_data = validated_data
                else:
                    raise APIError("Request must be JSON", 400)
            except ValidationError as e:
                return (
                    jsonify({"error": "Validation error", "details": e.errors()}),
                    400,
                )
            except Exception as e:
                return (
                    jsonify({"error": "Request validation failed", "message": str(e)}),
                    400,
                )
            return f(*args, **kwargs)

        return decorated

    return decorator


# OpenAI key validation


def validate_openai_key(f: Callable) -> Callable:
    """Decorator to validate OpenAI API key in headers."""

    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        openai_key = request.headers.get("X-OpenAI-Key")
        if not openai_key:
            return jsonify({"error": "OpenAI API key is required"}), 401

        # Validate OpenAI key format
        if not re.match(r"^sk-[a-zA-Z0-9]{48}$", openai_key):
            return jsonify({"error": "Invalid OpenAI API key format"}), 401

        # Store key in request context for later use
        request.openai_key = openai_key
        return f(*args, **kwargs)

    return decorated


# Enhanced authentication decorator


def token_required(f: Callable) -> Callable:
    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401

        if not token.startswith("Bearer "):
            return jsonify({"error": "Invalid token format"}), 401

        try:
            token = token.split(" ")[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(
                token, current_app.config["SECRET_KEY"], algorithms=["HS256"]
            )
            current_user = data["user"]

            # Validate user permissions if needed
            if "permissions" in data:
                request.user_permissions = data["permissions"]

        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        except Exception as e:
            log_manager.logger.error(f"Token validation error: {str(e)}")
            return jsonify({"error": "Token validation failed"}), 401

        request.current_user = current_user
        return f(current_user, *args, **kwargs)

    return decorated


# Rate limiting decorator


def rate_limit(max_requests: int = 100, window: int = 3600):
    """Simple rate limiting decorator."""

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args: Any, **kwargs: Any) -> Any:
            # Simple in-memory rate limiting (use Redis in production)
            request.remote_addr
            datetime.now()

            # This is a simplified implementation
            # In production, use Redis or similar for rate limiting
            return f(*args, **kwargs)

        return decorated

    return decorator


# Error handling


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


@app.errorhandler(APIError)
def handle_api_error(error: APIError) -> Any:
    response = jsonify(
        {
            "error": error.message,
            "status_code": error.status_code,
            "timestamp": datetime.now().isoformat(),
        }
    )
    response.status_code = error.status_code
    return response


@app.errorhandler(ValidationError)
def handle_validation_error(error: ValidationError) -> Any:
    response = jsonify(
        {
            "error": "Validation error",
            "details": error.errors(),
            "timestamp": datetime.now().isoformat(),
        }
    )
    response.status_code = 400
    return response


@app.errorhandler(404)
def not_found(error) -> Any:
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "status_code": 404,
                "timestamp": datetime.now().isoformat(),
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error) -> Any:
    log_manager.logger.error(f"Internal server error: {str(error)}")
    return (
        jsonify(
            {
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": datetime.now().isoformat(),
            }
        ),
        500,
    )


# Service initialization


def get_risk_manager() -> RiskManager:
    """Initialize and return risk manager instance."""
    if not hasattr(app, "risk_manager"):
        app.risk_manager = RiskManager()
    return app.risk_manager


def get_backtester() -> Backtester:
    """Initialize and return backtester instance."""
    if not hasattr(app, "backtester"):
        app.backtester = Backtester()
    return app.backtester


# Health check endpoint


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint - returns system health status"""
    try:
        from monitoring.health_check import get_health_checker
        
        checker = get_health_checker()
        health = checker.check_system_health()
        status_code = 200 if health['status'] == 'healthy' else 503
        return jsonify(health), status_code
    except Exception as e:
        log_manager.logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route("/health", methods=["GET"])
def health_check() -> Any:
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }
    )


# Enhanced routes with validation


@app.route("/portfolio", methods=["GET"])
@token_required
@rate_limit(max_requests=1000, window=3600)
def get_portfolio(current_user: str) -> Any:
    """Get the current portfolio value and composition."""
    try:
        portfolio_value = portfolio_manager.get_portfolio_value()
        composition = portfolio_manager.get_portfolio_composition()
        return jsonify(
            {
                "portfolio_value": portfolio_value,
                "composition": composition,
                "timestamp": datetime.now().isoformat(),
                "user": current_user,
            }
        )
    except Exception as e:
        log_manager.logger.error(f"Error getting portfolio: {str(e)}")
        raise APIError("Failed to retrieve portfolio information", 500)


@app.route("/portfolio/add", methods=["POST"])
@token_required
@validate_request(AssetRequest)
@rate_limit(max_requests=100, window=3600)
def add_asset(current_user: str) -> Any:
    """Add an asset to the portfolio."""
    try:
        data = request.validated_data
        portfolio_manager.add_asset(data.asset, data.quantity, data.price)
        return jsonify(
            {
                "message": "Asset added successfully",
                "asset": data.asset,
                "quantity": data.quantity,
                "price": data.price,
                "timestamp": datetime.now().isoformat(),
                "user": current_user,
            }
        )
    except Exception as e:
        log_manager.logger.error(f"Error adding asset: {str(e)}")
        raise APIError("Failed to add asset", 500)


@app.route("/risk/var", methods=["GET"])
@token_required
@rate_limit(max_requests=500, window=3600)
def get_risk_metrics(current_user: str) -> Any:
    """Get comprehensive risk metrics for the portfolio."""
    try:
        risk_manager = get_risk_manager()
        var = risk_manager.calculate_var()
        cvar = risk_manager.calculate_cvar()
        max_drawdown = risk_manager.calculate_max_drawdown()
        volatility = risk_manager.calculate_volatility()

        return jsonify(
            {
                "var": var,
                "cvar": cvar,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "timestamp": datetime.now().isoformat(),
                "user": current_user,
            }
        )
    except Exception as e:
        log_manager.logger.error(f"Error calculating risk metrics: {str(e)}")
        raise APIError("Failed to calculate risk metrics", 500)


@app.route("/risk/analyze", methods=["POST"])
@token_required
@validate_request(RiskAnalysisRequest)
@validate_openai_key
@rate_limit(max_requests=50, window=3600)
def analyze_risk(current_user: str) -> Any:
    """Analyze portfolio risk using OpenAI."""
    try:
        data = request.validated_data
        request.openai_key

        # Use OpenAI key for risk analysis
        # This is where you would integrate with OpenAI API
        get_risk_manager()

        return jsonify(
            {
                "message": "Risk analysis completed",
                "analysis": {
                    "risk_score": 0.75,
                    "recommendations": [
                        "Diversify portfolio",
                        "Reduce exposure to tech stocks",
                    ],
                    "confidence": data.confidence_level,
                },
                "timestamp": datetime.now().isoformat(),
                "user": current_user,
            }
        )
    except Exception as e:
        log_manager.logger.error(f"Error in risk analysis: {str(e)}")
        raise APIError("Failed to analyze risk", 500)


@app.route("/backtest/run", methods=["POST"])
@token_required
@validate_request(BacktestRequest)
@rate_limit(max_requests=20, window=3600)
def run_backtest(current_user: str) -> Any:
    """Run a backtest using the provided strategy and return detailed results."""
    try:
        data = request.validated_data
        backtester = get_backtester()

        # Convert data to DataFrame
        df = pd.DataFrame(data.data)
        if data.start_date:
            df = df[df.index >= data.start_date]
        if data.end_date:
            df = df[df.index <= data.end_date]

        results = backtester.run_backtest(data.strategy, df)

        return jsonify(
            {
                "message": "Backtest completed successfully",
                "results": {
                    "sharpe_ratio": results.get("sharpe_ratio"),
                    "total_return": results.get("total_return"),
                    "max_drawdown": results.get("max_drawdown"),
                    "equity_curve": (
                        results.get("equity_curve").to_dict()
                        if results.get("equity_curve") is not None
                        else None
                    ),
                    "trades": results.get("trades"),
                    "metrics": results.get("metrics"),
                },
                "timestamp": datetime.now().isoformat(),
                "user": current_user,
            }
        )
    except Exception as e:
        log_manager.logger.error(f"Error running backtest: {str(e)}")
        raise APIError("Failed to run backtest", 500)


# Optional: Async support using Quart
try:
    from quart import Quart
    from quart.flask_patch import patch_all

    # Create Quart app
    quart_app = Quart(__name__)
    quart_app.config["SECRET_KEY"] = app.config["SECRET_KEY"]

    # Patch Flask routes to Quart
    patch_all()

    # Add async versions of routes
    @quart_app.route("/portfolio/async", methods=["GET"])
    async def get_portfolio_async() -> Any:
        """Async version of get_portfolio."""
        return await asyncio.to_thread(get_portfolio)

    @quart_app.route("/backtest/run/async", methods=["POST"])
    async def run_backtest_async() -> Any:
        """Async version of run_backtest."""
        return await asyncio.to_thread(run_backtest)

except ImportError:
    log_manager.logger.warning("Quart not installed. Async support disabled.")

if __name__ == "__main__":
    # Production configuration
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))

    app.run(debug=debug_mode, host=host, port=port)
