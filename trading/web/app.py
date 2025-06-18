from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from functools import wraps
import jwt
from flask import Flask, jsonify, request, current_app
from pydantic import BaseModel, Field, validator
import pandas as pd
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.risk.risk_manager import RiskManager
from trading.backtesting.backtester import Backtester
from trading.utils.logging_utils import log_manager

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change in production
portfolio_manager = PortfolioManager()

# Pydantic models for request validation
class AssetRequest(BaseModel):
    asset: str = Field(..., description="Asset symbol")
    quantity: float = Field(..., gt=0, description="Quantity of asset")
    price: float = Field(..., gt=0, description="Price per unit")

    @validator('asset')
    def validate_asset(cls, v):
        if not v.isalnum():
            raise ValueError('Asset symbol must be alphanumeric')
        return v.upper()

class BacktestRequest(BaseModel):
    strategy: str = Field(..., description="Strategy name")
    data: Dict[str, Any] = Field(..., description="Historical data for backtesting")
    start_date: Optional[datetime] = Field(None, description="Start date for backtest")
    end_date: Optional[datetime] = Field(None, description="End date for backtest")

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['user']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Error handling
class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@app.errorhandler(APIError)
def handle_api_error(error):
    response = jsonify({'error': error.message})
    response.status_code = error.status_code
    return response

# Service initialization
def get_risk_manager() -> RiskManager:
    """Initialize and return risk manager instance."""
    if not hasattr(app, 'risk_manager'):
        app.risk_manager = RiskManager()
    return app.risk_manager

def get_backtester() -> Backtester:
    """Initialize and return backtester instance."""
    if not hasattr(app, 'backtester'):
        app.backtester = Backtester()
    return app.backtester

# Routes
@app.route('/portfolio', methods=['GET'])
@token_required
def get_portfolio(current_user):
    """Get the current portfolio value and composition."""
    try:
        portfolio_value = portfolio_manager.get_portfolio_value()
        composition = portfolio_manager.get_portfolio_composition()
        return jsonify({
            'portfolio_value': portfolio_value,
            'composition': composition,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        log_manager.logger.error(f"Error getting portfolio: {str(e)}")
        raise APIError("Failed to retrieve portfolio information", 500)

@app.route('/portfolio/add', methods=['POST'])
@token_required
def add_asset(current_user):
    """Add an asset to the portfolio."""
    try:
        data = AssetRequest(**request.json)
        portfolio_manager.add_asset(data.asset, data.quantity, data.price)
        return jsonify({
            'message': 'Asset added successfully',
            'asset': data.asset,
            'quantity': data.quantity,
            'price': data.price,
            'timestamp': datetime.now().isoformat()
        })
    except ValueError as e:
        raise APIError(str(e), 400)
    except Exception as e:
        log_manager.logger.error(f"Error adding asset: {str(e)}")
        raise APIError("Failed to add asset", 500)

@app.route('/risk/var', methods=['GET'])
@token_required
def get_risk_metrics(current_user):
    """Get comprehensive risk metrics for the portfolio."""
    try:
        risk_manager = get_risk_manager()
        var = risk_manager.calculate_var()
        cvar = risk_manager.calculate_cvar()
        max_drawdown = risk_manager.calculate_max_drawdown()
        volatility = risk_manager.calculate_volatility()
        
        return jsonify({
            'var': var,
            'cvar': cvar,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        log_manager.logger.error(f"Error calculating risk metrics: {str(e)}")
        raise APIError("Failed to calculate risk metrics", 500)

@app.route('/backtest/run', methods=['POST'])
@token_required
def run_backtest(current_user):
    """Run a backtest using the provided strategy and return detailed results."""
    try:
        data = BacktestRequest(**request.json)
        backtester = get_backtester()
        
        # Convert data to DataFrame
        df = pd.DataFrame(data.data)
        if data.start_date:
            df = df[df.index >= data.start_date]
        if data.end_date:
            df = df[df.index <= data.end_date]
            
        results = backtester.run_backtest(data.strategy, df)
        
        return jsonify({
            'message': 'Backtest completed successfully',
            'results': {
                'sharpe_ratio': results.get('sharpe_ratio'),
                'total_return': results.get('total_return'),
                'max_drawdown': results.get('max_drawdown'),
                'equity_curve': results.get('equity_curve').to_dict(),
                'trades': results.get('trades'),
                'metrics': results.get('metrics')
            },
            'timestamp': datetime.now().isoformat()
        })
    except ValueError as e:
        raise APIError(str(e), 400)
    except Exception as e:
        log_manager.logger.error(f"Error running backtest: {str(e)}")
        raise APIError("Failed to run backtest", 500)

# Optional: Async support using Quart
try:
    from quart import Quart
    from quart.flask_patch import patch_all
    
    # Create Quart app
    quart_app = Quart(__name__)
    quart_app.config['SECRET_KEY'] = app.config['SECRET_KEY']
    
    # Patch Flask routes to Quart
    patch_all()
    
    # Add async versions of routes
    @quart_app.route('/portfolio/async', methods=['GET'])
    async def get_portfolio_async():
        """Async version of get_portfolio."""
        return await asyncio.to_thread(get_portfolio)
    
    @quart_app.route('/backtest/run/async', methods=['POST'])
    async def run_backtest_async():
        """Async version of run_backtest."""
        return await asyncio.to_thread(run_backtest)
        
except ImportError:
    log_manager.logger.warning("Quart not installed. Async support disabled.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 