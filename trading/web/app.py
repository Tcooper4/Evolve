from flask import Flask, jsonify, request
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.risk.risk_manager import RiskManager
from trading.backtesting.backtester import Backtester
import pandas as pd

app = Flask(__name__)
portfolio_manager = PortfolioManager()
risk_manager = None
backtester = None

@app.route('/portfolio', methods=['GET'])
def get_portfolio():
    """Get the current portfolio value."""
    return jsonify({'portfolio_value': portfolio_manager.get_portfolio_value()})

@app.route('/portfolio/add', methods=['POST'])
def add_asset():
    """Add an asset to the portfolio."""
    data = request.json
    portfolio_manager.add_asset(data['asset'], data['quantity'], data['price'])
    return jsonify({'message': 'Asset added successfully'})

@app.route('/risk/var', methods=['GET'])
def get_var():
    """Get the Value at Risk (VaR) for the portfolio."""
    if risk_manager is None:
        return jsonify({'error': 'Risk manager not initialized'}), 400
    return jsonify({'var': risk_manager.calculate_var()})

@app.route('/backtest/run', methods=['POST'])
def run_backtest():
    """Run a backtest using the provided strategy."""
    data = request.json
    strategy = data.get('strategy')
    if strategy is None:
        return jsonify({'error': 'Strategy not provided'}), 400
    backtester = Backtester(pd.DataFrame(data['data']))
    backtester.run_backtest(strategy)
    return jsonify({'message': 'Backtest completed successfully'})

if __name__ == '__main__':
    app.run(debug=True) 