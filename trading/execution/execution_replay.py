"""
Execution Replay System

Enhanced with Batch 10 features: trade-by-trade visualization system for execution replay.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionReplay:
    """Trade-by-trade execution replay system with visualization."""

    def __init__(self, replay_dir: str = "replays"):
        """Initialize the execution replay system.
        
        Args:
            replay_dir: Directory for storing replay data
        """
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.replay_history = []
        self.current_replay = None
        
        logger.info(f"ExecutionReplay initialized with replay directory: {self.replay_dir}")

    def start_replay_session(
        self, 
        session_name: str, 
        initial_capital: float = 100000.0,
        symbols: Optional[List[str]] = None
    ) -> str:
        """Start a new replay session.
        
        Args:
            session_name: Name of the replay session
            initial_capital: Starting capital
            symbols: List of symbols to track
            
        Returns:
            Session ID
        """
        try:
            session_id = f"replay_{int(time.time())}_{session_name}"
            
            self.current_replay = {
                "session_id": session_id,
                "session_name": session_name,
                "start_time": datetime.now().isoformat(),
                "initial_capital": initial_capital,
                "current_capital": initial_capital,
                "symbols": symbols or [],
                "trades": [],
                "positions": {},
                "equity_curve": [],
                "market_data": {},
                "events": []
            }
            
            # Initialize equity curve
            self.current_replay["equity_curve"].append({
                "timestamp": datetime.now().isoformat(),
                "equity": initial_capital,
                "cash": initial_capital,
                "positions_value": 0.0
            })
            
            logger.info(f"Started replay session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting replay session: {e}")
            return None

    def record_trade(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: float, 
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None,
        slippage: float = 0.0,
        commission: float = 0.0
    ) -> bool:
        """Record a trade in the current replay session.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Number of shares/contracts
            price: Execution price
            timestamp: Trade timestamp
            order_id: Order identifier
            slippage: Execution slippage
            commission: Trading commission
            
        Returns:
            True if trade recorded successfully
        """
        try:
            if not self.current_replay:
                logger.error("No active replay session")
                return False
            
            trade_timestamp = timestamp or datetime.now()
            
            # Calculate trade value and update positions
            trade_value = quantity * price
            trade_cost = trade_value + commission
            
            if side.lower() == "buy":
                # Buying
                if self.current_replay["current_capital"] < trade_cost:
                    logger.warning(f"Insufficient capital for trade: {trade_cost} > {self.current_replay['current_capital']}")
                    return False
                
                self.current_replay["current_capital"] -= trade_cost
                
                # Update positions
                if symbol not in self.current_replay["positions"]:
                    self.current_replay["positions"][symbol] = {
                        "quantity": 0,
                        "avg_price": 0.0,
                        "total_cost": 0.0
                    }
                
                pos = self.current_replay["positions"][symbol]
                total_quantity = pos["quantity"] + quantity
                total_cost = pos["total_cost"] + trade_cost
                
                pos["quantity"] = total_quantity
                pos["avg_price"] = total_cost / total_quantity if total_quantity > 0 else 0.0
                pos["total_cost"] = total_cost
                
            else:  # sell
                # Selling
                if symbol not in self.current_replay["positions"] or self.current_replay["positions"][symbol]["quantity"] < quantity:
                    logger.warning(f"Insufficient position for sale: {symbol}")
                    return False
                
                # Calculate PnL
                pos = self.current_replay["positions"][symbol]
                realized_pnl = (price - pos["avg_price"]) * quantity - commission
                
                self.current_replay["current_capital"] += trade_value - commission
                
                # Update positions
                pos["quantity"] -= quantity
                if pos["quantity"] <= 0:
                    del self.current_replay["positions"][symbol]
            
            # Record trade
            trade_record = {
                "timestamp": trade_timestamp.isoformat(),
                "symbol": symbol,
                "side": side.lower(),
                "quantity": quantity,
                "price": price,
                "trade_value": trade_value,
                "commission": commission,
                "slippage": slippage,
                "order_id": order_id,
                "realized_pnl": realized_pnl if side.lower() == "sell" else 0.0,
                "cash_after": self.current_replay["current_capital"],
                "positions_after": self.current_replay["positions"].copy()
            }
            
            self.current_replay["trades"].append(trade_record)
            
            # Update equity curve
            positions_value = self._calculate_positions_value()
            total_equity = self.current_replay["current_capital"] + positions_value
            
            self.current_replay["equity_curve"].append({
                "timestamp": trade_timestamp.isoformat(),
                "equity": total_equity,
                "cash": self.current_replay["current_capital"],
                "positions_value": positions_value
            })
            
            # Record event
            self._record_event("trade_executed", trade_record)
            
            logger.info(f"Trade recorded: {side} {quantity} {symbol} @ {price}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False

    def record_market_data(
        self, 
        symbol: str, 
        price: float, 
        volume: float = 0.0,
        timestamp: Optional[datetime] = None
    ):
        """Record market data for replay.
        
        Args:
            symbol: Trading symbol
            price: Current price
            volume: Trading volume
            timestamp: Data timestamp
        """
        try:
            if not self.current_replay:
                return
            
            data_timestamp = timestamp or datetime.now()
            
            if symbol not in self.current_replay["market_data"]:
                self.current_replay["market_data"][symbol] = []
            
            market_record = {
                "timestamp": data_timestamp.isoformat(),
                "price": price,
                "volume": volume
            }
            
            self.current_replay["market_data"][symbol].append(market_record)
            
        except Exception as e:
            logger.error(f"Error recording market data: {e}")

    def _calculate_positions_value(self) -> float:
        """Calculate current positions value."""
        try:
            total_value = 0.0
            
            for symbol, position in self.current_replay["positions"].items():
                # Use latest market price if available
                if symbol in self.current_replay["market_data"] and self.current_replay["market_data"][symbol]:
                    latest_price = self.current_replay["market_data"][symbol][-1]["price"]
                    total_value += position["quantity"] * latest_price
                else:
                    # Use average price as fallback
                    total_value += position["quantity"] * position["avg_price"]
            
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating positions value: {e}")
            return 0.0

    def _record_event(self, event_type: str, event_data: Dict[str, Any]):
        """Record an event in the replay session."""
        try:
            event_record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "event_data": event_data
            }
            
            self.current_replay["events"].append(event_record)
            
        except Exception as e:
            logger.error(f"Error recording event: {e}")

    def end_replay_session(self) -> Optional[str]:
        """End the current replay session and save data.
        
        Returns:
            Path to saved replay file
        """
        try:
            if not self.current_replay:
                logger.error("No active replay session to end")
                return None
            
            # Add end time
            self.current_replay["end_time"] = datetime.now().isoformat()
            
            # Calculate final statistics
            self.current_replay["statistics"] = self._calculate_session_statistics()
            
            # Save replay data
            filename = f"{self.current_replay['session_id']}.json"
            filepath = self.replay_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.current_replay, f, indent=2, default=str)
            
            # Store in history
            self.replay_history.append({
                "session_id": self.current_replay["session_id"],
                "session_name": self.current_replay["session_name"],
                "filepath": str(filepath),
                "start_time": self.current_replay["start_time"],
                "end_time": self.current_replay["end_time"],
                "total_trades": len(self.current_replay["trades"]),
                "final_equity": self.current_replay["equity_curve"][-1]["equity"] if self.current_replay["equity_curve"] else 0.0
            })
            
            logger.info(f"Replay session ended: {filepath}")
            
            # Clear current replay
            self.current_replay = None
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error ending replay session: {e}")
            return None

    def _calculate_session_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive session statistics."""
        try:
            if not self.current_replay["trades"]:
                return {"error": "No trades recorded"}
            
            trades_df = pd.DataFrame(self.current_replay["trades"])
            equity_df = pd.DataFrame(self.current_replay["equity_curve"])
            
            # Basic statistics
            total_trades = len(trades_df)
            buy_trades = len(trades_df[trades_df["side"] == "buy"])
            sell_trades = len(trades_df[trades_df["side"] == "sell"])
            
            # PnL statistics
            total_realized_pnl = trades_df["realized_pnl"].sum()
            total_commission = trades_df["commission"].sum()
            total_slippage = trades_df["slippage"].sum()
            
            # Equity statistics
            if len(equity_df) > 1:
                initial_equity = equity_df.iloc[0]["equity"]
                final_equity = equity_df.iloc[-1]["equity"]
                total_return = (final_equity - initial_equity) / initial_equity
                
                # Calculate drawdown
                equity_df["peak"] = equity_df["equity"].expanding().max()
                equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
                max_drawdown = equity_df["drawdown"].min()
            else:
                total_return = 0.0
                max_drawdown = 0.0
            
            # Trade analysis
            winning_trades = len(trades_df[trades_df["realized_pnl"] > 0])
            losing_trades = len(trades_df[trades_df["realized_pnl"] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            avg_win = trades_df[trades_df["realized_pnl"] > 0]["realized_pnl"].mean() if winning_trades > 0 else 0.0
            avg_loss = trades_df[trades_df["realized_pnl"] < 0]["realized_pnl"].mean() if losing_trades > 0 else 0.0
            
            return {
                "total_trades": total_trades,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "total_realized_pnl": total_realized_pnl,
                "total_commission": total_commission,
                "total_slippage": total_slippage,
                "net_pnl": total_realized_pnl - total_commission,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                "final_equity": equity_df.iloc[-1]["equity"] if len(equity_df) > 0 else 0.0,
                "final_cash": equity_df.iloc[-1]["cash"] if len(equity_df) > 0 else 0.0,
                "final_positions_value": equity_df.iloc[-1]["positions_value"] if len(equity_df) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating session statistics: {e}")
            return {"error": str(e)}

    def load_replay(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a replay session from file.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Replay data dictionary
        """
        try:
            filename = f"{session_id}.json"
            filepath = self.replay_dir / filename
            
            if not filepath.exists():
                logger.error(f"Replay file not found: {filepath}")
                return None
            
            with open(filepath, 'r') as f:
                replay_data = json.load(f)
            
            logger.info(f"Replay loaded: {session_id}")
            return replay_data
            
        except Exception as e:
            logger.error(f"Error loading replay: {e}")
            return None

    def generate_replay_visualization(
        self, 
        session_id: str, 
        output_dir: str = "replay_visualizations"
    ) -> Optional[str]:
        """Generate visualization for a replay session.
        
        Args:
            session_id: Session ID to visualize
            output_dir: Output directory for visualizations
            
        Returns:
            Path to generated visualization file
        """
        try:
            replay_data = self.load_replay(session_id)
            if not replay_data:
                return None
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate HTML visualization
            html_content = self._generate_replay_html(replay_data)
            
            html_file = output_path / f"replay_{session_id}.html"
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Replay visualization generated: {html_file}")
            return str(html_file)
            
        except Exception as e:
            logger.error(f"Error generating replay visualization: {e}")
            return None

    def _generate_replay_html(self, replay_data: Dict[str, Any]) -> str:
        """Generate HTML visualization for replay data."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Execution Replay - {session_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric { padding: 15px; background: #f8f9fa; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
        .chart-container { margin: 20px 0; height: 400px; }
        .trade-log { max-height: 400px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #007bff; color: white; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Execution Replay: {session_name}</h1>
            <p>Session ID: {session_id}</p>
            <p>Duration: {start_time} to {end_time}</p>
        </div>
        
        <div class="section">
            <h2>Performance Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{total_return:.2%}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{final_equity:,.0f}</div>
                    <div class="metric-label">Final Equity</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_trades}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{win_rate:.1%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{max_drawdown:.2%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{net_pnl:,.0f}</div>
                    <div class="metric-label">Net PnL</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Equity Curve</h2>
            <div id="equityChart" class="chart-container"></div>
        </div>
        
        <div class="section">
            <h2>Trade Log</h2>
            <div class="trade-log">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th>Value</th>
                            <th>PnL</th>
                            <th>Cash</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Equity curve chart
        const equityData = {equity_data};
        const layout = {{
            title: 'Equity Curve',
            xaxis: {{ title: 'Time' }},
            yaxis: {{ title: 'Equity ($)' }},
            hovermode: 'x unified'
        }};
        Plotly.newPlot('equityChart', equityData, layout);
    </script>
</body>
</html>
        """
        
        # Prepare data for template
        stats = replay_data.get("statistics", {})
        
        # Format trade rows
        trade_rows = ""
        for trade in replay_data.get("trades", []):
            pnl_class = "positive" if trade.get("realized_pnl", 0) >= 0 else "negative"
            trade_rows += f"""
                <tr>
                    <td>{trade.get('timestamp', '')}</td>
                    <td>{trade.get('symbol', '')}</td>
                    <td>{trade.get('side', '').upper()}</td>
                    <td>{trade.get('quantity', 0):,.0f}</td>
                    <td>${trade.get('price', 0):.2f}</td>
                    <td>${trade.get('trade_value', 0):,.0f}</td>
                    <td class="{pnl_class}">${trade.get('realized_pnl', 0):,.2f}</td>
                    <td>${trade.get('cash_after', 0):,.0f}</td>
                </tr>
            """
        
        # Prepare equity curve data
        equity_curve = replay_data.get("equity_curve", [])
        equity_data = [{
            "x": [point["timestamp"] for point in equity_curve],
            "y": [point["equity"] for point in equity_curve],
            "type": "scatter",
            "mode": "lines",
            "name": "Equity",
            "line": {"color": "#007bff"}
        }]
        
        return html_template.format(
            session_name=replay_data.get("session_name", "Unknown"),
            session_id=replay_data.get("session_id", "Unknown"),
            start_time=replay_data.get("start_time", "Unknown"),
            end_time=replay_data.get("end_time", "Unknown"),
            total_return=stats.get("total_return", 0.0),
            final_equity=stats.get("final_equity", 0.0),
            total_trades=stats.get("total_trades", 0),
            win_rate=stats.get("win_rate", 0.0),
            max_drawdown=stats.get("max_drawdown", 0.0),
            net_pnl=stats.get("net_pnl", 0.0),
            trade_rows=trade_rows,
            equity_data=json.dumps(equity_data)
        )

    def get_replay_summary(self) -> Dict[str, Any]:
        """Get summary of all replay sessions."""
        if not self.replay_history:
            return {"total_replays": 0}
        
        total_replays = len(self.replay_history)
        total_trades = sum(replay["total_trades"] for replay in self.replay_history)
        
        # Calculate average performance
        returns = []
        for replay in self.replay_history:
            if replay["final_equity"] > 0:
                # Load replay to get initial equity
                replay_data = self.load_replay(replay["session_id"])
                if replay_data and replay_data.get("equity_curve"):
                    initial_equity = replay_data["equity_curve"][0]["equity"]
                    if initial_equity > 0:
                        returns.append((replay["final_equity"] - initial_equity) / initial_equity)
        
        avg_return = np.mean(returns) if returns else 0.0
        
        return {
            "total_replays": total_replays,
            "total_trades": total_trades,
            "average_return": avg_return,
            "recent_replays": self.replay_history[-5:] if len(self.replay_history) > 5 else self.replay_history
        } 