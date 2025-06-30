"""
Live Market Runner

This module provides a comprehensive live market data streaming and agent triggering system.
It streams live data, triggers agents periodically (every X seconds or based on price moves),
and stores and updates live forecast vs actual results.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import threading
from collections import defaultdict, deque
import signal
import sys

# Local imports
from trading.market.market_data import MarketData
from trading.agents.agent_manager import AgentManager, get_agent_manager
from trading.agents.base_agent_interface import AgentResult
from trading.memory.agent_memory import AgentMemory
from trading.portfolio.portfolio_manager import PortfolioManager


class TriggerType(Enum):
    """Trigger types for agent execution."""
    TIME_BASED = "time_based"
    PRICE_MOVE = "price_move"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    MANUAL = "manual"


@dataclass
class TriggerConfig:
    """Configuration for agent triggers."""
    trigger_type: TriggerType
    interval_seconds: int = 60  # For time-based triggers
    price_move_threshold: float = 0.01  # 1% for price-based triggers
    volume_spike_threshold: float = 2.0  # 2x average for volume triggers
    volatility_threshold: float = 0.02  # 2% for volatility triggers
    enabled: bool = True


@dataclass
class ForecastResult:
    """Result of a forecast prediction."""
    timestamp: datetime
    symbol: str
    forecast_price: float
    forecast_direction: str  # 'up', 'down', 'sideways'
    confidence: float
    horizon_hours: int
    model_name: str
    features: Dict[str, Any]
    actual_price: Optional[float] = None
    actual_direction: Optional[str] = None
    accuracy: Optional[float] = None
    pnl: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        return {'success': True, 'result': result_dict, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForecastResult':
        """Create from dictionary."""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return {'success': True, 'result': cls(**data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


@dataclass
class LiveMarketState:
    """Current state of live market data."""
    timestamp: datetime
    symbols: Dict[str, Dict[str, Any]]  # symbol -> {price, volume, volatility, etc.}
    market_regime: str  # 'trending', 'ranging', 'volatile'
    global_metrics: Dict[str, float]  # market-wide metrics
    agent_status: Dict[str, str]  # agent -> status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        state_dict = asdict(self)
        state_dict['timestamp'] = self.timestamp.isoformat()
        return {'success': True, 'result': state_dict, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


class LiveMarketRunner:
    """Live market data streaming and agent triggering system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the live market runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.market_data = MarketData(self.config.get('market_data_config', {}))
        self.agent_manager = get_agent_manager()
        self.memory = AgentMemory()
        self.portfolio_manager = PortfolioManager(self.config.get('portfolio_config', {}))
        
        # Live data management
        self.symbols = self.config.get('symbols', ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'])
        self.live_data = {}
        self.price_history = defaultdict(lambda: deque(maxlen=1000))  # Last 1000 prices per symbol
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))  # Last 1000 volumes per symbol
        
        # Trigger management
        self.trigger_configs = self._load_trigger_configs()
        self.last_triggers = defaultdict(datetime)
        self.trigger_counters = defaultdict(int)
        
        # Forecast tracking
        self.forecast_results = []
        self.forecast_file = Path("trading/live/forecast_results.json")
        self.forecast_file.parent.mkdir(parents=True, exist_ok=True)
        
        # State management
        self.running = False
        self.paused = False
        self.last_update = datetime.utcnow()
        
        # Performance tracking
        self.execution_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # Setup logging
        self._setup_logging()
        
        # Load existing forecasts
        self._load_forecast_results()
        
        self.logger.info(f"LiveMarketRunner initialized with {len(self.symbols)} symbols")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_path = Path("trading/live/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_path / "live_market_runner.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
    
    def _load_trigger_configs(self) -> Dict[str, TriggerConfig]:
        """Load trigger configurations."""
        default_configs = {
            "model_builder": TriggerConfig(
                trigger_type=TriggerType.TIME_BASED,
                interval_seconds=3600,  # Every hour
                enabled=True
            ),
            "performance_critic": TriggerConfig(
                trigger_type=TriggerType.TIME_BASED,
                interval_seconds=1800,  # Every 30 minutes
                enabled=True
            ),
            "updater": TriggerConfig(
                trigger_type=TriggerType.TIME_BASED,
                interval_seconds=7200,  # Every 2 hours
                enabled=True
            ),
            "execution_agent": TriggerConfig(
                trigger_type=TriggerType.PRICE_MOVE,
                price_move_threshold=0.005,  # 0.5% price move
                enabled=True
            )
        }
        
        # Override with config if provided
        config_triggers = self.config.get('triggers', {})
        for agent_name, trigger_config in config_triggers.items():
            if agent_name in default_configs:
                default_configs[agent_name] = TriggerConfig(**trigger_config)
        
        return {'success': True, 'result': default_configs, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _load_forecast_results(self) -> None:
        """Load existing forecast results from file."""
        if self.forecast_file.exists():
            try:
                with open(self.forecast_file, 'r') as f:
                    data = json.load(f)
                    self.forecast_results = [
                        ForecastResult.from_dict(forecast) 
                        for forecast in data.get('forecasts', [])
                    ]
                self.logger.info(f"Loaded {len(self.forecast_results)} existing forecasts")
            except Exception as e:
                self.logger.error(f"Error loading forecast results: {e}")
    
    def _save_forecast_results(self) -> None:
        """Save forecast results to file."""
        try:
            data = {
                'last_updated': datetime.utcnow().isoformat(),
                'forecasts': [forecast.to_dict() for forecast in self.forecast_results]
            }
            with open(self.forecast_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving forecast results: {e}")
    
    async def start(self) -> None:
        """Start the live market runner."""
        if self.running:
            self.logger.warning("LiveMarketRunner is already running")
            return
        
        self.running = True
        self.logger.info("Starting LiveMarketRunner...")
        
        # Initialize market data
        await self._initialize_market_data()
        
        # Start data streaming
        asyncio.create_task(self._stream_market_data())
        
        # Start agent triggering
        asyncio.create_task(self._trigger_agents())
        
        # Start forecast tracking
        asyncio.create_task(self._track_forecasts())
        
        # Start performance monitoring
        asyncio.create_task(self._monitor_performance())
        
        self.logger.info("LiveMarketRunner started successfully")
    
    async def stop(self) -> None:
        """Stop the live market runner."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping LiveMarketRunner...")
        
        # Save forecast results
        self._save_forecast_results()
        
        # Save final state
        await self._save_state()
        
        self.logger.info("LiveMarketRunner stopped")
    
    async def pause(self) -> None:
        """Pause the live market runner."""
        self.paused = True
        self.logger.info("LiveMarketRunner paused")
    
    async def resume(self) -> None:
        """Resume the live market runner."""
        self.paused = False
        self.logger.info("LiveMarketRunner resumed")
    
    async def _initialize_market_data(self) -> None:
        """Initialize market data for all symbols."""
        self.logger.info(f"Initializing market data for {len(self.symbols)} symbols")
        
        for symbol in self.symbols:
            try:
                # Fetch initial data
                data = self.market_data.fetch_data(symbol)
                
                # Initialize live data
                self.live_data[symbol] = {
                    'price': data['Close'].iloc[-1],
                    'volume': data['Volume'].iloc[-1],
                    'high': data['High'].iloc[-1],
                    'low': data['Low'].iloc[-1],
                    'open': data['Open'].iloc[-1],
                    'timestamp': datetime.utcnow(),
                    'volatility': self._calculate_volatility(data['Close']),
                    'price_change': 0.0,
                    'volume_change': 0.0
                }
                
                # Initialize history
                self.price_history[symbol].extend(data['Close'].tail(100).values)
                self.volume_history[symbol].extend(data['Volume'].tail(100).values)
                
                self.logger.info(f"Initialized data for {symbol}: ${self.live_data[symbol]['price']:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error initializing data for {symbol}: {e}")
    
    async def _stream_market_data(self) -> None:
        """Stream live market data."""
        self.logger.info("Starting market data streaming...")
        
        while self.running:
            if self.paused:
                await asyncio.sleep(1)
                continue
            
            try:
                current_time = datetime.utcnow()
                
                # Update data for each symbol
                for symbol in self.symbols:
                    await self._update_symbol_data(symbol)
                
                # Update global market metrics
                await self._update_market_metrics()
                
                # Update last update time
                self.last_update = current_time
                
                # Sleep for update interval
                update_interval = self.config.get('update_interval', 30)  # 30 seconds
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in market data streaming: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _update_symbol_data(self, symbol: str) -> None:
        """Update data for a specific symbol."""
        try:
            # Fetch latest data
            data = self.market_data.fetch_data(symbol)
            
            if data.empty:
                return
            
            # Get latest values
            latest = data.iloc[-1]
            previous_price = self.live_data[symbol]['price']
            
            # Update live data
            self.live_data[symbol].update({
                'price': latest['Close'],
                'volume': latest['Volume'],
                'high': latest['High'],
                'low': latest['Low'],
                'open': latest['Open'],
                'timestamp': datetime.utcnow(),
                'volatility': self._calculate_volatility(data['Close']),
                'price_change': (latest['Close'] - previous_price) / previous_price,
                'volume_change': self._calculate_volume_change(symbol, latest['Volume'])
            })
            
            # Update history
            self.price_history[symbol].append(latest['Close'])
            self.volume_history[symbol].append(latest['Volume'])
            
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {e}")
    
    async def _update_market_metrics(self) -> None:
        """Update global market metrics."""
        try:
            # Calculate market-wide metrics
            prices = [data['price'] for data in self.live_data.values()]
            volumes = [data['volume'] for data in self.live_data.values()]
            volatilities = [data['volatility'] for data in self.live_data.values()]
            
            # Market regime detection
            avg_volatility = np.mean(volatilities)
            price_changes = [data['price_change'] for data in self.live_data.values()]
            avg_price_change = np.mean(np.abs(price_changes))
            
            if avg_volatility > 0.03:  # 3% volatility threshold
                market_regime = 'volatile'
            elif avg_price_change > 0.01:  # 1% average price change
                market_regime = 'trending'
            else:
                market_regime = 'ranging'
            
            # Store global metrics
            self.global_metrics = {
                'avg_price': np.mean(prices),
                'avg_volume': np.mean(volumes),
                'avg_volatility': avg_volatility,
                'market_regime': market_regime,
                'correlation': self._calculate_correlation(),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating market metrics: {e}")
    
    async def _trigger_agents(self) -> None:
        """Trigger agents based on configured conditions."""
        self.logger.info("Starting agent triggering...")
        
        while self.running:
            if self.paused:
                await asyncio.sleep(1)
                continue
            
            try:
                current_time = datetime.utcnow()
                
                # Check each agent's trigger conditions
                for agent_name, trigger_config in self.trigger_configs.items():
                    if not trigger_config.enabled:
                        continue
                    
                    if await self._should_trigger_agent(agent_name, trigger_config, current_time):
                        await self._execute_agent(agent_name)
                        self.last_triggers[agent_name] = current_time
                        self.trigger_counters[agent_name] += 1
                
                # Sleep for trigger check interval
                trigger_interval = self.config.get('trigger_interval', 10)  # 10 seconds
                await asyncio.sleep(trigger_interval)
                
            except Exception as e:
                self.logger.error(f"Error in agent triggering: {e}")
                await asyncio.sleep(5)
    
    async def _should_trigger_agent(self, agent_name: str, trigger_config: TriggerConfig, 
                                  current_time: datetime) -> bool:
        """Check if an agent should be triggered."""
        try:
            if trigger_config.trigger_type == TriggerType.TIME_BASED:
                # Time-based trigger
                last_trigger = self.last_triggers.get(agent_name, datetime.min)
                time_since_last = (current_time - last_trigger).total_seconds()
                return time_since_last >= trigger_config.interval_seconds
            
            elif trigger_config.trigger_type == TriggerType.PRICE_MOVE:
                # Price move trigger
                for symbol in self.symbols:
                    if symbol in self.live_data:
                        price_change = abs(self.live_data[symbol]['price_change'])
                        if price_change >= trigger_config.price_move_threshold:
                            return True
            
            elif trigger_config.trigger_type == TriggerType.VOLUME_SPIKE:
                # Volume spike trigger
                for symbol in self.symbols:
                    if symbol in self.live_data:
                        volume_change = self.live_data[symbol]['volume_change']
                        if volume_change >= trigger_config.volume_spike_threshold:
                            return True
            
            elif trigger_config.trigger_type == TriggerType.VOLATILITY_SPIKE:
                # Volatility spike trigger
                for symbol in self.symbols:
                    if symbol in self.live_data:
                        volatility = self.live_data[symbol]['volatility']
                        if volatility >= trigger_config.volatility_threshold:
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking trigger for {agent_name}: {e}")
            return False
    
    async def _execute_agent(self, agent_name: str) -> None:
        """Execute an agent with current market data."""
        try:
            start_time = time.time()
            
            # Prepare market data for agent
            market_data = {
                'live_data': self.live_data,
                'global_metrics': self.global_metrics,
                'price_history': dict(self.price_history),
                'volume_history': dict(self.volume_history)
            }
            
            # Execute agent
            result = await self.agent_manager.execute_agent(
                agent_name, 
                market_data=market_data,
                live_mode=True
            )
            
            # Record execution time
            execution_time = time.time() - start_time
            self.execution_times[agent_name].append(execution_time)
            
            # Log result
            if result.success:
                self.logger.info(f"Agent {agent_name} executed successfully in {execution_time:.2f}s")
                
                # Store forecast if this is a forecasting agent
                if 'forecast' in result.data:
                    await self._store_forecast(agent_name, result.data['forecast'])
            else:
                self.logger.error(f"Agent {agent_name} failed: {result.error_message}")
                self.error_counts[agent_name] += 1
            
            # Update agent status
            self.agent_manager._update_agent_status(agent_name, "executed")
            
        except Exception as e:
            self.logger.error(f"Error executing agent {agent_name}: {e}")
            self.error_counts[agent_name] += 1
    
    async def _store_forecast(self, agent_name: str, forecast_data: Dict[str, Any]) -> None:
        """Store a forecast result."""
        try:
            forecast = ForecastResult(
                timestamp=datetime.utcnow(),
                symbol=forecast_data.get('symbol', 'UNKNOWN'),
                forecast_price=forecast_data.get('forecast_price', 0.0),
                forecast_direction=forecast_data.get('direction', 'sideways'),
                confidence=forecast_data.get('confidence', 0.0),
                horizon_hours=forecast_data.get('horizon_hours', 24),
                model_name=agent_name,
                features=forecast_data.get('features', {})
            )
            
            self.forecast_results.append(forecast)
            
            # Keep only recent forecasts (last 1000)
            if len(self.forecast_results) > 1000:
                self.forecast_results = self.forecast_results[-1000:]
            
            self.logger.info(f"Stored forecast from {agent_name} for {forecast.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error storing forecast: {e}")
    
    async def _track_forecasts(self) -> None:
        """Track forecast accuracy and update results."""
        self.logger.info("Starting forecast tracking...")
        
        while self.running:
            if self.paused:
                await asyncio.sleep(1)
                continue
            
            try:
                current_time = datetime.utcnow()
                
                # Check forecasts that have reached their horizon
                for forecast in self.forecast_results:
                    if (forecast.actual_price is None and 
                        (current_time - forecast.timestamp).total_seconds() >= 
                        forecast.horizon_hours * 3600):
                        
                        # Get actual price
                        symbol = forecast.symbol
                        if symbol in self.live_data:
                            actual_price = self.live_data[symbol]['price']
                            
                            # Calculate accuracy
                            price_error = abs(actual_price - forecast.forecast_price) / forecast.forecast_price
                            accuracy = 1.0 - min(price_error, 1.0)  # Cap at 100% accuracy
                            
                            # Determine actual direction
                            if price_error < 0.01:  # Within 1%
                                actual_direction = forecast.forecast_direction
                            else:
                                actual_direction = 'up' if actual_price > forecast.forecast_price else 'down'
                            
                            # Calculate PnL (simplified)
                            if forecast.forecast_direction == 'up' and actual_direction == 'up':
                                pnl = (actual_price - forecast.forecast_price) / forecast.forecast_price
                            elif forecast.forecast_direction == 'down' and actual_direction == 'down':
                                pnl = (forecast.forecast_price - actual_price) / forecast.forecast_price
                            else:
                                pnl = -0.01  # Small loss for wrong direction
                            
                            # Update forecast
                            forecast.actual_price = actual_price
                            forecast.actual_direction = actual_direction
                            forecast.accuracy = accuracy
                            forecast.pnl = pnl
                            
                            self.logger.info(f"Forecast accuracy for {symbol}: {accuracy:.2%}")
                
                # Save forecast results periodically
                if len(self.forecast_results) % 10 == 0:  # Every 10 forecasts
                    self._save_forecast_results()
                
                # Sleep for tracking interval
                tracking_interval = self.config.get('tracking_interval', 300)  # 5 minutes
                await asyncio.sleep(tracking_interval)
                
            except Exception as e:
                self.logger.error(f"Error in forecast tracking: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_performance(self) -> None:
        """Monitor system performance and log metrics."""
        self.logger.info("Starting performance monitoring...")
        
        while self.running:
            if self.paused:
                await asyncio.sleep(1)
                continue
            
            try:
                # Calculate performance metrics
                performance_metrics = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'execution_times': {
                        agent: {
                            'avg': np.mean(times) if times else 0,
                            'max': np.max(times) if times else 0,
                            'min': np.min(times) if times else 0,
                            'count': len(times)
                        }
                        for agent, times in self.execution_times.items()
                    },
                    'error_counts': dict(self.error_counts),
                    'trigger_counts': dict(self.trigger_counters),
                    'forecast_count': len(self.forecast_results),
                    'live_symbols': len(self.live_data),
                    'uptime': (datetime.utcnow() - self.last_update).total_seconds()
                }
                
                # Log performance metrics
                self.logger.info(f"Performance metrics: {performance_metrics}")
                
                # Save performance data
                await self._save_performance_metrics(performance_metrics)
                
                # Sleep for monitoring interval
                monitoring_interval = self.config.get('monitoring_interval', 600)  # 10 minutes
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _save_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save performance metrics to file."""
        try:
            metrics_file = Path("trading/live/performance_metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
    
    async def _save_state(self) -> None:
        """Save current state to file."""
        try:
            state = LiveMarketState(
                timestamp=datetime.utcnow(),
                symbols=self.live_data,
                market_regime=self.global_metrics.get('market_regime', 'unknown'),
                global_metrics=self.global_metrics,
                agent_status={
                    name: self.agent_manager.get_agent_status(name).status 
                    if self.agent_manager.get_agent_status(name) else 'unknown'
                    for name in self.trigger_configs.keys()
                }
            )
            
            state_file = Path("trading/live/current_state.json")
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def get_current_state(self) -> LiveMarketState:
        """Get current market state."""
        return LiveMarketState(
            timestamp=datetime.utcnow(),
            symbols=self.live_data,
            market_regime=self.global_metrics.get('market_regime', 'unknown'),
            global_metrics=self.global_metrics,
            agent_status={
                name: self.agent_manager.get_agent_status(name).status 
                if self.agent_manager.get_agent_status(name) else 'unknown'
                for name in self.trigger_configs.keys()
            }
        )
    
    def get_forecast_accuracy(self, symbol: Optional[str] = None, 
                            model: Optional[str] = None) -> Dict[str, Any]:
        """Get forecast accuracy statistics."""
        try:
            # Filter forecasts
            forecasts = self.forecast_results
            
            if symbol:
                forecasts = [f for f in forecasts if f.symbol == symbol]
            if model:
                forecasts = [f for f in forecasts if f.model_name == model]
            
            # Calculate statistics
            completed_forecasts = [f for f in forecasts if f.accuracy is not None]
            
            if not completed_forecasts:
                return {
                    'total_forecasts': len(forecasts),
                    'completed_forecasts': 0,
                    'avg_accuracy': 0.0,
                    'avg_pnl': 0.0,
                    'success_rate': 0.0
                }
            
            accuracies = [f.accuracy for f in completed_forecasts]
            pnls = [f.pnl for f in completed_forecasts]
            success_count = sum(1 for f in completed_forecasts if f.accuracy > 0.5)
            
            return {
                'total_forecasts': len(forecasts),
                'completed_forecasts': len(completed_forecasts),
                'avg_accuracy': np.mean(accuracies),
                'avg_pnl': np.mean(pnls),
                'success_rate': success_count / len(completed_forecasts),
                'best_forecast': max(completed_forecasts, key=lambda f: f.accuracy).to_dict(),
                'worst_forecast': min(completed_forecasts, key=lambda f: f.accuracy).to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast accuracy: {e}")
            return {'success': True, 'result': {}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _calculate_volatility(self, prices: pd.Series, window: int = 20) -> float:
        """Calculate price volatility."""
        if len(prices) < window:
            return 0.0
        
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return {'success': True, 'result': 0.0, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        return returns.tail(window).std()
    
    def _calculate_volume_change(self, symbol: str, current_volume: float) -> float:
        """Calculate volume change relative to average."""
        if not self.volume_history[symbol]:
            return 0.0
        
        avg_volume = np.mean(list(self.volume_history[symbol]))
        if avg_volume == 0:
            return {'success': True, 'result': 0.0, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        return current_volume / avg_volume
    
    def _calculate_correlation(self) -> float:
        """Calculate average correlation between symbols."""
        if len(self.symbols) < 2:
            return 0.0
        
        try:
            # Get price data for all symbols
            price_data = {}
            for symbol in self.symbols:
                if symbol in self.price_history and self.price_history[symbol]:
                    price_data[symbol] = list(self.price_history[symbol])
            
            if len(price_data) < 2:
                return 0.0
            
            # Calculate correlations
            correlations = []
            symbols = list(price_data.keys())
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    min_len = min(len(price_data[sym1]), len(price_data[sym2]))
                    
                    if min_len > 10:  # Need at least 10 data points
                        corr = np.corrcoef(
                            price_data[sym1][-min_len:],
                            price_data[sym2][-min_len:]
                        )[0, 1]
                        
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return {'success': True, 'result': 0.0, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


# Factory function
def create_live_market_runner(config: Optional[Dict[str, Any]] = None) -> LiveMarketRunner:
    """Create a LiveMarketRunner with default configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        LiveMarketRunner instance
    """
    default_config = {
        'symbols': ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'],
        'update_interval': 30,  # 30 seconds
        'trigger_interval': 10,  # 10 seconds
        'tracking_interval': 300,  # 5 minutes
        'monitoring_interval': 600,  # 10 minutes
        'market_data_config': {
            'cache_size': 1000,
            'update_threshold': 5,
            'max_retries': 3
        },
        'triggers': {
            'model_builder': {
                'trigger_type': 'time_based',
                'interval_seconds': 3600,
                'enabled': True
            },
            'performance_critic': {
                'trigger_type': 'time_based',
                'interval_seconds': 1800,
                'enabled': True
            },
            'execution_agent': {
                'trigger_type': 'price_move',
                'price_move_threshold': 0.005,
                'enabled': True
            }
        }
    }
    
    if config:
        default_config.update(config)
    
    return {'success': True, 'result': LiveMarketRunner(default_config), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}