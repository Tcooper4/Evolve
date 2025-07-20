ester - Hypothesis Testing Module

This module tests trading hypotheses across multiple tickers, timeframes, and market regimes.
It evaluates performance using Sharpe ratio, drawdown, win rate, and other metrics.
"

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult, AgentState
from trading.utils.error_handling import log_errors, retry_on_error
from trading.exceptions import StrategyError, ModelError
from trading.agents.alpha.alphagen_agent import Hypothesis

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
lt of a hypothesis test."""
    
    hypothesis_id: str
    ticker: str
    timeframe: str
    regime: str
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    volatility: float
    var_95: float  # 95 Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk
    
    # Additional metrics
    beta: float
    alpha: float
    information_ratio: float
    
    # Test metadata
    execution_time: float
    data_quality_score: float
    confidence_score: float
    
    # Raw data
    equity_curve: pd.Series
    trade_log: List[Dict[str, Any]]
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]     hypothesis_id:self.hypothesis_id,
          ticker": self.ticker,
           timeframe": self.timeframe,
            regime": self.regime,
            start_date:self.start_date.isoformat(),
          end_date": self.end_date.isoformat(),
         total_return": self.total_return,
         sharpe_ratio": self.sharpe_ratio,
         max_drawdown": self.max_drawdown,
            win_rate: self.win_rate,
          profit_factor": self.profit_factor,
         calmar_ratio": self.calmar_ratio,
          sortino_ratio": self.sortino_ratio,
         total_trades": self.total_trades,
           winning_trades": self.winning_trades,
          losing_trades": self.losing_trades,
          avg_win": self.avg_win,
           avg_loss: self.avg_loss,
        largest_win": self.largest_win,
         largest_loss": self.largest_loss,
            volatility: self.volatility,
            var_95ar_95
           cvar_95ar_95,
           beta": self.beta,
        alpha": self.alpha,
            information_ratio": self.information_ratio,
           execution_time": self.execution_time,
       data_quality_score": self.data_quality_score,
           confidence_score: self.confidence_score,
            created_at:self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestResult":
    te from dictionary."""
        # Convert datetime strings back to datetime objects
        for field in [start_date", "end_date, _at]:     if isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


@dataclass
class TestConfig:
  Configuration for hypothesis testing." # Test parameters
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000  transaction_cost: float =0.001  # 0.1er trade
    slippage: float = 0.05  # 0.5 slippage
    
    # Asset universe
    tickers: List[str] = field(default_factory=lambda:         SPY,QQQ, IWM, EFA, EEM, TLT, GLD, USO", VNQ", "XLE"
    ])
    timeframes: List[str] = field(default_factory=lambda: 
       1h,4h", 1d1w"
    ])
    
    # Risk parameters
    max_position_size: float = 0.1  #10% max position
    stop_loss_pct: float = 0.02# 2% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5   max_drawdown_threshold: float = 0.15#15% max drawdown
    min_win_rate: float = 0.4 min_profit_factor: float = 10.2  # Test execution
    parallel_tests: int = 4
    timeout_seconds: int = 30  
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return {
            start_date:self.start_date.isoformat(),
          end_date": self.end_date.isoformat(),
            initial_capital": self.initial_capital,
           transaction_cost": self.transaction_cost,
            slippage: self.slippage,
           tickers": self.tickers,
            timeframes: self.timeframes,
            max_position_size": self.max_position_size,
          stop_loss_pct": self.stop_loss_pct,
            take_profit_pct": self.take_profit_pct,
           min_sharpe_ratio: self.min_sharpe_ratio,
          max_drawdown_threshold": self.max_drawdown_threshold,
         min_win_rate": self.min_win_rate,
            min_profit_factor: self.min_profit_factor,
           parallel_tests: self.parallel_tests,
            timeout_seconds": self.timeout_seconds
        }


class SignalTester(BaseAgent):that tests trading hypotheses across multiple dimensions."""
    
    __version__ = "100    __author__ = SignalTester Team"
    __description__ = "Tests trading hypotheses across tickers, timeframes, and regimes"
    __tags__ = ["alpha", "testing", "backtesting,performance"]
    __capabilities__ = ["hypothesis_testing", "performance_analysis", "risk_metrics"]
    __dependencies__ =pandas", numpy",yfinance]   
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.test_config = None
        self.test_results = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _setup(self) -> None:
        tup the agent."""
        try:
            # Load test configuration
            test_config_data = self.config.custom_config.get("test_config, {})
            self.test_config = TestConfig(**test_config_data)
            
            logger.info("SignalTester agent setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup SignalTester agent: {e}")
            raise
    
    @log_errors()
    async def execute(self, **kwargs) -> AgentResult:
Execute hypothesis testing."""
        try:
            self.status.state = AgentState.RUNNING
            start_time = datetime.now()
            
            # Get hypotheses to test
            hypotheses = kwargs.get(hypotheses,            if not hypotheses:
                return AgentResult(
                    success=False,
                    error_message="No hypotheses provided for testing",
                    metadata={agent":signal_tester}                )
            
            # Test each hypothesis
            all_results =     for hypothesis in hypotheses:
                results = await self._test_hypothesis(hypothesis)
                all_results.extend(results)
            
            # Rank results
            ranked_results = self._rank_test_results(all_results)
            
            # Store results
            self.test_results.extend(ranked_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.status.state = AgentState.SUCCESS
            
            return AgentResult(
                success=True,
                data={
                    test_results": [r.to_dict() for r in ranked_results],
                   summary": self._generate_test_summary(ranked_results),
                   execution_time: execution_time
                },
                execution_time=execution_time,
                metadata={agent":signal_tester"}
            )
            
        except Exception as e:
            self.status.state = AgentState.ERROR
            return self.handle_error(e)
    
    async def _test_hypothesis(self, hypothesis: Hypothesis) -> List[TestResult]:
       single hypothesis across multiple dimensions."""
        try:
            results = []
            
            # Test across different tickers
            for ticker in self.test_config.tickers:
                # Test across different timeframes
                for timeframe in self.test_config.timeframes:
                    # Test across different market regimes
                    for regime in ["bull",bear", "sideways"]:
                        try:
                            result = await self._run_single_test(
                                hypothesis, ticker, timeframe, regime
                            )
                            if result:
                                results.append(result)
                        except Exception as e:
                            logger.error(fTest failed for {ticker}/{timeframe}/{regime}: {e}")
                            continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to test hypothesis [object Object]hypothesis.id}: {e}")
            return    
    async def _run_single_test(
        self, 
        hypothesis: Hypothesis, 
        ticker: str, 
        timeframe: str, 
        regime: str
    ) -> Optional[TestResult]:
     Run a single test for a hypothesis."""
        try:
            test_start = datetime.now()
            
            # Get market data
            data = await self._get_market_data(ticker, timeframe, regime)
            if data is None or data.empty:
                return None
            
            # Implement strategy based on hypothesis
            signals = self._generate_signals(data, hypothesis)
            
            # Run backtest
            equity_curve, trade_log = self._run_backtest(data, signals, hypothesis)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(equity_curve, trade_log)
            
            # Create test result
            result = TestResult(
                hypothesis_id=hypothesis.id,
                ticker=ticker,
                timeframe=timeframe,
                regime=regime,
                start_date=data.index[0],
                end_date=data.index[-1],
                equity_curve=equity_curve,
                trade_log=trade_log,
                execution_time=(datetime.now() - test_start).total_seconds(),
                data_quality_score=self._calculate_data_quality(data),
                confidence_score=hypothesis.confidence_score,
                **metrics
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Single test failed: {e}")
            return None
    
    def _generate_signals(self, data: pd.DataFrame, hypothesis: Hypothesis) -> pd.Series:
 te trading signals based on hypothesis."""
        try:
            signals = pd.Series(0, index=data.index)
            
            # Implement entry conditions
            entry_mask = self._evaluate_conditions(data, hypothesis.entry_conditions)
            signals[entry_mask] = 1
            
            # Implement exit conditions
            exit_mask = self._evaluate_conditions(data, hypothesis.exit_conditions)
            signals[exit_mask] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return pd.Series(0, index=data.index)
    
    def _evaluate_conditions(self, data: pd.DataFrame, conditions: List[str]) -> pd.Series:
        valuate entry/exit conditions."""
        try:
            mask = pd.Series(True, index=data.index)
            
            for condition in conditions:
                condition_mask = self._evaluate_single_condition(data, condition)
                mask = mask & condition_mask
            
            return mask
            
        except Exception as e:
            logger.error(f"Failed to evaluate conditions: {e}")
            return pd.Series(False, index=data.index)
    
    def _evaluate_single_condition(self, data: pd.DataFrame, condition: str) -> pd.Series:
   Evaluatea single condition."""
        try:
            # Simple condition parser - can be enhanced
            if RSI" in condition:
                if >                   threshold = float(condition.split(">")[1].strip())
                    return data.get("RSI", pd.Series(50, index=data.index)) > threshold
                elif <                   threshold = float(condition.split("<")[1].strip())
                    return data.get("RSI", pd.Series(50, index=data.index)) < threshold
            
            elif "moving average" in condition.lower():
                if "above" in condition.lower():
                    return data.get(Close", pd.Series(0, index=data.index)) > data.get(MA20", pd.Series(0, index=data.index))
                elif "below" in condition.lower():
                    return data.get(Close", pd.Series(0, index=data.index)) < data.get(MA20", pd.Series(0, index=data.index))
            
            # Default tofalseunknown conditions
            return pd.Series(False, index=data.index)
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition{condition}': {e}")
            return pd.Series(False, index=data.index)
    
    def _run_backtest(
        self, 
        data: pd.DataFrame, 
        signals: pd.Series, 
        hypothesis: Hypothesis
    ) -> Tuple[pd.Series, List[Dict[str, Any]]]:
     acktest simulation."""
        try:
            initial_capital = self.test_config.initial_capital
            current_capital = initial_capital
            position = 0
            equity_curve = []
            trade_log = []
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                signal = signals.iloc[i]
                price = row.get("Close", 0)
                
                # Execute trades based on signals
                if signal == 1 and position ==0                   position_size = current_capital * hypothesis.risk_parameters.get(position_size", 0.1)
                    shares = position_size / price
                    position = shares
                    current_capital -= position_size
                    
                    trade_log.append({
                  timestamp": timestamp,
                       action                   shares                 price                   capital: current_capital
                    })
                
                elif signal == -1 and position > 0                   position_value = position * price
                    current_capital += position_value
                    
                    trade_log.append({
                  timestamp": timestamp,
                        action                   shares": position,
                      price                   capital: current_capital
                    })
                    
                    position = 0
                
                # Calculate current equity
                current_equity = current_capital + (position * price)
                equity_curve.append(current_equity)
            
            # Close any remaining position
            if position > 0             final_price = data.iloc-1].get("Close", 0          position_value = position * final_price
                current_capital += position_value
                equity_curve[-1] = current_capital
            
            return pd.Series(equity_curve, index=data.index), trade_log
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return pd.Series([initial_capital] * len(data), index=data.index),    def _calculate_performance_metrics(
        self, 
        equity_curve: pd.Series, 
        trade_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
  late comprehensive performance metrics."""
        try:
            # Basic returns
            returns = equity_curve.pct_change().dropna()
            total_return = (equity_curve.iloc[-1 / equity_curve.iloc[0]) - 1
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252alized
            sharpe_ratio = (returns.mean() *252lity if volatility > 0 else 0
            
            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Trade statistics
            total_trades = len([t for t in trade_log if taction"] in ["BUY", "SELL"]])
            if total_trades == 0            return {
                 total_return": total_return,
                   sharpe_ratio": 0,
                   max_drawdown": 0,
                 win_rate                   profit_factor": 0,
                   calmar_ratio": 0,
                    sortino_ratio": 0,
                   total_trades": 0,
                    winning_trades": 0,
                    losing_trades": 0,
                avg_win                avg_loss                   largest_win                  largest_loss": 0,
               volatility": volatility,
                    var_95: returns.quantile(0.05),
                    cvar_95:returns[returns <= returns.quantile(0.05)].mean(),
             beta             alpha                   information_ratio":0                }
            
            # Calculate trade P&L
            trade_pnls = 
            for i in range(0, len(trade_log) - 1, 2):
                if i +1< len(trade_log):
                    buy_trade = trade_log[i]
                    sell_trade = trade_log[i + 1]
                    if buy_tradeaction"] == "BUY andsell_trade[action"] == "SELL":
                        pnl = (sell_trade["price"] - buy_trade["price"]) * buy_trade["shares"]
                        trade_pnls.append(pnl)
            
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0           avg_win = np.mean(winning_trades) if winning_trades else0          avg_loss = np.mean(losing_trades) if losing_trades else 0
            largest_win = max(winning_trades) if winning_trades else 0
            largest_loss = min(losing_trades) if losing_trades else 0
            
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) !=0else float('inf')
            
            # Additional metrics
            calmar_ratio = (total_return * 252/ abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0           sortino_ratio = (returns.mean() * 252downside_deviation if downside_deviation > 0 else 0
            
            return[object Object]
             total_return": total_return,
             sharpe_ratio": sharpe_ratio,
             max_drawdown": max_drawdown,
             win_rate": win_rate,
              profit_factor: profit_factor,
             calmar_ratio": calmar_ratio,
              sortino_ratio: sortino_ratio,
             total_trades": total_trades,
               winning_trades": len(winning_trades),
              losing_trades": len(losing_trades),
                avg_win": avg_win,
             avg_loss": avg_loss,
            largest_win": largest_win,
             largest_loss": largest_loss,
           volatility": volatility,
                var_95: returns.quantile(0.05),
                cvar_95:returns[returns <= returns.quantile(0.05)].mean(),
           beta": 0,  # Would need market data to calculate
            alpha": 0,  # Would need market data to calculate
                information_ratio: 0  # Would need benchmark to calculate
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return[object Object]    
    def _rank_test_results(self, results: List[TestResult]) -> List[TestResult]:
  Rank test results by composite score."""
        try:
            for result in results:
                score = self._calculate_composite_score(result)
                result.confidence_score = score
            
            # Sort by composite score (descending)
            ranked_results = sorted(
                results,
                key=lambda r: r.confidence_score,
                reverse=True
            )
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Failed to rank test results: {e}")
            return results
    
    def _calculate_composite_score(self, result: TestResult) -> float:
  alculate composite score for ranking."""
        try:
            score = 0
            
            # Sharpe ratio (30% weight)
            sharpe_score = min(result.sharpe_ratio / 2010)  # Normalize to 0-1
            score += sharpe_score *0.3      
            # Win rate (20% weight)
            score += result.win_rate *0.2      
            # Profit factor (15% weight)
            profit_factor_score = min(result.profit_factor / 3.0, 1.0)
            score += profit_factor_score * 0.15      
            # Drawdown penalty (15% weight)
            drawdown_penalty = max(0esult.max_drawdown / 00.2)  # Penalty for >20% drawdown
            score += (1 - drawdown_penalty) * 0.15      
            # Calmar ratio (10% weight)
            calmar_score = min(result.calmar_ratio / 2.0, 1.0)
            score += calmar_score *0.1      
            # Data quality (10% weight)
            score += result.data_quality_score *0.1      
            return max(0 min(score,1# Ensure 0-1 range
            
        except Exception as e:
            logger.error(f"Failed to calculate composite score: {e}")
            return 0.5
    def _generate_test_summary(self, results: List[TestResult]) -> Dict[str, Any]:
 enerate summary of test results."""
        try:
            if not results:
                return {"error": "No test results available"}
            
            # Aggregate metrics
            avg_sharpe = np.mean([r.sharpe_ratio for r in results])
            avg_win_rate = np.mean([r.win_rate for r in results])
            avg_drawdown = np.mean([r.max_drawdown for r in results])
            avg_return = np.mean([r.total_return for r in results])
            
            # Best performing tests
            best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
            best_win_rate = max(results, key=lambda r: r.win_rate)
            best_return = max(results, key=lambda r: r.total_return)
            
            return[object Object]
                total_tests": len(results),
               avg_sharpe_ratio": avg_sharpe,
             avg_win_rate": avg_win_rate,
               avg_max_drawdown": avg_drawdown,
               avg_total_return": avg_return,
               best_sharpe_test": {
                   ticker": best_sharpe.ticker,
                   timeframe": best_sharpe.timeframe,
                 sharpe_ratio": best_sharpe.sharpe_ratio
                },
            best_win_rate_test": {
                    ticker: best_win_rate.ticker,
                    timeframe": best_win_rate.timeframe,
                   win_rate: best_win_rate.win_rate
                },
               best_return_test": {
                ticker": best_return.ticker,
                   timeframe": best_return.timeframe,
                 total_return": best_return.total_return
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate test summary: {e}")
            return {"error": str(e)}
    
    async def _get_market_data(self, ticker: str, timeframe: str, regime: str) -> Optional[pd.DataFrame]:
        et market data for testing."""
        try:
            # Placeholder - implement actual data fetching
            # This would typically fetch from yfinance, Alpha Vantage, etc.
            
            # Generate synthetic data for demonstration
            dates = pd.date_range(
                start=self.test_config.start_date,
                end=self.test_config.end_date,
                freq="1   )
            
            np.random.seed(hash(ticker + timeframe + regime) % 2**32)
            
            # Generate price data
            returns = np.random.normal(0050.02ates))
            prices = 100 (1 + returns).cumprod()
            
            # Generate technical indicators
            data = pd.DataFrame({
                Open": prices * (1 + np.random.normal(0, 05),
                High": prices * (1 + np.abs(np.random.normal(001),
                Low": prices * (1 - np.abs(np.random.normal(001),
               Closes,
          Volume": np.random.lognormal(10ates))
            }, index=dates)
            
            # Add technical indicators
            data["MA20] = data["Close"].rolling(20).mean()
            data["RSI"] = self._calculate_rsi(data["Close"])
            
            return data.dropna()
            
        except Exception as e:
            logger.error(f"Failed to get market data for {ticker}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14pd.Series:
      late RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi =100 (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return pd.Series(50, index=prices.index)
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
       ulate data quality score."""
        try:
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape0 * data.shape1      
            # Check for outliers
            outlier_ratio =0           for col in ["Open", "High,Low]:
                if col in data.columns:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((data[col] < (q1 - 1.5 iqr)) | (data[col] > (q3 + 10.5                   outlier_ratio += outliers / len(data)
            
            outlier_ratio /= 4  # Average across columns
            
            # Calculate quality score
            quality_score = 1- (missing_ratio + outlier_ratio) / 2
            return max(0, min(quality_score,1          
        except Exception as e:
            logger.error(f"Failed to calculate data quality: {e}")
            return 0.5
    
    def validate_input(self, **kwargs) -> bool:
       e input parameters.""   required_params = ["hypotheses"]
        return all(param in kwargs for param in required_params)
    
    def validate_config(self) -> bool:
       gent configuration.""   required_config = ["test_config"]
        custom_config = self.config.custom_config or[object Object]        return all(key in custom_config for key in required_config)
    
    def handle_error(self, error: Exception) -> AgentResult:
      ndle errors during execution."""
        self.status.state = AgentState.ERROR
        self.status.current_error = str(error)
        
        return AgentResult(
            success=false
            error_message=str(error),
            error_type=type(error).__name__,
            metadata={agent":signal_tester}        )
    
    def get_capabilities(self) -> List[str]:
  agent capabilities.       return self.__capabilities__
    
    def get_requirements(self) -> Dict[str, Any]:
  agent requirements.
        return {
         dependencies": self.__dependencies__,
            data_sources: ["market_data", "technical_indicators"],
          config": ["test_config]        }
    
    def get_test_results(self) -> List[TestResult]:
     t all test results.       return self.test_results.copy()
    
    def clear_results(self) -> None:
     Clear stored test results."""
        self.test_results.clear()
