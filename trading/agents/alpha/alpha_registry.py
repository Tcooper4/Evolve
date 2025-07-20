AlphaRegistry - Strategy Lifecycle Management

This module tracks the lifecycle of alpha strategies and detects alpha decay.
It maintains a registry of strategies with performance history and decay detection.


import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult, AgentState
from trading.utils.error_handling import log_errors, retry_on_error
from trading.exceptions import StrategyError, ModelError
from trading.agents.alpha.alphagen_agent import Hypothesis
from trading.agents.alpha.signal_tester import TestResult
from trading.agents.alpha.risk_validator import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecord:
   strategy record in the registry."""
    
    strategy_id: str
    hypothesis_id: str
    name: str
    description: str
    strategy_type: str
    asset_class: str
    timeframe: str
    
    # Lifecycle information
    created_at: datetime
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    status: str = "created"  # created, testing, validated, deployed, retired, decayed
    
    # Performance history
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    current_performance: Optional[Dict[str, Any]] = None
    
    # Decay metrics
    decay_score: float = 0.0
    decay_factors: List[str] = field(default_factory=list)
    last_decay_check: Optional[datetime] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]     strategy_id:self.strategy_id,
           hypothesis_id:self.hypothesis_id,
           name": self.name,
           description": self.description,
           strategy_type": self.strategy_type,
           asset_class": self.asset_class,
           timeframe": self.timeframe,
           created_at:self.created_at.isoformat(),
           deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
           retired_at:self.retired_at.isoformat() if self.retired_at else None,
           status": self.status,
           performance_history": self.performance_history,
           current_performance": self.current_performance,
           decay_score": self.decay_score,
           decay_factors": self.decay_factors,
           last_decay_check: self.last_decay_check.isoformat() if self.last_decay_check else None,
           tags": self.tags,
           metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyRecord":
    te from dictionary."""
        # Convert datetime strings back to datetime objects
        for field in ["created_at", "deployed_at, retired_at",last_decay_check]:           if data.get(field) and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


@dataclass
class DecayAnalysis:
   of alpha decay analysis."""
    
    strategy_id: str
    analysis_date: datetime
    
    # Decay metrics
    overall_decay_score: float  # 0igher = more decay
    performance_decay: float
    correlation_decay: float
    market_regime_decay: float
    competition_decay: float
    
    # Decay factors
    decay_factors: List[str]
    decay_severity: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]
    action_required: str  # none, monitor, adjust, retire
    
    # Historical context
    decay_trend: str  # improving, stable, worsening
    time_to_decay: Optional[int] = None  # days until significant decay
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]     strategy_id:self.strategy_id,
           analysis_date": self.analysis_date.isoformat(),
           overall_decay_score": self.overall_decay_score,
           performance_decay": self.performance_decay,
           correlation_decay": self.correlation_decay,
           market_regime_decay:self.market_regime_decay,
           competition_decay": self.competition_decay,
           decay_factors": self.decay_factors,
           decay_severity": self.decay_severity,
           recommendations": self.recommendations,
           action_required": self.action_required,
           decay_trend": self.decay_trend,
           time_to_decay": self.time_to_decay,
           created_at:self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecayAnalysis":
    te from dictionary."""
        # Convert datetime strings back to datetime objects
        for field in ["analysis_date", "_at]:           if data.get(field) and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class AlphaRegistry(BaseAgent):ent that manages strategy lifecycle and detects alpha decay."""
    
    __version__ = 10    __author__ = "AlphaRegistry Team"
    __description__ =Managesstrategy lifecycle and detects alpha decay"
    __tags__ = [alpha",registry",lifecycle", "decay_detection"]
    __capabilities__ = ["strategy_registration", "lifecycle_management", "decay_detection"]
    __dependencies__ =pandas",numpy",sqlite3"]   
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.db_path = None
        self.strategies =[object Object]      self.decay_analyses = []
        self.connection = None
        
    def _setup(self) -> None:
        tup the agent."""
        try:
            # Initialize database
            db_config = self.config.custom_config.get("database, {})
            self.db_path = db_config.get(path,alpha_registry.db")
            
            # Create database connection
            self.connection = sqlite3.connect(self.db_path)
            self._create_tables()
            
            # Load existing strategies
            self._load_strategies()
            
            logger.info("AlphaRegistry agent setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup AlphaRegistry agent: {e}")
            raise
    
    def _create_tables(self) -> None:
     database tables."""
        try:
            cursor = self.connection.cursor()
            
            # Strategies table
            cursor.execute("            CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT,
                    name TEXT,
                    description TEXT,
                    strategy_type TEXT,
                    asset_class TEXT,
                    timeframe TEXT,
                    created_at TEXT,
                    deployed_at TEXT,
                    retired_at TEXT,
                    status TEXT,
                    performance_history TEXT,
                    current_performance TEXT,
                    decay_score REAL,
                    decay_factors TEXT,
                    last_decay_check TEXT,
                    tags TEXT,
                    metadata TEXT
                )
                  
            # Decay analyses table
            cursor.execute("            CREATE TABLE IF NOT EXISTS decay_analyses (
                    analysis_id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    analysis_date TEXT,
                    overall_decay_score REAL,
                    performance_decay REAL,
                    correlation_decay REAL,
                    market_regime_decay REAL,
                    competition_decay REAL,
                    decay_factors TEXT,
                    decay_severity TEXT,
                    recommendations TEXT,
                    action_required TEXT,
                    decay_trend TEXT,
                    time_to_decay INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
                )
                  
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def _load_strategies(self) -> None:
     existing strategies from database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM strategies")
            rows = cursor.fetchall()
            
            for row in rows:
                strategy_data = {
                    strategy_id": row[0],
                  hypothesis_id": row[1],
                  name                   description": row[3],
                  strategy_type": row[4],
                    asset_class": row[5],
                    timeframe": row[6],
                    created_at": row[7],
                    deployed_at": row[8],
                    retired_at": row[9],
                  status": row[10],
              performance_history: json.loads(row[11]) if row[11] else [],
              current_performance: json.loads(row[12]) if row[12] else None,
                    decay_score": row[13],
                  decay_factors: json.loads(row[14]) if row[14] else [],
                   last_decay_check": row[15],
                    tags: json.loads(row[16]) if row[16] else [],
                  metadata: json.loads(row[17]) if row[17] else [object Object]                }
                
                strategy = StrategyRecord.from_dict(strategy_data)
                self.strategies[strategy.strategy_id] = strategy
            
            logger.info(f"Loaded {len(self.strategies)} strategies from database")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
    
    @log_errors()
    async def execute(self, **kwargs) -> AgentResult:
     ute registry operations."""
        try:
            self.status.state = AgentState.RUNNING
            start_time = datetime.now()
            
            operation = kwargs.get("operation", "update")
            
            if operation == "register:            result = await self._register_strategy(kwargs)
            elif operation == "update:            result = await self._update_strategies(kwargs)
            elif operation == decay_analysis:            result = await self._perform_decay_analysis(kwargs)
            elif operation ==lifecycle_update:            result = await self._update_lifecycle(kwargs)
            else:
                return AgentResult(
                    success=False,
                    error_message=fUnknown operation: {operation}",
                    metadata={"agent:alpha_registry}                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.status.state = AgentState.SUCCESS
            
            return AgentResult(
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={"agent:alpha_registry"}
            )
            
        except Exception as e:
            self.status.state = AgentState.ERROR
            return self.handle_error(e)
    
    async def _register_strategy(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
     a new strategy to the registry."""
        try:
            hypothesis = kwargs.get("hypothesis")
            test_result = kwargs.get("test_result")
            validation_result = kwargs.get(validation_result")
            
            if not all([hypothesis, test_result, validation_result]):
                raise ValueError("Missing required data for strategy registration")
            
            # Create strategy record
            strategy_id = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S)}_{hypothesis.id}"
            
            strategy = StrategyRecord(
                strategy_id=strategy_id,
                hypothesis_id=hypothesis.id,
                name=hypothesis.title,
                description=hypothesis.description,
                strategy_type=hypothesis.strategy_type,
                asset_class=hypothesis.asset_class,
                timeframe=hypothesis.timeframe,
                created_at=datetime.now(),
                status="created,              tags=hypothesis.tags,
                metadata={
                   entry_conditions": hypothesis.entry_conditions,
                    exit_conditions: hypothesis.exit_conditions,
                    risk_parameters: hypothesis.risk_parameters,
                   confidence_score: hypothesis.confidence_score
                }
            )
            
            # Add initial performance data
            if test_result:
                strategy.current_performance = {
                 sharpe_ratio": test_result.sharpe_ratio,
                 total_return": test_result.total_return,
                 max_drawdown: test_result.max_drawdown,
                  win_rate: test_result.win_rate,
                  profit_factor:test_result.profit_factor,
                 last_updated:datetime.now().isoformat()
                }
                
                strategy.performance_history.append({
                  date:datetime.now().isoformat(),
                performance": strategy.current_performance.copy()
                })
            
            # Store in database
            self._save_strategy(strategy)
            
            # Add to memory
            self.strategies[strategy_id] = strategy
            
            return[object Object]
            strategy_id": strategy_id,
         status": "registered,
         strategy": strategy.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to register strategy: {e}")
            raise
    
    async def _update_strategies(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
     strategies with new performance data."""
        try:
            updates = kwargs.get("updates, [])           updated_count = 0
            
            for update in updates:
                strategy_id = update.get("strategy_id)       performance_data = update.get("performance")
                
                if strategy_id in self.strategies and performance_data:
                    strategy = self.strategies[strategy_id]
                    
                    # Update current performance
                    strategy.current_performance = performance_data
                    strategy.current_performance["last_updated"] = datetime.now().isoformat()
                    
                    # Add to performance history
                    strategy.performance_history.append({
                      date:datetime.now().isoformat(),
                    performance": performance_data.copy()
                    })
                    
                    # Keep only last 100 performance records
                    if len(strategy.performance_history) > 100:
                        strategy.performance_history = strategy.performance_history[-100:]
                    
                    # Update in database
                    self._save_strategy(strategy)
                    updated_count += 1
            
            return[object Object]
              updated_count: updated_count,
               total_strategies": len(self.strategies)
            }
            
        except Exception as e:
            logger.error(f"Failed to update strategies: {e}")
            raise
    
    async def _perform_decay_analysis(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
     decay analysis on all strategies."""
        try:
            strategy_ids = kwargs.get("strategy_ids")
            if not strategy_ids:
                strategy_ids = list(self.strategies.keys())
            
            decay_analyses = []
            
            for strategy_id in strategy_ids:
                if strategy_id in self.strategies:
                    analysis = await self._analyze_strategy_decay(strategy_id)
                    if analysis:
                        decay_analyses.append(analysis)
                        self.decay_analyses.append(analysis)
            
            return[object Object]
           analyses_performed": len(decay_analyses),
               decay_analyses": [a.to_dict() for a in decay_analyses]
            }
            
        except Exception as e:
            logger.error(fFailed to perform decay analysis: {e}")
            raise
    
    async def _analyze_strategy_decay(self, strategy_id: str) -> Optional[DecayAnalysis]:
     decay analysis for a single strategy."""
        try:
            strategy = self.strategies.get(strategy_id)
            if not strategy or not strategy.performance_history:
                return None
            
            # Calculate decay metrics
            performance_decay = self._calculate_performance_decay(strategy)
            correlation_decay = self._calculate_correlation_decay(strategy)
            market_regime_decay = self._calculate_market_regime_decay(strategy)
            competition_decay = self._calculate_competition_decay(strategy)
            
            # Calculate overall decay score
            weights =[object Object]
                performance": 0.4
                correlation": 0.2
                market_regime": 0.2
                competition:0.2   }
            
            overall_decay_score = (
                performance_decay * weightsperformance +
                correlation_decay * weightscorrelation +
                market_regime_decay * weights["market_regime"] +
                competition_decay * weights["competition"]
            )
            
            # Identify decay factors
            decay_factors = []
            decay_severity = {}
            
            if performance_decay > 0.3             decay_factors.append("performance_degradation)             decay_severity["performance"] = performance_decay
            
            if correlation_decay > 0.5             decay_factors.append("increased_correlation)             decay_severity["correlation"] = correlation_decay
            
            if market_regime_decay > 0.4             decay_factors.append("market_regime_change)             decay_severity[market_regime] = market_regime_decay
            
            if competition_decay > 0.3             decay_factors.append("increased_competition)             decay_severity["competition"] = competition_decay
            
            # Generate recommendations
            recommendations = self._generate_decay_recommendations(overall_decay_score, decay_factors)
            
            # Determine action required
            action_required = self._determine_action_required(overall_decay_score, decay_factors)
            
            # Analyze decay trend
            decay_trend = self._analyze_decay_trend(strategy)
            
            # Estimate time to significant decay
            time_to_decay = self._estimate_time_to_decay(strategy, overall_decay_score)
            
            # Update strategy decay score
            strategy.decay_score = overall_decay_score
            strategy.decay_factors = decay_factors
            strategy.last_decay_check = datetime.now()
            
            # Save updated strategy
            self._save_strategy(strategy)
            
            analysis = DecayAnalysis(
                strategy_id=strategy_id,
                analysis_date=datetime.now(),
                overall_decay_score=overall_decay_score,
                performance_decay=performance_decay,
                correlation_decay=correlation_decay,
                market_regime_decay=market_regime_decay,
                competition_decay=competition_decay,
                decay_factors=decay_factors,
                decay_severity=decay_severity,
                recommendations=recommendations,
                action_required=action_required,
                decay_trend=decay_trend,
                time_to_decay=time_to_decay
            )
            
            # Save analysis to database
            self._save_decay_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(fFailed to analyze decay for strategy {strategy_id}: {e}")
            return None
    
    def _calculate_performance_decay(self, strategy: StrategyRecord) -> float:
     late performance decay score."""
        try:
            if len(strategy.performance_history) < 2            return0      
            # Get recent performance vs historical
            recent_performance = strategy.performance_history[-1ance"]
            historical_performance = strategy.performance_history[0ce"]
            
            # Calculate decay in key metrics
            sharpe_decay = max(0orical_performance.get(sharpe_ratio", 0) - 
                                  recent_performance.get("sharpe_ratio", 0)) / 
                                 max(historical_performance.get("sharpe_ratio", 1),1      
            return_decay = max(0orical_performance.get(total_return", 0) - 
                                  recent_performance.get("total_return", 0)) / 
                                 max(abs(historical_performance.get("total_return", 1)),1      
            # Average decay score
            return (sharpe_decay + return_decay) / 2
            
        except Exception as e:
            logger.error(f"Failed to calculate performance decay: {e}")
            return 0    def _calculate_correlation_decay(self, strategy: StrategyRecord) -> float:
     late correlation decay score."""
        try:
            # This would typically involve analyzing correlation with market indices
            # For now, use a simplified approach based on performance consistency
            
            if len(strategy.performance_history) < 10            return0      
            # Calculate performance consistency over time
            returns =         for record in strategy.performance_history[-10:]:
                returns.append(record[performance"].get("total_return",0      
            # Higher variance indicates lower correlation stability
            variance = np.var(returns)
            correlation_decay = min(variance / 00.110)  # Normalize to0      
            return correlation_decay
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation decay: {e}")
            return 0    def _calculate_market_regime_decay(self, strategy: StrategyRecord) -> float:
     late market regime decay score."""
        try:
            # This would typically involve analyzing performance across different market regimes
            # For now, use a simplified approach
            
            if len(strategy.performance_history) < 5            return0      
            # Check if strategy performance is consistent across different periods
            recent_performance = strategy.performance_history[-1ance]           avg_performance = np.mean([
                record[performance"].get(sharpe_ratio", 0) 
                for record in strategy.performance_history[-5:]
            ])
            
            # Calculate deviation from average
            deviation = abs(recent_performance.get(sharpe_ratio", 0) - avg_performance)
            regime_decay = min(deviation / 00.510)  # Normalize to0      
            return regime_decay
            
        except Exception as e:
            logger.error(f"Failed to calculate market regime decay: {e}")
            return 0    def _calculate_competition_decay(self, strategy: StrategyRecord) -> float:
     late competition decay score."""
        try:
            # This would typically involve analyzing market competition and strategy uniqueness
            # For now, use a simplified approach based on strategy age and performance
            
            strategy_age_days = (datetime.now() - strategy.created_at).days
            
            # Older strategies are more likely to face competition
            age_factor = min(strategy_age_days / 365, 1      
            # Performance degradation also indicates competition
            performance_factor = strategy.decay_score
            
            competition_decay = (age_factor + performance_factor) / 2
            
            return competition_decay
            
        except Exception as e:
            logger.error(f"Failed to calculate competition decay: {e}")
            return 0
    def _generate_decay_recommendations(self, decay_score: float, decay_factors: List[str]) -> List[str]:
     enerate recommendations based on decay analysis."""
        recommendations = []
        
        if decay_score > 0.7:
            recommendations.append(Consider retiring strategy due to significant decay")
        elif decay_score > 0.5:
            recommendations.append("Monitor strategy closely and consider parameter adjustments")
        elif decay_score > 0.3:
            recommendations.append("Review strategy performance and consider minor optimizations")
        
        if "performance_degradationin decay_factors:
            recommendations.append("Investigate recent performance decline and adjust parameters")
        
        if "increased_correlationin decay_factors:
            recommendations.append("Consider diversifying strategy or reducing position size")
        
        if "market_regime_changein decay_factors:
            recommendations.append(Adapt strategy to current market conditions")
        
        ifincreased_competitionin decay_factors:
            recommendations.append("Explore strategy variations or alternative approaches")
        
        return recommendations
    
    def _determine_action_required(self, decay_score: float, decay_factors: List[str]) -> str:
     ermine required action based on decay analysis.""    if decay_score > 0.7:
            return "retire"
        elif decay_score > 0.5:
            return "adjust"
        elif decay_score > 0.3:
            returnmonitor"
        else:
            return "none"
    
    def _analyze_decay_trend(self, strategy: StrategyRecord) -> str:
     alyze decay trend over time."""
        try:
            if len(strategy.performance_history) < 5            return "stable"
            
            # Get recent decay scores (if available)
            recent_scores =         for record in strategy.performance_history[-5:]:
                if "decay_score" in record:
                    recent_scores.append(record["decay_score"])
            
            if len(recent_scores) < 3            return "stable"
            
            # Calculate trend
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            if trend > 0.01            return "worsening        elif trend < -0.01            return "improving            else:
                return "stable"
                
        except Exception as e:
            logger.error(fFailed to analyze decay trend: {e}")
            returnstable"
    
    def _estimate_time_to_decay(self, strategy: StrategyRecord, current_decay: float) -> Optional[int]:
     imate time until significant decay."""
        try:
            if current_decay < 0.3            return None  # No significant decay expected soon
            
            # Simple linear extrapolation
            if len(strategy.performance_history) >= 2            recent_decay = current_decay
                older_decay = strategy.performance_history[0].get(decay_score
                
                if recent_decay > older_decay:
                    decay_rate = (recent_decay - older_decay) / len(strategy.performance_history)
                    if decay_rate > 0:
                        days_to_significant = int((0.7- current_decay) / decay_rate)
                        return max(0, days_to_significant)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to estimate time to decay: {e}")
            return None
    
    async def _update_lifecycle(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
     te strategy lifecycle status."""
        try:
            strategy_id = kwargs.get("strategy_id)        new_status = kwargs.get("status")
            
            if not strategy_id or strategy_id not in self.strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            strategy = self.strategies[strategy_id]
            
            # Update status
            strategy.status = new_status
            
            # Update timestamps
            if new_status == deployed and not strategy.deployed_at:
                strategy.deployed_at = datetime.now()
            elif new_status ==retired and not strategy.retired_at:
                strategy.retired_at = datetime.now()
            
            # Save to database
            self._save_strategy(strategy)
            
            return[object Object]
            strategy_id": strategy_id,
               new_status": new_status,
           updated_at:datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update lifecycle: {e}")
            raise
    
    def _save_strategy(self, strategy: StrategyRecord) -> None:
     strategy to database."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("            INSERT OR REPLACE INTO strategies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  (
                strategy.strategy_id,
                strategy.hypothesis_id,
                strategy.name,
                strategy.description,
                strategy.strategy_type,
                strategy.asset_class,
                strategy.timeframe,
                strategy.created_at.isoformat(),
                strategy.deployed_at.isoformat() if strategy.deployed_at else None,
                strategy.retired_at.isoformat() if strategy.retired_at else None,
                strategy.status,
                json.dumps(strategy.performance_history),
                json.dumps(strategy.current_performance),
                strategy.decay_score,
                json.dumps(strategy.decay_factors),
                strategy.last_decay_check.isoformat() if strategy.last_decay_check else None,
                json.dumps(strategy.tags),
                json.dumps(strategy.metadata)
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to save strategy: {e}")
            raise
    
    def _save_decay_analysis(self, analysis: DecayAnalysis) -> None:
     decay analysis to database."""
        try:
            cursor = self.connection.cursor()
            
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{analysis.strategy_id}"
            
            cursor.execute("            INSERT INTO decay_analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  (
                analysis_id,
                analysis.strategy_id,
                analysis.analysis_date.isoformat(),
                analysis.overall_decay_score,
                analysis.performance_decay,
                analysis.correlation_decay,
                analysis.market_regime_decay,
                analysis.competition_decay,
                json.dumps(analysis.decay_factors),
                json.dumps(analysis.decay_severity),
                json.dumps(analysis.recommendations),
                analysis.action_required,
                analysis.decay_trend,
                analysis.time_to_decay,
                analysis.created_at.isoformat()
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to save decay analysis: {e}")
            raise
    
    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
     strategy by ID.       return self.strategies.get(strategy_id)
    
    def get_all_strategies(self) -> Dict[str, StrategyRecord]:
     all strategies.       return self.strategies.copy()
    
    def get_strategies_by_status(self, status: str) -> List[StrategyRecord]:
     strategies by status.
        return [s for s in self.strategies.values() if s.status == status]
    
    def get_decay_analyses(self, strategy_id: str = None) -> List[DecayAnalysis]:
     decay analyses."""
        if strategy_id:
            return [a for a in self.decay_analyses if a.strategy_id == strategy_id]
        return self.decay_analyses.copy()
    
    def validate_input(self, **kwargs) -> bool:
       e input parameters."""
        operation = kwargs.get("operation")
        valid_operations = ["register",update,decay_analysis",lifecycle_update"]
        return operation in valid_operations
    
    def validate_config(self) -> bool:
       gent configuration."# Basic validation - database path should be specified
        returntrue   
    def handle_error(self, error: Exception) -> AgentResult:
      ndle errors during execution."""
        self.status.state = AgentState.ERROR
        self.status.current_error = str(error)
        
        return AgentResult(
            success=false
            error_message=str(error),
            error_type=type(error).__name__,
            metadata={"agent:alpha_registry}        )
    
    def get_capabilities(self) -> List[str]:
  agent capabilities.       return self.__capabilities__
    
    def get_requirements(self) -> Dict[str, Any]:
  agent requirements.
        return {
         dependencies": self.__dependencies__,
      config": ["database]        }
    
    def __del__(self):
atabase connection."if self.connection:
            self.connection.close()
