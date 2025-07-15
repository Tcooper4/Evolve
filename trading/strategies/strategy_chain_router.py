"""
Strategy Chain Router - Batch 17
Enhanced strategy routing with fallback mechanisms and comprehensive logging
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Types of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    POSITIONAL = "positional"

class StrategyStatus(Enum):
    """Status of strategy execution."""
    VALID = "valid"
    INVALID = "invalid"
    MISSING = "missing"
    DISABLED = "disabled"
    ERROR = "error"

@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    enabled: bool = True
    priority: int = 1
    fallback_strategies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyResult:
    """Result of strategy execution."""
    strategy_name: str
    status: StrategyStatus
    signal: Optional[str] = None
    confidence: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PromptStrategyMapping:
    """Mapping between prompts and strategies."""
    prompt_hash: str
    strategy_name: str
    confidence: float
    created_at: datetime
    last_used: datetime
    usage_count: int = 0

class StrategyChainRouter:
    """
    Enhanced strategy chain router with fallback mechanisms.
    
    Features:
    - Prompt-linked strategy routing
    - Fallback to base strategies (RSI, SMA, etc.)
    - Comprehensive logging and warnings
    - Strategy validation and error handling
    - Performance tracking
    """
    
    def __init__(self, 
                 enable_fallbacks: bool = True,
                 log_warnings: bool = True,
                 max_fallback_depth: int = 3):
        """
        Initialize strategy chain router.
        
        Args:
            enable_fallbacks: Enable fallback to base strategies
            log_warnings: Enable warning logging
            max_fallback_depth: Maximum fallback chain depth
        """
        self.enable_fallbacks = enable_fallbacks
        self.log_warnings = log_warnings
        self.max_fallback_depth = max_fallback_depth
        
        # Strategy registry
        self.strategies: Dict[str, StrategyConfig] = {}
        
        # Prompt-strategy mappings
        self.prompt_mappings: Dict[str, PromptStrategyMapping] = {}
        
        # Base strategies (fallbacks)
        self.base_strategies = {
            "RSI": StrategyConfig(
                name="RSI",
                strategy_type=StrategyType.MEAN_REVERSION,
                parameters={"period": 14, "overbought": 70, "oversold": 30},
                enabled=True,
                priority=1,
                fallback_strategies=[]
            ),
            "SMA": StrategyConfig(
                name="SMA",
                strategy_type=StrategyType.TREND_FOLLOWING,
                parameters={"short_period": 10, "long_period": 20},
                enabled=True,
                priority=1,
                fallback_strategies=[]
            ),
            "MACD": StrategyConfig(
                name="MACD",
                strategy_type=StrategyType.TREND_FOLLOWING,
                parameters={"fast": 12, "slow": 26, "signal": 9},
                enabled=True,
                priority=1,
                fallback_strategies=[]
            ),
            "Bollinger": StrategyConfig(
                name="Bollinger",
                strategy_type=StrategyType.MEAN_REVERSION,
                parameters={"period": 20, "std_dev": 2},
                enabled=True,
                priority=1,
                fallback_strategies=[]
            )
        }
        
        # Performance tracking
        self.execution_history: List[StrategyResult] = []
        self.fallback_usage: Dict[str, int] = {}
        
        logger.info("StrategyChainRouter initialized with fallback mechanisms")
    
    def _generate_prompt_hash(self, prompt: str) -> str:
        """
        Generate hash for prompt content.
        
        Args:
            prompt: Prompt content
            
        Returns:
            Hash string
        """
        return hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    def register_strategy(self, strategy_config: StrategyConfig) -> bool:
        """
        Register a strategy.
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            True if registration successful
        """
        try:
            # Validate strategy
            if not self._validate_strategy(strategy_config):
                logger.warning(f"Strategy validation failed: {strategy_config.name}")
                return False
            
            self.strategies[strategy_config.name] = strategy_config
            logger.info(f"Strategy registered: {strategy_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering strategy {strategy_config.name}: {e}")
            return False
    
    def _validate_strategy(self, strategy_config: StrategyConfig) -> bool:
        """
        Validate strategy configuration.
        
        Args:
            strategy_config: Strategy to validate
            
        Returns:
            True if valid
        """
        if not strategy_config.name:
            return False
        
        if strategy_config.name in self.strategies:
            logger.warning(f"Strategy already exists: {strategy_config.name}")
            return False
        
        if not strategy_config.enabled:
            logger.info(f"Strategy disabled: {strategy_config.name}")
            return False
        
        return True
    
    def link_prompt_to_strategy(self, 
                               prompt: str, 
                               strategy_name: str, 
                               confidence: float = 1.0) -> bool:
        """
        Link a prompt to a specific strategy.
        
        Args:
            prompt: Prompt content
            strategy_name: Name of the strategy
            confidence: Confidence in the mapping
            
        Returns:
            True if linking successful
        """
        prompt_hash = self._generate_prompt_hash(prompt)
        
        # Check if strategy exists
        if strategy_name not in self.strategies and strategy_name not in self.base_strategies:
            if self.log_warnings:
                logger.warning(f"Attempted to link prompt to non-existent strategy: {strategy_name}")
            return False
        
        # Create or update mapping
        if prompt_hash in self.prompt_mappings:
            mapping = self.prompt_mappings[prompt_hash]
            mapping.strategy_name = strategy_name
            mapping.confidence = confidence
            mapping.last_used = datetime.now()
            mapping.usage_count += 1
        else:
            mapping = PromptStrategyMapping(
                prompt_hash=prompt_hash,
                strategy_name=strategy_name,
                confidence=confidence,
                created_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=1
            )
            self.prompt_mappings[prompt_hash] = mapping
        
        logger.info(f"Linked prompt (hash: {prompt_hash}) to strategy: {strategy_name}")
        return True
    
    def route_strategy(self, 
                      prompt: str, 
                      context: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """
        Route prompt to appropriate strategy with fallback.
        
        Args:
            prompt: Prompt content
            context: Additional context
            
        Returns:
            StrategyResult with execution details
        """
        start_time = datetime.now()
        prompt_hash = self._generate_prompt_hash(prompt)
        
        # Try to find linked strategy
        linked_strategy = self._get_linked_strategy(prompt_hash)
        
        if linked_strategy:
            result = self._execute_strategy(linked_strategy, context)
            if result.status == StrategyStatus.VALID:
                return result
            else:
                if self.log_warnings:
                    logger.warning(
                        f"Linked strategy '{linked_strategy}' is invalid or missing for prompt (hash: {prompt_hash}). "
                        f"Error: {result.error_message}. Falling back to base strategies."
                    )
        
        # Fallback to base strategies
        if self.enable_fallbacks:
            fallback_result = self._execute_fallback_strategies(prompt, context)
            if fallback_result:
                return fallback_result
        
        # If all else fails, return error result
        execution_time = (datetime.now() - start_time).total_seconds()
        return StrategyResult(
            strategy_name="unknown",
            status=StrategyStatus.ERROR,
            error_message="No valid strategy found and fallbacks disabled",
            execution_time=execution_time
        )
    
    def _get_linked_strategy(self, prompt_hash: str) -> Optional[str]:
        """
        Get linked strategy for prompt hash.
        
        Args:
            prompt_hash: Hash of the prompt
            
        Returns:
            Strategy name or None
        """
        if prompt_hash in self.prompt_mappings:
            mapping = self.prompt_mappings[prompt_hash]
            strategy_name = mapping.strategy_name
            
            # Check if strategy exists and is valid
            if strategy_name in self.strategies or strategy_name in self.base_strategies:
                return strategy_name
            else:
                if self.log_warnings:
                    logger.warning(f"Linked strategy '{strategy_name}' not found for prompt hash: {prompt_hash}")
        
        return None
    
    def _execute_strategy(self, 
                         strategy_name: str, 
                         context: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """
        Execute a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            context: Execution context
            
        Returns:
            StrategyResult
        """
        start_time = datetime.now()
        
        # Get strategy config
        strategy_config = self.strategies.get(strategy_name) or self.base_strategies.get(strategy_name)
        
        if not strategy_config:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StrategyResult(
                strategy_name=strategy_name,
                status=StrategyStatus.MISSING,
                error_message=f"Strategy '{strategy_name}' not found",
                execution_time=execution_time
            )
        
        if not strategy_config.enabled:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StrategyResult(
                strategy_name=strategy_name,
                status=StrategyStatus.DISABLED,
                error_message=f"Strategy '{strategy_name}' is disabled",
                execution_time=execution_time
            )
        
        try:
            # Simulate strategy execution
            signal = self._simulate_strategy_execution(strategy_config, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            return StrategyResult(
                strategy_name=strategy_name,
                status=StrategyStatus.VALID,
                signal=signal,
                confidence=0.8,
                parameters=strategy_config.parameters,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            if self.log_warnings:
                logger.warning(f"Error executing strategy '{strategy_name}': {e}")
            
            return StrategyResult(
                strategy_name=strategy_name,
                status=StrategyStatus.ERROR,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _execute_fallback_strategies(self, 
                                   prompt: str, 
                                   context: Optional[Dict[str, Any]] = None) -> Optional[StrategyResult]:
        """
        Execute fallback strategies in order of priority.
        
        Args:
            prompt: Original prompt
            context: Execution context
            
        Returns:
            StrategyResult or None if no fallback succeeds
        """
        # Try base strategies in order
        fallback_order = ["RSI", "SMA", "MACD", "Bollinger"]
        
        for strategy_name in fallback_order:
            if strategy_name in self.base_strategies:
                result = self._execute_strategy(strategy_name, context)
                
                if result.status == StrategyStatus.VALID:
                    # Track fallback usage
                    self.fallback_usage[strategy_name] = self.fallback_usage.get(strategy_name, 0) + 1
                    
                    if self.log_warnings:
                        logger.info(f"Successfully used fallback strategy: {strategy_name}")
                    
                    return result
                else:
                    if self.log_warnings:
                        logger.warning(f"Fallback strategy '{strategy_name}' failed: {result.error_message}")
        
        return None
    
    def _simulate_strategy_execution(self, 
                                   strategy_config: StrategyConfig, 
                                   context: Optional[Dict[str, Any]] = None) -> str:
        """
        Simulate strategy execution (placeholder for actual implementation).
        
        Args:
            strategy_config: Strategy configuration
            context: Execution context
            
        Returns:
            Simulated signal
        """
        # This is a placeholder - in practice, you'd implement actual strategy logic
        strategy_type = strategy_config.strategy_type
        
        if strategy_type == StrategyType.MEAN_REVERSION:
            return "BUY" if context and context.get("rsi", 50) < 30 else "SELL"
        elif strategy_type == StrategyType.TREND_FOLLOWING:
            return "BUY" if context and context.get("trend", "up") == "up" else "SELL"
        elif strategy_type == StrategyType.MOMENTUM:
            return "BUY" if context and context.get("momentum", 0) > 0 else "SELL"
        else:
            return "HOLD"
    
    def get_strategy_status(self, strategy_name: str) -> StrategyStatus:
        """
        Get status of a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy status
        """
        if strategy_name in self.strategies:
            strategy_config = self.strategies[strategy_name]
            return StrategyStatus.VALID if strategy_config.enabled else StrategyStatus.DISABLED
        elif strategy_name in self.base_strategies:
            strategy_config = self.base_strategies[strategy_name]
            return StrategyStatus.VALID if strategy_config.enabled else StrategyStatus.DISABLED
        else:
            return StrategyStatus.MISSING
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fallback usage.
        
        Returns:
            Dictionary with fallback statistics
        """
        total_fallbacks = sum(self.fallback_usage.values())
        
        return {
            "total_fallbacks": total_fallbacks,
            "fallback_usage": self.fallback_usage.copy(),
            "most_used_fallback": max(self.fallback_usage.items(), key=lambda x: x[1])[0] if self.fallback_usage else None,
            "fallback_rate": total_fallbacks / max(len(self.execution_history), 1)
        }
    
    def get_prompt_mapping_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about prompt-strategy mappings.
        
        Returns:
            Dictionary with mapping statistics
        """
        if not self.prompt_mappings:
            return {}
        
        total_mappings = len(self.prompt_mappings)
        valid_mappings = 0
        invalid_mappings = 0
        
        for mapping in self.prompt_mappings.values():
            if self.get_strategy_status(mapping.strategy_name) == StrategyStatus.VALID:
                valid_mappings += 1
            else:
                invalid_mappings += 1
        
        return {
            "total_mappings": total_mappings,
            "valid_mappings": valid_mappings,
            "invalid_mappings": invalid_mappings,
            "mapping_validity_rate": valid_mappings / total_mappings if total_mappings > 0 else 0.0
        }
    
    def clear_invalid_mappings(self):
        """Clear invalid prompt-strategy mappings."""
        invalid_hashes = []
        
        for prompt_hash, mapping in self.prompt_mappings.items():
            if self.get_strategy_status(mapping.strategy_name) != StrategyStatus.VALID:
                invalid_hashes.append(prompt_hash)
        
        for prompt_hash in invalid_hashes:
            del self.prompt_mappings[prompt_hash]
        
        if invalid_hashes and self.log_warnings:
            logger.warning(f"Cleared {len(invalid_hashes)} invalid prompt-strategy mappings")
    
    def reset_statistics(self):
        """Reset all statistics and history."""
        self.execution_history.clear()
        self.fallback_usage.clear()
        logger.info("Strategy router statistics reset")

def create_strategy_chain_router(enable_fallbacks: bool = True) -> StrategyChainRouter:
    """Factory function to create a strategy chain router."""
    return StrategyChainRouter(enable_fallbacks=enable_fallbacks) 