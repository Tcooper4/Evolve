RiskValidator - Risk Validation Agent

This agent validates trading hypotheses by checking correlation, stability, and real-world viability.
It ensures hypotheses meet risk management standards before deployment.


import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult, AgentState
from trading.utils.error_handling import log_errors, retry_on_error
from trading.exceptions import StrategyError, ModelError
from trading.agents.alpha.signal_tester import TestResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
   of risk validation."""
    
    hypothesis_id: str
    test_result_id: str
    
    # Validation scores (0-1, higher is better)
    correlation_score: float
    stability_score: float
    viability_score: float
    overall_score: float
    
    # Detailed analysis
    correlation_analysis: Dict[str, Any]
    stability_analysis: Dict[str, Any]
    viability_analysis: Dict[str, Any]
    
    # Risk flags
    risk_flags: Liststr]
    warnings: List[str]
    recommendations: List[str]
    
    # Validation metadata
    validation_date: datetime
    validator_version: str
    confidence_level: float
    
    # Final decision
    is_approved: bool
    approval_reason: str
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]     hypothesis_id:self.hypothesis_id,
           test_result_id": self.test_result_id,
            correlation_score": self.correlation_score,
            stability_score": self.stability_score,
            viability_score": self.viability_score,
          overall_score": self.overall_score,
       correlation_analysis": self.correlation_analysis,
     stability_analysis: self.stability_analysis,
     viability_analysis: self.viability_analysis,
            risk_flags: self.risk_flags,
            warnings: self.warnings,
            recommendations": self.recommendations,
            validation_date: self.validation_date.isoformat(),
            validator_version": self.validator_version,
           confidence_level: self.confidence_level,
        is_approved": self.is_approved,
            approval_reason": self.approval_reason,
            created_at:self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->ValidationResult":
    te from dictionary."""
        # Convert datetime strings back to datetime objects
        for field invalidation_date, _at]:     if isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


@dataclass
class ValidationConfig:
  Configuration for risk validation."""
    
    # Correlation thresholds
    max_correlation_threshold: float = 0.7
    min_correlation_threshold: float = -0.3
    
    # Stability thresholds
    min_stability_periods: int = 3
    max_parameter_sensitivity: float = 0.2
    
    # Viability thresholds
    min_liquidity_requirement: float = 100# $1M daily volume
    max_slippage_threshold: float =01  # 00.1%
    min_market_cap: float =1000000  # $1B market cap
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5   max_drawdown_threshold: float = 0.15
    min_win_rate: float = 0.4 min_profit_factor: float = 10.2 # Risk management
    max_position_size: float = 0.1  #10% max position
    max_leverage: float = 2in_diversification: int = 5  # Minimum number of uncorrelated strategies
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]          max_correlation_threshold": self.max_correlation_threshold,
          min_correlation_threshold": self.min_correlation_threshold,
            min_stability_periods": self.min_stability_periods,
          max_parameter_sensitivity": self.max_parameter_sensitivity,
         min_liquidity_requirement": self.min_liquidity_requirement,
          max_slippage_threshold": self.max_slippage_threshold,
           min_market_cap: self.min_market_cap,
           min_sharpe_ratio: self.min_sharpe_ratio,
          max_drawdown_threshold": self.max_drawdown_threshold,
         min_win_rate": self.min_win_rate,
            min_profit_factor: self.min_profit_factor,
            max_position_size": self.max_position_size,
         max_leverage": self.max_leverage,
     min_diversification: self.min_diversification
        }


class RiskValidator(BaseAgent):ent that validates trading hypotheses for risk management."""
    
    __version__ = "100    __author__ = "RiskValidator Team"
    __description__ = Validatestrading hypotheses for correlation, stability, and viability"
    __tags__ =alpharisk,validation",compliance"]
    __capabilities__ =risk_validation", "correlation_analysis,stability_testing"]
    __dependencies__ =pandas,numpy",scipy]   
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.validation_config = None
        self.validation_results =      self.existing_strategies = []
        
    def _setup(self) -> None:
        tup the agent."""
        try:
            # Load validation configuration
            validation_config_data = self.config.custom_config.get(validation_config, {})
            self.validation_config = ValidationConfig(**validation_config_data)
            
            # Load existing strategies for correlation analysis
            self.existing_strategies = self.config.custom_config.get("existing_strategies",      
            logger.info("RiskValidator agent setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup RiskValidator agent: {e}")
            raise
    
    @log_errors()
    async def execute(self, **kwargs) -> AgentResult:
     te risk validation."""
        try:
            self.status.state = AgentState.RUNNING
            start_time = datetime.now()
            
            # Get test results to validate
            test_results = kwargs.get("test_results,            if not test_results:
                return AgentResult(
                    success=False,
                    error_message="No test results provided for validation",
                    metadata={"agent:risk_validator}                )
            
            # Validate each test result
            validation_results =           for test_result in test_results:
                validation_result = await self._validate_test_result(test_result)
                if validation_result:
                    validation_results.append(validation_result)
            
            # Generate validation summary
            summary = self._generate_validation_summary(validation_results)
            
            # Store results
            self.validation_results.extend(validation_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.status.state = AgentState.SUCCESS
            
            return AgentResult(
                success=True,
                data={
             validation_results": [r.to_dict() for r in validation_results],
                    summary": summary,
                   execution_time: execution_time
                },
                execution_time=execution_time,
                metadata={"agent:risk_validator"}
            )
            
        except Exception as e:
            self.status.state = AgentState.ERROR
            return self.handle_error(e)
    
    async def _validate_test_result(self, test_result: TestResult) -> Optional[ValidationResult]:
   Validate a single test result."""
        try:
            # Perform correlation analysis
            correlation_score, correlation_analysis = self._analyze_correlation(test_result)
            
            # Perform stability analysis
            stability_score, stability_analysis = self._analyze_stability(test_result)
            
            # Perform viability analysis
            viability_score, viability_analysis = self._analyze_viability(test_result)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                correlation_score, stability_score, viability_score
            )
            
            # Generate risk flags and recommendations
            risk_flags, warnings, recommendations = self._generate_risk_assessment(
                test_result, correlation_analysis, stability_analysis, viability_analysis
            )
            
            # Make approval decision
            is_approved, approval_reason = self._make_approval_decision(
                overall_score, risk_flags, test_result
            )
            
            validation_result = ValidationResult(
                hypothesis_id=test_result.hypothesis_id,
                test_result_id=f"{test_result.hypothesis_id}_{test_result.ticker}_{test_result.timeframe},       correlation_score=correlation_score,
                stability_score=stability_score,
                viability_score=viability_score,
                overall_score=overall_score,
                correlation_analysis=correlation_analysis,
                stability_analysis=stability_analysis,
                viability_analysis=viability_analysis,
                risk_flags=risk_flags,
                warnings=warnings,
                recommendations=recommendations,
                validation_date=datetime.now(),
                validator_version=self.__version__,
                confidence_level=overall_score,
                is_approved=is_approved,
                approval_reason=approval_reason
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate test result: {e}")
            return None
    
    def _analyze_correlation(self, test_result: TestResult) -> Tuple[float, Dict[str, Any]]:
nalyze correlation with existing strategies."""
        try:
            # Calculate correlation with existing strategies
            correlations =       for strategy in self.existing_strategies:
                if hasattr(strategy, 'equity_curve') and strategy.equity_curve is not None:
                    # Align time periods
                    aligned_data = self._align_time_series(
                        test_result.equity_curve, strategy.equity_curve
                    )
                    if aligned_data is not None:
                        corr = pearsonr(aligned_data0, aligned_data[1])[0]
                        correlations.append(abs(corr))
            
            # Calculate average correlation
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Calculate correlation score (lower correlation is better)
            correlation_score = max(0, 1 - avg_correlation / self.validation_config.max_correlation_threshold)
            
            analysis =[object Object]
        average_correlation: avg_correlation,
                max_correlation": max(correlations) if correlations else 0
                correlation_count": len(correlations),
             correlation_threshold: self.validation_config.max_correlation_threshold
            }
            
            return correlation_score, analysis
            
        except Exception as e:
            logger.error(fFailed to analyze correlation: {e}")
            return 05{"error": str(e)}
    
    def _analyze_stability(self, test_result: TestResult) -> Tuple[float, Dict[str, Any]]:
strategy stability across different conditions."""
        try:
            stability_metrics = []
            
            # 1. Parameter sensitivity analysis
            param_sensitivity = self._analyze_parameter_sensitivity(test_result)
            stability_metrics.append(param_sensitivity)
            
            # 2. Regime stability analysis
            regime_stability = self._analyze_regime_stability(test_result)
            stability_metrics.append(regime_stability)
            
            # 3. Time period stability analysis
            time_stability = self._analyze_time_stability(test_result)
            stability_metrics.append(time_stability)
            
            # 4. Market condition stability
            market_stability = self._analyze_market_stability(test_result)
            stability_metrics.append(market_stability)
            
            # Calculate overall stability score
            stability_score = np.mean(stability_metrics)
            
            analysis =[object Object]
          parameter_sensitivity": param_sensitivity,
               regime_stability": regime_stability,
               time_stability": time_stability,
               market_stability": market_stability,
                overall_stability: stability_score
            }
            
            return stability_score, analysis
            
        except Exception as e:
            logger.error(fFailed to analyze stability: {e}")
            return 05{"error": str(e)}
    
    def _analyze_viability(self, test_result: TestResult) -> Tuple[float, Dict[str, Any]]:
     Analyze real-world viability."""
        try:
            viability_metrics = []
            
            # 1. Liquidity analysis
            liquidity_score = self._analyze_liquidity(test_result)
            viability_metrics.append(liquidity_score)
            
            # 2. Transaction cost analysis
            transaction_score = self._analyze_transaction_costs(test_result)
            viability_metrics.append(transaction_score)
            
            # 3. Market impact analysis
            market_impact_score = self._analyze_market_impact(test_result)
            viability_metrics.append(market_impact_score)
            
            # 4. Regulatory compliance
            compliance_score = self._analyze_regulatory_compliance(test_result)
            viability_metrics.append(compliance_score)
            
            # 5. Operational feasibility
            operational_score = self._analyze_operational_feasibility(test_result)
            viability_metrics.append(operational_score)
            
            # Calculate overall viability score
            viability_score = np.mean(viability_metrics)
            
            analysis =[object Object]
                liquidity_score: liquidity_score,
                transaction_score": transaction_score,
            market_impact_score": market_impact_score,
               compliance_score": compliance_score,
                operational_score": operational_score,
                overall_viability: viability_score
            }
            
            return viability_score, analysis
            
        except Exception as e:
            logger.error(fFailed to analyze viability: {e}")
            return 05{"error": str(e)}
    
    def _analyze_parameter_sensitivity(self, test_result: TestResult) -> float:
nalyze sensitivity to parameter changes."""
        try:
            # This would typically involve testing with slightly different parameters
            # For now, use a simplified approach based on performance metrics
            
            # Higher Sharpe ratio indicates lower sensitivity
            sharpe_sensitivity = min(test_result.sharpe_ratio / 2.0, 1      
            # Lower drawdown indicates lower sensitivity
            drawdown_sensitivity = max(0 - test_result.max_drawdown / 0.2      
            # Higher win rate indicates lower sensitivity
            win_rate_sensitivity = test_result.win_rate
            
            return np.mean([sharpe_sensitivity, drawdown_sensitivity, win_rate_sensitivity])
            
        except Exception as e:
            logger.error(fFailed to analyze parameter sensitivity: {e}")
            return 0.5   
    def _analyze_regime_stability(self, test_result: TestResult) -> float:
Analyze stability across different market regimes."""
        try:
            # This would typically involve testing across bull/bear/sideways markets
            # For now, use a simplified approach
            
            # Check if performance is consistent across different time periods
            if test_result.equity_curve is not None and len(test_result.equity_curve) > 100
                # Split into quarters and check consistency
                quarters = len(test_result.equity_curve) //4           quarter_returns = []
                
                for i in range(4):
                    start_idx = i * quarters
                    end_idx = (i + 1 * quarters if i < 3 else len(test_result.equity_curve)
                    quarter_data = test_result.equity_curve.iloc[start_idx:end_idx]
                    quarter_return = (quarter_data.iloc[-1 / quarter_data.iloc[0]) - 1
                    quarter_returns.append(quarter_return)
                
                # Calculate consistency (lower variance is better)
                return_std = np.std(quarter_returns)
                consistency_score = max(0, 1 - return_std / 0.1
                
                return consistency_score
            
            return0.5          
        except Exception as e:
            logger.error(fFailed to analyze regime stability: {e}")
            return 0.5   
    def _analyze_time_stability(self, test_result: TestResult) -> float:
Analyze stability over time."""
        try:
            # Check for performance degradation over time
            if test_result.equity_curve is not None and len(test_result.equity_curve) > 50
                # Split into early and late periods
                mid_point = len(test_result.equity_curve) // 2
                
                early_period = test_result.equity_curve.iloc[:mid_point]
                late_period = test_result.equity_curve.iloc[mid_point:]
                
                early_return = (early_period.iloc[-1 / early_period.iloc[0]) -1              late_return = (late_period.iloc[-1] / late_period.iloc[0]) - 1
                
                # Calculate performance consistency
                performance_ratio = late_return / early_return if early_return != 0 else1         stability_score = max(0, 1 - abs(1 - performance_ratio))
                
                return stability_score
            
            return0.5          
        except Exception as e:
            logger.error(fFailed to analyze time stability: {e}")
            return 0.5   
    def _analyze_market_stability(self, test_result: TestResult) -> float:
Analyze stability under different market conditions."""
        try:
            # This would typically involve testing under various market conditions
            # For now, use volatility as a proxy for market stability
            
            if test_result.equity_curve is not None:
                returns = test_result.equity_curve.pct_change().dropna()
                volatility = returns.std()
                
                # Lower volatility indicates better stability
                stability_score = max(0, 1 - volatility / 00.05
                
                return stability_score
            
            return0.5          
        except Exception as e:
            logger.error(fFailed to analyze market stability: {e}")
            return 0.5   
    def _analyze_liquidity(self, test_result: TestResult) -> float:
Analyze liquidity requirements."""
        try:
            # Check if the strategy can be executed with reasonable liquidity
            # This would typically involve checking actual market data
            
            # Simplified approach based on position size and frequency
            avg_position_size = test_result.total_trades / max(1 len(test_result.equity_curve))
            
            # Smaller position sizes are better for liquidity
            liquidity_score = max(0, 1 - avg_position_size / 0.1      
            return liquidity_score
            
        except Exception as e:
            logger.error(fFailed to analyze liquidity: {e}")
            return 0.5   
    def _analyze_transaction_costs(self, test_result: TestResult) -> float:
       Analyze impact of transaction costs."""
        try:
            # Check if strategy is profitable after transaction costs
            # This would typically involve detailed cost analysis
            
            # Simplified approach based on trade frequency
            trade_frequency = test_result.total_trades / max(1 len(test_result.equity_curve))
            
            # Lower trade frequency is better for transaction costs
            cost_score = max(0, 1 - trade_frequency / 0.1      
            return cost_score
            
        except Exception as e:
            logger.error(fFailed to analyze transaction costs: {e}")
            return 0.5   
    def _analyze_market_impact(self, test_result: TestResult) -> float:
Analyze potential market impact."""
        try:
            # Check if strategy trades would have significant market impact
            # This would typically involve volume analysis
            
            # Simplified approach based on position size
            avg_trade_size = test_result.total_trades / max(1 len(test_result.equity_curve))
            
            # Smaller trade sizes have less market impact
            impact_score = max(0, 1- avg_trade_size / 00.05      
            return impact_score
            
        except Exception as e:
            logger.error(fFailed to analyze market impact: {e}")
            return 0.5   
    def _analyze_regulatory_compliance(self, test_result: TestResult) -> float:
Analyze regulatory compliance."""
        try:
            # Check if strategy complies with regulatory requirements
            # This would typically involve compliance checks
            
            compliance_score = 1.0  # Assume compliant for now
            
            # Add specific checks as needed
            if test_result.max_drawdown > 0.5:  # 50% drawdown
                compliance_score *=0.8      
            if test_result.sharpe_ratio < 0:  # Negative Sharpe
                compliance_score *=0.9      
            return compliance_score
            
        except Exception as e:
            logger.error(fFailed toanalyze regulatory compliance: {e}")
            return 0.5   
    def _analyze_operational_feasibility(self, test_result: TestResult) -> float:
nalyze operational feasibility."""
        try:
            # Check if strategy can be operationally implemented
            # This would typically involve operational analysis
            
            feasibility_score =1      
            # Check for reasonable trade frequency
            if test_result.total_trades >1000 # Too many trades
                feasibility_score *=0.7      
            # Check for reasonable holding periods
            avg_holding_period = len(test_result.equity_curve) / max(1, test_result.total_trades)
            if avg_holding_period < 1:  # Very short holding periods
                feasibility_score *=0.8      
            return feasibility_score
            
        except Exception as e:
            logger.error(fFailed to analyze operational feasibility: {e}")
            return 0.5    def _calculate_overall_score(
        self, 
        correlation_score: float, 
        stability_score: float, 
        viability_score: float
    ) -> float:
  Calculateoverall validation score."""
        try:
            # Weighted average of all scores
            weights =[object Object]
                correlation": 0.3
                stability4
               viability:0.3   }
            
            overall_score = (
                correlation_score * weightscorrelation +
                stability_score * weights["stability"] +
                viability_score * weights["viability"]
            )
            
            return max(0, min(overall_score,1          
        except Exception as e:
            logger.error(f"Failed to calculate overall score: {e}")
            return 0.5
    def _generate_risk_assessment(
        self,
        test_result: TestResult,
        correlation_analysis: Dict[str, Any],
        stability_analysis: Dict[str, Any],
        viability_analysis: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
      enerate risk flags, warnings, and recommendations."""
        try:
            risk_flags = []
            warnings = []
            recommendations = []
            
            # Correlation flags
            if correlation_analysis.get("average_correlation", 0) > self.validation_config.max_correlation_threshold:
                risk_flags.append("HIGH_CORRELATION)          warnings.append("Strategy highly correlated with existing strategies)   recommendations.append(Consider reducing position size or diversifying")
            
            # Stability flags
            if stability_analysis.get(overall_stability", 0) < 0.6              risk_flags.append("LOW_STABILITY)          warnings.append("Strategy shows low stability across different conditions)   recommendations.append("Test with different parameters and market conditions")
            
            # Viability flags
            if viability_analysis.get(overall_viability", 0) < 0.7              risk_flags.append("LOW_VIABILITY)          warnings.append("Strategy may not be viable in real-world conditions)   recommendations.append("Review liquidity requirements and transaction costs")
            
            # Performance flags
            if test_result.sharpe_ratio < self.validation_config.min_sharpe_ratio:
                risk_flags.append("LOW_SHARPE)          warnings.append("Sharpe ratio below minimum threshold)   recommendations.append(Optimize strategy parameters or consider alternative approaches")
            
            if test_result.max_drawdown > self.validation_config.max_drawdown_threshold:
                risk_flags.append("HIGH_DRAWDOWN)          warnings.append("Maximum drawdown exceeds threshold)   recommendations.append("Implement stricter risk management or reduce position sizes")
            
            if test_result.win_rate < self.validation_config.min_win_rate:
                risk_flags.append("LOW_WIN_RATE)          warnings.append("Win rate below minimum threshold)   recommendations.append(Review entry/exit conditions and risk management")
            
            return risk_flags, warnings, recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate risk assessment: {e}")
            return [], [],    
    def _make_approval_decision(
        self, 
        overall_score: float, 
        risk_flags: List[str], 
        test_result: TestResult
    ) -> Tuple[bool, str]:
   ake final approval decision."""
        try:
            # Check if overall score meets minimum threshold
            if overall_score < 0.7            return False, "Overall validation score below threshold"
            
            # Check for critical risk flags
            critical_flags = ["HIGH_CORRELATION", LOW_STABILITY",LOW_VIABILITY"]
            if any(flag in risk_flags for flag in critical_flags):
                return False, f"Critical risk flags detected: {[f for f in risk_flags if f in critical_flags]}"
            
            # Check performance metrics
            if (test_result.sharpe_ratio < self.validation_config.min_sharpe_ratio or
                test_result.max_drawdown > self.validation_config.max_drawdown_threshold or
                test_result.win_rate < self.validation_config.min_win_rate):
                return False, "Performance metrics below minimum thresholds"
            
            return True, Strategy approved for deployment"
            
        except Exception as e:
            logger.error(f"Failed to make approval decision: {e}")
            returnfalseError in approval decision: {str(e)}"
    
    def _align_time_series(self, series1: pd.Series, series2: pd.Series) -> Optional[Tuple[pd.Series, pd.Series]]:
       gn two time series for correlation analysis."""
        try:
            # Find common index
            common_index = series1.index.intersection(series2.index)
            
            if len(common_index) < 10 Need minimum data points
                return None
            
            aligned_series1 = series1loc[common_index]
            aligned_series2 = series2loc[common_index]
            
            return aligned_series1ies2
            
        except Exception as e:
            logger.error(f"Failed to align time series: {e}")
            return None
    
    def _generate_validation_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
 enerate summary of validation results."""
        try:
            if not validation_results:
                return {"error": "No validation results available"}
            
            # Calculate statistics
            approved_count = sum(1 for r in validation_results if r.is_approved)
            total_count = len(validation_results)
            
            avg_correlation_score = np.mean([r.correlation_score for r in validation_results])
            avg_stability_score = np.mean([r.stability_score for r in validation_results])
            avg_viability_score = np.mean([r.viability_score for r in validation_results])
            avg_overall_score = np.mean([r.overall_score for r in validation_results])
            
            # Risk flag analysis
            all_risk_flags =         for result in validation_results:
                all_risk_flags.extend(result.risk_flags)
            
            risk_flag_counts = [object Object]          for flag in all_risk_flags:
                risk_flag_counts[flag] = risk_flag_counts.get(flag, 0) + 1
            
            return[object Object]
                total_validated": total_count,
               approved_count": approved_count,
              approval_rate": approved_count / total_count if total_count > 0 else 0
               average_scores": {
                    correlation: avg_correlation_score,
                  stability": avg_stability_score,
                   viability": avg_viability_score,
                overall": avg_overall_score
                },
                risk_flag_summary": risk_flag_counts,
            top_recommendations": self._get_top_recommendations(validation_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate validation summary: {e}")
            return {"error": str(e)}
    
    def _get_top_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
op recommendations from validation results."""
        try:
            all_recommendations =         for result in validation_results:
                all_recommendations.extend(result.recommendations)
            
            # Count recommendations
            recommendation_counts = [object Object]           for rec in all_recommendations:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
            
            # Return top 5ations
            sorted_recommendations = sorted(
                recommendation_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return [rec[0] for rec in sorted_recommendations[:5]]
            
        except Exception as e:
            logger.error(fFailed to get top recommendations: {e}")
            return []
    
    def validate_input(self, **kwargs) -> bool:
       e input parameters.""   required_params = ["test_results"]
        return all(param in kwargs for param in required_params)
    
    def validate_config(self) -> bool:
       gent configuration.""   required_config = [validation_config"]
        custom_config = self.config.custom_config or[object Object]        return all(key in custom_config for key in required_config)
    
    def handle_error(self, error: Exception) -> AgentResult:
      ndle errors during execution."""
        self.status.state = AgentState.ERROR
        self.status.current_error = str(error)
        
        return AgentResult(
            success=false
            error_message=str(error),
            error_type=type(error).__name__,
            metadata={"agent:risk_validator}        )
    
    def get_capabilities(self) -> List[str]:
  agent capabilities.       return self.__capabilities__
    
    def get_requirements(self) -> Dict[str, Any]:
  agent requirements.
        return {
         dependencies": self.__dependencies__,
            data_sources: ["market_data",existing_strategies"],
      config": [validation_config]        }
    
    def get_validation_results(self) -> List[ValidationResult]:
Get all validation results.       return self.validation_results.copy()
    
    def clear_results(self) -> None:
      stored validation results."""
        self.validation_results.clear()
