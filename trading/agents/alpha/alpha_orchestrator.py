AlphaOrchestrator - Autonomous Alpha Strategy Orchestrator

This is the top-level orchestrator that manages the complete alpha strategy lifecycle:
idea â†’ test â†’ validate â†’ register â†’ monitor

It coordinates all other alpha agents and ensures transparent logging of every decision.


import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import uuid

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult, AgentState
from trading.utils.error_handling import log_errors, retry_on_error
from trading.exceptions import StrategyError, ModelError
from trading.agents.alpha.alphagen_agent import AlphaGenAgent, Hypothesis
from trading.agents.alpha.signal_tester import SignalTester, TestResult
from trading.agents.alpha.risk_validator import RiskValidator, ValidationResult
from trading.agents.alpha.alpha_registry import AlphaRegistry, StrategyRecord
from trading.agents.alpha.sentiment_ingestion import SentimentIngestion

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationCycle:
   of a complete alpha strategy cycle.
    
    cycle_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Cycle stages
    idea_generation: Dict[str, Any] = field(default_factory=dict)
    testing: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    registration: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    hypotheses_generated: int = 0 hypotheses_tested: int = 0
    hypotheses_validated: int = 0
    strategies_registered: int = 0
    strategies_deployed: int = 0
    
    # Status
    status: str = "running"  # running, completed, failed
    current_stage: str = idea_generation"
    
    # Metadata
    cycle_type: str = "full"  # full, quick, maintenance
    priority: str = "normal"  # low, normal, high, critical
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]     cycle_id:self.cycle_id,
           start_time:self.start_time.isoformat(),
           end_time": self.end_time.isoformat() if self.end_time else None,
           idea_generation": self.idea_generation,
           testing": self.testing,
           validation: self.validation,
           registration": self.registration,
           monitoring: self.monitoring,
           hypotheses_generated: self.hypotheses_generated,
           hypotheses_tested: self.hypotheses_tested,
           hypotheses_validated: self.hypotheses_validated,
           strategies_registered: self.strategies_registered,
           strategies_deployed:self.strategies_deployed,
           status": self.status,
           current_stage:self.current_stage,
           cycle_type: self.cycle_type,
           priority: self.priority,
           created_at:self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrchestrationCycle":
    te from dictionary."""
        # Convert datetime strings back to datetime objects
        for field in [start_time", end_time, "_at]:           if data.get(field) and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


@dataclass
class DecisionLog:
   of a decision made during orchestration."""
    
    decision_id: str
    cycle_id: str
    stage: str
    decision_type: str
    description: str
    reasoning: str
    data_used: Dict[str, Any]
    outcome: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]     decision_id:self.decision_id,
           cycle_id: self.cycle_id,
           stage": self.stage,
           decision_type": self.decision_type,
           description": self.description,
           reasoning": self.reasoning,
           data_used": self.data_used,
           outcome": self.outcome,
           confidence: self.confidence,
           timestamp": self.timestamp.isoformat(),
           metadata": self.metadata
        }


class AlphaOrchestrator(BaseAgent):ent that orchestrates the complete alpha strategy lifecycle."""
    
    __version__ = 10    __author__ = AlphaOrchestrator Team"
    __description__ = "Orchestrates complete alpha strategy lifecycle __tags__ = [alpha",orchestrator",lifecycle",automation"]
    __capabilities__ = [full_lifecycle_orchestration",decision_logging", "agent_coordination"]
    __dependencies__ = ["alphagen_agent",signal_tester,risk_validator,alpha_registry",sentiment_ingestion"]   
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.alphagen_agent = None
        self.signal_tester = None
        self.risk_validator = None
        self.alpha_registry = None
        self.sentiment_ingestion = None
        
        self.active_cycles =[object Object]      self.completed_cycles =      self.decision_logs = []
        
    def _setup(self) -> None:
        tup the agent."""
        try:
            # Initialize all alpha agents
            self._initialize_agents()
            
            logger.info(AlphaOrchestrator agent setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup AlphaOrchestrator agent: {e}")
            raise
    
    def _initialize_agents(self) -> None:
     all alpha agents."""
        try:
            # Initialize AlphaGen agent
            alphagen_config = AgentConfig(
                name=alphagen_orchestrator,            custom_config=self.config.custom_config.get("alphagen,[object Object]     )
            self.alphagen_agent = AlphaGenAgent(alphagen_config)
            
            # Initialize SignalTester agent
            tester_config = AgentConfig(
                name="signal_tester_orchestrator,            custom_config=self.config.custom_config.get(signal_tester,[object Object]     )
            self.signal_tester = SignalTester(tester_config)
            
            # Initialize RiskValidator agent
            validator_config = AgentConfig(
                name=risk_validator_orchestrator,            custom_config=self.config.custom_config.get("risk_validator,[object Object]     )
            self.risk_validator = RiskValidator(validator_config)
            
            # Initialize AlphaRegistry agent
            registry_config = AgentConfig(
                name=alpha_registry_orchestrator,            custom_config=self.config.custom_config.get("alpha_registry,[object Object]     )
            self.alpha_registry = AlphaRegistry(registry_config)
            
            # Initialize SentimentIngestion agent
            sentiment_config = AgentConfig(
                name="sentiment_ingestion_orchestrator,            custom_config=self.config.custom_config.get("sentiment_ingestion,[object Object]     )
            self.sentiment_ingestion = SentimentIngestion(sentiment_config)
            
            logger.info("All alpha agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    @log_errors()
    async def execute(self, **kwargs) -> AgentResult:
     ute the alpha orchestration cycle."""
        try:
            self.status.state = AgentState.RUNNING
            start_time = datetime.now()
            
            # Get cycle parameters
            cycle_type = kwargs.get("cycle_type", "full")
            priority = kwargs.get("priority", "normal")
            
            # Create new orchestration cycle
            cycle = OrchestrationCycle(
                cycle_id=str(uuid.uuid4()),
                start_time=datetime.now(),
                cycle_type=cycle_type,
                priority=priority
            )
            
            self.active_cycles[cycle.cycle_id] = cycle
            
            # Execute the full cycle
            result = await self._execute_full_cycle(cycle)
            
            # Complete the cycle
            cycle.end_time = datetime.now()
            cycle.status = "completed"
            
            # Move to completed cycles
            self.completed_cycles.append(cycle)
            del self.active_cycles[cycle.cycle_id]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.status.state = AgentState.SUCCESS
            
            return AgentResult(
                success=True,
                data={
                    cycle": cycle.to_dict(),
                    result": result,
                   execution_time: execution_time
                },
                execution_time=execution_time,
                metadata={agent:alpha_orchestrator"}
            )
            
        except Exception as e:
            self.status.state = AgentState.ERROR
            return self.handle_error(e)
    
    async def _execute_full_cycle(self, cycle: OrchestrationCycle) -> Dict[str, Any]:
     ute the complete alpha strategy cycle."""
        try:
            cycle_results = {}
            
            # Stage 1: Idea Generation
            cycle.current_stage = idea_generation       idea_result = await self._execute_idea_generation(cycle)
            cycle.idea_generation = idea_result
            cycle_results[idea_generation"] = idea_result
            
            # Stage 2: Testing
            cycle.current_stage = "testing       test_result = await self._execute_testing(cycle, idea_result)
            cycle.testing = test_result
            cycle_results["testing"] = test_result
            
            # Stage 3: Validation
            cycle.current_stage = "validation"
            validation_result = await self._execute_validation(cycle, test_result)
            cycle.validation = validation_result
            cycle_results["validation"] = validation_result
            
            # Stage 4: Registration
            cycle.current_stage = "registration"
            registration_result = await self._execute_registration(cycle, validation_result)
            cycle.registration = registration_result
            cycle_results["registration"] = registration_result
            
            # Stage 5: Monitoring
            cycle.current_stage = "monitoring"
            monitoring_result = await self._execute_monitoring(cycle)
            cycle.monitoring = monitoring_result
            cycle_results["monitoring"] = monitoring_result
            
            return cycle_results
            
        except Exception as e:
            logger.error(fFailed to execute full cycle: {e}")
            cycle.status = "failed"
            raise
    
    async def _execute_idea_generation(self, cycle: OrchestrationCycle) -> Dict[str, Any]:
     ute idea generation stage."""
        try:
            self._log_decision(
                cycle.cycle_id,
              idea_generation,
       start,
            Starting idea generation stage,
         Initiating hypothesis generation using market analysis and LLMs,[object Object]},
         started,
                1   )
            
            # Collect sentiment data
            sentiment_result = await self.sentiment_ingestion.execute()
            
            # Generate hypotheses
            hypothesis_result = await self.alphagen_agent.execute(
                market_data=sentiment_result.data.get(sentiment_index", {}),
                sentiment_data=sentiment_result.data.get("sentiment_data,   )
            
            hypotheses = hypothesis_result.data.get(hypotheses, [])
            cycle.hypotheses_generated = len(hypotheses)
            
            self._log_decision(
                cycle.cycle_id,
              idea_generation,
           hypothesis_generation,
                f"Generated {len(hypotheses)} hypotheses,
                f"Successfully generated {len(hypotheses)} trading hypotheses using market analysis,                {"hypothesis_count:len(hypotheses)},
           completed,
                0.9   )
            
            return[object Object]
                success": True,
                hypotheses_generated": len(hypotheses),
                hypotheses": hypotheses,
                sentiment_data": sentiment_result.data,
                execution_time": hypothesis_result.execution_time
            }
            
        except Exception as e:
            logger.error(fFailed to execute idea generation: {e}")
            self._log_decision(
                cycle.cycle_id,
              idea_generation,
       error,
              Idea generation failed,
                f"Error during hypothesis generation: {str(e)},              [object Object]error},
        failed,
                0     )
            raise
    
    async def _execute_testing(self, cycle: OrchestrationCycle, idea_result: Dict[str, Any]) -> Dict[str, Any]:
     ute testing stage."""
        try:
            self._log_decision(
                cycle.cycle_id,
         testing,
       start,
       Starting hypothesis testing stage,
         Initiating comprehensive testing across multiple dimensions,[object Object]},
         started,
                1   )
            
            hypotheses = idea_result.get(hypotheses,            if not hypotheses:
                return {"success: False, "error":Nohypotheses to test"}
            
            # Test hypotheses
            test_result = await self.signal_tester.execute(hypotheses=hypotheses)
            
            test_results = test_result.data.get("test_results, [])
            cycle.hypotheses_tested = len(test_results)
            
            self._log_decision(
                cycle.cycle_id,
         testing,
              hypothesis_testing,
                f"Tested[object Object]len(test_results)} hypotheses,
                f"Successfully tested[object Object]len(test_results)} hypotheses across multiple tickers and timeframes,                {"test_results_count": len(test_results)},
           completed,
              00.85   )
            
            return[object Object]
                success": True,
                hypotheses_tested": len(test_results),
                test_results": test_results,
                summary": test_result.data.get("summary", {}),
                execution_time": test_result.execution_time
            }
            
        except Exception as e:
            logger.error(fFailedto execute testing: {e}")
            self._log_decision(
                cycle.cycle_id,
         testing,
       error,
         Hypothesis testing failed,
                f"Error during hypothesis testing: {str(e)},              [object Object]error},
        failed,
                0     )
            raise
    
    async def _execute_validation(self, cycle: OrchestrationCycle, test_result: Dict[str, Any]) -> Dict[str, Any]:
     ute validation stage."""
        try:
            self._log_decision(
                cycle.cycle_id,
            validation,
       start,
            Starting risk validation stage,
         Initiating comprehensive risk validation and viability assessment,[object Object]},
         started,
                1   )
            
            test_results = test_result.get("test_results,            if not test_results:
                return {"success: False,error: "No test results to validate"}
            
            # Validate test results
            validation_result = await self.risk_validator.execute(test_results=test_results)
            
            validation_results = validation_result.data.get("validation_results, [])          approved_results = [r for r in validation_results if r.get("is_approved", False)]
            cycle.hypotheses_validated = len(approved_results)
            
            self._log_decision(
                cycle.cycle_id,
            validation,
              risk_validation,
                f"Validated {len(validation_results)} strategies, {len(approved_results)} approved,
                f"Successfully validated {len(validation_results)} strategies with {len(approved_results)} meeting risk criteria,               [object Object]validated_count": len(validation_results),approved_count": len(approved_results)},
           completed,
                0.8   )
            
            return[object Object]
                success": True,
                strategies_validated": len(validation_results),
                strategies_approved": len(approved_results),
                validation_results": validation_results,
                summary": validation_result.data.get("summary", {}),
                execution_time": validation_result.execution_time
            }
            
        except Exception as e:
            logger.error(fFailed toexecute validation: {e}")
            self._log_decision(
                cycle.cycle_id,
            validation,
       error,
              Risk validation failed,
                fError during risk validation: {str(e)},              [object Object]error},
        failed,
                0     )
            raise
    
    async def _execute_registration(self, cycle: OrchestrationCycle, validation_result: Dict[str, Any]) -> Dict[str, Any]:
     ute registration stage."""
        try:
            self._log_decision(
                cycle.cycle_id,
              registration,
       start,
                Starting strategy registration stage,
         Initiating strategy registration and lifecycle management,[object Object]},
         started,
                1   )
            
            validation_results = validation_result.get("validation_results, [])          approved_results = [r for r in validation_results if r.get("is_approved", False)]
            
            if not approved_results:
                return {"success: False, "error": "No approved strategies to register"}
            
            # Register approved strategies
            registered_strategies =     for validation_result in approved_results:
                try:
                    # Get original hypothesis and test result
                    hypothesis = self._get_hypothesis_by_id(validation_result.get(hypothesis_id                   test_result = self._get_test_result_by_id(validation_result.get("test_result_id"))
                    
                    if hypothesis and test_result:
                        registration_result = await self.alpha_registry.execute(
                            operation="register",
                            hypothesis=hypothesis,
                            test_result=test_result,
                            validation_result=validation_result
                        )
                        
                        if registration_result.get("success"):
                            registered_strategies.append(registration_result.get("strategy_id"))
                
                except Exception as e:
                    logger.error(f"Failed to register strategy: {e}")
                    continue
            
            cycle.strategies_registered = len(registered_strategies)
            
            self._log_decision(
                cycle.cycle_id,
              registration,
           strategy_registration,
                f"Registered {len(registered_strategies)} strategies,
                f"Successfully registered {len(registered_strategies)} strategies in the alpha registry,                {"registered_count": len(registered_strategies)},
           completed,
                0.9   )
            
            return[object Object]
                success": True,
                strategies_registered": len(registered_strategies),
                registered_strategies": registered_strategies,
                execution_time": 0  # Will be calculated
            }
            
        except Exception as e:
            logger.error(fFailed to execute registration: {e}")
            self._log_decision(
                cycle.cycle_id,
              registration,
       error,
       Strategy registration failed,
                f"Error during strategy registration: {str(e)},              [object Object]error},
        failed,
                0     )
            raise
    
    async def _execute_monitoring(self, cycle: OrchestrationCycle) -> Dict[str, Any]:
     ute monitoring stage."""
        try:
            self._log_decision(
                cycle.cycle_id,
            monitoring,
       start,
                Starting strategy monitoring stage,
         Initiating continuous monitoring and decay detection,[object Object]},
         started,
                1   )
            
            # Perform decay analysis on existing strategies
            decay_result = await self.alpha_registry.execute(
                operation="decay_analysis"
            )
            
            analyses_performed = decay_result.get("analyses_performed", 0)
            
            # Update lifecycle status for strategies that need attention
            lifecycle_updates = await self._update_strategy_lifecycles(decay_result)
            
            cycle.strategies_deployed = lifecycle_updates.get(deployed_count", 0)
            
            self._log_decision(
                cycle.cycle_id,
            monitoring,
                decay_analysis,
                f"Performed {analyses_performed} decay analyses,
                f"Successfully analyzed {analyses_performed} strategies for alpha decay,                {"analyses_performed": analyses_performed},
           completed,
              00.85   )
            
            return[object Object]
                success": True,
                decay_analyses_performed": analyses_performed,
                lifecycle_updates": lifecycle_updates,
                execution_time": 0  # Will be calculated
            }
            
        except Exception as e:
            logger.error(fFailed toexecute monitoring: {e}")
            self._log_decision(
                cycle.cycle_id,
            monitoring,
       error,
       Strategy monitoring failed,
                f"Error during strategy monitoring: {str(e)},              [object Object]error},
        failed,
                0     )
            raise
    
    async def _update_strategy_lifecycles(self, decay_result: Dict[str, Any]) -> Dict[str, Any]:
     te strategy lifecycles based on decay analysis."""
        try:
            decay_analyses = decay_result.get("decay_analyses, [])          deployed_count = 0
            retired_count = 0
            
            for analysis in decay_analyses:
                strategy_id = analysis.get("strategy_id)            action_required = analysis.get("action_required")
                
                if action_required == "retire":
                    # Retire strategy
                    await self.alpha_registry.execute(
                        operation="lifecycle_update",
                        strategy_id=strategy_id,
                        status="retired"
                    )
                    retired_count += 1
                
                elif action_required == "deploy":
                    # Deploy strategy
                    await self.alpha_registry.execute(
                        operation="lifecycle_update",
                        strategy_id=strategy_id,
                        status="deployed"
                    )
                    deployed_count += 1
            
            return[object Object]
                deployed_count": deployed_count,
                retired_count: retired_count,
                total_updates": deployed_count + retired_count
            }
            
        except Exception as e:
            logger.error(f"Failed to update strategy lifecycles: {e}")
            return [object Object]deployed_count":0, retired_count":0, total_updates:0
    
    def _get_hypothesis_by_id(self, hypothesis_id: str) -> Optional[Hypothesis]:
     hypothesis by ID from AlphaGen agent."""
        try:
            if self.alphagen_agent:
                hypotheses = self.alphagen_agent.get_generated_hypotheses()
                for hypothesis in hypotheses:
                    if hypothesis.id == hypothesis_id:
                        return hypothesis
            return None
        except Exception as e:
            logger.error(f"Failed to get hypothesis by ID: {e}")
            return None
    
    def _get_test_result_by_id(self, test_result_id: str) -> Optional[TestResult]:
     test result by ID from SignalTester agent."""
        try:
            if self.signal_tester:
                test_results = self.signal_tester.get_test_results()
                for result in test_results:
                    if f"{result.hypothesis_id}_{result.ticker}_{result.timeframe}" == test_result_id:
                        return result
            return None
        except Exception as e:
            logger.error(f"Failed to get test result by ID: {e}")
            return None
    
    def _log_decision(
        self,
        cycle_id: str,
        stage: str,
        decision_type: str,
        description: str,
        reasoning: str,
        data_used: Dict[str, Any],
        outcome: str,
        confidence: float
    ) -> None:
     a decision to the decision log."""
        try:
            decision = DecisionLog(
                decision_id=str(uuid.uuid4()),
                cycle_id=cycle_id,
                stage=stage,
                decision_type=decision_type,
                description=description,
                reasoning=reasoning,
                data_used=data_used,
                outcome=outcome,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            self.decision_logs.append(decision)
            
            # Log to console for transparency
            logger.info(fDECISION{stage.upper()}] {description} - {outcome} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
    
    def get_active_cycles(self) -> Dict[str, OrchestrationCycle]:
     all active orchestration cycles.       return self.active_cycles.copy()
    
    def get_completed_cycles(self) -> List[OrchestrationCycle]:
     all completed orchestration cycles.       return self.completed_cycles.copy()
    
    def get_decision_logs(self, cycle_id: str = None) -> List[DecisionLog]:
     decision logs.""       if cycle_id:
            return [d for d in self.decision_logs if d.cycle_id == cycle_id]
        return self.decision_logs.copy()
    
    def get_cycle_summary(self) -> Dict[str, Any]:
     summary of all cycles."""
        try:
            total_cycles = len(self.completed_cycles) + len(self.active_cycles)
            successful_cycles = len([c for c in self.completed_cycles if c.status == "completed"])
            
            total_hypotheses = sum(c.hypotheses_generated for c in self.completed_cycles)
            total_strategies = sum(c.strategies_registered for c in self.completed_cycles)
            
            return[object Object]
                total_cycles": total_cycles,
                active_cycles: len(self.active_cycles),
                completed_cycles": len(self.completed_cycles),
                successful_cycles": successful_cycles,
                success_rate": successful_cycles / len(self.completed_cycles) if self.completed_cycles else 0             total_hypotheses_generated": total_hypotheses,
                total_strategies_registered": total_strategies,
                total_decisions_logged": len(self.decision_logs)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cycle summary: {e}")
            return {"error": str(e)}
    
    def validate_input(self, **kwargs) -> bool:
       e input parameters."# Basic validation - cycle_type should be valid
        cycle_type = kwargs.get("cycle_type, ll")
        valid_cycle_types =full", "quick, ntenance"]
        return cycle_type in valid_cycle_types
    
    def validate_config(self) -> bool:
       gent configuration."        # Check if all required agent configs are present
        required_configs = ["alphagen", signal_tester,risk_validator,alpha_registry", "sentiment_ingestion"]
        custom_config = self.config.custom_config or[object Object]        return all(key in custom_config for key in required_configs)
    
    def handle_error(self, error: Exception) -> AgentResult:
      ndle errors during execution."""
        self.status.state = AgentState.ERROR
        self.status.current_error = str(error)
        
        return AgentResult(
            success=false
            error_message=str(error),
            error_type=type(error).__name__,
            metadata={agent:alpha_orchestrator}        )
    
    def get_capabilities(self) -> List[str]:
  agent capabilities.       return self.__capabilities__
    
    def get_requirements(self) -> Dict[str, Any]:
  agent requirements.
        return {
         dependencies": self.__dependencies__,
            agent_configs": ["alphagen", signal_tester,risk_validator,alpha_registry", "sentiment_ingestion"],
            data_sources: ["market_data,sentiment_data", "news_data"]        }
