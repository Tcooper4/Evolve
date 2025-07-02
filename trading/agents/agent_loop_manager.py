"""
Agent Loop Manager

This module orchestrates the 3-agent system for autonomous model management:
- ModelBuilderAgent: builds models from scratch
- PerformanceCriticAgent: evaluates model performance
- UpdaterAgent: updates models based on evaluation results

The system runs autonomously with full reasoning loops and state persistence.
"""

import json
import logging
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
import threading
from queue import Queue
import signal
import sys

# Local imports
from trading.agents.model_builder_agent import ModelBuilderAgent, ModelBuildRequest
from trading.agents.performance_critic_agent import PerformanceCriticAgent, ModelEvaluationRequest
from trading.agents.updater_agent import UpdaterAgent, UpdateRequest
from trading.memory.performance_memory import PerformanceMemory
from core.utils.common_helpers import timer, handle_exceptions
from trading.data.data_listener import DataListener
from trading.agents.base_agent_interface import AgentConfig

@dataclass
class AgentLoopState:
    """State of the agent loop."""
    loop_id: str
    start_timestamp: str
    current_cycle: int
    total_models_built: int
    total_models_evaluated: int
    total_models_updated: int
    active_models: List[str]
    failed_operations: List[Dict[str, Any]]
    last_cycle_timestamp: str
    status: str = "running"

@dataclass
class AgentCommunication:
    """Communication between agents."""
    from_agent: str
    to_agent: str
    message_type: str
    data: Dict[str, Any]
    timestamp: str
    priority: str = "normal"

class AgentLoopManager:
    """Manages the autonomous 3-agent loop for model management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Agent Loop Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents with AgentConfig
        self.model_builder = ModelBuilderAgent(AgentConfig(name="ModelBuilderAgent"))
        self.performance_critic = PerformanceCriticAgent(AgentConfig(name="PerformanceCriticAgent"))
        self.updater = UpdaterAgent(AgentConfig(name="UpdaterAgent"))
        
        # Initialize memory
        self.memory = PerformanceMemory()
        
        # State management
        self.state = AgentLoopState(
            loop_id=str(uuid.uuid4()),
            start_timestamp=datetime.now().isoformat(),
            current_cycle=0,
            total_models_built=0,
            total_models_evaluated=0,
            total_models_updated=0,
            active_models=[],
            failed_operations=[],
            last_cycle_timestamp=datetime.now().isoformat()
        )
        
        # Communication queue
        self.communication_queue = Queue()
        
        # Control flags
        self.running = False
        self.paused = False
        
        # Configuration
        self.cycle_interval = self.config.get('cycle_interval', 3600)  # 1 hour
        self.max_models = self.config.get('max_models', 10)
        self.evaluation_threshold = self.config.get('evaluation_threshold', 0.5)
        
        # Data paths
        self.data_dir = Path("data")
        self.state_file = Path("trading/agents/loop_state.json")
        self.communication_file = Path("trading/agents/communication_log.json")
        
        # Create directories
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # DataListener integration
        self.data_listener = DataListener(
            on_price=self._on_price_update,
            on_news=self._on_news_event,
            volatility_threshold=self.config.get('volatility_threshold', 0.05)
        )
        self.data_listener_symbols = self.config.get('data_listener_symbols', ['BTC'])
        self.data_listener_keywords = self.config.get('data_listener_keywords', ['bitcoin', 'fed', 'market'])
        self.data_listener_enabled = self.config.get('data_listener_enabled', True)
        
        self.logger.info("AgentLoopManager initialized")
    
    async def start_loop(self) -> None:
        """Start the autonomous agent loop."""
        self.logger.info("Starting autonomous agent loop")
        self.running = True
        
        # Load previous state
        self._load_state()
        
        # Start DataListener if enabled
        if self.data_listener_enabled:
            self.data_listener.start(self.data_listener_symbols, self.data_listener_keywords)
        
        # Start communication handler
        communication_thread = threading.Thread(target=self._handle_communications, daemon=True)
        communication_thread.start()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running:
                if not self.paused and not self.data_listener.paused:
                    await self._execute_cycle()
                else:
                    self.logger.info("Agent loop paused due to manual pause or real-time data event.")
                
                # Wait for next cycle
                await asyncio.sleep(self.cycle_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        finally:
            await self._shutdown()
    
    async def _execute_cycle(self) -> None:
        """Execute one complete cycle of the agent loop."""
        cycle_start = datetime.now()
        self.state.current_cycle += 1
        self.logger.info(f"Starting cycle {self.state.current_cycle}")
        
        try:
            # Step 1: Model Builder Phase
            await self._model_builder_phase()
            
            # Step 2: Performance Critic Phase
            await self._performance_critic_phase()
            
            # Step 3: Updater Phase
            await self._updater_phase()
            
            # Step 4: State Management
            self._update_cycle_state(cycle_start)
            
            self.logger.info(f"Completed cycle {self.state.current_cycle}")
            
        except Exception as e:
            self.logger.error(f"Error in cycle {self.state.current_cycle}: {str(e)}")
            self._record_failed_operation("cycle_execution", str(e))
    
    async def _model_builder_phase(self) -> None:
        """Execute model builder phase."""
        self.logger.info("Starting Model Builder phase")
        
        # Check if we need new models
        if len(self.state.active_models) < self.max_models:
            # Determine what type of model to build
            model_type = self._determine_model_type_to_build()
            
            # Create build request
            build_request = ModelBuildRequest(
                model_type=model_type,
                data_path=self._get_latest_data_path(),
                target_column='close',
                hyperparameters=self._get_model_hyperparameters(model_type),
                request_id=f"cycle_{self.state.current_cycle}_{model_type}"
            )
            # Use async execute method
            result = await self.model_builder.execute(request=build_request)
            
            if result.build_status == "success":
                self.state.total_models_built += 1
                self.state.active_models.append(result.model_id)
                
                # Send communication to critic
                self._send_communication(
                    "model_builder", "performance_critic",
                    "model_built", {
                        "model_id": result.model_id,
                        "model_type": result.model_type,
                        "model_path": result.model_path,
                        "training_metrics": result.training_metrics
                    }
                )
                
                self.logger.info(f"Built new {model_type} model: {result.model_id}")
            else:
                self._record_failed_operation("model_building", result.error_message)
    
    async def _performance_critic_phase(self) -> None:
        """Execute performance critic phase."""
        self.logger.info("Starting Performance Critic phase")
        
        # Get models that need evaluation
        models_to_evaluate = self._get_models_for_evaluation()
        
        for model_id in models_to_evaluate:
            try:
                # Get model metadata
                model_metadata = self.memory.get_model_metadata(model_id)
                if not model_metadata:
                    continue
                
                # Create evaluation request
                eval_request = ModelEvaluationRequest(
                    model_id=model_id,
                    model_path=model_metadata.get('model_path', ''),
                    model_type=model_metadata.get('model_type', ''),
                    test_data_path=self._get_latest_data_path(),
                    request_id=f"cycle_{self.state.current_cycle}_eval_{model_id}"
                )
                # Use async execute method for evaluation
                result = await self.performance_critic.execute(request=eval_request)
                
                if result.evaluation_status == "success":
                    self.state.total_models_evaluated += 1
                    
                    # Send communication to updater
                    self._send_communication(
                        "performance_critic", "updater",
                        "model_evaluated", {
                            "model_id": model_id,
                            "evaluation_result": asdict(result)
                        }
                    )
                    
                    self.logger.info(f"Evaluated model {model_id}")
                else:
                    self._record_failed_operation("model_evaluation", result.error_message)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_id}: {str(e)}")
                self._record_failed_operation("model_evaluation", str(e))
    
    async def _updater_phase(self) -> None:
        """Execute updater phase."""
        self.logger.info("Starting Updater phase")
        
        # Process communications from critic
        while not self.communication_queue.empty():
            try:
                comm = self.communication_queue.get_nowait()
                
                if (comm.from_agent == "performance_critic" and 
                    comm.to_agent == "updater" and 
                    comm.message_type == "model_evaluated"):
                    
                    evaluation_result = ModelEvaluationResult(**comm.data["evaluation_result"])
                    
                    # Process evaluation and determine if update is needed
                    update_request = self.updater.process_evaluation(evaluation_result)
                    
                    if update_request:
                        # Execute update
                        result = await self.updater.execute(request=update_request)
                        
                        if result.update_status == "success":
                            self.state.total_models_updated += 1
                            
                            # Update active models list
                            if result.original_model_id in self.state.active_models:
                                self.state.active_models.remove(result.original_model_id)
                            self.state.active_models.append(result.new_model_id)
                            
                            self.logger.info(f"Updated model {result.original_model_id} -> {result.new_model_id}")
                        else:
                            self._record_failed_operation("model_update", result.error_message)
                    
            except Exception as e:
                self.logger.error(f"Error in updater phase: {str(e)}")
                self._record_failed_operation("updater_phase", str(e))
    
    def _determine_model_type_to_build(self) -> str:
        """Determine what type of model to build next.
        
        Returns:
            Model type to build
        """
        # Count existing models by type
        model_counts = {}
        for model_id in self.state.active_models:
            metadata = self.memory.get_model_metadata(model_id)
            if metadata:
                model_type = metadata.get('model_type', 'unknown')
                model_counts[model_type] = model_counts.get(model_type, 0) + 1
        
        # Build missing model types or the least represented one
        if 'lstm' not in model_counts:
            return 'lstm'
        elif 'xgboost' not in model_counts:
            return 'xgboost'
        elif 'ensemble' not in model_counts:
            return 'ensemble'
        else:
            # Return the least represented model type
            return min(model_counts, key=model_counts.get)
    
    def _get_model_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameters for model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Hyperparameters
        """
        if model_type == 'lstm':
            return {
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif model_type == 'ensemble':
            return {
                'models': ['lstm', 'xgboost'],
                'weights': [0.5, 0.5],
                'voting_method': 'weighted_average'
            }
        else:
            return {}
    
    def _get_models_for_evaluation(self) -> List[str]:
        """Get list of models that need evaluation.
        
        Returns:
            List of model IDs to evaluate
        """
        models_to_evaluate = []
        
        for model_id in self.state.active_models:
            # Check if model needs evaluation based on time or performance
            last_evaluation = self._get_last_evaluation_time(model_id)
            
            if last_evaluation is None or self._should_evaluate_model(model_id, last_evaluation):
                models_to_evaluate.append(model_id)
        
        return models_to_evaluate
    
    def _get_last_evaluation_time(self, model_id: str) -> Optional[datetime]:
        """Get last evaluation time for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Last evaluation time or None
        """
        evaluation_history = self.performance_critic.get_evaluation_history(model_id)
        if evaluation_history:
            return datetime.fromisoformat(evaluation_history[-1].evaluation_timestamp)
        return None
    
    def _should_evaluate_model(self, model_id: str, last_evaluation: datetime) -> bool:
        """Determine if model should be evaluated.
        
        Args:
            model_id: Model ID
            last_evaluation: Last evaluation time
            
        Returns:
            Whether model should be evaluated
        """
        # Evaluate if more than 24 hours have passed
        time_since_evaluation = datetime.now() - last_evaluation
        return time_since_evaluation > timedelta(hours=24)
    
    def _get_latest_data_path(self) -> str:
        """Get path to latest data.
        
        Returns:
            Path to latest data file
        """
        # This should be configurable and dynamic
        return "data/latest_market_data.csv"
    
    def _send_communication(self, from_agent: str, to_agent: str, 
                          message_type: str, data: Dict[str, Any]) -> None:
        """Send communication between agents.
        
        Args:
            from_agent: Source agent
            to_agent: Target agent
            message_type: Type of message
            data: Message data
        """
        communication = AgentCommunication(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            data=data,
            timestamp=datetime.now().isoformat()
        )
        
        self.communication_queue.put(communication)
        self._log_communication(communication)
    
    def _handle_communications(self) -> None:
        """Handle communications in a separate thread."""
        while self.running:
            try:
                # Process communications (this is handled in the main loop)
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in communication handler: {str(e)}")
    
    def _log_communication(self, communication: AgentCommunication) -> None:
        """Log communication to file.
        
        Args:
            communication: Communication to log
        """
        try:
            log_entry = asdict(communication)
            
            if self.communication_file.exists():
                with open(self.communication_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            log_data.append(log_entry)
            
            with open(self.communication_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging communication: {str(e)}")
    
    def _update_cycle_state(self, cycle_start: datetime) -> None:
        """Update cycle state.
        
        Args:
            cycle_start: Cycle start time
        """
        self.state.last_cycle_timestamp = datetime.now().isoformat()
        self._save_state()
    
    def _record_failed_operation(self, operation_type: str, error_message: str) -> None:
        """Record a failed operation.
        
        Args:
            operation_type: Type of operation
            error_message: Error message
        """
        failed_op = {
            'operation_type': operation_type,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat(),
            'cycle': self.state.current_cycle
        }
        
        self.state.failed_operations.append(failed_op)
        
        # Keep only last 100 failed operations
        if len(self.state.failed_operations) > 100:
            self.state.failed_operations = self.state.failed_operations[-100:]
    
    def _save_state(self) -> None:
        """Save current state to file."""
        try:
            state_dict = asdict(self.state)
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
    
    def _load_state(self) -> None:
        """Load state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_dict = json.load(f)
                
                # Update state with loaded data
                for key, value in state_dict.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
                
                self.logger.info("Loaded previous state")
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Frame object
        """
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    async def _shutdown(self) -> None:
        """Shutdown the agent loop."""
        self.logger.info("Shutting down agent loop...")
        
        self.running = False
        self.state.status = "stopped"
        self._save_state()
        
        # Stop DataListener
        if self.data_listener_enabled:
            self.data_listener.stop()
        
        self.logger.info("Agent loop shutdown complete")
    
    def pause_loop(self) -> None:
        """Pause the agent loop."""
        self.paused = True
        self.logger.info("Agent loop paused")
    
    def resume_loop(self) -> None:
        """Resume the agent loop."""
        self.paused = False
        self.logger.info("Agent loop resumed")
    
    def get_loop_status(self) -> Dict[str, Any]:
        """Get current loop status.
        
        Returns:
            Loop status dictionary
        """
        return {'success': True, 'result': {
            'running': self.running,
            'paused': self.paused,
            'state': asdict(self.state),
            'active_models_count': len(self.state.active_models),
            'queue_size': self.communication_queue.qsize()
        }, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents.
        
        Returns:
            Agent status dictionary
        """
        return {'success': True, 'result': {
            'model_builder': {
                'models_built': self.state.total_models_built,
                'active_models': self.model_builder.list_models()
            },
            'performance_critic': {
                'models_evaluated': self.state.total_models_evaluated,
                'evaluation_history': len(self.performance_critic.evaluation_history)
            },
            'updater': {
                'models_updated': self.state.total_models_updated,
                'active_models': self.updater.get_active_models()
            }
        }, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _on_price_update(self, data: Dict[str, Any]):
        # Optionally log or act on price updates
        if data.get('paused') and not self.paused:
            self.logger.warning("Trading paused due to volatility spike.")
    
    def _on_news_event(self, news: Dict[str, Any]):
        # Optionally log or act on news events
        self.logger.info(f"News event: {news.get('title')}")
        if self.data_listener.paused and not self.paused:
            self.logger.warning("Trading paused due to significant news event.")

# Main entry point
async def main():
    """Main entry point for the agent loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Model Management Agent Loop")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--cycle-interval", type=int, default=3600, help="Cycle interval in seconds")
    parser.add_argument("--max-models", type=int, default=10, help="Maximum number of active models")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    config.update({
        'cycle_interval': args.cycle_interval,
        'max_models': args.max_models
    })
    
    # Create and start agent loop manager
    manager = AgentLoopManager(config)
    
    try:
        await manager.start_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 