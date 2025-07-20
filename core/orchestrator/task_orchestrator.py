"""
Task Orchestrator

This module provides a centralized task orchestration system for the Evolve trading platform.
It uses modular components for scheduling, execution, monitoring, and condition checking.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

# Local imports
try:
    from utils.common_helpers import safe_json_save, load_config
    from utils.cache_utils import cache_result
except ImportError:
    # Fallback implementations if utils not available
    def safe_json_save(file_path: str, data: Any):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_config(config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def cache_result(func):
        return func

# Modular imports
from .task_models import TaskConfig, TaskExecution, AgentStatus, TaskStatus, TaskType
from .task_scheduler import TaskScheduler
from .task_executor import TaskExecutor
from .task_monitor import TaskMonitor
from .task_conditions import TaskConditions
from .task_providers import create_task_provider, TaskProvider


class TaskOrchestrator:
    """
    Centralized task orchestration system for Evolve platform
    """
    
    def __init__(self, config_path: str = "config/task_schedule.yaml"):
        # Load configuration
        self.config_path = config_path
        self.config = self._load_task_config()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory
        self.logs_dir = Path("logs/orchestrator")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scheduler = TaskScheduler(self.config.get("scheduler_config", {}))
        self.executor = TaskExecutor(self.config.get("executor_config", {}))
        self.monitor = TaskMonitor(self.config.get("monitor_config", {}))
        self.conditions = TaskConditions(self.config.get("conditions_config", {}))
        
        # Initialize task providers
        self.task_providers: Dict[str, TaskProvider] = {}
        self._initialize_task_providers()
        
        # Execution state
        self.is_running = False
        
        # Load task configurations
        self._load_tasks()
        
        # Initialize scheduling
        self._initialize_scheduling()
    
    def _load_task_config(self) -> Dict[str, Any]:
        """Load task configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Create default configuration
                default_config = self._create_default_config()
                self._save_task_config(default_config)
                return default_config
        except Exception as e:
            self.logger.error(f"Failed to load task config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default task configuration"""
        return {
            'orchestrator': {
                'enabled': True,
                'max_concurrent_tasks': 5,
                'default_timeout_minutes': 15,
                'health_check_interval_minutes': 5,
                'performance_monitoring': True,
                'error_alerting': True
            },
            'scheduler_config': {
                'max_workers': 10
            },
            'executor_config': {
                'max_workers': 10
            },
            'monitor_config': {
                'health_check_interval_minutes': 5,
                'performance_threshold': 0.8,
                'error_threshold': 5
            },
            'conditions_config': {},
            'tasks': {
                'model_innovation': {
                    'enabled': True,
                    'interval_minutes': 120,
                    'priority': 'high',
                    'dependencies': [],
                    'conditions': {'market_hours': True}
                },
                'strategy_research': {
                    'enabled': True,
                    'interval_minutes': 60,
                    'priority': 'medium',
                    'dependencies': [],
                    'conditions': {'market_hours': True}
                },
                'sentiment_fetch': {
                    'enabled': True,
                    'interval_minutes': 30,
                    'priority': 'medium',
                    'dependencies': [],
                    'conditions': {}
                },
                'meta_control': {
                    'enabled': True,
                    'interval_minutes': 15,
                    'priority': 'high',
                    'dependencies': ['sentiment_fetch'],
                    'conditions': {'system_health': 0.7}
                },
                'risk_management': {
                    'enabled': True,
                    'interval_minutes': 5,
                    'priority': 'critical',
                    'dependencies': [],
                    'conditions': {}
                },
                'execution': {
                    'enabled': True,
                    'interval_minutes': 1,
                    'priority': 'critical',
                    'dependencies': ['risk_management'],
                    'conditions': {'market_hours': True, 'system_health': 0.8}
                },
                'explanation': {
                    'enabled': True,
                    'interval_minutes': 60,
                    'priority': 'low',
                    'dependencies': ['execution'],
                    'conditions': {}
                },
                'system_health': {
                    'enabled': True,
                    'interval_minutes': 5,
                    'priority': 'high',
                    'dependencies': [],
                    'conditions': {}
                },
                'data_sync': {
                    'enabled': True,
                    'interval_minutes': 30,
                    'priority': 'medium',
                    'dependencies': [],
                    'conditions': {}
                },
                'performance_analysis': {
                    'enabled': True,
                    'interval_minutes': 60,
                    'priority': 'low',
                    'dependencies': [],
                    'conditions': {}
                }
            }
        }
    
    def _save_task_config(self, config: Dict[str, Any]):
        """Save task configuration to YAML file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save task config: {e}")
    
    def _initialize_task_providers(self):
        """Initialize task providers"""
        try:
            # Initialize agent task provider
            agent_provider = create_task_provider("agent", self.config.get("agent_provider_config", {}))
            self.task_providers["agent"] = agent_provider
            
            # Initialize system task provider
            system_provider = create_task_provider("system", self.config.get("system_provider_config", {}))
            self.task_providers["system"] = system_provider
            
            # Register providers with executor
            self.executor.register_task_provider("agent", agent_provider.execute_task)
            self.executor.register_task_provider("system", system_provider.execute_task)
            
            self.logger.info("Initialized task providers")
        except Exception as e:
            self.logger.error(f"Failed to initialize task providers: {e}")
    
    def _load_tasks(self):
        """Load task configurations and build dynamic DAG"""
        try:
            tasks_config = self.config.get('tasks', {})
            
            # Build task dependency graph
            task_graph = {}
            
            for task_name, task_data in tasks_config.items():
                task_config = TaskConfig(
                    name=task_name,
                    task_type=TaskType(task_name),
                    enabled=task_data.get('enabled', True),
                    interval_minutes=task_data.get('interval_minutes', 60),
                    priority=task_data.get('priority', 'medium'),
                    dependencies=task_data.get('dependencies', []),
                    conditions=task_data.get('conditions', {}),
                    parameters=task_data.get('parameters', {}),
                    retry_attempts=task_data.get('retry_attempts', 3),
                    retry_delay=task_data.get('retry_delay', 60),
                    skip_on_failure=task_data.get('skip_on_failure', False)
                )
                
                task_graph[task_name] = task_config
                self.scheduler.add_task(task_config)
            
            # Validate DAG and detect cycles
            self._validate_task_dependencies(task_graph)
            
            # Build execution order
            self.execution_order = self._build_execution_order(task_graph)
            
            self.logger.info(f"Loaded {len(tasks_config)} task configurations with dynamic DAG")
        except Exception as e:
            self.logger.error(f"Failed to load tasks: {e}")
    
    def _validate_task_dependencies(self, task_graph: Dict[str, TaskConfig]) -> None:
        """Validate task dependencies and detect cycles."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
                
            visited.add(node)
            rec_stack.add(node)
            
            for dep in task_graph[node].dependencies:
                if dep in task_graph and has_cycle(dep):
                    return True
                    
            rec_stack.remove(node)
            return False
        
        for task_name in task_graph:
            if has_cycle(task_name):
                raise ValueError(f"Circular dependency detected involving task: {task_name}")
    
    def _build_execution_order(self, task_graph: Dict[str, TaskConfig]) -> List[List[str]]:
        """Build execution order using topological sort."""
        # Calculate in-degrees
        in_degree = {task: 0 for task in task_graph}
        for task_name, task_config in task_graph.items():
            for dep in task_config.dependencies:
                if dep in task_graph:
                    in_degree[task_name] += 1
        
        # Topological sort
        execution_levels = []
        queue = [task for task, degree in in_degree.items() if degree == 0]
        
        while queue:
            level = []
            next_queue = []
            
            for task in queue:
                level.append(task)
                
                # Reduce in-degree for dependent tasks
                for task_name, task_config in task_graph.items():
                    if task in task_config.dependencies:
                        in_degree[task_name] -= 1
                        if in_degree[task_name] == 0:
                            next_queue.append(task_name)
            
            execution_levels.append(level)
            queue = next_queue
        
        return execution_levels
    
    def _initialize_scheduling(self):
        """Initialize task scheduling"""
        try:
            self.scheduler.start()
            self.logger.info("Initialized task scheduling")
        except Exception as e:
            self.logger.error(f"Failed to initialize scheduling: {e}")
    
    async def start(self):
        """Start the orchestrator"""
        try:
            self.is_running = True
            self.logger.info("Starting Task Orchestrator")
            
            # Start monitoring loops
            asyncio.create_task(self._health_monitor_loop())
            asyncio.create_task(self._performance_monitor_loop())
            
            self.logger.info("Task Orchestrator started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start orchestrator: {e}")
            self.is_running = False
    
    async def stop(self):
        """Stop the orchestrator"""
        try:
            self.is_running = False
            self.scheduler.stop()
            self.executor.executor.shutdown(wait=True)
            self.logger.info("Task Orchestrator stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop orchestrator: {e}")
    
    async def _health_monitor_loop(self):
        """Health monitoring loop"""
        while self.is_running:
            try:
                health_status = await self.monitor.check_system_health()
                
                # Log health status
                if health_status.get("overall_health", 1.0) < 0.7:
                    self.logger.warning(f"System health degraded: {health_status}")
                
                await asyncio.sleep(self.config.get("health_check_interval_minutes", 5) * 60)
            except Exception as e:
                self.logger.error(f"Health monitor loop error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                await self.monitor.update_performance_metrics()
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                self.logger.error(f"Performance monitor loop error: {e}")
                await asyncio.sleep(60)
    
    async def execute_task_now(self, task_name: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Execute a task immediately"""
        try:
            task_id = await self.executor.execute_task_now(task_name, parameters)
            self.logger.info(f"Executed task immediately: {task_name} (ID: {task_id})")
            return task_id
        except Exception as e:
            self.logger.error(f"Failed to execute task {task_name}: {e}")
            raise
    
    def get_task_status(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a task"""
        return self.executor.get_task_status(task_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            return {
                "orchestrator_status": "running" if self.is_running else "stopped",
                "scheduled_tasks": self.scheduler.get_scheduled_tasks(),
                "current_executions": self.executor.get_current_executions(),
                "agent_status": self.monitor.get_agent_status(),
                "performance_summary": self.monitor.get_performance_summary(),
                "system_health": asyncio.run(self.monitor.check_system_health()),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    def update_task_config(self, task_name: str, updates: Dict[str, Any]):
        """Update task configuration"""
        try:
            self.scheduler.update_task(task_name, updates)
            self.logger.info(f"Updated task configuration: {task_name}")
        except Exception as e:
            self.logger.error(f"Failed to update task config {task_name}: {e}")
    
    def export_status_report(self) -> str:
        """Export status report"""
        try:
            report_data = self.get_system_status()
            
            report_file = self.logs_dir / f"orchestrator_status_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            safe_json_save(str(report_file), report_data)
            
            self.logger.info(f"Exported status report: {report_file}")
            return str(report_file)
        except Exception as e:
            self.logger.error(f"Failed to export status report: {e}")
            return ""


def create_task_orchestrator(config_path: str = "config/task_schedule.yaml") -> TaskOrchestrator:
    """Factory function to create a task orchestrator"""
    return TaskOrchestrator(config_path)


async def start_orchestrator(config_path: str = "config/task_schedule.yaml") -> TaskOrchestrator:
    """Start the orchestrator with the given configuration"""
    orchestrator = create_task_orchestrator(config_path)
    await orchestrator.start()
    return orchestrator


if __name__ == "__main__":
    async def main():
        orchestrator = await start_orchestrator()
        try:
            # Keep the orchestrator running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await orchestrator.stop()
    
    asyncio.run(main())
