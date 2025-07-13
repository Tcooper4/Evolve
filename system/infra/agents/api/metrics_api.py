import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import psutil
from automation.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class MetricsAPI:
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self._metrics_history: List[Dict] = []
        self._max_history_size = 1000async def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_total': memory.total,
                'memory_available': memory.available,
                'disk_usage': disk.percent,
                'disk_total': disk.total,
                'disk_free': disk.free,
                'active_tasks': len(await self.orchestrator.get_running_tasks()),
                'system_status': self._get_system_status(cpu_percent, memory.percent)
            }

            # Store in history
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history_size:
                self._metrics_history.pop(0)

            return metrics
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            raise

    async def get_task_metrics(self, task_id: str) -> Optional[Dict]:
        """Get metrics for a specific task"""
        try:
            task = await self.orchestrator.get_task(task_id)
            if not task:
                return None

            metrics = {
                'task_id': task.task_id,
                'name': task.name,
                'status': task.status,
                'progress': task.progress,
                'start_time': task.start_time.isoformat() if task.start_time else None,
                'end_time': task.end_time.isoformat() if task.end_time else None,
                'execution_time': (task.end_time - task.start_time).total_seconds() if task.end_time and task.start_time else None,
                'error_message': task.error_message
            }

            return metrics
        except Exception as e:
            logger.error(f"Error getting task metrics: {str(e)}")
            raise

    async def get_agent_metrics(self) -> List[Dict]:
        """Get metrics for all agents"""
        try:
            agents = await self.orchestrator.get_agents()
            metrics = []

            for agent in agents:
                agent_metrics = {
                    'agent_id': agent.agent_id,
                    'name': agent.name,
                    'status': agent.status,
                    'active_tasks': len(agent.get_tasks()),
                    'last_heartbeat': agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                    'cpu_usage': agent.cpu_usage if hasattr(agent, 'cpu_usage') else None,
                    'memory_usage': agent.memory_usage if hasattr(agent, 'memory_usage') else None
                }
                metrics.append(agent_metrics)

            return metrics
        except Exception as e:
            logger.error(f"Error getting agent metrics: {str(e)}")
            raise

    async def get_model_metrics(self) -> List[Dict]:
        """Get metrics for all models"""
        try:
            models = await self.orchestrator.get_models()
            metrics = []

            for model in models:
                model_metrics = {
                    'model_id': model.model_id,
                    'name': model.name,
                    'type': model.type,
                    'status': model.status,
                    'accuracy': model.accuracy if hasattr(model, 'accuracy') else None,
                    'last_trained': model.last_trained.isoformat() if model.last_trained else None,
                    'training_time': model.training_time if hasattr(model, 'training_time') else None
                }
                metrics.append(model_metrics)

            return metrics
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            raise

    async def get_metrics_history(self, limit: int = 100) -> List[Dict]:
        """Get historical metrics data"""
        try:
            return self._metrics_history[-limit:]
        except Exception as e:
            logger.error(f"Error getting metrics history: {str(e)}")
            raise

    def _get_system_status(self, cpu_percent: float, memory_percent: float) -> str:
        """Determine system status based on resource usage"""
        if cpu_percent > 90 or memory_percent > 90:
            return 'error'
        elif cpu_percent > 70 or memory_percent > 70:
            return 'warning'
        else:
            return 'active'

    async def get_all_metrics(self) -> Dict:
        """Get all metrics in one call"""
        try:
            system_metrics = await self.get_system_metrics()
            agent_metrics = await self.get_agent_metrics()
            model_metrics = await self.get_model_metrics()

            return {
                'system': system_metrics,
                'agents': agent_metrics,
                'models': model_metrics,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting all metrics: {str(e)}")
            raise
