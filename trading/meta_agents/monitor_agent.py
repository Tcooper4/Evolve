"""
Monitor Agent

This module implements a specialized agent for monitoring system performance,
collecting metrics, and managing alerts.

Note: This module was adapted from the legacy automation/monitoring/metrics_collector.py file.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from .base_agent import BaseAgent
from .metrics_collector import MetricsCollector
from .alert_manager import AlertManager

class MonitorAgent(BaseAgent):
    """Agent responsible for system monitoring and metrics collection."""
    
    def __init__(self, config: Dict):
        """Initialize the monitor agent."""
        super().__init__(config)
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config, self.metrics_collector)
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for monitoring."""
        log_path = Path("logs/monitoring")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "monitor_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the monitor agent."""
        try:
            await self.metrics_collector.initialize()
            await self.alert_manager.initialize()
            self.logger.info("Monitor agent initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize monitor agent: {str(e)}")
            raise
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        try:
            metrics = await self.metrics_collector.collect_all_metrics()
            self.logger.info("Collected system metrics")
            return metrics
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            raise
    
    async def get_metrics(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical metrics data."""
        try:
            metrics = await self.metrics_collector.get_metrics(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            raise
    
    async def get_metric_summary(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get summary statistics for metrics."""
        try:
            summary = await self.metrics_collector.get_metric_summary(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time
            )
            return summary
        except Exception as e:
            self.logger.error(f"Error getting metric summary: {str(e)}")
            raise
    
    async def create_alert_rule(self, rule: Dict[str, Any]) -> str:
        """Create a new alert rule."""
        try:
            alert_rule = await self.alert_manager.create_alert_rule(rule)
            self.logger.info(f"Created alert rule: {alert_rule.id}")
            return alert_rule.id
        except Exception as e:
            self.logger.error(f"Error creating alert rule: {str(e)}")
            raise
    
    async def get_alert_history(
        self,
        rule_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get alert history."""
        try:
            history = await self.alert_manager.get_alert_history(
                rule_id=rule_id,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            return history
        except Exception as e:
            self.logger.error(f"Error getting alert history: {str(e)}")
            raise
    
    async def monitor_system(self):
        """Monitor system performance and handle alerts."""
        try:
            while True:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Check for alerts
                await self.alert_manager.check_alerts()
                
                # Log system status
                self.logger.info(f"System status: {metrics['system']}")
                
                await asyncio.sleep(self.config.get("monitor_interval", 60))
                
        except Exception as e:
            self.logger.error(f"Error monitoring system: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Start the monitor agent."""
        try:
            await self.initialize()
            self.logger.info("Monitor agent started")
            
            # Start system monitoring
            asyncio.create_task(self.monitor_system())
            
        except Exception as e:
            self.logger.error(f"Error starting monitor agent: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the monitor agent."""
        try:
            await self.metrics_collector.stop()
            await self.alert_manager.stop()
            self.logger.info("Monitor agent stopped")
        except Exception as e:
            self.logger.error(f"Error stopping monitor agent: {str(e)}")
            raise 