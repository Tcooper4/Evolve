"""
Alert Agent

This module implements a specialized agent for managing alerts and delivering
notifications through various channels.

Note: This module was adapted from the legacy automation/monitoring/alert_manager.py file.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from trading.base_agent import BaseAgent
from trading.alert_manager import AlertManager

class AlertAgent(BaseAgent):
    """Agent responsible for managing alerts and notifications."""
    
    def __init__(self, config: Dict):
        """Initialize the alert agent."""
        super().__init__(config)
        self.alert_manager = AlertManager(config)
        self.setup_logging()
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logging(self):
        """Configure logging for alert management."""
        log_path = Path("logs/alerts")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "alert_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    async def initialize(self) -> None:
        """Initialize the alert agent."""
        try:
            await self.alert_manager.initialize()
            self.logger.info("Alert agent initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize alert agent: {str(e)}")
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
    
    async def update_alert_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update an alert rule."""
        try:
            rule = await self.alert_manager.update_alert_rule(rule_id, **updates)
            if rule:
                self.logger.info(f"Updated alert rule: {rule_id}")
            return rule
        except Exception as e:
            self.logger.error(f"Error updating alert rule: {str(e)}")
            raise
    
    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule."""
        try:
            success = await self.alert_manager.delete_alert_rule(rule_id)
            if success:
                self.logger.info(f"Deleted alert rule: {rule_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error deleting alert rule: {str(e)}")
            raise
    
    async def get_alert_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get an alert rule by ID."""
        try:
            rule = await self.alert_manager.get_alert_rule(rule_id)
            return rule
        except Exception as e:
            self.logger.error(f"Error getting alert rule: {str(e)}")
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
    
    async def send_notification(self, alert: Dict[str, Any]) -> None:
        """Send a notification for an alert."""
        try:
            await self.alert_manager.process_alerts([alert])
            self.logger.info(f"Sent notification for alert: {alert['rule_id']}")
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            raise
    
    async def monitor_alerts(self):
        """Monitor and process alerts."""
        try:
            while True:
                # Check for new alerts
                await self.alert_manager.check_alerts()
                
                # Process any pending notifications
                alerts = await self.alert_manager.get_alert_history(limit=10)
                for alert in alerts:
                    if alert.get("status") == "pending":
                        await self.send_notification(alert)
                
                await asyncio.sleep(self.config.get("monitor_interval", 60))
                
        except Exception as e:
            self.logger.error(f"Error monitoring alerts: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Start the alert agent."""
        try:
            await self.initialize()
            self.logger.info("Alert agent started")
            
            # Start alert monitoring
            asyncio.create_task(self.monitor_alerts())
            
        except Exception as e:
            self.logger.error(f"Error starting alert agent: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the alert agent."""
        try:
            await self.alert_manager.stop()
            self.logger.info("Alert agent stopped")
        except Exception as e:
            self.logger.error(f"Error stopping alert agent: {str(e)}")
            raise 