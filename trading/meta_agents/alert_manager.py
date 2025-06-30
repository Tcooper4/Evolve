"""
Alert Manager

This module implements a system for managing alerts and notifications based on
monitored metrics and predefined rules.

Note: This module was adapted from the legacy automation/monitoring/alert_manager.py file.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import redis
from pathlib import Path
from dataclasses import dataclass
from trading.metrics_collector import MetricsCollector

@dataclass
class AlertRule:
    """Alert rule model."""
    id: str
    name: str
    metric: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: str
    enabled: bool = True
    created_at: datetime = None
    last_triggered: Optional[datetime] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def to_dict(self) -> Dict:
        """Convert rule to dictionary."""
        return {'success': True, 'result': {, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            "id": self.id,
            "name": self.name,
            "metric": self.metric,
            "condition": self.condition,
            "threshold": self.threshold,
            "duration": self.duration,
            "severity": self.severity,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AlertRule':
        """Create rule from dictionary."""
        return {'success': True, 'result': cls(, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            id=data["id"],
            name=data["name"],
            metric=data["metric"],
            condition=data["condition"],
            threshold=data["threshold"],
            duration=data["duration"],
            severity=data["severity"],
            enabled=data["enabled"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_triggered=datetime.fromisoformat(data["last_triggered"]) if data.get("last_triggered") else None,
            metadata=data.get("metadata", {})
        )

class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(
        self,
        config: Dict,
        metrics_collector: MetricsCollector,
        redis_client: Optional[redis.Redis] = None
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    ):
        """Initialize alert manager."""
        self.config = config
        self.metrics_collector = metrics_collector
        self.redis_client = redis_client
        self.setup_logging()
        self.http_session = None
        
        # Default alert rules
        self.default_rules = [
            AlertRule(
                id="high_cpu",
                name="High CPU Usage",
                metric="cpu_usage",
                condition=">",
                threshold=80.0,
                duration=300,  # 5 minutes
                severity="warning"
            ),
            AlertRule(
                id="high_memory",
                name="High Memory Usage",
                metric="memory_usage",
                condition=">",
                threshold=85.0,
                duration=300,
                severity="warning"
            ),
            AlertRule(
                id="high_disk",
                name="High Disk Usage",
                metric="disk_usage",
                condition=">",
                threshold=90.0,
                duration=300,
                severity="critical"
            )
        ]
        
    def setup_logging(self):
        """Setup logging for the alert manager."""
        log_path = Path("logs/alerts")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "alert_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    async def initialize(self) -> None:
        """Initialize HTTP session."""
        try:
            self.http_session = aiohttp.ClientSession()
            self.logger.info("Alert manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize alert manager: {str(e)}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup HTTP session."""
        try:
            if self.http_session:
                await self.http_session.close()
            self.logger.info("Alert manager cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup alert manager: {str(e)}")
            raise
            
    async def start_monitoring(self):
        """Start alert monitoring."""
        try:
            # Initialize default rules
            for rule in self.default_rules:
                await self.create_alert_rule(rule)
                
            while True:
                await self.check_alerts()
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Error in alert monitoring: {str(e)}")
            raise
            
    async def create_alert_rule(self, rule: AlertRule) -> AlertRule:
        """Create alert rule."""
        try:
            if self.redis_client:
                # Store in Redis
                await self.redis_client.set(
                    f"alert_rule:{rule.id}",
                    json.dumps(rule.to_dict())
                )
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Error creating alert rule: {str(e)}")
            raise
            
    async def get_alert_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get alert rule by ID."""
        try:
            if not self.redis_client:
                return None
                
            rule_data = await self.redis_client.get(f"alert_rule:{rule_id}")
            if not rule_data:
                return None
                
            return AlertRule.from_dict(json.loads(rule_data))
            
        except Exception as e:
            self.logger.error(f"Error getting alert rule: {str(e)}")
            return None
            
    async def update_alert_rule(
        self,
        rule_id: str,
        name: Optional[str] = None,
        metric: Optional[str] = None,
        condition: Optional[str] = None,
        threshold: Optional[float] = None,
        duration: Optional[int] = None,
        severity: Optional[str] = None,
        enabled: Optional[bool] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[AlertRule]:
        """Update alert rule."""
        try:
            # Get rule
            rule = await self.get_alert_rule(rule_id)
            if not rule:
                return None
                
            # Update fields
            if name is not None:
                rule.name = name
            if metric is not None:
                rule.metric = metric
            if condition is not None:
                rule.condition = condition
            if threshold is not None:
                rule.threshold = threshold
            if duration is not None:
                rule.duration = duration
            if severity is not None:
                rule.severity = severity
            if enabled is not None:
                rule.enabled = enabled
            if metadata is not None:
                rule.metadata.update(metadata)
                
            if self.redis_client:
                # Store in Redis
                await self.redis_client.set(
                    f"alert_rule:{rule.id}",
                    json.dumps(rule.to_dict())
                )
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Error updating alert rule: {str(e)}")
            raise
            
    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete alert rule."""
        try:
            if not self.redis_client:
                return False
                
            # Check if rule exists
            if not await self.get_alert_rule(rule_id):
                return False
                
            # Delete from Redis
            await self.redis_client.delete(f"alert_rule:{rule_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting alert rule: {str(e)}")
            return False
            
    async def check_alerts(self):
        """Check all alert rules."""
        try:
            if not self.redis_client:
                return
                
            # Get all rules
            rule_ids = await self.redis_client.keys("alert_rule:*")
            
            for rule_id in rule_ids:
                rule = await self.get_alert_rule(rule_id.decode().split(":")[1])
                if not rule or not rule.enabled:
                    continue
                    
                await self.check_rule(rule)
                
        except Exception as e:
            self.logger.error(f"Error checking alerts: {str(e)}")
            
    async def check_rule(self, rule: AlertRule):
        """Check single alert rule."""
        try:
            # Get current metric value
            metrics = await self.metrics_collector.collect_all_metrics()
            metric_value = metrics.get("system", {}).get(rule.metric)
            
            if metric_value is None:
                return
                
            # Check condition
            triggered = False
            if rule.condition == ">":
                triggered = metric_value > rule.threshold
            elif rule.condition == "<":
                triggered = metric_value < rule.threshold
            elif rule.condition == ">=":
                triggered = metric_value >= rule.threshold
            elif rule.condition == "<=":
                triggered = metric_value <= rule.threshold
            elif rule.condition == "==":
                triggered = metric_value == rule.threshold
                
            if triggered:
                await self.trigger_alert(rule, {"value": metric_value})
                
        except Exception as e:
            self.logger.error(f"Error checking rule: {str(e)}")
            
    async def trigger_alert(self, rule: AlertRule, metric: Dict):
        """Trigger an alert."""
        try:
            alert = {
                "rule_id": rule.id,
                "name": rule.name,
                "metric": rule.metric,
                "value": metric["value"],
                "threshold": rule.threshold,
                "severity": rule.severity,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update rule's last triggered time
            rule.last_triggered = datetime.now()
            if self.redis_client:
                await self.redis_client.set(
                    f"alert_rule:{rule.id}",
                    json.dumps(rule.to_dict())
                )
            
            # Store alert in history
            if self.redis_client:
                await self.redis_client.lpush(
                    "alert_history",
                    json.dumps(alert)
                )
                await self.redis_client.ltrim("alert_history", 0, 999)  # Keep last 1000 alerts
            
            # Send notifications
            await self.process_alerts([alert])
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {str(e)}")
            
    async def get_alert_history(
        self,
        rule_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get alert history."""
        try:
            if not self.redis_client:
                return []
                
            # Get alerts from Redis
            alerts = await self.redis_client.lrange("alert_history", 0, -1)
            
            # Parse and filter alerts
            filtered_alerts = []
            for alert_data in alerts:
                alert = json.loads(alert_data)
                
                # Filter by rule ID
                if rule_id and alert["rule_id"] != rule_id:
                    continue
                    
                # Filter by time range
                alert_time = datetime.fromisoformat(alert["timestamp"])
                if start_time and alert_time < start_time:
                    continue
                if end_time and alert_time > end_time:
                    continue
                    
                filtered_alerts.append(alert)
            
            # Sort by timestamp and limit results
            filtered_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            return filtered_alerts[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting alert history: {str(e)}")
            raise
            
    async def process_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Process and send alerts."""
        try:
            for alert in alerts:
                # Send email alert if configured
                if "email" in self.config.get("notifications", {}):
                    await self.send_email_alert(alert)
                
                # Send Slack alert if configured
                if "slack" in self.config.get("notifications", {}):
                    await self.send_slack_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Error processing alerts: {str(e)}")
            
    async def send_email_alert(self, alert: Dict[str, Any]) -> None:
        """Send email alert."""
        try:
            email_config = self.config["notifications"]["email"]
            
            msg = MIMEMultipart()
            msg["From"] = email_config["from"]
            msg["To"] = email_config["to"]
            msg["Subject"] = f"Alert: {alert['name']} - {alert['severity']}"
            
            body = f"""
            Alert Details:
            - Name: {alert['name']}
            - Metric: {alert['metric']}
            - Value: {alert['value']}
            - Threshold: {alert['threshold']}
            - Severity: {alert['severity']}
            - Time: {alert['timestamp']}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
                server.starttls()
                server.login(email_config["username"], email_config["password"])
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
            
    async def send_slack_alert(self, alert: Dict[str, Any]) -> None:
        """Send Slack alert."""
        try:
            slack_config = self.config["notifications"]["slack"]
            
            message = {
                "text": f"*Alert: {alert['name']}*\n"
                       f"Severity: {alert['severity']}\n"
                       f"Metric: {alert['metric']}\n"
                       f"Value: {alert['value']}\n"
                       f"Threshold: {alert['threshold']}\n"
                       f"Time: {alert['timestamp']}"
            }
            
            async with self.http_session.post(
                slack_config["webhook_url"],
                json=message
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to send Slack alert: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {str(e)}")
            
    async def start(self) -> None:
        """Start the alert manager."""
        try:
            await self.initialize()
            self.logger.info("Alert manager started")
            await self.start_monitoring()
        except Exception as e:
            self.logger.error(f"Error starting alert manager: {str(e)}")
            raise
            
    async def stop(self) -> None:
        """Stop the alert manager."""
        try:
            await self.cleanup()
            self.logger.info("Alert manager stopped")
        except Exception as e:
            self.logger.error(f"Error stopping alert manager: {str(e)}")
            raise 