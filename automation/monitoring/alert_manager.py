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
from .metrics_collector import MetricsCollector
from ..notifications.notification_manager import NotificationManager

from ..config.config import load_config

logger = logging.getLogger(__name__)

class AlertRule(BaseModel):
    """Alert rule model."""
    id: str
    name: str
    metric: str
    condition: str
    threshold: float
    duration: int  # seconds
    severity: str
    enabled: bool = True
    created_at: datetime
    last_triggered: Optional[datetime] = None
    metadata: Dict = {}

class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        metrics_collector: MetricsCollector,
        notification_manager: NotificationManager
    ):
        """Initialize alert manager."""
        self.redis = redis_client
        self.metrics_collector = metrics_collector
        self.notification_manager = notification_manager
        self.logger = logging.getLogger(__name__)
        
        # Default alert rules
        self.default_rules = [
            AlertRule(
                id="high_cpu",
                name="High CPU Usage",
                metric="cpu_usage",
                condition=">",
                threshold=80.0,
                duration=300,  # 5 minutes
                severity="warning",
                created_at=datetime.now()
            ),
            AlertRule(
                id="high_memory",
                name="High Memory Usage",
                metric="memory_usage",
                condition=">",
                threshold=85.0,
                duration=300,
                severity="warning",
                created_at=datetime.now()
            ),
            AlertRule(
                id="high_disk",
                name="High Disk Usage",
                metric="disk_usage",
                condition=">",
                threshold=90.0,
                duration=300,
                severity="critical",
                created_at=datetime.now()
            )
        ]
        
    def setup_logging(self):
        """Setup logging for the alert manager."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('automation/logs/alert_manager.log'),
                logging.StreamHandler()
            ]
        )
        
    async def initialize(self) -> None:
        """Initialize Redis connection and HTTP session."""
        try:
            # Setup Redis
            redis_config = self.config["redis"]
            self.redis_client = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                decode_responses=True
            )
            
            # Setup HTTP session
            self.http_session = aiohttp.ClientSession()
            
            logger.info("Alert manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize alert manager: {str(e)}")
            raise
            
    async def cleanup(self) -> None:
        """Cleanup Redis connection and HTTP session."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.http_session:
                await self.http_session.close()
            logger.info("Alert manager cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup alert manager: {str(e)}")
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
            # Store in Redis
            await self.redis.set(
                f"alert_rule:{rule.id}",
                rule.json()
            )
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Error creating alert rule: {str(e)}")
            raise
            
    async def get_alert_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get alert rule by ID."""
        try:
            rule_data = await self.redis.get(f"alert_rule:{rule_id}")
            if not rule_data:
                return None
                
            return AlertRule.parse_raw(rule_data)
            
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
                
            # Store in Redis
            await self.redis.set(
                f"alert_rule:{rule.id}",
                rule.json()
            )
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Error updating alert rule: {str(e)}")
            raise
            
    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete alert rule."""
        try:
            # Check if rule exists
            if not await self.get_alert_rule(rule_id):
                return False
                
            # Delete from Redis
            await self.redis.delete(f"alert_rule:{rule_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting alert rule: {str(e)}")
            return False
            
    async def check_alerts(self):
        """Check all alert rules."""
        try:
            # Get all rules
            rule_ids = await self.redis.keys("alert_rule:*")
            
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
            # Get metrics
            metrics = await self.metrics_collector.get_metrics(
                rule.metric,
                start_time=datetime.now() - timedelta(seconds=rule.duration)
            )
            
            if not metrics:
                return
                
            # Check condition
            triggered = False
            for metric in metrics:
                value = metric["value"]
                if rule.condition == ">" and value > rule.threshold:
                    triggered = True
                    break
                elif rule.condition == "<" and value < rule.threshold:
                    triggered = True
                    break
                elif rule.condition == ">=" and value >= rule.threshold:
                    triggered = True
                    break
                elif rule.condition == "<=" and value <= rule.threshold:
                    triggered = True
                    break
                elif rule.condition == "==" and value == rule.threshold:
                    triggered = True
                    break
                    
            if triggered:
                await self.trigger_alert(rule, metrics[-1])
                
        except Exception as e:
            self.logger.error(f"Error checking rule: {str(e)}")
            
    async def trigger_alert(self, rule: AlertRule, metric: Dict):
        """Trigger alert."""
        try:
            # Update last triggered
            rule.last_triggered = datetime.now()
            await self.redis.set(
                f"alert_rule:{rule.id}",
                rule.json()
            )
            
            # Create notification
            await self.notification_manager.create_notification(
                title=f"Alert: {rule.name}",
                message=(
                    f"Metric {rule.metric} is {rule.condition} {rule.threshold}\n"
                    f"Current value: {metric['value']}\n"
                    f"Severity: {rule.severity}"
                ),
                severity=rule.severity,
                metadata={
                    "rule_id": rule.id,
                    "metric": rule.metric,
                    "value": metric["value"],
                    "threshold": rule.threshold
                }
            )
            
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
            # Get notifications
            notifications = await self.notification_manager.get_notifications(
                notification_type="alert",
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Filter by rule
            if rule_id:
                notifications = [
                    n for n in notifications
                    if n.metadata.get("rule_id") == rule_id
                ]
                
            return notifications
            
        except Exception as e:
            self.logger.error(f"Error getting alert history: {str(e)}")
            return []
            
    async def send_email_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert via email."""
        try:
            email_config = self.config["notifications"]["email"]
            
            msg = MIMEMultipart()
            msg["From"] = email_config["from"]
            msg["To"] = email_config["to"]
            msg["Subject"] = f"Alert: {alert['type']} - {alert['level']}"
            
            body = f"""
            Alert Details:
            Type: {alert['type']}
            Level: {alert['level']}
            Message: {alert['message']}
            Timestamp: {alert['timestamp']}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
                server.starttls()
                server.login(email_config["username"], email_config["password"])
                server.send_message(msg)
                
            logger.info(f"Email alert sent: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            raise
            
    async def send_slack_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert via Slack."""
        try:
            slack_config = self.config["notifications"]["slack"]
            
            message = {
                "text": f"*{alert['level'].upper()} Alert*\n"
                       f"Type: {alert['type']}\n"
                       f"Message: {alert['message']}\n"
                       f"Timestamp: {alert['timestamp']}"
            }
            
            async with self.http_session.post(slack_config["webhook_url"], json=message) as response:
                if response.status != 200:
                    raise Exception(f"Failed to send Slack message: {response.status}")
                    
            logger.info(f"Slack alert sent: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            raise
            
    async def process_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Process and send alerts."""
        try:
            for alert in alerts:
                # Store alert in Redis
                await self.redis_client.hset(
                    "alerts",
                    f"{alert['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    json.dumps(alert)
                )
                
                # Send notifications based on alert level
                if alert["level"] == "error":
                    await self.send_email_alert(alert)
                    await self.send_slack_alert(alert)
                elif alert["level"] == "warning":
                    await self.send_slack_alert(alert)
                    
        except Exception as e:
            logger.error(f"Failed to process alerts: {str(e)}")
            raise
            
    async def start(self) -> None:
        """Start the alert manager."""
        try:
            self.running = True
            await self.initialize()
            logger.info("Alert manager started")
            
            while self.running:
                # Get latest metrics
                metrics_data = await self.redis_client.hgetall("metrics_history")
                if metrics_data:
                    latest_metrics = json.loads(metrics_data[max(metrics_data.keys())])
                    
                    # Check metrics and generate alerts
                    alerts = await self.check_metrics(latest_metrics)
                    if alerts:
                        await self.process_alerts(alerts)
                        
                await asyncio.sleep(self.config["monitoring"]["alert_check_interval"])
                
        except Exception as e:
            logger.error(f"Alert manager failed: {str(e)}")
            raise
        finally:
            self.running = False
            await self.cleanup()
            
    async def stop(self) -> None:
        """Stop the alert manager."""
        self.running = False
        logger.info("Alert manager stopped") 