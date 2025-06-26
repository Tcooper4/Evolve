"""
Report Service

Redis pub/sub service for automated report generation after forecast and strategy execution.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import redis
from pathlib import Path
import sys
import os

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from report.report_generator import ReportGenerator, generate_trade_report

logger = logging.getLogger(__name__)


class ReportService:
    """
    Redis pub/sub service for automated report generation.
    
    Listens for forecast and strategy completion events and generates
    comprehensive reports automatically.
    """
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 service_name: str = 'report_service',
                 **kwargs):
        """
        Initialize the ReportService.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database
            service_name: Service name for logging
            **kwargs: Additional arguments for ReportGenerator
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.service_name = service_name
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        # Initialize report generator
        self.report_generator = ReportGenerator(**kwargs)
        
        # Service status
        self.running = False
        self.last_heartbeat = time.time()
        
        # Channels to listen to
        self.channels = [
            'forecast_completed',
            'strategy_completed',
            'backtest_completed',
            'model_evaluation_completed'
        ]
        
        logger.info(f"ReportService initialized: {service_name}")
    
    def start(self):
        """Start the report service."""
        try:
            self.running = True
            logger.info(f"Starting {self.service_name}")
            
            # Start heartbeat
            self._start_heartbeat()
            
            # Start listening for events
            self._listen_for_events()
            
        except Exception as e:
            logger.error(f"Error starting {self.service_name}: {e}")
            self.running = False
            raise
    
    def stop(self):
        """Stop the report service."""
        self.running = False
        logger.info(f"Stopping {self.service_name}")
    
    def _start_heartbeat(self):
        """Start heartbeat monitoring."""
        import threading
        
        def heartbeat():
            while self.running:
                try:
                    self.last_heartbeat = time.time()
                    self.redis_client.set(
                        f"service:{self.service_name}:heartbeat",
                        self.last_heartbeat,
                        ex=60
                    )
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    time.sleep(30)
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
    
    def _listen_for_events(self):
        """Listen for Redis events and generate reports."""
        pubsub = self.redis_client.pubsub()
        
        try:
            # Subscribe to channels
            for channel in self.channels:
                pubsub.subscribe(channel)
                logger.info(f"Subscribed to {channel}")
            
            # Listen for messages
            for message in pubsub.listen():
                if not self.running:
                    break
                
                if message['type'] == 'message':
                    try:
                        self._handle_event(message['channel'], message['data'])
                    except Exception as e:
                        logger.error(f"Error handling event: {e}")
                        
        except Exception as e:
            logger.error(f"Error in event listener: {e}")
        finally:
            pubsub.close()
    
    def _handle_event(self, channel: str, data: str):
        """Handle incoming events."""
        try:
            event_data = json.loads(data)
            logger.info(f"Received event on {channel}: {event_data.get('event_id', 'unknown')}")
            
            if channel == 'forecast_completed':
                self._handle_forecast_completed(event_data)
            elif channel == 'strategy_completed':
                self._handle_strategy_completed(event_data)
            elif channel == 'backtest_completed':
                self._handle_backtest_completed(event_data)
            elif channel == 'model_evaluation_completed':
                self._handle_model_evaluation_completed(event_data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding event data: {e}")
        except Exception as e:
            logger.error(f"Error handling event on {channel}: {e}")
    
    def _handle_forecast_completed(self, event_data: Dict[str, Any]):
        """Handle forecast completion event."""
        try:
            # Extract data from event
            forecast_data = event_data.get('forecast_data', {})
            symbol = event_data.get('symbol', 'Unknown')
            timeframe = event_data.get('timeframe', 'Unknown')
            period = event_data.get('period', 'Unknown')
            event_id = event_data.get('event_id', 'unknown')
            
            # Generate model report
            model_data = {
                'predictions': forecast_data.get('predictions', []),
                'actuals': forecast_data.get('actuals', []),
                'model_name': forecast_data.get('model_name', 'Unknown'),
                'model_params': forecast_data.get('model_params', {})
            }
            
            # Create empty trade data for model-only report
            trade_data = {'trades': []}
            strategy_data = {
                'strategy_name': 'Forecast Only',
                'symbol': symbol,
                'timeframe': timeframe,
                'signals': [],
                'market_conditions': {},
                'performance': {},
                'parameters': {}
            }
            
            # Generate report
            report_data = self.report_generator.generate_comprehensive_report(
                trade_data=trade_data,
                model_data=model_data,
                strategy_data=strategy_data,
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                report_id=f"forecast_{event_id}"
            )
            
            # Publish report completion
            self._publish_report_completed(report_data, 'forecast_report')
            
            logger.info(f"Generated forecast report: {report_data['report_id']}")
            
        except Exception as e:
            logger.error(f"Error handling forecast completed: {e}")
    
    def _handle_strategy_completed(self, event_data: Dict[str, Any]):
        """Handle strategy completion event."""
        try:
            # Extract data from event
            strategy_data = event_data.get('strategy_data', {})
            trade_data = event_data.get('trade_data', {})
            model_data = event_data.get('model_data', {})
            symbol = event_data.get('symbol', 'Unknown')
            timeframe = event_data.get('timeframe', 'Unknown')
            period = event_data.get('period', 'Unknown')
            event_id = event_data.get('event_id', 'unknown')
            
            # Generate comprehensive report
            report_data = self.report_generator.generate_comprehensive_report(
                trade_data=trade_data,
                model_data=model_data,
                strategy_data=strategy_data,
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                report_id=f"strategy_{event_id}"
            )
            
            # Publish report completion
            self._publish_report_completed(report_data, 'strategy_report')
            
            logger.info(f"Generated strategy report: {report_data['report_id']}")
            
        except Exception as e:
            logger.error(f"Error handling strategy completed: {e}")
    
    def _handle_backtest_completed(self, event_data: Dict[str, Any]):
        """Handle backtest completion event."""
        try:
            # Extract data from event
            backtest_data = event_data.get('backtest_data', {})
            symbol = event_data.get('symbol', 'Unknown')
            timeframe = event_data.get('timeframe', 'Unknown')
            period = event_data.get('period', 'Unknown')
            event_id = event_data.get('event_id', 'unknown')
            
            # Generate backtest report
            trade_data = {
                'trades': backtest_data.get('trades', []),
                'equity_curve': backtest_data.get('equity_curve', [])
            }
            
            model_data = {
                'predictions': backtest_data.get('predictions', []),
                'actuals': backtest_data.get('actuals', []),
                'model_name': backtest_data.get('model_name', 'Unknown')
            }
            
            strategy_data = {
                'strategy_name': backtest_data.get('strategy_name', 'Backtest'),
                'symbol': symbol,
                'timeframe': timeframe,
                'signals': backtest_data.get('signals', []),
                'market_conditions': backtest_data.get('market_conditions', {}),
                'performance': backtest_data.get('performance', {}),
                'parameters': backtest_data.get('parameters', {})
            }
            
            # Generate report
            report_data = self.report_generator.generate_comprehensive_report(
                trade_data=trade_data,
                model_data=model_data,
                strategy_data=strategy_data,
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                report_id=f"backtest_{event_id}"
            )
            
            # Publish report completion
            self._publish_report_completed(report_data, 'backtest_report')
            
            logger.info(f"Generated backtest report: {report_data['report_id']}")
            
        except Exception as e:
            logger.error(f"Error handling backtest completed: {e}")
    
    def _handle_model_evaluation_completed(self, event_data: Dict[str, Any]):
        """Handle model evaluation completion event."""
        try:
            # Extract data from event
            evaluation_data = event_data.get('evaluation_data', {})
            symbol = event_data.get('symbol', 'Unknown')
            timeframe = event_data.get('timeframe', 'Unknown')
            period = event_data.get('period', 'Unknown')
            event_id = event_data.get('event_id', 'unknown')
            
            # Generate evaluation report
            model_data = {
                'predictions': evaluation_data.get('predictions', []),
                'actuals': evaluation_data.get('actuals', []),
                'model_name': evaluation_data.get('model_name', 'Unknown'),
                'model_params': evaluation_data.get('model_params', {}),
                'metrics': evaluation_data.get('metrics', {})
            }
            
            # Create empty trade data for evaluation-only report
            trade_data = {'trades': []}
            strategy_data = {
                'strategy_name': 'Model Evaluation',
                'symbol': symbol,
                'timeframe': timeframe,
                'signals': [],
                'market_conditions': {},
                'performance': evaluation_data.get('performance', {}),
                'parameters': evaluation_data.get('parameters', {})
            }
            
            # Generate report
            report_data = self.report_generator.generate_comprehensive_report(
                trade_data=trade_data,
                model_data=model_data,
                strategy_data=strategy_data,
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                report_id=f"evaluation_{event_id}"
            )
            
            # Publish report completion
            self._publish_report_completed(report_data, 'evaluation_report')
            
            logger.info(f"Generated evaluation report: {report_data['report_id']}")
            
        except Exception as e:
            logger.error(f"Error handling model evaluation completed: {e}")
    
    def _publish_report_completed(self, report_data: Dict[str, Any], report_type: str):
        """Publish report completion event."""
        try:
            event_data = {
                'event_id': f"report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'report_type': report_type,
                'report_id': report_data['report_id'],
                'symbol': report_data['symbol'],
                'files': report_data.get('files', {}),
                'summary': {
                    'total_pnl': report_data['trade_metrics'].total_pnl,
                    'win_rate': report_data['trade_metrics'].win_rate,
                    'sharpe_ratio': report_data['trade_metrics'].sharpe_ratio,
                    'model_accuracy': report_data['model_metrics'].accuracy
                }
            }
            
            self.redis_client.publish('report_completed', json.dumps(event_data))
            logger.info(f"Published report completion: {report_data['report_id']}")
            
        except Exception as e:
            logger.error(f"Error publishing report completion: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'service_name': self.service_name,
            'running': self.running,
            'last_heartbeat': self.last_heartbeat,
            'channels': self.channels,
            'redis_connected': self.redis_client.ping()
        }


def main():
    """Main function to run the report service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Report Service')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--redis-db', type=int, default=0, help='Redis database')
    parser.add_argument('--service-name', default='report_service', help='Service name')
    parser.add_argument('--openai-api-key', help='OpenAI API key')
    parser.add_argument('--notion-token', help='Notion API token')
    parser.add_argument('--slack-webhook', help='Slack webhook URL')
    parser.add_argument('--output-dir', default='reports', help='Output directory')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create email config if provided
    email_config = {}
    if os.getenv('EMAIL_SMTP_SERVER'):
        email_config = {
            'smtp_server': os.getenv('EMAIL_SMTP_SERVER'),
            'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'from_email': os.getenv('EMAIL_FROM'),
            'to_email': os.getenv('EMAIL_TO')
        }
    
    # Initialize and start service
    service = ReportService(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        service_name=args.service_name,
        openai_api_key=args.openai_api_key or os.getenv('OPENAI_API_KEY'),
        notion_token=args.notion_token or os.getenv('NOTION_TOKEN'),
        slack_webhook=args.slack_webhook or os.getenv('SLACK_WEBHOOK'),
        email_config=email_config,
        output_dir=args.output_dir
    )
    
    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        service.stop()
    except Exception as e:
        logger.error(f"Service error: {e}")
        service.stop()


if __name__ == '__main__':
    main() 