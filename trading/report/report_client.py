"""
Report Client

Client for interacting with the report service and generating reports on demand.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import redis
from pathlib import Path
import sys

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from report.report_generator import ReportGenerator, generate_trade_report

logger = logging.getLogger(__name__)

class ReportClient:
    """
    Client for interacting with the report service.
    
    Provides methods to generate reports on demand and interact with
    the automated report service.
    """
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 **kwargs):
        """
        Initialize the ReportClient.
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database
            **kwargs: Additional arguments for ReportGenerator
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        # Initialize report generator
        self.report_generator = ReportGenerator(**kwargs)
        
        logger.info("ReportClient initialized")
    
    def generate_report(self,
                       trade_data: Dict[str, Any],
                       model_data: Dict[str, Any],
                       strategy_data: Dict[str, Any],
                       symbol: str,
                       timeframe: str,
                       period: str,
                       report_id: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report.
        
        Args:
            trade_data: Trade performance data
            model_data: Model performance data
            strategy_data: Strategy execution data
            symbol: Trading symbol
            timeframe: Timeframe used
            period: Analysis period
            report_id: Unique report identifier
            
        Returns:
            Dictionary containing report data and file paths
        """
        try:
            report_data = self.report_generator.generate_comprehensive_report(
                trade_data=trade_data,
                model_data=model_data,
                strategy_data=strategy_data,
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                report_id=report_id
            )
            
            logger.info(f"Generated report: {report_data['report_id']}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def trigger_forecast_report(self,
                               forecast_data: Dict[str, Any],
                               symbol: str,
                               timeframe: str,
                               period: str) -> str:
        """
        Trigger a forecast report generation.
        
        Args:
            forecast_data: Forecast data
            symbol: Trading symbol
            timeframe: Timeframe used
            period: Analysis period
            
        Returns:
            Event ID for tracking
        """
        try:
            event_id = f"forecast_{int(time.time())}"
            
            event_data = {
                'event_id': event_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'forecast_data': forecast_data
            }
            
            self.redis_client.publish('forecast_completed', json.dumps(event_data))
            
            logger.info(f"Triggered forecast report: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error triggering forecast report: {e}")
            raise
    
    def trigger_strategy_report(self,
                               strategy_data: Dict[str, Any],
                               trade_data: Dict[str, Any],
                               model_data: Dict[str, Any],
                               symbol: str,
                               timeframe: str,
                               period: str) -> str:
        """
        Trigger a strategy report generation.
        
        Args:
            strategy_data: Strategy execution data
            trade_data: Trade performance data
            model_data: Model performance data
            symbol: Trading symbol
            timeframe: Timeframe used
            period: Analysis period
            
        Returns:
            Event ID for tracking
        """
        try:
            event_id = f"strategy_{int(time.time())}"
            
            event_data = {
                'event_id': event_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'strategy_data': strategy_data,
                'trade_data': trade_data,
                'model_data': model_data
            }
            
            self.redis_client.publish('strategy_completed', json.dumps(event_data))
            
            logger.info(f"Triggered strategy report: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error triggering strategy report: {e}")
            raise
    
    def trigger_backtest_report(self,
                               backtest_data: Dict[str, Any],
                               symbol: str,
                               timeframe: str,
                               period: str) -> str:
        """
        Trigger a backtest report generation.
        
        Args:
            backtest_data: Backtest results data
            symbol: Trading symbol
            timeframe: Timeframe used
            period: Analysis period
            
        Returns:
            Event ID for tracking
        """
        try:
            event_id = f"backtest_{int(time.time())}"
            
            event_data = {
                'event_id': event_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'backtest_data': backtest_data
            }
            
            self.redis_client.publish('backtest_completed', json.dumps(event_data))
            
            logger.info(f"Triggered backtest report: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error triggering backtest report: {e}")
            raise
    
    def wait_for_report(self, event_id: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
        """
        Wait for a report to be completed.
        
        Args:
            event_id: Event ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            Report data if completed, None if timeout
        """
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe('report_completed')
            
            start_time = time.time()
            
            for message in pubsub.listen():
                if time.time() - start_time > timeout:
                    logger.warning(f"Timeout waiting for report: {event_id}")
                    break
                
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        if event_data.get('event_id') == event_id:
                            pubsub.close()
                            return event_data
                    except json.JSONDecodeError:
                        continue
            
            pubsub.close()
            return None
            
        except Exception as e:
            logger.error(f"Error waiting for report: {e}")
            return None
    
    def get_recent_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent reports from Redis.
        
        Args:
            limit: Maximum number of reports to return
            
        Returns:
            List of recent report summaries
        """
        try:
            # Get recent report events from Redis
            pattern = "report:*"
            keys = self.redis_client.keys(pattern)
            
            reports = []
            for key in sorted(keys, reverse=True)[:limit]:
                try:
                    report_data = json.loads(self.redis_client.get(key))
                    reports.append(report_data)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            return reports
            
        except Exception as e:
            logger.error(f"Error getting recent reports: {e}")
            return []
    
    def get_report_files(self, report_id: str) -> Dict[str, str]:
        """
        Get file paths for a specific report.
        
        Args:
            report_id: Report ID
            
        Returns:
            Dictionary of file paths
        """
        try:
            # Check if report exists in Redis
            key = f"report:{report_id}"
            report_data = self.redis_client.get(key)
            
            if report_data:
                data = json.loads(report_data)
                return data.get('files', {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting report files: {e}")
            return {}
    
    def list_available_reports(self) -> List[Dict[str, Any]]:
        """
        List all available reports.
        
        Args:
            List of available reports with metadata
        """
        try:
            reports = []
            
            # Check markdown directory
            markdown_dir = Path("reports/markdown")
            if markdown_dir.exists():
                for file in markdown_dir.glob("*.md"):
                    report_info = {
                        'report_id': file.stem,
                        'format': 'markdown',
                        'file_path': str(file),
                        'created': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                    }
                    reports.append(report_info)
            
            # Check HTML directory
            html_dir = Path("reports/html")
            if html_dir.exists():
                for file in html_dir.glob("*.html"):
                    report_info = {
                        'report_id': file.stem,
                        'format': 'html',
                        'file_path': str(file),
                        'created': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                    }
                    reports.append(report_info)
            
            # Check PDF directory
            pdf_dir = Path("reports/pdf")
            if pdf_dir.exists():
                for file in pdf_dir.glob("*.pdf"):
                    report_info = {
                        'report_id': file.stem,
                        'format': 'pdf',
                        'file_path': str(file),
                        'created': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                    }
                    reports.append(report_info)
            
            return reports
            
        except Exception as e:
            logger.error(f"Error listing available reports: {e}")
            return []
    
    def delete_report(self, report_id: str) -> bool:
        """
        Delete a report and its files.
        
        Args:
            report_id: Report ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from Redis
            key = f"report:{report_id}"
            self.redis_client.delete(key)
            
            # Delete files
            deleted = False
            
            # Markdown
            markdown_file = Path(f"reports/markdown/{report_id}.md")
            if markdown_file.exists():
                markdown_file.unlink()
                deleted = True
            
            # HTML
            html_file = Path(f"reports/html/{report_id}.html")
            if html_file.exists():
                html_file.unlink()
                deleted = True
            
            # PDF
            pdf_file = Path(f"reports/pdf/{report_id}.pdf")
            if pdf_file.exists():
                pdf_file.unlink()
                deleted = True
            
            logger.info(f"Deleted report: {report_id}")
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting report: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get report service status.
        
        Returns:
            Service status information
        """
        try:
            # Check if service is running
            heartbeat_key = os.getenv('KEY', '')
            heartbeat = self.redis_client.get(heartbeat_key)
            
            status = {
                'service_name': 'report_service',
                'running': heartbeat is not None,
                'last_heartbeat': float(heartbeat) if heartbeat else None,
                'redis_connected': self.redis_client.ping()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                'service_name': 'report_service',
                'running': False,
                'last_heartbeat': None,
                'redis_connected': False,
                'error': str(e)
            }

# Convenience functions
def generate_quick_report(trade_data: Dict[str, Any],
                         model_data: Dict[str, Any],
                         strategy_data: Dict[str, Any],
                         symbol: str,
                         timeframe: str = "1h",
                         period: str = "7d") -> Dict[str, Any]:
    """
    Generate a quick report without Redis service.
    
    Args:
        trade_data: Trade performance data
        model_data: Model performance data
        strategy_data: Strategy execution data
        symbol: Trading symbol
        timeframe: Timeframe used
        period: Analysis period
        
    Returns:
        Report data
    """
    return generate_trade_report(
        trade_data=trade_data,
        model_data=model_data,
        strategy_data=strategy_data,
        symbol=symbol,
        timeframe=timeframe,
        period=period
    ) 
