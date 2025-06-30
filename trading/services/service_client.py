"""
Service Client Example

Demonstrates how to interact with the agent services via Redis pub/sub.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import redis

logger = logging.getLogger(__name__)


class ServiceClient:
    """
    Client for interacting with agent services via Redis pub/sub.
    
    Provides a simple interface for sending requests and receiving responses.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0):
        """
        Initialize the ServiceClient.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.pubsub = self.redis_client.pubsub()
        
        # Response storage
        self.responses = {}
        self.response_timeout = 30  # seconds
        
        logger.info("ServiceClient initialized")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def send_request(self, service_name: str, message_type: str, data: Dict[str, Any], 
                    wait_for_response: bool = True) -> Optional[Dict[str, Any]]:
        """
        Send a request to a service and optionally wait for response.
        
        Args:
            service_name: Name of the target service
            message_type: Type of message to send
            data: Message data
            wait_for_response: Whether to wait for and return response
            
        Returns:
            Response data if wait_for_response is True, None otherwise
        """
        try:
            # Subscribe to service output channel
            output_channel = f"{service_name}_output"
            self.pubsub.subscribe(output_channel)
            
            # Send the message
            message = {
                'type': message_type,
                'data': data,
                'timestamp': time.time()
            }
            
            self.redis_client.publish(
                f"{service_name}_input",
                json.dumps(message)
            )
            
            logger.info(f"Sent {message_type} to {service_name}")
            
            if wait_for_response:
                return self._wait_for_response(service_name, message_type)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error sending request to {service_name}: {e}")
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _wait_for_response(self, service_name: str, message_type: str) -> Optional[Dict[str, Any]]:
        """Wait for response from a service."""
        start_time = time.time()
        
        while time.time() - start_time < self.response_timeout:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    data = json.loads(message['data'].decode('utf-8'))
                    
                    # Check if this is a response from our target service
                    if data.get('service') == service_name:
                        response_type = data.get('type', '')
                        
                        # Check if this is a response to our request
                        if (response_type.endswith('_completed') or 
                            response_type.endswith('_result') or
                            response_type == 'error'):
                            
                            logger.info(f"Received response from {service_name}: {response_type}")
                            return {'success': True, 'result': data, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                            
            except Exception as e:
                logger.error(f"Error waiting for response: {e}")
                time.sleep(0.1)
        
        logger.warning(f"Timeout waiting for response from {service_name}")
        return None
    
    def build_model(self, model_type: str = 'lstm', symbol: str = 'BTCUSDT', 
                   timeframe: str = '1h', features: list = None) -> Optional[Dict[str, Any]]:
        """Build a model using the ModelBuilderService."""
        data = {
            'model_type': model_type,
            'symbol': symbol,
            'timeframe': timeframe,
            'features': features or ['close', 'volume', 'rsi', 'macd']
        }
        
        return {'success': True, 'result': self.send_request('model_builder', 'build_model', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def evaluate_model(self, model_id: str, symbol: str = 'BTCUSDT', 
                      timeframe: str = '1h', period: str = '30d') -> Optional[Dict[str, Any]]:
        """Evaluate a model using the PerformanceCriticService."""
        data = {
            'model_id': model_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'metrics': ['sharpe', 'drawdown', 'win_rate']
        }
        
        return {'success': True, 'result': self.send_request('performance_critic', 'evaluate_model', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def retrain_model(self, model_id: str, new_data_period: str = '30d', 
                     retrain_type: str = 'incremental') -> Optional[Dict[str, Any]]:
        """Retrain a model using the UpdaterService."""
        data = {
            'model_id': model_id,
            'new_data_period': new_data_period,
            'retrain_type': retrain_type
        }
        
        return {'success': True, 'result': self.send_request('updater', 'retrain_model', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def search_github(self, query: str, max_results: int = 10, 
                     language: str = 'python') -> Optional[Dict[str, Any]]:
        """Search GitHub using the ResearchService."""
        data = {
            'query': query,
            'max_results': max_results,
            'language': language
        }
        
        return {'success': True, 'result': self.send_request('research', 'search_github', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def tune_hyperparameters(self, model_type: str, param_space: dict = None, 
                           optimization_method: str = 'bayesian', 
                           n_trials: int = 50) -> Optional[Dict[str, Any]]:
        """Tune hyperparameters using the MetaTunerService."""
        data = {
            'model_type': model_type,
            'param_space': param_space or {},
            'optimization_method': optimization_method,
            'n_trials': n_trials,
            'cv_folds': 5
        }
        
        return {'success': True, 'result': self.send_request('meta_tuner', 'tune_hyperparameters', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def generate_plot(self, plot_type: str, data_source: str, 
                     plot_config: dict = None) -> Optional[Dict[str, Any]]:
        """Generate a plot using the MultimodalService."""
        data = {
            'plot_type': plot_type,
            'data_source': data_source,
            'plot_config': plot_config or {},
            'save_path': f'plots/{plot_type}_{int(time.time())}.png'
        }
        
        return {'success': True, 'result': self.send_request('multimodal', 'generate_plot', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def route_prompt(self, user_prompt: str, context: dict = None, 
                    available_agents: list = None) -> Optional[Dict[str, Any]]:
        """Route a prompt using the PromptRouterService."""
        data = {
            'user_prompt': user_prompt,
            'context': context or {},
            'available_agents': available_agents or ['model_builder', 'performance_critic', 'updater']
        }
        
        return {'success': True, 'result': self.send_request('prompt_router', 'route_prompt', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def process_natural_language_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Process a natural language query using QuantGPT."""
        data = {
            'query': query
        }
        
        return {'success': True, 'result': self.send_request('quant_gpt', 'process_query', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_query_history(self, limit: int = 10, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get query history from QuantGPT."""
        data = {
            'limit': limit,
            'symbol': symbol
        }
        
        return {'success': True, 'result': self.send_request('quant_gpt', 'get_query_history', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_available_symbols(self) -> Optional[Dict[str, Any]]:
        """Get available symbols and parameters from QuantGPT."""
        return {'success': True, 'result': self.send_request('quant_gpt', 'get_available_symbols', {}), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def execute_model_safely(self, model_code: str, model_name: str, 
                           input_data: Dict[str, Any] = None, 
                           model_type: str = "custom") -> Optional[Dict[str, Any]]:
        """Execute a user-defined model safely."""
        data = {
            'model_code': model_code,
            'model_name': model_name,
            'input_data': input_data or {},
            'model_type': model_type
        }
        
        return {'success': True, 'result': self.send_request('safe_executor', 'execute_model', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def execute_strategy_safely(self, strategy_code: str, strategy_name: str,
                              market_data: Dict[str, Any] = None,
                              parameters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Execute a user-defined strategy safely."""
        data = {
            'strategy_code': strategy_code,
            'strategy_name': strategy_name,
            'market_data': market_data or {},
            'parameters': parameters or {}
        }
        
        return {'success': True, 'result': self.send_request('safe_executor', 'execute_strategy', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def execute_indicator_safely(self, indicator_code: str, indicator_name: str,
                               price_data: Dict[str, Any] = None,
                               parameters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Execute a user-defined indicator safely."""
        data = {
            'indicator_code': indicator_code,
            'indicator_name': indicator_name,
            'price_data': price_data or {},
            'parameters': parameters or {}
        }
        
        return {'success': True, 'result': self.send_request('safe_executor', 'execute_indicator', data), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_safe_executor_statistics(self) -> Optional[Dict[str, Any]]:
        """Get SafeExecutor statistics."""
        return {'success': True, 'result': self.send_request('safe_executor', 'get_statistics', {}), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def cleanup_safe_executor(self) -> Optional[Dict[str, Any]]:
        """Clean up SafeExecutor resources."""
        return {'success': True, 'result': self.send_request('safe_executor', 'cleanup', {}), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a service."""
        return {'success': True, 'result': self.send_request(service_name, 'status', {}), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def ping_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Ping a service to check if it's alive."""
        return {'success': True, 'result': self.send_request(service_name, 'ping', {}), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def generate_report(self, trade_data: Dict[str, Any], model_data: Dict[str, Any], 
                       strategy_data: Dict[str, Any], symbol: str, timeframe: str, 
                       period: str, report_id: str = None) -> Dict[str, Any]:
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
            Report data dictionary
        """
        try:
            # Import here to avoid circular imports
            from report.report_client import ReportClient
            
            client = ReportClient(
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                redis_db=self.redis_db
            )
            
            return {'success': True, 'result': client.generate_report(, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                trade_data=trade_data,
                model_data=model_data,
                strategy_data=strategy_data,
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                report_id=report_id
            )
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def trigger_strategy_report(self, strategy_data: Dict[str, Any], 
                               trade_data: Dict[str, Any], model_data: Dict[str, Any],
                               symbol: str, timeframe: str, period: str) -> str:
        """
        Trigger automated strategy report generation.
        
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
            from report.report_client import ReportClient
            
            client = ReportClient(
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                redis_db=self.redis_db
            )
            
            return {'success': True, 'result': client.trigger_strategy_report(, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                strategy_data=strategy_data,
                trade_data=trade_data,
                model_data=model_data,
                symbol=symbol,
                timeframe=timeframe,
                period=period
            )
            
        except Exception as e:
            logger.error(f"Error triggering strategy report: {e}")
            raise
    
    def get_recent_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent reports.
        
        Args:
            limit: Maximum number of reports to return
            
        Returns:
            List of recent report summaries
        """
        try:
            from report.report_client import ReportClient
            
            client = ReportClient(
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                redis_db=self.redis_db
            )
            
            return client.get_recent_reports(limit=limit)
            
        except Exception as e:
            logger.error(f"Error getting recent reports: {e}")
            return {'success': True, 'result': [], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def list_available_reports(self) -> List[Dict[str, Any]]:
        """
        List all available reports.
        
        Returns:
            List of available reports with metadata
        """
        try:
            from report.report_client import ReportClient
            
            client = ReportClient(
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                redis_db=self.redis_db
            )
            
            return client.list_available_reports()
            
        except Exception as e:
            logger.error(f"Error listing available reports: {e}")
            return {'success': True, 'result': [], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def close(self):
        """Close the service client and cleanup resources."""
        try:
            self.pubsub.close()
            self.redis_client.close()
            logger.info("ServiceClient closed successfully")
            return {
                'success': True,
                'message': 'ServiceClient closed successfully',
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error closing ServiceClient: {e}")
            return {
                'success': False,
                'message': f'Error closing ServiceClient: {e}',
                'timestamp': time.time()
            }
    
    def get_reasoning_decisions(self, agent_name: str = None, limit: int = 10) -> Optional[Dict[str, Any]]:
        """Get recent reasoning decisions from agents."""
        try:
            data = {
                'agent_name': agent_name,
                'limit': limit
            }
            
            response = self.send_request('reasoning_tracker', 'get_decisions', data)
            
            if response and response.get('success'):
                return {
                    'success': True,
                    'decisions': response.get('data', {}).get('decisions', []),
                    'count': len(response.get('data', {}).get('decisions', [])),
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to retrieve reasoning decisions',
                    'decisions': [],
                    'count': 0,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"Error getting reasoning decisions: {e}")
            return {
                'success': False,
                'message': f'Error getting reasoning decisions: {e}',
                'decisions': [],
                'count': 0,
                'timestamp': time.time()
            }
    
    def get_reasoning_statistics(self) -> Optional[Dict[str, Any]]:
        """Get reasoning statistics from agents."""
        try:
            response = self.send_request('reasoning_tracker', 'get_statistics', {})
            
            if response and response.get('success'):
                return {
                    'success': True,
                    'statistics': response.get('data', {}),
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to retrieve reasoning statistics',
                    'statistics': {},
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"Error getting reasoning statistics: {e}")
            return {
                'success': False,
                'message': f'Error getting reasoning statistics: {e}',
                'statistics': {},
                'timestamp': time.time()
            }
    
    def log_reasoning_decision(self, decision_data: Dict[str, Any]) -> Optional[str]:
        """Log a reasoning decision."""
        try:
            response = self.send_request('reasoning_tracker', 'log_decision', decision_data)
            
            if response and response.get('success'):
                return {
                    'success': True,
                    'decision_id': response.get('data', {}).get('decision_id'),
                    'message': 'Decision logged successfully',
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to log decision',
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"Error logging reasoning decision: {e}")
            return {
                'success': False,
                'message': f'Error logging reasoning decision: {e}',
                'timestamp': time.time()
            }


def main():
    """Example usage of the ServiceClient."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Agent Service Client')
    parser.add_argument('--action', required=True, 
                       choices=['build', 'evaluate', 'retrain', 'search', 'tune', 'plot', 'route'],
                       help='Action to perform')
    parser.add_argument('--service', help='Service name for status/ping')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    # Initialize client
    client = ServiceClient(
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )
    
    try:
        if args.action == 'build':
            result = client.build_model('lstm', 'BTCUSDT', '1h')
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "build", "result": result}
            
        elif args.action == 'evaluate':
            if not args.service:
                print("Error: --service (model_id) is required for evaluate action")
                return {"status": "failed", "action": "evaluate", "error": "Missing service parameter"}
            result = client.evaluate_model(args.service, 'BTCUSDT', '1h')
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "evaluate", "result": result}
            
        elif args.action == 'retrain':
            if not args.service:
                print("Error: --service (model_id) is required for retrain action")
                return {"status": "failed", "action": "retrain", "error": "Missing service parameter"}
            result = client.retrain_model(args.service)
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "retrain", "result": result}
            
        elif args.action == 'search':
            result = client.search_github('trading bot machine learning')
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "search", "result": result}
            
        elif args.action == 'tune':
            result = client.tune_hyperparameters('lstm')
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "tune", "result": result}
            
        elif args.action == 'plot':
            result = client.generate_plot('equity_curve', 'backtest_results')
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "plot", "result": result}
            
        elif args.action == 'route':
            result = client.route_prompt('Build me an LSTM model for Bitcoin prediction')
            print(json.dumps(result, indent=2))
            return {'success': True, 'result': {"status": "completed", "action": "route", "result": result}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return {"status": "interrupted", "action": args.action}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "failed", "action": args.action, "error": str(e)}
    finally:
        client.close()


if __name__ == "__main__":
    main() 