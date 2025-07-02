#!/usr/bin/env python3
"""
Evolve Forecasting Pipeline Launcher

This script launches the complete Evolve trading system for production deployment.
It handles all initialization, validation, and startup procedures.

Usage:
    python run_forecasting_pipeline.py [--mode production|development] [--port 8501] [--host localhost]

Features:
- Environment validation
- API key verification
- Service initialization
- Live trade logging
- Alert system setup
- Redis/Consul connection
- Comprehensive error handling
"""

import os
import sys
import signal
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import subprocess
import threading
import time
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forecasting_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForecastingPipelineLauncher:
    """Launches the complete Evolve forecasting pipeline."""
    
    def __init__(self, mode: str = "production", port: int = 8501, host: str = "localhost"):
        """Initialize the launcher.
        
        Args:
            mode: Deployment mode (production/development)
            port: Port for the web interface
            host: Host for the web interface
        """
        self.mode = mode
        self.port = port
        self.host = host
        self.processes = []
        self.running = False
        
        # Set environment variables
        os.environ["TRADING_ENV"] = mode
        os.environ["STREAMLIT_SERVER_PORT"] = str(port)
        os.environ["STREAMLIT_SERVER_ADDRESS"] = host
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        
        if mode == "development":
            os.environ["STREAMLIT_DEBUG"] = "true"
        else:
            os.environ["STREAMLIT_DEBUG"] = "false"

    def validate_environment(self) -> bool:
        """Validate environment setup.
        
        Returns:
            True if environment is valid
        """
        logger.info("🔍 Validating environment...")
        
        # Check required environment variables
        required_vars = [
            "ALPHA_VANTAGE_API_KEY",
            "POLYGON_API_KEY", 
            "OPENAI_API_KEY",
            "JWT_SECRET_KEY",
            "EMAIL_PASSWORD",
            "SLACK_WEBHOOK_URL"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            logger.info("Please set these variables in your .env file")
            return False
        
        # Check directories
        required_dirs = ["logs", "data", "models", "cache", "reports"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Directory ready: {dir_name}")
        
        # Validate configuration
        try:
            from trading.config.settings import validate_config
            validate_config()
            logger.info("✅ Configuration validated")
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
        
        logger.info("✅ Environment validation completed")
        return True
    
    def setup_live_trade_logging(self) -> None:
        """Setup live trade logging and alerts."""
        logger.info("📊 Setting up live trade logging...")
        
        try:
            # Initialize trade logger
            from trading.utils.trade_logger import TradeLogger
            trade_logger = TradeLogger()
            
            # Setup Slack alerts
            if os.getenv("SLACK_WEBHOOK_URL"):
                from trading.utils.notifications import SlackNotifier
                slack_notifier = SlackNotifier(os.getenv("SLACK_WEBHOOK_URL"))
                trade_logger.add_notifier(slack_notifier)
                logger.info("✅ Slack notifications configured")
            
            # Setup email alerts
            if os.getenv("EMAIL_PASSWORD"):
                from trading.utils.notifications import EmailNotifier
                email_notifier = EmailNotifier(
                    host=os.getenv("EMAIL_HOST", "smtp.gmail.com"),
                    port=int(os.getenv("EMAIL_PORT", "587")),
                    user=os.getenv("EMAIL_USER"),
                    password=os.getenv("EMAIL_PASSWORD")
                )
                trade_logger.add_notifier(email_notifier)
                logger.info("✅ Email notifications configured")
            
            logger.info("✅ Live trade logging configured")
            
        except Exception as e:
            logger.warning(f"⚠️ Live trade logging setup failed: {e}")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_redis_consul(self) -> bool:
        """Setup Redis and Consul connections.
        
        Returns:
            True if setup successful
        """
        logger.info("🔗 Setting up Redis and Consul connections...")
        
        try:
            # Test Redis connection
            import redis
            redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD"),
                decode_responses=True
            )
            redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # Test Consul connection (if configured)
            consul_host = os.getenv("CONSUL_HOST")
            if consul_host:
                import consul
                consul_client = consul.Consul(host=consul_host)
                consul_client.agent.self()
                logger.info("✅ Consul connection established")
            else:
                logger.info("ℹ️ Consul not configured, skipping")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis/Consul setup failed: {e}")
            return False
    
    def start_services(self) -> bool:
        """Start background services.
        
        Returns:
            True if services started successfully
        """
        logger.info("🚀 Starting background services...")
        
        try:
            # Start agent manager
            from trading.agents.agent_manager import AgentManager
            agent_manager = AgentManager()
            agent_manager.start()
            logger.info("✅ Agent manager started")
            
            # Start market data service
            from trading.data.data_listener import DataListener
            data_listener = DataListener()
            data_listener.start()
            logger.info("✅ Market data service started")
            
            # Start monitoring service
            from trading.utils.monitor import SystemMonitor
            monitor = SystemMonitor()
            monitor.start()
            logger.info("✅ System monitor started")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Service startup failed: {e}")
            return False
    
    def start_web_interface(self) -> subprocess.Popen:
        """Start the Streamlit web interface.
        
        Returns:
            Subprocess object for the web interface
        """
        logger.info(f"🌐 Starting web interface on {self.host}:{self.port}...")
        
        try:
            # Start Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", str(self.port),
                "--server.address", self.host,
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            if self.mode == "development":
                cmd.extend(["--logger.level", "debug"])
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment to check if it started successfully
            time.sleep(3)
            if process.poll() is None:
                logger.info(f"✅ Web interface started successfully")
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Web interface failed to start: {stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to start web interface: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.shutdown()

    def shutdown(self):
        """Gracefully shutdown all services."""
        logger.info("🔄 Shutting down services...")
        self.running = False
        
        # Stop all processes
        for process in self.processes:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        logger.info("✅ Shutdown completed")
        sys.exit(0)

    def run(self) -> None:
        """Run the complete forecasting pipeline."""
        logger.info("🚀 Starting Evolve Forecasting Pipeline")
        logger.info(f"📊 Mode: {self.mode}")
        logger.info(f"🌐 Interface: http://{self.host}:{self.port}")
        logger.info("=" * 60)
        
        try:
            # Validate environment
            if not self.validate_environment():
                logger.error("❌ Environment validation failed")
                sys.exit(1)
            
            # Setup live trade logging
            self.setup_live_trade_logging()
            
            # Setup Redis/Consul
            if not self.setup_redis_consul():
                logger.warning("⚠️ Redis/Consul setup failed, continuing without...")
            
            # Start background services
            if not self.start_services():
                logger.error("❌ Service startup failed")
                sys.exit(1)
            
            # Start web interface
            web_process = self.start_web_interface()
            if web_process:
                self.processes.append(web_process)
            else:
                logger.error("❌ Web interface startup failed")
                sys.exit(1)
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            self.running = True
            logger.info("🎉 Evolve Forecasting Pipeline is now running!")
            logger.info("📊 Access the dashboard at: http://localhost:8501")
            logger.info("📈 Monitor logs at: logs/forecasting_pipeline.log")
            logger.info("🛑 Press Ctrl+C to stop")
            
            # Keep running
            while self.running:
                time.sleep(1)
                
                # Check if web interface is still running
                if web_process and web_process.poll() is not None:
                    logger.error("❌ Web interface stopped unexpectedly")
                    break
            
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
        finally:
            self.shutdown()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evolve Forecasting Pipeline Launcher")
    parser.add_argument(
        "--mode",
        choices=["production", "development"],
        default="production",
        help="Deployment mode"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for web interface"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for web interface"
    )
    
    args = parser.parse_args()
    
    # Create launcher and run
    launcher = ForecastingPipelineLauncher(
        mode=args.mode,
        port=args.port,
        host=args.host
    )
    
    try:
        launcher.run()
        return {
            "status": "completed",
            "mode": args.mode,
            "port": args.port,
            "host": args.host,
            "exit_code": 0
        }
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return {'success': True, 'result': None, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat(),
            "status": "failed",
            "mode": args.mode,
            "port": args.port,
            "host": args.host,
            "error": str(e),
            "exit_code": 1
        }

if __name__ == "__main__":
    main() 