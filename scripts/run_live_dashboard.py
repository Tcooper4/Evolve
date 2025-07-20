#!/usr/bin/env python3
"""
Live Dashboard Runner

Enhanced dashboard launcher with:
- Port selection via argparse
- Robust error handling for streamlit imports
- Dynamic refresh control
- Health monitoring and recovery
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class DashboardRunner:
    """Enhanced dashboard runner with error handling and monitoring."""
    
    def __init__(self, port: int = 8501, host: str = "localhost", 
                 config_file: Optional[str] = None, 
                 refresh_interval: int = 30):
        """Initialize the dashboard runner."""
        self.port = port
        self.host = host
        self.config_file = config_file
        self.refresh_interval = refresh_interval
        self.process = None
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info(f"DashboardRunner initialized: {host}:{port}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "dashboard.log"),
                logging.StreamHandler()
            ]
        )
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import streamlit
            logger.info(f"âœ… Streamlit version: {streamlit.__version__}")
            return True
        except ImportError as e:
            logger.error(f"âŒ Streamlit not available: {e}")
            logger.error("Please install streamlit: pip install streamlit")
            return False
    
    def _check_app_file(self) -> bool:
        """Check if the main app file exists."""
        app_file = Path("app.py")
        if not app_file.exists():
            logger.error(f"âŒ App file not found: {app_file}")
            return False
        
        logger.info(f"âœ… App file found: {app_file}")
        return True
    
    def _validate_port(self) -> bool:
        """Validate port number."""
        if not (1024 <= self.port <= 65535):
            logger.error(f"âŒ Invalid port number: {self.port}. Must be between 1024-65535")
            return False
        
        # Check if port is available
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.host, self.port))
                logger.info(f"âœ… Port {self.port} is available")
                return True
        except OSError:
            logger.warning(f"âš ï¸ Port {self.port} may be in use")
            return True  # Continue anyway, let streamlit handle it
    
    def _create_config_file(self):
        """Create streamlit config file with custom settings."""
        if not self.config_file:
            return
        
        config_content = f"""
[server]
port = {self.port}
address = "{self.host}"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
base = "light"

[client]
showErrorDetails = true
"""
        
        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"âœ… Created config file: {config_path}")
    
    def _get_streamlit_command(self) -> list:
        """Build streamlit command with arguments."""
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(self.port),
            "--server.address", self.host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        if self.config_file:
            cmd.extend(["--config", self.config_file])
        
        return cmd
    
    def start(self) -> bool:
        """Start the dashboard with error handling."""
        logger.info("ðŸš€ Starting Live Dashboard...")
        
        # Pre-flight checks
        if not self._check_dependencies():
            return False
        
        if not self._check_app_file():
            return False
        
        if not self._validate_port():
            return False
        
        # Create config file if specified
        if self.config_file:
            self._create_config_file()
        
        try:
            import subprocess
            import threading
            
            # Build command
            cmd = self._get_streamlit_command()
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.running = True
            
            # Start monitoring threads
            self._start_monitoring()
            
            logger.info(f"âœ… Dashboard started successfully on http://{self.host}:{self.port}")
            logger.info("Press Ctrl+C to stop")
            
            # Wait for process
            try:
                stdout, stderr = self.process.communicate()
                if stdout:
                    logger.info(f"STDOUT: {stdout}")
                if stderr:
                    logger.error(f"STDERR: {stderr}")
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self.stop()
            
            return True
            
        except Exception as e:
            logger.error(f"âŒ Failed to start dashboard: {e}")
            return False
    
    def _start_monitoring(self):
        """Start monitoring threads."""
        import threading
        
        # Health check thread
        def health_check():
            while self.running and self.process:
                try:
                    if self.process.poll() is not None:
                        logger.error("Dashboard process terminated unexpectedly")
                        self.running = False
                        break
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
        
        # Start health check thread
        health_thread = threading.Thread(target=health_check, daemon=True)
        health_thread.start()
        
        logger.info("âœ… Monitoring started")
    
    def stop(self):
        """Stop the dashboard gracefully."""
        logger.info("ðŸ›‘ Stopping dashboard...")
        
        self.running = False
        
        if self.process:
            try:
                # Try graceful shutdown first
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown timeout, forcing kill")
                    self.process.kill()
                    self.process.wait()
                
                logger.info("âœ… Dashboard stopped successfully")
                
            except Exception as e:
                logger.error(f"Error stopping dashboard: {e}")
            finally:
                self.process = None
    
    def get_status(self) -> dict:
        """Get dashboard status."""
        return {
            "running": self.running,
            "port": self.port,
            "host": self.host,
            "refresh_interval": self.refresh_interval,
            "process_pid": self.process.pid if self.process else None,
            "process_returncode": self.process.returncode if self.process else None
        }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Live Dashboard Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run on default port 8501
  %(prog)s -p 8080           # Run on port 8080
  %(prog)s -p 8080 -H 0.0.0.0 # Run on all interfaces
  %(prog)s -r 60              # Set refresh interval to 60 seconds
        """
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8501,
        help="Port to run the dashboard on (default: 8501)"
    )
    
    parser.add_argument(
        "-H", "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to streamlit config file"
    )
    
    parser.add_argument(
        "-r", "--refresh",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start dashboard runner
    runner = DashboardRunner(
        port=args.port,
        host=args.host,
        config_file=args.config,
        refresh_interval=args.refresh
    )
    
    try:
        success = runner.start()
        if not success:
            logger.error("Failed to start dashboard")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
