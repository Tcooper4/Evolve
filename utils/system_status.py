"""System status and metrics calculation utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import psutil
import platform
import os
import time
import subprocess
import socket

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SystemStatus:
    def __init__(self) -> None:
        """Initialize system status monitor."""
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def get_uptime(self) -> str:
        """Get system uptime."""
        try:
            uptime_seconds = time.time() - self.start_time
            return str(timedelta(seconds=int(uptime_seconds)))
        except Exception as e:
            self.logger.error(f"Error getting uptime: {str(e)}")
            return "Unknown"
            
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        try:
            return {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "load_avg": psutil.getloadavg()
            }
        except Exception as e:
            self.logger.error(f"Error getting CPU info: {str(e)}")
            return {
                "usage_percent": None,
                "count": None,
                "frequency": None,
                "load_avg": None
            }
            
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
        except Exception as e:
            self.logger.error(f"Error getting memory info: {str(e)}")
            return {
                "total": None,
                "available": None,
                "used": None,
                "percent": None
            }
            
    def get_disk_info(self) -> Dict[str, Any]:
        """Get disk information."""
        try:
            disk = psutil.disk_usage('/')
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
        except Exception as e:
            self.logger.error(f"Error getting disk info: {str(e)}")
            return {
                "total": None,
                "used": None,
                "free": None,
                "percent": None
            }
            
    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            self.logger.error(f"Error getting network info: {str(e)}")
            return {
                "bytes_sent": None,
                "bytes_recv": None,
                "packets_sent": None,
                "packets_recv": None
            }
            
    def get_process_info(self) -> Dict[str, Any]:
        """Get process information."""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting process info: {str(e)}")
            return {
                "pid": None,
                "name": None,
                "status": None,
                "cpu_percent": None,
                "memory_percent": None,
                "create_time": None
            }
            
    def get_agent_liveness(self) -> Dict[str, Any]:
        """Get agent liveness information."""
        try:
            # Check if agent process is running
            agent_process = None
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'agent' in proc.info['name'].lower():
                    agent_process = proc
                    break
                    
            if agent_process:
                return {
                    "status": "running",
                    "pid": agent_process.info['pid'],
                    "uptime": str(timedelta(seconds=int(time.time() - agent_process.create_time()))),
                    "last_heartbeat": self._get_last_heartbeat()
                }
            else:
                return {
                    "status": "stopped",
                    "pid": None,
                    "uptime": None,
                    "last_heartbeat": None
                }
        except Exception as e:
            self.logger.error(f"Error getting agent liveness: {str(e)}")
            return {
                "status": "unknown",
                "pid": None,
                "uptime": None,
                "last_heartbeat": None
            }
            
    def _get_last_heartbeat(self) -> Optional[str]:
        """Get last agent heartbeat timestamp."""
        try:
            heartbeat_file = "logs/agent_heartbeat.log"
            if os.path.exists(heartbeat_file):
                with open(heartbeat_file, 'r') as f:
                    last_line = f.readlines()[-1]
                    return last_line.strip()
            return None
        except Exception as e:
            self.logger.error(f"Error getting last heartbeat: {str(e)}")
            return None
            
    def get_system_info(self) -> Dict[str, Any]:
        """Get complete system information."""
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": self.get_uptime(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "disk": self.get_disk_info(),
            "network": self.get_network_info(),
            "process": self.get_process_info(),
            "agent": self.get_agent_liveness()
        }
        
    def save_status_report(self, filepath: str) -> None:
        """Save system status report to file."""
        try:
            status = self.get_system_info()
            with open(filepath, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving status report: {str(e)}")
            
    def print_status(self) -> None:
        """Print system status to console."""
        try:
            status = self.get_system_info()
            print("\nSystem Status Report")
            print("===================")
            print(f"Timestamp: {status['timestamp']}")
            print(f"Uptime: {status['uptime']}")
            print("\nPlatform:")
            for key, value in status['platform'].items():
                print(f"  {key}: {value}")
            print("\nCPU:")
            for key, value in status['cpu'].items():
                print(f"  {key}: {value}")
            print("\nMemory:")
            for key, value in status['memory'].items():
                print(f"  {key}: {value}")
            print("\nDisk:")
            for key, value in status['disk'].items():
                print(f"  {key}: {value}")
            print("\nNetwork:")
            for key, value in status['network'].items():
                print(f"  {key}: {value}")
            print("\nProcess:")
            for key, value in status['process'].items():
                print(f"  {key}: {value}")
            print("\nAgent:")
            for key, value in status['agent'].items():
                print(f"  {key}: {value}")
        except Exception as e:
            self.logger.error(f"Error printing status: {str(e)}")
            
def get_system_scorecard() -> Dict[str, Any]:
    """Calculate system performance metrics from logs and goals.
    
    Returns:
        Dictionary containing:
        - sharpe_7d: Average Sharpe ratio over last 7 days
        - sharpe_30d: Average Sharpe ratio over last 30 days
        - win_rate: Percentage of profitable trades
        - mse_avg: Average Mean Squared Error
        - goal_status: Dictionary of goal achievement status
        - last_10_entries: DataFrame of last 10 performance entries
        - trades_per_day: Series of trades per day
    """
    try:
        # Load performance log
        log_file = Path("memory/logs/performance_log.csv")
        if not log_file.exists():
            return {
                "sharpe_7d": 0.0,
                "sharpe_30d": 0.0,
                "win_rate": 0.0,
                "mse_avg": 0.0,
                "goal_status": {},
                "last_10_entries": pd.DataFrame(),
                "trades_per_day": pd.Series()
            }
            
        df = pd.read_csv(log_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate date ranges
        now = datetime.now()
        date_7d = now - timedelta(days=7)
        date_30d = now - timedelta(days=30)
        
        # Calculate metrics
        sharpe_7d = df[df['timestamp'] >= date_7d]['sharpe'].mean()
        sharpe_30d = df[df['timestamp'] >= date_30d]['sharpe'].mean()
        
        # Calculate win rate
        profitable_trades = df[df['sharpe'] > 0].shape[0]
        total_trades = df.shape[0]
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate MSE average
        mse_avg = df['mse'].mean()
        
        # Get last 10 entries
        last_10_entries = df.sort_values('timestamp', ascending=False).head(10)
        
        # Calculate trades per day
        trades_per_day = df.groupby(df['timestamp'].dt.date).size()
        
        # Load goal status
        goal_file = Path("memory/goals/status.json")
        goal_status = {}
        if goal_file.exists():
            with open(goal_file, 'r') as f:
                goal_status = json.load(f)
        
        return {
            "sharpe_7d": round(sharpe_7d, 2),
            "sharpe_30d": round(sharpe_30d, 2),
            "win_rate": round(win_rate * 100, 1),  # Convert to percentage
            "mse_avg": round(mse_avg, 4),
            "goal_status": goal_status,
            "last_10_entries": last_10_entries,
            "trades_per_day": trades_per_day
        }
        
    except Exception as e:
        error_msg = f"Error calculating system metrics: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and print status
    status = SystemStatus()
    status.print_status()
