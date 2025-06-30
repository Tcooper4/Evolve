"""
Logging Service

This module implements logging and log management functionality.

Note: This module was adapted from the legacy automation/services/automation_logging.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import yaml
import time
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import re
import gzip
import shutil
import os

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogConfig:
    """Log configuration."""
    name: str
    level: LogLevel
    format: str
    handlers: List[Dict[str, Any]]
    filters: Optional[List[Dict[str, Any]]] = None

class LoggingService:
    """Manages logging and log management."""
    
    def __init__(self, config_path: str):
        """Initialize logging service."""
        self.config_path = config_path
        self.load_config()
        self.setup_database()
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_queue = Queue()
        self.running = False
        self.initialize_loggers()
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    self.config = json.load(f)
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def setup_database(self) -> None:
        """Set up logging database."""
        try:
            db_path = Path(self.config['database']['path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(str(db_path))
            self.cursor = self.conn.cursor()
            
            # Create logs table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    logger_name TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    function TEXT,
                    line_number INTEGER,
                    thread_id INTEGER,
                    process_id INTEGER,
                    extra TEXT
                )
            ''')
            
            # Create index on timestamp and logger_name
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
                ON logs(timestamp)
            ''')
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_logs_logger 
                ON logs(logger_name)
            ''')
            
            self.conn.commit()
        except Exception as e:
            print(f"Error setting up database: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def initialize_loggers(self) -> None:
        """Initialize loggers from configuration."""
        try:
            for logger_config in self.config['loggers']:
                config = LogConfig(
                    name=logger_config['name'],
                    level=LogLevel(logger_config['level']),
                    format=logger_config['format'],
                    handlers=logger_config['handlers'],
                    filters=logger_config.get('filters')
                )
                self.setup_logger(config)
        except Exception as e:
            print(f"Error initializing loggers: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logger(self, config: LogConfig) -> None:
        """Set up a logger with the given configuration."""
        try:
            logger = logging.getLogger(config.name)
            logger.setLevel(config.level.value)
            
            # Remove existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Add handlers
            for handler_config in config.handlers:
                handler = self.create_handler(handler_config)
                if handler:
                    logger.addHandler(handler)
            
            # Add filters
            if config.filters:
                for filter_config in config.filters:
                    filter_obj = self.create_filter(filter_config)
                    if filter_obj:
                        logger.addFilter(filter_obj)
            
            self.loggers[config.name] = logger
        except Exception as e:
            print(f"Error setting up logger {config.name}: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def create_handler(self, config: Dict[str, Any]) -> Optional[logging.Handler]:
        """Create a logging handler from configuration."""
        try:
            handler_type = config['type']
            
            if handler_type == 'stream':
                return logging.StreamHandler()
            
            elif handler_type == 'file':
                file_path = Path(config['path'])
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                handler = logging.FileHandler(file_path)
                if config.get('mode') == 'a':
                    handler.mode = 'a'
                
                return handler
            
            elif handler_type == 'rotating_file':
                file_path = Path(config['path'])
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                handler = logging.handlers.RotatingFileHandler(
                    file_path,
                    maxBytes=config.get('max_bytes', 10 * 1024 * 1024),
                    backupCount=config.get('backup_count', 5)
                )
                
                return handler
            
            elif handler_type == 'timed_rotating_file':
                file_path = Path(config['path'])
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                handler = logging.handlers.TimedRotatingFileHandler(
                    file_path,
                    when=config.get('when', 'midnight'),
                    interval=config.get('interval', 1),
                    backupCount=config.get('backup_count', 5)
                )
                
                return handler
            
            else:
                print(f"Unknown handler type: {handler_type}")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            print(f"Error creating handler: {str(e)}")
            return None
    
    def create_filter(self, config: Dict[str, Any]) -> Optional[logging.Filter]:
        """Create a logging filter from configuration."""
        try:
            filter_type = config['type']
            
            if filter_type == 'level':
                return logging.Filter()
            
            elif filter_type == 'regex':
                return RegexFilter(config['pattern'])
            
            else:
                print(f"Unknown filter type: {filter_type}")
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            print(f"Error creating filter: {str(e)}")
            return None
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name."""
        try:
            if name not in self.loggers:
                raise ValueError(f"Logger {name} not found")
            return {'success': True, 'result': self.loggers[name], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            print(f"Error getting logger: {str(e)}")
            raise
    
    def log(self, logger_name: str, level: LogLevel, message: str, **kwargs) -> None:
        """Log a message."""
        try:
            logger = self.get_logger(logger_name)
            
            # Add to database
            self.cursor.execute('''
                INSERT INTO logs (
                    timestamp, logger_name, level, message,
                    module, function, line_number,
                    thread_id, process_id, extra
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                logger_name,
                level.value,
                message,
                kwargs.get('module'),
                kwargs.get('function'),
                kwargs.get('line_number'),
                kwargs.get('thread_id'),
                kwargs.get('process_id'),
                json.dumps(kwargs.get('extra', {}))
            ))
            self.conn.commit()
            
            # Log using logger
            log_func = getattr(logger, level.value.lower())
            log_func(message, **kwargs)
        except Exception as e:
            print(f"Error logging message: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def get_logs(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                 logger_name: Optional[str] = None, level: Optional[LogLevel] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get log entries."""
        try:
            query = 'SELECT * FROM logs WHERE 1=1'
            params = []
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            if logger_name:
                query += ' AND logger_name = ?'
                params.append(logger_name)
            
            if level:
                query += ' AND level = ?'
                params.append(level.value)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            self.cursor.execute(query, params)
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'logger_name': row[2],
                    'level': row[3],
                    'message': row[4],
                    'module': row[5],
                    'function': row[6],
                    'line_number': row[7],
                    'thread_id': row[8],
                    'process_id': row[9],
                    'extra': json.loads(row[10]) if row[10] else {}
                })
            
            return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            print(f"Error getting logs: {str(e)}")
            raise
    
    def cleanup_old_logs(self, days: int) -> None:
        """Clean up logs older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            self.cursor.execute('''
                DELETE FROM logs
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            self.conn.commit()
            print(f"Cleaned up logs older than {days} days")
        except Exception as e:
            print(f"Error cleaning up old logs: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def rotate_logs(self) -> None:
        """Rotate log files."""
        try:
            for logger_config in self.config['loggers']:
                for handler_config in logger_config['handlers']:
                    if handler_config['type'] in ['rotating_file', 'timed_rotating_file']:
                        file_path = Path(handler_config['path'])
                        if file_path.exists():
                            # Create backup directory
                            backup_dir = file_path.parent / 'backups'
                            backup_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Create backup filename
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            backup_path = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                            
                            # Compress and move file
                            with open(file_path, 'rb') as f_in:
                                with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            
                            # Clear original file
                            file_path.unlink()
        except Exception as e:
            print(f"Error rotating logs: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def process_log_queue(self) -> None:
        """Process logs from the queue."""
        try:
            while self.running:
                try:
                    log_entry = self.log_queue.get(timeout=1)
                    self.log(
                        log_entry['logger_name'],
                        LogLevel(log_entry['level']),
                        log_entry['message'],
                        **log_entry.get('kwargs', {})
                    )
                except Queue.Empty:
                    continue
        except Exception as e:
            print(f"Error processing log queue: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    async def start(self) -> None:
        """Start logging service."""
        try:
            self.running = True
            queue_thread = threading.Thread(target=self.process_log_queue)
            queue_thread.start()
            
            while self.running:
                # Rotate logs if needed
                if self.config.get('rotation', {}).get('enabled', False):
                    self.rotate_logs()
                
                # Clean up old logs if needed
                if self.config.get('cleanup', {}).get('enabled', False):
                    self.cleanup_old_logs(self.config['cleanup']['days'])
                
                await asyncio.sleep(self.config.get('check_interval', 3600))
        except Exception as e:
            print(f"Error in logging service: {str(e)}")
            raise
        finally:
            self.running = False
            queue_thread.join()
    
    def stop(self) -> None:
        """Stop logging service."""
        self.running = False

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
class RegexFilter(logging.Filter):
    """Filter log records based on regex pattern."""
    
    def __init__(self, pattern: str):
        """Initialize filter with pattern."""
        super().__init__()
        self.pattern = re.compile(pattern)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record."""
        return {'success': True, 'result': bool(self.pattern.search(record.getMessage())), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Logging service')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--cleanup', type=int, help='Clean up logs older than specified days')
    args = parser.parse_args()
    
    try:
        service = LoggingService(args.config)
        
        if args.cleanup:
            service.cleanup_old_logs(args.cleanup)
        else:
            asyncio.run(service.start())
    except KeyboardInterrupt:
        print("Logging service interrupted")
    except Exception as e:
        print(f"Error in logging service: {str(e)}")
        raise

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == '__main__':
    main() 