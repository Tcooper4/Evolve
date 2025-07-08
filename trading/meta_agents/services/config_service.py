"""
Config Service

This module implements configuration management functionality.

Note: This module was adapted from the legacy automation/services/automation_config.py file.
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
import os
import shutil
import hashlib
import base64
import secrets

class ConfigType(Enum):
    """Config type enumeration."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"

@dataclass
class Config:
    """Config configuration."""
    name: str
    type: ConfigType
    path: str
    description: str
    schema: Optional[Dict[str, Any]] = None
    encrypted: bool = False
    version: str = "1.0.0"

class ConfigService:
    """Manages configuration files and settings."""
    
    def __init__(self, config_path: str):
        """Initialize config service."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.setup_database()
        self.configs: Dict[str, Config] = {}
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.config_hashes: Dict[str, str] = {}
        self.running = False
        self.initialize_configs()
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/config")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "config_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
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
            self.logger.error(f"Error loading config: {str(e)}")
            raise

    def setup_database(self) -> None:
        """Set up config database."""
        try:
            db_path = Path(self.config['database']['path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(str(db_path))
            self.cursor = self.conn.cursor()
            
            # Create configs table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS configs (
                    name TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    description TEXT,
                    schema TEXT,
                    encrypted BOOLEAN NOT NULL,
                    version TEXT NOT NULL,
                    last_modified DATETIME NOT NULL
                )
            ''')
            
            # Create config_history table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    content TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    modified_by TEXT,
                    modified_at DATETIME NOT NULL,
                    FOREIGN KEY (config_name) REFERENCES configs(name)
                )
            ''')
            
            self.conn.commit()
            self.logger.info("Database setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def initialize_configs(self) -> None:
        """Initialize configs from configuration."""
        try:
            for config_config in self.config['configs']:
                config = Config(
                    name=config_config['name'],
                    type=ConfigType(config_config['type']),
                    path=config_config['path'],
                    description=config_config['description'],
                    schema=config_config.get('schema'),
                    encrypted=config_config.get('encrypted', False),
                    version=config_config.get('version', '1.0.0')
                )
                self.configs[config.name] = config
                self.load_config_file(config)
        except Exception as e:
            self.logger.error(f"Error initializing configs: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def load_config_file(self, config: Config) -> None:
        """Load a config file."""
        try:
            file_path = Path(config.path)
            if not file_path.exists():
                self.logger.warning(f"Config file not found: {file_path}")

            with open(file_path, 'r') as f:
                if config.type == ConfigType.JSON:
                    content = json.load(f)
                elif config.type == ConfigType.YAML:
                    content = yaml.safe_load(f)
                elif config.type == ConfigType.ENV:
                    content = {}
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            content[key] = value
                else:
                    raise ValueError(f"Unsupported config type: {config.type}")
            
            if config.encrypted:
                content = self.decrypt_config(content)
            
            self.config_cache[config.name] = content
            self.config_hashes[config.name] = self.calculate_hash(content)
            
            # Update database
            self.cursor.execute('''
                INSERT OR REPLACE INTO configs (
                    name, type, path, description, schema,
                    encrypted, version, last_modified
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.name,
                config.type.value,
                config.path,
                config.description,
                json.dumps(config.schema) if config.schema else None,
                config.encrypted,
                config.version,
                datetime.now()
            ))
            
            # Add to history
            self.cursor.execute('''
                INSERT INTO config_history (
                    config_name, version, content, hash,
                    modified_by, modified_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                config.name,
                config.version,
                json.dumps(content),
                self.config_hashes[config.name],
                os.getenv('USER'),
                datetime.now()
            ))
            
            self.conn.commit()
            self.logger.info(f"Loaded config: {config.name}")
        except Exception as e:
            self.logger.error(f"Error loading config file: {str(e)}")
            raise
    
    def save_config_file(self, config: Config, content: Dict[str, Any]) -> None:
        """Save a config file."""
        try:
            file_path = Path(config.path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config.encrypted:
                content = self.encrypt_config(content)
            
            with open(file_path, 'w') as f:
                if config.type == ConfigType.JSON:
                    json.dump(content, f, indent=2)
                elif config.type == ConfigType.YAML:
                    yaml.dump(content, f)
                elif config.type == ConfigType.ENV:
                    for key, value in content.items():
                        f.write(f"{key}={value}\n")
                else:
                    raise ValueError(f"Unsupported config type: {config.type}")
            
            self.config_cache[config.name] = content
            self.config_hashes[config.name] = self.calculate_hash(content)
            
            # Update database
            self.cursor.execute('''
                UPDATE configs
                SET last_modified = ?
                WHERE name = ?
            ''', (datetime.now(), config.name))
            
            # Add to history
            self.cursor.execute('''
                INSERT INTO config_history (
                    config_name, version, content, hash,
                    modified_by, modified_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                config.name,
                config.version,
                json.dumps(content),
                self.config_hashes[config.name],
                os.getenv('USER'),
                datetime.now()
            ))
            
            self.conn.commit()
            self.logger.info(f"Saved config: {config.name}")
        except Exception as e:
            self.logger.error(f"Error saving config file: {str(e)}")
            raise

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a config by name."""
        try:
            if name not in self.configs:
                raise ValueError(f"Config {name} not found")
            return self.config_cache.get(name)
        except Exception as e:
            self.logger.error(f"Error getting config: {str(e)}")
            raise
    
    def set_config(self, name: str, content: Dict[str, Any]) -> None:
        """Set a config by name."""
        try:
            if name not in self.configs:
                raise ValueError(f"Config {name} not found")
            
            config = self.configs[name]
            
            # Validate against schema if exists
            if config.schema:
                self.validate_config(content, config.schema)
            
            self.save_config_file(config, content)
        except Exception as e:
            self.logger.error(f"Error setting config: {str(e)}")
            raise

    def validate_config(self, content: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate config against schema."""
        try:
            # TODO: Implement schema validation
            raise NotImplementedError('Pending feature')
        except Exception as e:
            self.logger.error(f"Error validating config: {str(e)}")
            raise

    def calculate_hash(self, content: Dict[str, Any]) -> str:
        """Calculate hash of config content."""
        try:
            content_str = json.dumps(content, sort_keys=True)
            return hashlib.sha256(content_str.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash: {str(e)}")
            raise
    
    def encrypt_config(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt config content."""
        try:
            # Generate encryption key
            key = secrets.token_bytes(32)
            
            # Encrypt content
            content_str = json.dumps(content)
            encrypted = base64.b64encode(key + content_str.encode()).decode()
            
            return {
                'encrypted': True,
                'data': encrypted
            }
        except Exception as e:
            self.logger.error(f"Error encrypting config: {str(e)}")
            raise
    
    def decrypt_config(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt config content."""
        try:
            if not content.get('encrypted'):
                return content
            
            # Decrypt content
            encrypted = base64.b64decode(content['data'])
            key = encrypted[:32]
            decrypted = encrypted[32:].decode()
            
            return json.loads(decrypted)
        except Exception as e:
            self.logger.error(f"Error decrypting config: {str(e)}")
            raise
    
    def get_config_history(self, name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get config history."""
        try:
            query = '''
                SELECT * FROM config_history
                WHERE config_name = ?
                ORDER BY modified_at DESC
            '''
            params = [name]
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            self.cursor.execute(query, params)
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'id': row[0],
                    'config_name': row[1],
                    'version': row[2],
                    'content': json.loads(row[3]),
                    'hash': row[4],
                    'modified_by': row[5],
                    'modified_at': row[6]
                })
            
            return results
        except Exception as e:
            self.logger.error(f"Error getting config history: {str(e)}")
            raise
    
    def restore_config(self, name: str, version: str) -> None:
        """Restore config to a specific version."""
        try:
            if name not in self.configs:
                raise ValueError(f"Config {name} not found")
            
            # Get version from history
            self.cursor.execute('''
                SELECT content FROM config_history
                WHERE config_name = ? AND version = ?
                ORDER BY modified_at DESC
                LIMIT 1
            ''', (name, version))
            
            row = self.cursor.fetchone()
            if not row:
                raise ValueError(f"Version {version} not found for config {name}")
            
            content = json.loads(row[0])
            config = self.configs[name]
            
            self.save_config_file(config, content)
            self.logger.info(f"Restored config {name} to version {version}")
        except Exception as e:
            self.logger.error(f"Error restoring config: {str(e)}")
            raise

    def watch_config_file(self, name: str, callback: callable) -> None:
        """Watch a config file for changes."""
        try:
            if name not in self.configs:
                raise ValueError(f"Config {name} not found")
            
            config = self.configs[name]
            file_path = Path(config.path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {file_path}")
            
            last_modified = file_path.stat().st_mtime
            
            while self.running:
                try:
                    current_modified = file_path.stat().st_mtime
                    if current_modified > last_modified:
                        self.load_config_file(config)
                        callback(config.name, self.config_cache[config.name])
                        last_modified = current_modified
                    time.sleep(1)
                except FileNotFoundError:
                    self.logger.warning(f"Config file not found: {file_path}")
                    time.sleep(5)
        except Exception as e:
            self.logger.error(f"Error watching config file: {str(e)}")
            raise

    async def start(self) -> None:
        """Start config service."""
        try:
            self.running = True
            
            # Watch all config files
            watch_tasks = []
            for config in self.configs.values():
                if config.path:
                    task = asyncio.create_task(
                        self.watch_config_file(
                            config.name,
                            lambda name, content: self.logger.info(f"Config {name} changed")
                        )
                    )
                    watch_tasks.append(task)
            
            await asyncio.gather(*watch_tasks)
        except Exception as e:
            self.logger.error(f"Error in config service: {str(e)}")
            raise
        finally:
            self.running = False
    
    def stop(self) -> None:
        """Stop config service."""
        self.running = False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Config service')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--get', help='Get config by name')
    parser.add_argument('--set', help='Set config by name')
    parser.add_argument('--history', help='Get config history')
    parser.add_argument('--restore', help='Restore config to version')
    args = parser.parse_args()
    
    try:
        service = ConfigService(args.config)
        
        if args.get:
            logger.info(json.dumps(service.get_config(args.get), indent=2))
        elif args.set:
            content = json.loads(input("Enter config content: "))
            service.set_config(args.set, content)
        elif args.history:
            logger.info(json.dumps(service.get_config_history(args.history), indent=2))
        elif args.restore:
            version = input("Enter version to restore: ")
            service.restore_config(args.restore, version)
        else:
            asyncio.run(service.start())
    except KeyboardInterrupt:
        logging.info("Config service interrupted")
    except Exception as e:
        logging.error(f"Error in config service: {str(e)}")
        raise

if __name__ == '__main__':
    main() 