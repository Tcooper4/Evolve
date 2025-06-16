"""
System Backup

This module implements system backup functionality.

Note: This module was adapted from the legacy automation/scripts/backup_system.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
import shutil
import tarfile
import datetime
import os
import subprocess

class BackupManager:
    """Manages system backups."""
    
    def __init__(self, config_path: str):
        """Initialize backup manager."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
    
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/backup")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "backup_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
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
    
    def create_backup(self) -> str:
        """Create a system backup."""
        try:
            # Create backup directory
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = Path(self.config['backup']['directory']) / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup files
            for item in self.config['backup']['items']:
                source = Path(item['source'])
                if source.exists():
                    if source.is_file():
                        shutil.copy2(source, backup_dir)
                    else:
                        shutil.copytree(source, backup_dir / source.name)
                    self.logger.info(f"Backed up: {source}")
                else:
                    self.logger.warning(f"Source not found: {source}")
            
            # Create archive
            archive_path = backup_dir.with_suffix('.tar.gz')
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_dir.name)
            
            # Remove backup directory
            shutil.rmtree(backup_dir)
            
            self.logger.info(f"Created backup: {archive_path}")
            return str(archive_path)
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            raise
    
    def restore_backup(self, backup_path: str) -> None:
        """Restore a system backup."""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_path}")
            
            # Extract backup
            extract_dir = backup_path.parent / backup_path.stem
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            
            # Restore files
            backup_dir = extract_dir / backup_path.stem
            for item in self.config['backup']['items']:
                source = backup_dir / Path(item['source']).name
                target = Path(item['source'])
                
                if source.exists():
                    if source.is_file():
                        shutil.copy2(source, target)
                    else:
                        if target.exists():
                            shutil.rmtree(target)
                        shutil.copytree(source, target)
                    self.logger.info(f"Restored: {target}")
                else:
                    self.logger.warning(f"Backup item not found: {source}")
            
            # Clean up
            shutil.rmtree(extract_dir)
            
            self.logger.info(f"Restored backup: {backup_path}")
        except Exception as e:
            self.logger.error(f"Error restoring backup: {str(e)}")
            raise
    
    def cleanup_old_backups(self) -> None:
        """Clean up old backups."""
        try:
            backup_dir = Path(self.config['backup']['directory'])
            if not backup_dir.exists():
                return
            
            # Get backup files
            backup_files = list(backup_dir.glob('*.tar.gz'))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent backups
            max_backups = self.config['backup'].get('max_backups', 5)
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                self.logger.info(f"Deleted old backup: {old_backup}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {str(e)}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage system backups')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--action', required=True, choices=['create', 'restore', 'cleanup'],
                      help='Action to perform')
    parser.add_argument('--backup', help='Path to backup file (required for restore action)')
    args = parser.parse_args()
    
    try:
        manager = BackupManager(args.config)
        
        if args.action == 'create':
            manager.create_backup()
        elif args.action == 'restore':
            if not args.backup:
                raise ValueError("Backup path is required for restore action")
            manager.restore_backup(args.backup)
        elif args.action == 'cleanup':
            manager.cleanup_old_backups()
    except KeyboardInterrupt:
        logging.info("Backup management interrupted")
    except Exception as e:
        logging.error(f"Error managing backups: {str(e)}")
        raise

if __name__ == '__main__':
    main() 