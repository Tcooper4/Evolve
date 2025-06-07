import os
import shutil
import datetime
import logging
import json
from pathlib import Path
import tarfile
import hashlib
from typing import Dict, List, Optional

class BackupSystem:
    def __init__(self, config_path: str = "automation/config/backup_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        
    def _load_config(self) -> Dict:
        """Load backup configuration."""
        if not os.path.exists(self.config_path):
            # Create default config
            config = {
                "backup_dirs": [
                    "trading",
                    "models",
                    "config",
                    "alerts",
                    "feature_engineering"
                ],
                "exclude_patterns": [
                    "*.pyc",
                    "__pycache__",
                    "*.log",
                    "*.tmp",
                    "*.bak"
                ],
                "backup_location": "backups",
                "max_backups": 7,  # Keep last 7 days of backups
                "compression": "gz"
            }
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            return config
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def setup_logging(self):
        """Setup logging for backup system."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "backup_system.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("BackupSystem")
    
    def create_backup(self) -> str:
        """Create a new backup."""
        try:
            # Create backup directory
            backup_dir = Path(self.config["backup_location"])
            backup_dir.mkdir(exist_ok=True)
            
            # Generate backup filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}.tar.{self.config['compression']}"
            backup_path = backup_dir / backup_name
            
            # Create tar archive
            with tarfile.open(backup_path, f"w:{self.config['compression']}") as tar:
                for dir_name in self.config["backup_dirs"]:
                    if os.path.exists(dir_name):
                        self.logger.info(f"Backing up {dir_name}")
                        tar.add(dir_name, 
                               filter=lambda x: self._filter_file(x, self.config["exclude_patterns"]))
            
            # Create backup manifest
            manifest = self._create_manifest(backup_path)
            manifest_path = backup_dir / f"{backup_name}.manifest"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=4)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            self.logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            raise
    
    def restore_backup(self, backup_path: str, target_dir: Optional[str] = None) -> None:
        """Restore from a backup."""
        try:
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Verify backup integrity
            manifest_path = f"{backup_path}.manifest"
            if not os.path.exists(manifest_path):
                self.logger.warning("No manifest file found, skipping integrity check")
            else:
                self._verify_backup(backup_path, manifest_path)
            
            # Extract backup
            target = target_dir or "."
            with tarfile.open(backup_path, f"r:{self.config['compression']}") as tar:
                tar.extractall(path=target)
            
            self.logger.info(f"Backup restored successfully to {target}")
            
        except Exception as e:
            self.logger.error(f"Restore failed: {str(e)}")
            raise
    
    def _filter_file(self, tarinfo: tarfile.TarInfo, exclude_patterns: List[str]) -> Optional[tarfile.TarInfo]:
        """Filter files based on exclude patterns."""
        for pattern in exclude_patterns:
            if pattern in tarinfo.name:
                return None
        return tarinfo
    
    def _create_manifest(self, backup_path: str) -> Dict:
        """Create a manifest file for the backup."""
        manifest = {
            "timestamp": datetime.datetime.now().isoformat(),
            "backup_file": os.path.basename(backup_path),
            "files": {}
        }
        
        with tarfile.open(backup_path, f"r:{self.config['compression']}") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    manifest["files"][member.name] = {
                        "size": member.size,
                        "mtime": member.mtime
                    }
        
        return manifest
    
    def _verify_backup(self, backup_path: str, manifest_path: str) -> None:
        """Verify backup integrity using manifest."""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        with tarfile.open(backup_path, f"r:{self.config['compression']}") as tar:
            for filename, fileinfo in manifest["files"].items():
                try:
                    member = tar.getmember(filename)
                    if member.size != fileinfo["size"]:
                        raise ValueError(f"File size mismatch for {filename}")
                except KeyError:
                    raise ValueError(f"File {filename} missing from backup")
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backups keeping only the most recent ones."""
        backup_dir = Path(self.config["backup_location"])
        backups = sorted(backup_dir.glob(f"backup_*.tar.{self.config['compression']}"))
        
        if len(backups) > self.config["max_backups"]:
            for backup in backups[:-self.config["max_backups"]]:
                backup.unlink()
                manifest = backup.with_suffix(f".tar.{self.config['compression']}.manifest")
                if manifest.exists():
                    manifest.unlink()
                self.logger.info(f"Removed old backup: {backup}")

if __name__ == "__main__":
    # Example usage
    backup_system = BackupSystem()
    
    # Create a backup
    backup_path = backup_system.create_backup()
    print(f"Backup created at: {backup_path}")
    
    # Restore from backup (uncomment to use)
    # backup_system.restore_backup(backup_path) 