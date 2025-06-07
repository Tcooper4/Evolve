import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import re
from datetime import datetime
import json
import yaml
import tarfile
import zipfile
import shutil
import hashlib
from dataclasses import dataclass
import aiofiles
import aiohttp
import asyncio
import boto3
import google.cloud.storage
import azure.storage.blob
import cryptography
from cryptography.fernet import Fernet
import gnupg
import rsa
import pyzipper
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Backup:
    id: str
    timestamp: str
    type: str
    size: int
    hash: str
    metadata: Dict
    encryption_key: Optional[str]

@dataclass
class BackupConfig:
    id: str
    name: str
    description: str
    type: str
    storage: Dict[str, Any]
    schedule: Dict[str, Any]
    encryption: Dict[str, Any]
    retention: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    status: str
    stats: Dict[str, Any]

@dataclass
class BackupResult:
    id: str
    config_id: str
    start_time: datetime
    end_time: datetime
    status: str
    size: int
    checksum: str
    path: str
    metadata: Dict[str, Any]
    error: Optional[str]

class DocumentationBackup:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.backup_config = config.get('documentation', {}).get('backup', {})
        self.setup_storage()
        self.setup_encryption()
        self.configs: Dict[str, BackupConfig] = {}
        self.results: Dict[str, BackupResult] = {}
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.backups: Dict[str, Backup] = {}

    def setup_logging(self):
        """Configure logging for the documentation backup system."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "backup.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_storage(self):
        """Setup backup storage based on configuration."""
        storage_type = self.backup_config.get('storage', 'local')
        
        if storage_type == 'local':
            self.storage_path = Path(self.backup_config.get('local', {}).get('path', 'backups'))
            self.storage_path.mkdir(parents=True, exist_ok=True)
        elif storage_type == 's3':
            self.s3_client = boto3.client('s3')
        elif storage_type == 'gcs':
            self.gcs_client = google.cloud.storage.Client()
        elif storage_type == 'azure':
            self.azure_client = azure.storage.blob.BlobServiceClient.from_connection_string(
                self.backup_config.get('azure', {}).get('connection_string', '')
            )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    def setup_encryption(self):
        """Setup encryption for backups."""
        encryption_type = self.backup_config.get('encryption', {}).get('type', 'none')
        
        if encryption_type == 'fernet':
            key = self.backup_config.get('encryption', {}).get('key')
            if not key:
                key = Fernet.generate_key()
            self.encryption = Fernet(key)
        elif encryption_type == 'gpg':
            self.gpg = gnupg.GPG()
            self.gpg.import_keys(self.backup_config.get('encryption', {}).get('key'))
        elif encryption_type == 'rsa':
            key_size = self.backup_config.get('encryption', {}).get('key_size', 2048)
            self.public_key, self.private_key = rsa.newkeys(key_size)
        elif encryption_type == 'aes':
            self.encryption_key = self.backup_config.get('encryption', {}).get('key')
            if not self.encryption_key:
                self.encryption_key = os.urandom(32)
        else:
            self.encryption = None

    async def create_backup_config(
        self,
        name: str,
        description: str,
        type: str = 'full',
        storage: Optional[Dict] = None,
        schedule: Optional[Dict] = None,
        encryption: Optional[Dict] = None,
        retention: Optional[Dict] = None
    ) -> BackupConfig:
        """Create a new backup configuration."""
        try:
            async with self.lock:
                # Generate config ID
                config_id = f"backup_{hash(f'{name}_{datetime.now()}')}"
                
                # Create config
                backup_config = BackupConfig(
                    id=config_id,
                    name=name,
                    description=description,
                    type=type,
                    storage=storage or {},
                    schedule=schedule or {},
                    encryption=encryption or {},
                    retention=retention or {},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status='active',
                    stats={}
                )
                
                self.configs[config_id] = backup_config
                self.logger.info(f"Created backup config: {config_id}")
                return backup_config
                
        except Exception as e:
            self.logger.error(f"Backup config creation error: {str(e)}")
            raise

    async def delete_backup_config(self, config_id: str) -> bool:
        """Delete a backup configuration."""
        try:
            async with self.lock:
                if config_id not in self.configs:
                    raise ValueError(f"Backup config not found: {config_id}")
                
                # Delete config
                del self.configs[config_id]
                
                self.logger.info(f"Deleted backup config: {config_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Backup config deletion error: {str(e)}")
            return False

    async def create_backup(self, config_id: str) -> BackupResult:
        """Create a backup."""
        try:
            async with self.lock:
                if config_id not in self.configs:
                    raise ValueError(f"Backup config not found: {config_id}")
                
                backup_config = self.configs[config_id]
                
                # Generate backup ID
                backup_id = f"backup_{hash(f'{config_id}_{datetime.now()}')}"
                
                # Create backup result
                backup_result = BackupResult(
                    id=backup_id,
                    config_id=config_id,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    status='in_progress',
                    size=0,
                    checksum='',
                    path='',
                    metadata={},
                    error=None
                )
                
                # Create backup based on type
                if backup_config.type == 'full':
                    await self._create_full_backup(backup_config, backup_result)
                elif backup_config.type == 'incremental':
                    await self._create_incremental_backup(backup_config, backup_result)
                elif backup_config.type == 'differential':
                    await self._create_differential_backup(backup_config, backup_result)
                
                # Update backup result
                backup_result.end_time = datetime.now()
                backup_result.status = 'completed'
                
                self.results[backup_id] = backup_result
                self.logger.info(f"Created backup: {backup_id}")
                return backup_result
                
        except Exception as e:
            self.logger.error(f"Backup creation error: {str(e)}")
            if backup_result:
                backup_result.status = 'failed'
                backup_result.error = str(e)
            raise

    async def _create_full_backup(self, config: BackupConfig, result: BackupResult):
        """Create a full backup."""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Archive documentation
                archive_path = Path(temp_dir) / f"backup_{result.id}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add("documentation", arcname="documentation")
                
                # Encrypt archive if needed
                if config.encryption.get('enabled', False):
                    archive_path = await self._encrypt_file(archive_path)
                
                # Calculate checksum
                result.checksum = await self._calculate_checksum(archive_path)
                result.size = archive_path.stat().st_size
                
                # Upload to storage
                result.path = await self._upload_to_storage(
                    config.storage,
                    archive_path,
                    f"backups/{result.id}.tar.gz"
                )
                
        except Exception as e:
            self.logger.error(f"Full backup creation error: {str(e)}")
            raise

    async def _create_incremental_backup(self, config: BackupConfig, result: BackupResult):
        """Create an incremental backup."""
        try:
            # Get last backup
            last_backup = await self._get_last_backup(config.id)
            if not last_backup:
                # If no previous backup, create full backup
                await self._create_full_backup(config, result)
                return
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get changed files
                changed_files = await self._get_changed_files(last_backup)
                
                # Archive changed files
                archive_path = Path(temp_dir) / f"backup_{result.id}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    for file in changed_files:
                        tar.add(file, arcname=file)
                
                # Encrypt archive if needed
                if config.encryption.get('enabled', False):
                    archive_path = await self._encrypt_file(archive_path)
                
                # Calculate checksum
                result.checksum = await self._calculate_checksum(archive_path)
                result.size = archive_path.stat().st_size
                
                # Upload to storage
                result.path = await self._upload_to_storage(
                    config.storage,
                    archive_path,
                    f"backups/{result.id}.tar.gz"
                )
                
        except Exception as e:
            self.logger.error(f"Incremental backup creation error: {str(e)}")
            raise

    async def _create_differential_backup(self, config: BackupConfig, result: BackupResult):
        """Create a differential backup."""
        try:
            # Get last full backup
            last_full_backup = await self._get_last_full_backup(config.id)
            if not last_full_backup:
                # If no previous full backup, create full backup
                await self._create_full_backup(config, result)
                return
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get changed files since last full backup
                changed_files = await self._get_changed_files(last_full_backup)
                
                # Archive changed files
                archive_path = Path(temp_dir) / f"backup_{result.id}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    for file in changed_files:
                        tar.add(file, arcname=file)
                
                # Encrypt archive if needed
                if config.encryption.get('enabled', False):
                    archive_path = await self._encrypt_file(archive_path)
                
                # Calculate checksum
                result.checksum = await self._calculate_checksum(archive_path)
                result.size = archive_path.stat().st_size
                
                # Upload to storage
                result.path = await self._upload_to_storage(
                    config.storage,
                    archive_path,
                    f"backups/{result.id}.tar.gz"
                )
                
        except Exception as e:
            self.logger.error(f"Differential backup creation error: {str(e)}")
            raise

    async def _restore_full_backup(self, result: BackupResult, target_path: str):
        """Restore a full backup."""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download from storage
                archive_path = Path(temp_dir) / f"backup_{result.id}.tar.gz"
                await self._download_from_storage(
                    result.metadata['storage'],
                    result.path,
                    archive_path
                )
                
                # Verify checksum
                if not await self._verify_checksum(archive_path, result.checksum):
                    raise ValueError("Checksum verification failed")
                
                # Decrypt archive if needed
                if result.metadata.get('encrypted', False):
                    archive_path = await self._decrypt_file(archive_path)
                
                # Extract archive
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(target_path)
                
        except Exception as e:
            self.logger.error(f"Full backup restoration error: {str(e)}")
            raise

    async def _restore_incremental_backup(self, result: BackupResult, target_path: str):
        """Restore an incremental backup."""
        try:
            # Get base backup
            base_backup = await self._get_base_backup(result.config_id)
            if not base_backup:
                raise ValueError("Base backup not found")
            
            # Restore base backup
            await self._restore_full_backup(base_backup, target_path)
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download from storage
                archive_path = Path(temp_dir) / f"backup_{result.id}.tar.gz"
                await self._download_from_storage(
                    result.metadata['storage'],
                    result.path,
                    archive_path
                )
                
                # Verify checksum
                if not await self._verify_checksum(archive_path, result.checksum):
                    raise ValueError("Checksum verification failed")
                
                # Decrypt archive if needed
                if result.metadata.get('encrypted', False):
                    archive_path = await self._decrypt_file(archive_path)
                
                # Extract archive
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(target_path)
                
        except Exception as e:
            self.logger.error(f"Incremental backup restoration error: {str(e)}")
            raise

    async def _restore_differential_backup(self, result: BackupResult, target_path: str):
        """Restore a differential backup."""
        try:
            # Get base backup
            base_backup = await self._get_base_backup(result.config_id)
            if not base_backup:
                raise ValueError("Base backup not found")
            
            # Restore base backup
            await self._restore_full_backup(base_backup, target_path)
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download from storage
                archive_path = Path(temp_dir) / f"backup_{result.id}.tar.gz"
                await self._download_from_storage(
                    result.metadata['storage'],
                    result.path,
                    archive_path
                )
                
                # Verify checksum
                if not await self._verify_checksum(archive_path, result.checksum):
                    raise ValueError("Checksum verification failed")
                
                # Decrypt archive if needed
                if result.metadata.get('encrypted', False):
                    archive_path = await self._decrypt_file(archive_path)
                
                # Extract archive
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(target_path)
                
        except Exception as e:
            self.logger.error(f"Differential backup restoration error: {str(e)}")
            raise

    async def _get_last_backup(self, config_id: str) -> Optional[BackupResult]:
        """Get the last backup for a configuration."""
        try:
            backups = [
                result for result in self.results.values()
                if result.config_id == config_id
            ]
            if not backups:
                return None
            return max(backups, key=lambda x: x.end_time)
            
        except Exception as e:
            self.logger.error(f"Last backup retrieval error: {str(e)}")
            return None

    async def _get_last_full_backup(self, config_id: str) -> Optional[BackupResult]:
        """Get the last full backup for a configuration."""
        try:
            backups = [
                result for result in self.results.values()
                if result.config_id == config_id and
                self.configs[result.config_id].type == 'full'
            ]
            if not backups:
                return None
            return max(backups, key=lambda x: x.end_time)
            
        except Exception as e:
            self.logger.error(f"Last full backup retrieval error: {str(e)}")
            return None

    async def _get_base_backup(self, config_id: str) -> Optional[BackupResult]:
        """Get the base backup for a configuration."""
        try:
            return await self._get_last_full_backup(config_id)
            
        except Exception as e:
            self.logger.error(f"Base backup retrieval error: {str(e)}")
            return None

    async def _get_changed_files(self, last_backup: BackupResult) -> List[str]:
        """Get files changed since last backup."""
        try:
            changed_files = []
            
            # Get last backup files
            last_backup_files = set()
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download last backup
                archive_path = Path(temp_dir) / f"backup_{last_backup.id}.tar.gz"
                await self._download_from_storage(
                    last_backup.metadata['storage'],
                    last_backup.path,
                    archive_path
                )
                
                # Decrypt if needed
                if last_backup.metadata.get('encrypted', False):
                    archive_path = await self._decrypt_file(archive_path)
                
                # Extract archive
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
                
                # Get file list
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        last_backup_files.add(
                            os.path.relpath(os.path.join(root, file), temp_dir)
                        )
            
            # Get current files
            current_files = set()
            for root, _, files in os.walk("documentation"):
                for file in files:
                    current_files.add(
                        os.path.relpath(os.path.join(root, file), "documentation")
                    )
            
            # Find changed files
            for file in current_files:
                if file not in last_backup_files:
                    changed_files.append(file)
                else:
                    # Check if file was modified
                    current_path = os.path.join("documentation", file)
                    last_backup_path = os.path.join(temp_dir, file)
                    if os.path.getmtime(current_path) > os.path.getmtime(last_backup_path):
                        changed_files.append(file)
            
            return changed_files
            
        except Exception as e:
            self.logger.error(f"Changed files retrieval error: {str(e)}")
            return []

    async def _encrypt_file(self, file_path: Path) -> Path:
        """Encrypt a file."""
        try:
            if not self.encryption:
                return file_path
            
            # Read file
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Encrypt data
            encrypted_data = self.encryption.encrypt(data)
            
            # Write encrypted data
            encrypted_path = file_path.with_suffix('.enc')
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            return encrypted_path
            
        except Exception as e:
            self.logger.error(f"File encryption error: {str(e)}")
            raise

    async def _decrypt_file(self, file_path: Path) -> Path:
        """Decrypt a file."""
        try:
            if not self.encryption:
                return file_path
            
            # Read file
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Decrypt data
            decrypted_data = self.encryption.decrypt(data)
            
            # Write decrypted data
            decrypted_path = file_path.with_suffix('.dec')
            with open(decrypted_path, 'wb') as f:
                f.write(decrypted_data)
            
            return decrypted_path
            
        except Exception as e:
            self.logger.error(f"File decryption error: {str(e)}")
            raise

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        try:
            sha256_hash = hashlib.sha256()
            
            # Read file in chunks
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Checksum calculation error: {str(e)}")
            raise

    async def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            actual_checksum = await self._calculate_checksum(file_path)
            return actual_checksum == expected_checksum
            
        except Exception as e:
            self.logger.error(f"Checksum verification error: {str(e)}")
            return False

    async def _upload_to_storage(
        self,
        storage_config: Dict,
        file_path: Path,
        remote_path: str
    ) -> str:
        """Upload file to storage."""
        try:
            storage_type = storage_config.get('type', 'local')
            
            if storage_type == 's3':
                # Upload to S3
                self.s3_client.upload_file(
                    str(file_path),
                    storage_config['bucket'],
                    remote_path
                )
                return f"s3://{storage_config['bucket']}/{remote_path}"
                
            elif storage_type == 'azure':
                # Upload to Azure Blob Storage
                container = self.azure_client.get_container_client(
                    storage_config['container']
                )
                blob = container.get_blob_client(remote_path)
                with open(file_path, 'rb') as f:
                    blob.upload_blob(f)
                return f"azure://{storage_config['container']}/{remote_path}"
                
            elif storage_type == 'gcs':
                # Upload to Google Cloud Storage
                bucket = self.gcs_client.bucket(storage_config['bucket'])
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(str(file_path))
                return f"gcs://{storage_config['bucket']}/{remote_path}"
                
            elif storage_type == 'dropbox':
                # Upload to Dropbox
                with open(file_path, 'rb') as f:
                    self.dropbox.files_upload(f.read(), remote_path)
                return f"dropbox://{remote_path}"
                
            else:
                # Local storage
                target_path = Path(storage_config.get('path', 'backups')) / remote_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_path)
                return str(target_path)
            
        except Exception as e:
            self.logger.error(f"Storage upload error: {str(e)}")
            raise

    async def _download_from_storage(
        self,
        storage_config: Dict,
        remote_path: str,
        file_path: Path
    ):
        """Download file from storage."""
        try:
            storage_type = storage_config.get('type', 'local')
            
            if storage_type == 's3':
                # Download from S3
                self.s3_client.download_file(
                    storage_config['bucket'],
                    remote_path,
                    str(file_path)
                )
                
            elif storage_type == 'azure':
                # Download from Azure Blob Storage
                container = self.azure_client.get_container_client(
                    storage_config['container']
                )
                blob = container.get_blob_client(remote_path)
                with open(file_path, 'wb') as f:
                    f.write(blob.download_blob().readall())
                
            elif storage_type == 'gcs':
                # Download from Google Cloud Storage
                bucket = self.gcs_client.bucket(storage_config['bucket'])
                blob = bucket.blob(remote_path)
                blob.download_to_filename(str(file_path))
                
            elif storage_type == 'dropbox':
                # Download from Dropbox
                self.dropbox.files_download_to_file(
                    str(file_path),
                    remote_path
                )
                
            else:
                # Local storage
                source_path = Path(storage_config.get('path', 'backups')) / remote_path
                shutil.copy2(source_path, file_path)
            
        except Exception as e:
            self.logger.error(f"Storage download error: {str(e)}")
            raise

    async def cleanup_old_backups(self, config_id: str):
        """Clean up old backups based on retention policy."""
        try:
            async with self.lock:
                if config_id not in self.configs:
                    raise ValueError(f"Backup config not found: {config_id}")
                
                backup_config = self.configs[config_id]
                retention = backup_config.retention
                
                # Get backups to delete
                backups = [
                    result for result in self.results.values()
                    if result.config_id == config_id
                ]
                
                # Sort by end time
                backups.sort(key=lambda x: x.end_time)
                
                # Keep backups based on retention policy
                if retention.get('keep_last', 0) > 0:
                    backups_to_keep = backups[-retention['keep_last']:]
                else:
                    backups_to_keep = []
                
                if retention.get('keep_daily', 0) > 0:
                    daily_backups = {}
                    for backup in backups:
                        day = backup.end_time.date()
                        if day not in daily_backups:
                            daily_backups[day] = []
                        daily_backups[day].append(backup)
                    
                    for day in sorted(daily_backups.keys(), reverse=True)[:retention['keep_daily']]:
                        backups_to_keep.extend(daily_backups[day])
                
                if retention.get('keep_weekly', 0) > 0:
                    weekly_backups = {}
                    for backup in backups:
                        week = backup.end_time.isocalendar()[1]
                        if week not in weekly_backups:
                            weekly_backups[week] = []
                        weekly_backups[week].append(backup)
                    
                    for week in sorted(weekly_backups.keys(), reverse=True)[:retention['keep_weekly']]:
                        backups_to_keep.extend(weekly_backups[week])
                
                if retention.get('keep_monthly', 0) > 0:
                    monthly_backups = {}
                    for backup in backups:
                        month = backup.end_time.month
                        if month not in monthly_backups:
                            monthly_backups[month] = []
                        monthly_backups[month].append(backup)
                    
                    for month in sorted(monthly_backups.keys(), reverse=True)[:retention['keep_monthly']]:
                        backups_to_keep.extend(monthly_backups[month])
                
                # Delete backups not in keep list
                for backup in backups:
                    if backup not in backups_to_keep:
                        await self._delete_backup(backup)
                
                self.logger.info(f"Cleaned up old backups for config: {config_id}")
                
        except Exception as e:
            self.logger.error(f"Backup cleanup error: {str(e)}")
            raise

    async def _delete_backup(self, backup: BackupResult):
        """Delete a backup."""
        try:
            # Delete from storage
            await self._delete_from_storage(
                backup.metadata['storage'],
                backup.path
            )
            
            # Remove from results
            del self.results[backup.id]
            
            self.logger.info(f"Deleted backup: {backup.id}")
            
        except Exception as e:
            self.logger.error(f"Backup deletion error: {str(e)}")
            raise

    async def _delete_from_storage(self, storage_config: Dict, remote_path: str):
        """Delete file from storage."""
        try:
            storage_type = storage_config.get('type', 'local')
            
            if storage_type == 's3':
                # Delete from S3
                self.s3_client.delete_object(
                    Bucket=storage_config['bucket'],
                    Key=remote_path
                )
                
            elif storage_type == 'azure':
                # Delete from Azure Blob Storage
                container = self.azure_client.get_container_client(
                    storage_config['container']
                )
                blob = container.get_blob_client(remote_path)
                blob.delete_blob()
                
            elif storage_type == 'gcs':
                # Delete from Google Cloud Storage
                bucket = self.gcs_client.bucket(storage_config['bucket'])
                blob = bucket.blob(remote_path)
                blob.delete()
                
            elif storage_type == 'dropbox':
                # Delete from Dropbox
                self.dropbox.files_delete(remote_path)
                
            else:
                # Local storage
                file_path = Path(storage_config.get('path', 'backups')) / remote_path
                if file_path.exists():
                    file_path.unlink()
            
        except Exception as e:
            self.logger.error(f"Storage deletion error: {str(e)}")
            raise 