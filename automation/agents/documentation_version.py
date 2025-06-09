import logging
from typing import Dict, List, Optional, Union, Set
from pathlib import Path
import re
from datetime import datetime, timedelta
import json
import yaml
import git
import semver
import difflib
import hashlib
from dataclasses import dataclass, field
import aiofiles
import aiohttp
import asyncio
import boto3
import google.cloud.storage
import azure.storage.blob
from cachetools import TTLCache, LRUCache
from pydantic import BaseModel, Field, validator
import jwt
from .base_version_manager import BaseVersionManager, Version

class VersionValidationError(Exception):
    """Raised when version validation fails."""
    pass

class VersionAccessError(Exception):
    """Raised when access to a version is denied."""
    pass

class VersionMetrics(BaseModel):
    """Metrics for version operations."""
    views: int = 0
    comparisons: int = 0
    rollbacks: int = 0
    last_viewed: Optional[datetime] = None
    last_compared: Optional[datetime] = None
    last_rolled_back: Optional[datetime] = None

@dataclass
class Version:
    id: str
    doc_id: str
    version: str
    content: str
    changes: List[Dict]
    author: str
    timestamp: str
    metadata: Dict
    hash: str
    metrics: VersionMetrics = field(default_factory=VersionMetrics)
    access_control: Dict[str, List[str]] = field(default_factory=dict)

class DocumentationVersion(BaseVersionManager):
    """Documentation version management system."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.setup_storage()
        self.setup_caching()
        self.setup_security()
        self.setup_metrics()
    
    def setup_storage(self):
        """Setup version storage."""
        try:
            # Initialize version storage
            self.versions_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initialized version storage at {self.versions_path}")
        except Exception as e:
            self.logger.error(f"Storage setup error: {str(e)}")
            raise
    
    def setup_caching(self):
        """Setup version caching."""
        try:
            self.version_cache = TTLCache(
                maxsize=self.version_config.get('cache_size', 1000),
                ttl=self.version_config.get('cache_ttl', 3600)
            )
            self.logger.info("Initialized version cache")
        except Exception as e:
            self.logger.error(f"Cache setup error: {str(e)}")
            raise
    
    def setup_security(self):
        """Setup version security."""
        try:
            self.jwt_secret = self.version_config.get('jwt_secret', 'your-secret-key')
            self.logger.info("Initialized version security")
        except Exception as e:
            self.logger.error(f"Security setup error: {str(e)}")
            raise
    
    def setup_metrics(self):
        """Setup version metrics."""
        try:
            self.metrics = {
                'version_operations': {},
                'cache_hits': 0,
                'cache_misses': 0,
                'security_operations': {}
            }
            self.logger.info("Initialized version metrics")
        except Exception as e:
            self.logger.error(f"Metrics setup error: {str(e)}")
            raise
    
    def validate_version(self, version: Version) -> bool:
        """Validate version fields and content."""
        try:
            # Validate required fields
            if not version.version or not version.content:
                raise VersionValidationError("Version number and content are required")
            
            # Validate version format
            if not re.match(r'^\d+\.\d+\.\d+$', version.version):
                raise VersionValidationError("Invalid version format")
            
            # Validate content hash
            if version.hash != self._calculate_hash(version.content):
                raise VersionValidationError("Invalid content hash")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Version validation error: {str(e)}")
            raise
    
    def check_access(self, user_id: str, version: Version, operation: str) -> bool:
        """Check if user has access to perform operation on version."""
        try:
            # Get access control rules
            rules = version.access_control.get(operation, [])
            
            # Check if user has access
            return user_id in rules or 'admin' in rules
            
        except Exception as e:
            self.logger.error(f"Access check error: {str(e)}")
            return False
    
    def record_metrics(self, operation: str, version_id: str, success: bool):
        """Record operation metrics."""
        try:
            if operation not in self.metrics['version_operations']:
                self.metrics['version_operations'][operation] = {
                    'total': 0,
                    'success': 0,
                    'failure': 0,
                    'last_operation': None
                }
            
            metrics = self.metrics['version_operations'][operation]
            metrics['total'] += 1
            if success:
                metrics['success'] += 1
            else:
                metrics['failure'] += 1
            metrics['last_operation'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Metrics recording error: {str(e)}")
    
    async def create_version(
        self,
        doc_id: str,
        content: str,
        author: str,
        metadata: Optional[Dict] = None,
        user_id: str = None
    ) -> Version:
        """Create a new version of documentation with enhanced validation and security."""
        try:
            # Get current version
            current_version = self.get_latest_version(doc_id)
            
            # Calculate new version number
            if current_version:
                new_version = self._increment_version(current_version.version)
            else:
                new_version = '1.0.0'
            
            # Calculate content hash
            content_hash = self._calculate_hash(content)
            
            # Calculate changes
            changes = []
            if current_version:
                changes = self._calculate_changes(current_version.content, content)
            
            # Create version
            version = Version(
                id=f"{doc_id}_{new_version}",
                doc_id=doc_id,
                version=new_version,
                content=content,
                changes=changes,
                author=author,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {},
                hash=content_hash,
                access_control={
                    'read': [user_id] if user_id else [],
                    'write': [user_id] if user_id else [],
                    'delete': [user_id] if user_id else []
                }
            )
            
            # Validate version
            self.validate_version(version)
            
            # Store version
            await self._store_version(version)
            
            # Update versions cache
            if doc_id not in self.versions:
                self.versions[doc_id] = []
            self.versions[doc_id].append(version)
            
            # Cache version
            self.version_cache[version.id] = version
            
            # Record metrics
            self.record_metrics('create', version.id, True)
            
            self.logger.info(f"Created version {new_version} for document {doc_id}")
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to create version: {str(e)}")
            self.record_metrics('create', f"{doc_id}_{new_version}", False)
            raise
    
    async def get_version(self, doc_id: str, version: str, user_id: str = None) -> Optional[Version]:
        """Get a specific version with enhanced caching and security."""
        try:
            version_id = f"{doc_id}_{version}"
            
            # Check cache first
            if version_id in self.version_cache:
                cached_version = self.version_cache[version_id]
                self.metrics['cache_hits'] += 1
                
                # Check access
                if not self.check_access(user_id, cached_version, 'read'):
                    raise VersionAccessError(f"User {user_id} does not have read access to version {version_id}")
                
                # Update metrics
                cached_version.metrics.views += 1
                cached_version.metrics.last_viewed = datetime.now()
                
                return cached_version
            
            self.metrics['cache_misses'] += 1
            
            # Load version from storage
            version_path = self.versions_path / doc_id / f"{version}.md"
            if not version_path.exists():
                return None
            
            async with aiofiles.open(version_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse version
            version_data = json.loads(content)
            loaded_version = Version(**version_data)
            
            # Check access
            if not self.check_access(user_id, loaded_version, 'read'):
                raise VersionAccessError(f"User {user_id} does not have read access to version {version_id}")
            
            # Update metrics
            loaded_version.metrics.views += 1
            loaded_version.metrics.last_viewed = datetime.now()
            
            # Cache version
            self.version_cache[version_id] = loaded_version
            
            return loaded_version
            
        except Exception as e:
            self.logger.error(f"Failed to get version: {str(e)}")
            return None
    
    async def compare_versions(
        self,
        doc_id: str,
        version1: str,
        version2: str,
        user_id: str = None
    ) -> Dict:
        """Compare two versions with enhanced security."""
        try:
            v1 = await self.get_version(doc_id, version1, user_id)
            v2 = await self.get_version(doc_id, version2, user_id)
            
            if not v1 or not v2:
                raise ValueError("One or both versions not found")
            
            # Update metrics
            v1.metrics.comparisons += 1
            v1.metrics.last_compared = datetime.now()
            v2.metrics.comparisons += 1
            v2.metrics.last_compared = datetime.now()
            
            return {
                'version1': v1.version,
                'version2': v2.version,
                'changes': self._calculate_changes(v1.content, v2.content),
                'metadata_changes': self._compare_metadata(v1.metadata, v2.metadata)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {str(e)}")
            raise
    
    async def rollback_version(
        self,
        doc_id: str,
        version: str,
        user_id: str = None
    ) -> Version:
        """Rollback to a specific version with enhanced security."""
        try:
            target_version = await self.get_version(doc_id, version, user_id)
            if not target_version:
                raise ValueError(f"Version {version} not found")
            
            # Check access
            if not self.check_access(user_id, target_version, 'write'):
                raise VersionAccessError(f"User {user_id} does not have write access to version {version}")
            
            # Create new version with target content
            new_version = await self.create_version(
                doc_id=doc_id,
                content=target_version.content,
                author=f"Rollback to {version}",
                metadata={
                    'rollback': True,
                    'from_version': self.get_latest_version(doc_id).version,
                    'to_version': version
                },
                user_id=user_id
            )
            
            # Update metrics
            target_version.metrics.rollbacks += 1
            target_version.metrics.last_rolled_back = datetime.now()
            
            return new_version
            
        except Exception as e:
            self.logger.error(f"Failed to rollback version: {str(e)}")
            raise
    
    async def delete_version(
        self,
        doc_id: str,
        version: str,
        user_id: str = None
    ):
        """Delete a specific version with enhanced security."""
        try:
            version_id = f"{doc_id}_{version}"
            
            # Get version
            version_obj = await self.get_version(doc_id, version, user_id)
            if not version_obj:
                return
            
            # Check access
            if not self.check_access(user_id, version_obj, 'delete'):
                raise VersionAccessError(f"User {user_id} does not have delete access to version {version_id}")
            
            # Remove from storage
            storage_type = self.version_config.get('storage', 'local')
            
            if storage_type == 'local':
                version_path = self.storage_path / doc_id / f"{version}.json"
                if version_path.exists():
                    version_path.unlink()
            elif storage_type == 's3':
                bucket = self.version_config.get('s3', {}).get('bucket', '')
                key = f"{doc_id}/{version}.json"
                await asyncio.to_thread(
                    self.s3_client.delete_object,
                    Bucket=bucket,
                    Key=key
                )
            elif storage_type == 'gcs':
                bucket = self.gcs_client.bucket(
                    self.version_config.get('gcs', {}).get('bucket', '')
                )
                blob = bucket.blob(f"{doc_id}/{version}.json")
                await asyncio.to_thread(blob.delete)
            elif storage_type == 'azure':
                container = self.azure_client.get_container_client(
                    self.version_config.get('azure', {}).get('container', '')
                )
                blob = container.get_blob_client(f"{doc_id}/{version}.json")
                await asyncio.to_thread(blob.delete_blob)
            
            # Remove from cache
            if version_id in self.version_cache:
                del self.version_cache[version_id]
            
            # Remove from versions list
            if doc_id in self.versions:
                self.versions[doc_id] = [
                    v for v in self.versions[doc_id]
                    if v.version != version
                ]
            
            # Record metrics
            self.record_metrics('delete', version_id, True)
            
            self.logger.info(f"Deleted version {version} of document {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete version: {str(e)}")
            self.record_metrics('delete', version_id, False)
            raise
    
    async def cleanup_versions(
        self,
        doc_id: str,
        keep_versions: int = 5,
        user_id: str = None
    ):
        """Cleanup old versions with enhanced security."""
        try:
            versions = self.get_version_history(doc_id)
            
            if len(versions) <= keep_versions:
                return
            
            # Delete old versions
            for version in versions[:-keep_versions]:
                await self.delete_version(doc_id, version.version, user_id)
            
            # Record metrics
            self.record_metrics('cleanup', doc_id, True)
            
            self.logger.info(f"Cleaned up versions for document {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup versions: {str(e)}")
            self.record_metrics('cleanup', doc_id, False)
            raise

    def _increment_version(self, current_version: str) -> str:
        """Increment version number."""
        try:
            version = semver.VersionInfo.parse(current_version)
            return str(version.bump_patch())
        except ValueError:
            return '1.0.0'

    def _calculate_hash(self, content: str) -> str:
        """Calculate content hash."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_changes(self, old_content: str, new_content: str) -> List[Dict]:
        """Calculate changes between versions."""
        changes = []
        
        # Split content into lines
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        # Calculate diff
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm='',
            n=3
        )
        
        # Parse diff
        current_change = None
        for line in diff:
            if line.startswith('@@'):
                if current_change:
                    changes.append(current_change)
                current_change = {
                    'type': 'modification',
                    'lines': [],
                    'context': []
                }
            elif line.startswith('+'):
                current_change['lines'].append({
                    'type': 'addition',
                    'content': line[1:]
                })
            elif line.startswith('-'):
                current_change['lines'].append({
                    'type': 'deletion',
                    'content': line[1:]
                })
            else:
                current_change['context'].append(line)
        
        if current_change:
            changes.append(current_change)
        
        return changes

    def get_version_history(self, doc_id: str) -> List[Version]:
        """Get version history for documentation."""
        if doc_id in self.versions:
            return sorted(
                self.versions[doc_id],
                key=lambda v: semver.VersionInfo.parse(v.version)
            )
        return []

    def _compare_metadata(self, metadata1: Dict, metadata2: Dict) -> Dict:
        """Compare metadata between versions."""
        changes = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        # Find added and modified keys
        for key, value in metadata2.items():
            if key not in metadata1:
                changes['added'][key] = value
            elif metadata1[key] != value:
                changes['modified'][key] = {
                    'old': metadata1[key],
                    'new': value
                }
        
        # Find removed keys
        for key, value in metadata1.items():
            if key not in metadata2:
                changes['removed'][key] = value
        
        return changes

    async def _store_version(self, version: Version):
        """Store a version to disk."""
        try:
            version_dir = self.versions_path / version.doc_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            version_path = version_dir / f"{version.version}.md"
            with open(version_path, 'w', encoding='utf-8') as f:
                f.write(self._format_version(version))
            
            self.logger.info(f"Stored version {version.version} for document {version.doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store version: {str(e)}")
            raise

    def _format_version(self, version: Version) -> str:
        """Format a version as a string."""
        return f"{version.version}\n{version.content}"

    async def _store_version(self, version: Version):
        """Store a version to disk."""
        try:
            version_dir = self.versions_path / version.doc_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            version_path = version_dir / f"{version.version}.md"
            with open(version_path, 'w', encoding='utf-8') as f:
                f.write(self._format_version(version))
            
            self.logger.info(f"Stored version {version.version} for document {version.doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store version: {str(e)}")
            raise 