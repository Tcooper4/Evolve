import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import re
from datetime import datetime
import json
import yaml
import git
import semver
import difflib
import hashlib
from dataclasses import dataclass
import aiofiles
import aiohttp
import asyncio
import boto3
import google.cloud.storage
import azure.storage.blob
from .base_version_manager import BaseVersionManager, Version

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

class DocumentationVersion(BaseVersionManager):
    """Documentation version management system."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.setup_storage()
    
    def setup_storage(self):
        """Setup version storage."""
        try:
            # Initialize version storage
            self.versions_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initialized version storage at {self.versions_path}")
        except Exception as e:
            self.logger.error(f"Storage setup error: {str(e)}")
            raise
    
    async def create_version(
        self,
        doc_id: str,
        content: str,
        author: str,
        metadata: Optional[Dict] = None
    ) -> Version:
        """Create a new version of documentation."""
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
                hash=content_hash
            )
            
            # Store version
            await self._store_version(version)
            
            # Update versions cache
            if doc_id not in self.versions:
                self.versions[doc_id] = []
            self.versions[doc_id].append(version)
            
            self.logger.info(f"Created version {new_version} for document {doc_id}")
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to create version: {str(e)}")
            raise
    
    def get_latest_version(self, doc_id: str) -> Optional[Version]:
        """Get the latest version of a document."""
        if doc_id in self.versions and self.versions[doc_id]:
            return self.versions[doc_id][-1]
        return None
    
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

    async def compare_versions(
        self,
        doc_id: str,
        version1: str,
        version2: str
    ) -> Dict:
        """Compare two versions of documentation."""
        v1 = await self.get_version(doc_id, version1)
        v2 = await self.get_version(doc_id, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        return {
            'version1': v1.version,
            'version2': v2.version,
            'changes': self._calculate_changes(v1.content, v2.content),
            'metadata_changes': self._compare_metadata(v1.metadata, v2.metadata)
        }

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

    async def rollback_version(self, doc_id: str, version: str) -> Version:
        """Rollback to a specific version."""
        target_version = await self.get_version(doc_id, version)
        if not target_version:
            raise ValueError(f"Version {version} not found")
        
        # Create new version with target content
        return await self.create_version(
            doc_id=doc_id,
            content=target_version.content,
            author=f"Rollback to {version}",
            metadata={
                'rollback': True,
                'from_version': self.get_latest_version(doc_id).version,
                'to_version': version
            }
        )

    async def delete_version(self, doc_id: str, version: str):
        """Delete a specific version."""
        try:
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
            if doc_id in self.versions:
                self.versions[doc_id] = [
                    v for v in self.versions[doc_id]
                    if v.version != version
                ]
            
            self.logger.info(f"Deleted version {version} of document {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete version: {str(e)}")
            raise

    async def cleanup_versions(
        self,
        doc_id: str,
        keep_versions: int = 5
    ):
        """Cleanup old versions, keeping only the specified number of latest versions."""
        try:
            versions = self.get_version_history(doc_id)
            
            if len(versions) <= keep_versions:
                return
            
            # Delete old versions
            for version in versions[:-keep_versions]:
                await self.delete_version(doc_id, version.version)
            
            self.logger.info(f"Cleaned up versions for document {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup versions: {str(e)}")
            raise 