import logging
from typing import Dict, List, Optional, Union, Any, Set
from pathlib import Path
import json
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass, field
import git
from git import Repo
import yaml
import frontmatter
import mistune
from mistune import Markdown
import markdown
from bs4 import BeautifulSoup
import aiofiles
import babel
from babel import Locale
import polib
import sphinx
from sphinx.application import Sphinx
import docutils
from docutils.core import publish_doctree
import rst2html
import rst2pdf
import doc8
import restructuredtext_lint
import markdownlint
import remark
import prettier
import black
import isort
import mypy
import pyright
import pytype
import pyre
import jedi
import rope
import autopep8
import yapf
import flake8
import pydocstyle
import radon
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.visitors import ComplexityVisitor
import mccabe
import xenon
import vulture
import pyflakes
import pycodestyle
import shutil
from cachetools import TTLCache, LRUCache
from ratelimit import limits, sleep_and_retry
from pydantic import BaseModel, Field, validator
import jwt
import bleach
from .documentation_version import DocumentationVersion, Version

class DocumentValidationError(Exception):
    """Raised when document validation fails."""
    pass

class DocumentAccessError(Exception):
    """Raised when access to a document is denied."""
    pass

class DocumentLockError(Exception):
    """Raised when document locking fails."""
    pass

class DocumentMetrics(BaseModel):
    """Metrics for document operations."""
    views: int = 0
    edits: int = 0
    comments: int = 0
    reviews: int = 0
    last_viewed: Optional[datetime] = None
    last_edited: Optional[datetime] = None
    last_commented: Optional[datetime] = None
    last_reviewed: Optional[datetime] = None
    average_edit_time: Optional[float] = None
    total_edit_time: float = 0.0
    edit_count: int = 0

class DocumentLock(BaseModel):
    """Lock information for a document."""
    user_id: str
    locked_at: datetime
    expires_at: datetime
    reason: Optional[str] = None

class DocumentComment(BaseModel):
    """Comment on a document."""
    id: str
    user_id: str
    content: str
    created_at: datetime
    updated_at: datetime
    parent_id: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

class DocumentReview(BaseModel):
    """Review of a document."""
    id: str
    user_id: str
    status: str
    comments: List[DocumentComment] = []
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    feedback: Optional[str] = None

@dataclass
class Document:
    id: str
    title: str
    content: str
    language: str
    version: str
    status: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    author: str
    reviewers: List[str]
    tags: List[str]
    categories: List[str]
    dependencies: List[str]
    translations: Dict[str, str]
    history: List[Dict[str, Any]]
    authors: List[str]
    metrics: DocumentMetrics = field(default_factory=DocumentMetrics)
    lock: Optional[DocumentLock] = None
    comments: List[DocumentComment] = field(default_factory=list)
    reviews: List[DocumentReview] = field(default_factory=list)
    access_control: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class DocumentTemplate:
    id: str
    name: str
    description: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    author: str
    tags: List[str]
    categories: List[str]
    variables: List[str]
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    access_control: Dict[str, List[str]] = field(default_factory=dict)

class DocumentationManager:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.docs_config = config.get('documentation', {}).get('management', {})
        self.setup_repository()
        self.setup_templates()
        self.setup_languages()
        self.documents: Dict[str, Document] = {}
        self.templates: Dict[str, DocumentTemplate] = {}
        self.version_manager = DocumentationVersion(config)
        
        # Setup caching
        self.document_cache = TTLCache(
            maxsize=self.docs_config.get('cache_size', 1000),
            ttl=self.docs_config.get('cache_ttl', 3600)
        )
        self.template_cache = LRUCache(
            maxsize=self.docs_config.get('template_cache_size', 100)
        )
        
        # Setup locks
        self.lock = asyncio.Lock()
        self.document_locks: Dict[str, asyncio.Lock] = {}
        
        # Setup paths
        self.docs_path = Path(config.get('documentation', {}).get('docs_path', 'documentation'))
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.templates_path = self.docs_path / 'templates'
        self.templates_path.mkdir(parents=True, exist_ok=True)
        
        # Setup security
        self.jwt_secret = self.docs_config.get('jwt_secret', 'your-secret-key')
        self.allowed_tags = self.docs_config.get('allowed_tags', ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'a', 'code', 'pre'])
        self.allowed_attrs = self.docs_config.get('allowed_attrs', {'a': ['href', 'title'], 'img': ['src', 'alt']})
        
        # Setup metrics
        self.metrics = {
            'document_operations': {},
            'template_operations': {},
            'version_operations': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'lock_operations': {},
            'security_operations': {}
        }

    def setup_logging(self):
        """Configure logging for the documentation management system."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "management.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_repository(self):
        """Setup Git repository for version control."""
        try:
            repo_path = Path(self.docs_config.get('repository_path', 'documentation'))
            if not repo_path.exists():
                repo_path.mkdir(parents=True)
                self.repo = Repo.init(repo_path)
            else:
                self.repo = Repo(repo_path)
            
            self.logger.info(f"Initialized Git repository at {repo_path}")
            
        except Exception as e:
            self.logger.error(f"Repository setup error: {str(e)}")
            raise

    def setup_templates(self):
        """Load document templates."""
        try:
            templates_path = Path(self.docs_config.get('templates_path', 'documentation/templates'))
            if templates_path.exists():
                for template_file in templates_path.glob('*.md'):
                    template = self._load_template(template_file)
                    if template:
                        self.templates[template.id] = template
                        self.template_cache[template.id] = template
            
            self.logger.info(f"Loaded {len(self.templates)} templates")
            
        except Exception as e:
            self.logger.error(f"Template setup error: {str(e)}")
            raise

    def setup_languages(self):
        """Setup supported languages."""
        try:
            self.languages = self.docs_config.get('languages', ['en'])
            self.default_language = self.docs_config.get('default_language', 'en')
            
            # Initialize language-specific directories
            for lang in self.languages:
                lang_path = Path(f"documentation/{lang}")
                lang_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Setup {len(self.languages)} languages")
            
        except Exception as e:
            self.logger.error(f"Language setup error: {str(e)}")
            raise

    def validate_document(self, document: Document) -> bool:
        """Validate document fields and content."""
        try:
            # Validate required fields
            if not document.title or not document.content:
                raise DocumentValidationError("Title and content are required")
            
            # Validate language
            if document.language not in self.languages:
                raise DocumentValidationError(f"Unsupported language: {document.language}")
            
            # Validate version format
            if not re.match(r'^\d+\.\d+\.\d+$', document.version):
                raise DocumentValidationError("Invalid version format")
            
            # Validate status
            valid_statuses = ['draft', 'review', 'approved', 'published', 'archived']
            if document.status not in valid_statuses:
                raise DocumentValidationError(f"Invalid status: {document.status}")
            
            # Validate content
            if len(document.content) > self.docs_config.get('max_content_length', 1000000):
                raise DocumentValidationError("Content exceeds maximum length")
            
            # Sanitize content
            document.content = self.sanitize_content(document.content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Document validation error: {str(e)}")
            raise

    def sanitize_content(self, content: str) -> str:
        """Sanitize document content."""
        try:
            # Convert to HTML if needed
            if not content.startswith('<'):
                content = markdown.markdown(content)
            
            # Sanitize HTML
            content = bleach.clean(
                content,
                tags=self.allowed_tags,
                attributes=self.allowed_attrs,
                strip=True
            )
            
            return content
            
        except Exception as e:
            self.logger.error(f"Content sanitization error: {str(e)}")
            return content

    def check_access(self, user_id: str, document: Document, operation: str) -> bool:
        """Check if user has access to perform operation on document."""
        try:
            # Get access control rules
            rules = document.access_control.get(operation, [])
            
            # Check if user has access
            return user_id in rules or 'admin' in rules
            
        except Exception as e:
            self.logger.error(f"Access check error: {str(e)}")
            return False

    async def acquire_lock(self, doc_id: str, user_id: str, duration: int = 300) -> bool:
        """Acquire lock on document."""
        try:
            if doc_id not in self.document_locks:
                self.document_locks[doc_id] = asyncio.Lock()
            
            async with self.document_locks[doc_id]:
                document = await self.get_document(doc_id)
                if not document:
                    return False
                
                # Check if document is already locked
                if document.lock and document.lock.expires_at > datetime.now():
                    return False
                
                # Create new lock
                document.lock = DocumentLock(
                    user_id=user_id,
                    locked_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=duration)
                )
                
                # Save document
                await self._save_document(document)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Lock acquisition error: {str(e)}")
            return False

    async def release_lock(self, doc_id: str, user_id: str) -> bool:
        """Release lock on document."""
        try:
            if doc_id not in self.document_locks:
                return False
            
            async with self.document_locks[doc_id]:
                document = await self.get_document(doc_id)
                if not document or not document.lock:
                    return False
                
                # Check if user owns the lock
                if document.lock.user_id != user_id:
                    return False
                
                # Remove lock
                document.lock = None
                
                # Save document
                await self._save_document(document)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Lock release error: {str(e)}")
            return False

    def record_metrics(self, operation: str, document_id: str, success: bool):
        """Record operation metrics."""
        try:
            if operation not in self.metrics['document_operations']:
                self.metrics['document_operations'][operation] = {
                    'total': 0,
                    'success': 0,
                    'failure': 0,
                    'last_operation': None
                }
            
            metrics = self.metrics['document_operations'][operation]
            metrics['total'] += 1
            if success:
                metrics['success'] += 1
            else:
                metrics['failure'] += 1
            metrics['last_operation'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Metrics recording error: {str(e)}")

    def get_health_status(self) -> Dict:
        """Get health status of documentation system."""
        try:
            return {
                'status': 'healthy',
                'metrics': {
                    'documents': len(self.documents),
                    'templates': len(self.templates),
                    'cache': {
                        'size': len(self.document_cache),
                        'hits': self.metrics['cache_hits'],
                        'misses': self.metrics['cache_misses']
                    },
                    'operations': self.metrics['document_operations']
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check error: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    async def create_document(
        self,
        title: str,
        content: str,
        language: str = None,
        template_id: str = None,
        metadata: Optional[Dict] = None,
        user_id: str = None
    ) -> Document:
        """Create a new document with enhanced validation and security."""
        try:
            async with self.lock:
                # Generate document ID
                doc_id = f"doc_{hash(f'{title}_{datetime.now()}')}"
                
                # Set language
                if not language:
                    language = self.default_language
                
                # Apply template if specified
                if template_id and template_id in self.templates:
                    template = self.templates[template_id]
                    content = self._apply_template(content, template)
                
                # Create document
                document = Document(
                    id=doc_id,
                    title=title,
                    content=content,
                    language=language,
                    version='1.0.0',
                    status='draft',
                    metadata=metadata or {},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    author=user_id or self.docs_config.get('default_author', 'system'),
                    reviewers=[],
                    tags=[],
                    categories=[],
                    dependencies=[],
                    translations={},
                    history=[],
                    authors=[user_id or self.docs_config.get('default_author', 'system')],
                    metrics=DocumentMetrics(),
                    access_control={
                        'read': [user_id] if user_id else [],
                        'write': [user_id] if user_id else [],
                        'delete': [user_id] if user_id else []
                    }
                )
                
                # Validate document
                self.validate_document(document)
                
                # Save document
                await self._save_document(document)
                
                # Create initial version
                await self._create_version(document)
                
                # Cache document
                self.document_cache[doc_id] = document
                
                # Record metrics
                self.record_metrics('create', doc_id, True)
                
                self.logger.info(f"Created document: {doc_id}")
                return document
                
        except Exception as e:
            self.logger.error(f"Document creation error: {str(e)}")
            self.record_metrics('create', doc_id, False)
            raise

    async def update_document(
        self,
        doc_id: str,
        content: str = None,
        metadata: Optional[Dict] = None,
        user_id: str = None
    ) -> Document:
        """Update an existing document with enhanced validation and security."""
        try:
            async with self.lock:
                if doc_id not in self.documents:
                    raise ValueError(f"Document not found: {doc_id}")
                
                document = self.documents[doc_id]
                
                # Check access
                if not self.check_access(user_id, document, 'write'):
                    raise DocumentAccessError(f"User {user_id} does not have write access to document {doc_id}")
                
                # Check lock
                if document.lock and document.lock.user_id != user_id:
                    raise DocumentLockError(f"Document {doc_id} is locked by {document.lock.user_id}")
                
                # Update content if provided
                if content:
                    document.content = content
                
                # Update metadata if provided
                if metadata:
                    document.metadata.update(metadata)
                
                # Update timestamp
                document.updated_at = datetime.now()
                
                # Update metrics
                document.metrics.edits += 1
                document.metrics.last_edited = datetime.now()
                
                # Add to history
                document.history.append({
                    'action': 'update',
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'changes': self._get_changes(document, content) if content else []
                })
                
                # Validate document
                self.validate_document(document)
                
                # Save document
                await self._save_document(document)
                
                # Create new version
                await self._create_version(document)
                
                # Update cache
                self.document_cache[doc_id] = document
                
                # Record metrics
                self.record_metrics('update', doc_id, True)
                
                self.logger.info(f"Updated document: {doc_id}")
                return document
                
        except Exception as e:
            self.logger.error(f"Document update error: {str(e)}")
            self.record_metrics('update', doc_id, False)
            raise

    async def delete_document(self, doc_id: str, user_id: str = None) -> bool:
        """Delete a document with enhanced security."""
        try:
            async with self.lock:
                if doc_id not in self.documents:
                    raise ValueError(f"Document not found: {doc_id}")
                
                document = self.documents[doc_id]
                
                # Check access
                if not self.check_access(user_id, document, 'delete'):
                    raise DocumentAccessError(f"User {user_id} does not have delete access to document {doc_id}")
                
                # Check lock
                if document.lock and document.lock.user_id != user_id:
                    raise DocumentLockError(f"Document {doc_id} is locked by {document.lock.user_id}")
                
                # Delete document file
                doc_path = Path(f"documentation/{document.language}/{doc_id}.md")
                if doc_path.exists():
                    doc_path.unlink()
                
                # Remove from memory
                del self.documents[doc_id]
                
                # Remove from cache
                if doc_id in self.document_cache:
                    del self.document_cache[doc_id]
                
                # Remove versions
                if doc_id in self.version_manager.versions:
                    del self.version_manager.versions[doc_id]
                
                # Record metrics
                self.record_metrics('delete', doc_id, True)
                
                self.logger.info(f"Deleted document: {doc_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Document deletion error: {str(e)}")
            self.record_metrics('delete', doc_id, False)
            return False

    async def get_document(self, doc_id: str, user_id: str = None) -> Optional[Document]:
        """Get a document with enhanced caching and security."""
        try:
            # Check cache first
            if doc_id in self.document_cache:
                document = self.document_cache[doc_id]
                self.metrics['cache_hits'] += 1
                
                # Check access
                if not self.check_access(user_id, document, 'read'):
                    raise DocumentAccessError(f"User {user_id} does not have read access to document {doc_id}")
                
                # Update metrics
                document.metrics.views += 1
                document.metrics.last_viewed = datetime.now()
                
                return document
            
            self.metrics['cache_misses'] += 1
            
            # Try to load from file
            for lang in self.languages:
                doc_path = Path(f"documentation/{lang}/{doc_id}.md")
                if doc_path.exists():
                    document = await self._load_document(doc_path)
                    if document:
                        # Check access
                        if not self.check_access(user_id, document, 'read'):
                            raise DocumentAccessError(f"User {user_id} does not have read access to document {doc_id}")
                        
                        # Update metrics
                        document.metrics.views += 1
                        document.metrics.last_viewed = datetime.now()
                        
                        # Cache document
                        self.document_cache[doc_id] = document
                        self.documents[doc_id] = document
                        
                        return document
            
            return None
                
        except Exception as e:
            self.logger.error(f"Document retrieval error: {str(e)}")
            return None

    async def list_documents(
        self,
        language: str = None,
        status: str = None,
        tags: List[str] = None,
        categories: List[str] = None,
        user_id: str = None
    ) -> List[Document]:
        """List documents with enhanced filtering and security."""
        try:
            documents = list(self.documents.values())
            
            # Filter by access
            if user_id:
                documents = [d for d in documents if self.check_access(user_id, d, 'read')]
            
            # Apply filters
            if language:
                documents = [d for d in documents if d.language == language]
            
            if status:
                documents = [d for d in documents if d.status == status]
            
            if tags:
                documents = [d for d in documents if all(tag in d.tags for tag in tags)]
            
            if categories:
                documents = [d for d in documents if all(cat in d.categories for cat in categories)]
            
            # Update metrics
            for doc in documents:
                doc.metrics.views += 1
                doc.metrics.last_viewed = datetime.now()
            
            return documents
                
        except Exception as e:
            self.logger.error(f"Document listing error: {str(e)}")
            return []

    async def create_template(
        self,
        name: str,
        description: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> DocumentTemplate:
        """Create a new document template."""
        try:
            async with self.lock:
                # Generate template ID
                template_id = f"template_{hash(f'{name}_{datetime.now()}')}"
                
                # Create template
                template = DocumentTemplate(
                    id=template_id,
                    name=name,
                    description=description,
                    content=content,
                    metadata=metadata or {},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    author=self.docs_config.get('default_author', 'system'),
                    tags=[],
                    categories=[],
                    variables=[]
                )
                
                # Save template
                await self._save_template(template)
                
                self.logger.info(f"Created template: {template_id}")
                return template
                
        except Exception as e:
            self.logger.error(f"Template creation error: {str(e)}")
            raise

    async def update_template(
        self,
        template_id: str,
        content: str = None,
        metadata: Optional[Dict] = None
    ) -> DocumentTemplate:
        """Update an existing template."""
        try:
            async with self.lock:
                if template_id not in self.templates:
                    raise ValueError(f"Template not found: {template_id}")
                
                template = self.templates[template_id]
                
                # Update content if provided
                if content:
                    template.content = content
                
                # Update metadata if provided
                if metadata:
                    template.metadata.update(metadata)
                
                # Update timestamp
                template.updated_at = datetime.now()
                
                # Save template
                await self._save_template(template)
                
                self.logger.info(f"Updated template: {template_id}")
                return template
                
        except Exception as e:
            self.logger.error(f"Template update error: {str(e)}")
            raise

    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        try:
            async with self.lock:
                if template_id not in self.templates:
                    raise ValueError(f"Template not found: {template_id}")
                
                # Delete template file
                template_path = Path(f"documentation/templates/{template_id}.md")
                if template_path.exists():
                    template_path.unlink()
                
                # Remove from memory
                del self.templates[template_id]
                
                self.logger.info(f"Deleted template: {template_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Template deletion error: {str(e)}")
            return False

    async def get_template(self, template_id: str) -> Optional[DocumentTemplate]:
        """Get a template by ID."""
        try:
            if template_id in self.templates:
                return self.templates[template_id]
            
            # Try to load from file
            template_path = Path(f"documentation/templates/{template_id}.md")
            if template_path.exists():
                template = await self._load_template(template_path)
                if template:
                    self.templates[template_id] = template
                    return template
            
            return None
                
        except Exception as e:
            self.logger.error(f"Template retrieval error: {str(e)}")
            return None

    async def list_templates(
        self,
        tags: List[str] = None,
        categories: List[str] = None
    ) -> List[DocumentTemplate]:
        """List templates with optional filters."""
        try:
            templates = list(self.templates.values())
            
            # Apply filters
            if tags:
                templates = [t for t in templates if all(tag in t.tags for tag in tags)]
            
            if categories:
                templates = [t for t in templates if all(cat in t.categories for cat in categories)]
            
            return templates
                
        except Exception as e:
            self.logger.error(f"Template listing error: {str(e)}")
            return []

    async def create_version(self, doc_id: str, content: str, author: str, change_log: str) -> Optional[Version]:
        """Create a new version of a document."""
        try:
            return await self.version_manager.create_version(
                doc_id=doc_id,
                content=content,
                author=author,
                metadata={'change_log': change_log}
            )
        except Exception as e:
            self.logger.error(f"Failed to create version: {str(e)}")
            return None

    def get_versions(self, doc_id: str) -> List[Version]:
        """Get all versions of a document."""
        return self.version_manager.get_version_history(doc_id)

    async def get_version(self, doc_id: str, version: str) -> Optional[Version]:
        """Get a specific version of a document."""
        try:
            return await self.version_manager.get_version(doc_id, version)
        except Exception as e:
            self.logger.error(f"Failed to get version: {str(e)}")
            return None

    async def compare_versions(self, doc_id: str, version1: str, version2: str) -> Dict:
        """Compare two versions of a document."""
        try:
            return await self.version_manager.compare_versions(doc_id, version1, version2)
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {str(e)}")
            return {}

    async def rollback_version(self, doc_id: str, version: str) -> Optional[Version]:
        """Rollback to a previous version."""
        try:
            return await self.version_manager.rollback_version(doc_id, version)
        except Exception as e:
            self.logger.error(f"Failed to rollback version: {str(e)}")
            return None

    async def delete_version(self, doc_id: str, version: str):
        """Delete a version."""
        try:
            await self.version_manager.delete_version(doc_id, version)
        except Exception as e:
            self.logger.error(f"Failed to delete version: {str(e)}")

    async def cleanup_versions(self, doc_id: str, keep_versions: int = 5):
        """Clean up old versions."""
        try:
            await self.version_manager.cleanup_versions(doc_id, keep_versions)
        except Exception as e:
            self.logger.error(f"Failed to cleanup versions: {str(e)}")

    async def _save_document(self, document: Document):
        """Save a document to file."""
        try:
            # Create directory if it doesn't exist
            doc_dir = Path(f"documentation/{document.language}")
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Save document
            doc_path = doc_dir / f"{document.id}.md"
            async with aiofiles.open(doc_path, 'w', encoding='utf-8') as f:
                await f.write(self._format_document(document))
            
            # Update Git repository
            self.repo.index.add([str(doc_path)])
            self.repo.index.commit(f"Update document: {document.id}")
            
        except Exception as e:
            self.logger.error(f"Document save error: {str(e)}")
            raise

    async def _load_document(self, doc_path: Path) -> Optional[Document]:
        """Load a document from file."""
        try:
            async with aiofiles.open(doc_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse frontmatter
            doc = frontmatter.loads(content)
            
            # Create document
            return Document(
                id=doc_path.stem,
                title=doc.metadata.get('title', ''),
                content=doc.content,
                language=doc.metadata.get('language', self.default_language),
                version=doc.metadata.get('version', '1.0.0'),
                status=doc.metadata.get('status', 'draft'),
                metadata=doc.metadata,
                created_at=datetime.fromisoformat(doc.metadata.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(doc.metadata.get('updated_at', datetime.now().isoformat())),
                author=doc.metadata.get('author', self.docs_config.get('default_author', 'system')),
                reviewers=doc.metadata.get('reviewers', []),
                tags=doc.metadata.get('tags', []),
                categories=doc.metadata.get('categories', []),
                dependencies=doc.metadata.get('dependencies', []),
                translations=doc.metadata.get('translations', {}),
                history=doc.metadata.get('history', []),
                authors=[self.docs_config.get('default_author', 'system')]
            )
            
        except Exception as e:
            self.logger.error(f"Document load error: {str(e)}")
            return None

    async def _save_template(self, template: DocumentTemplate):
        """Save a template to file."""
        try:
            # Create directory if it doesn't exist
            template_dir = Path("documentation/templates")
            template_dir.mkdir(parents=True, exist_ok=True)
            
            # Save template
            template_path = template_dir / f"{template.id}.md"
            async with aiofiles.open(template_path, 'w', encoding='utf-8') as f:
                await f.write(self._format_template(template))
            
            # Update Git repository
            self.repo.index.add([str(template_path)])
            self.repo.index.commit(f"Update template: {template.id}")
            
        except Exception as e:
            self.logger.error(f"Template save error: {str(e)}")
            raise

    async def _load_template(self, template_path: Path) -> Optional[DocumentTemplate]:
        """Load a template from file."""
        try:
            async with aiofiles.open(template_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse frontmatter
            template = frontmatter.loads(content)
            
            # Create template
            return DocumentTemplate(
                id=template_path.stem,
                name=template.metadata.get('name', ''),
                description=template.metadata.get('description', ''),
                content=template.content,
                metadata=template.metadata,
                created_at=datetime.fromisoformat(template.metadata.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(template.metadata.get('updated_at', datetime.now().isoformat())),
                author=template.metadata.get('author', self.docs_config.get('default_author', 'system')),
                tags=template.metadata.get('tags', []),
                categories=template.metadata.get('categories', []),
                variables=template.metadata.get('variables', [])
            )
            
        except Exception as e:
            self.logger.error(f"Template load error: {str(e)}")
            return None

    async def _save_version(self, doc_version: Version):
        """Save a version to file."""
        try:
            # Create directory if it doesn't exist
            version_dir = Path(f"documentation/versions/{doc_version.document_id}")
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save version
            version_path = version_dir / f"{doc_version.version}.md"
            async with aiofiles.open(version_path, 'w', encoding='utf-8') as f:
                await f.write(self._format_version(doc_version))
            
            # Update Git repository
            self.repo.index.add([str(version_path)])
            self.repo.index.commit(f"Create version {doc_version.version} for document: {doc_version.document_id}")
            
        except Exception as e:
            self.logger.error(f"Version save error: {str(e)}")
            raise

    async def _load_version(self, version_path: Path) -> Optional[Version]:
        """Load a version from file."""
        try:
            async with aiofiles.open(version_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse frontmatter
            version = frontmatter.loads(content)
            
            # Create version
            return Version(
                id=version_path.stem,
                document_id=version.metadata.get('document_id', ''),
                version=version.metadata.get('version', ''),
                content=version.content,
                changes=version.metadata.get('changes', []),
                created_at=datetime.fromisoformat(version.metadata.get('created_at', datetime.now().isoformat())),
                author=version.metadata.get('author', self.docs_config.get('default_author', 'system')),
                commit_hash=version.metadata.get('commit_hash', ''),
                status=version.metadata.get('status', 'draft'),
                metadata=version.metadata,
                change_log=version.metadata.get('change_log', '')
            )
            
        except Exception as e:
            self.logger.error(f"Version load error: {str(e)}")
            return None

    def _format_document(self, document: Document) -> str:
        """Format a document for storage."""
        frontmatter_data = {
            'title': document.title,
            'language': document.language,
            'version': document.version,
            'status': document.status,
            'created_at': document.created_at.isoformat(),
            'updated_at': document.updated_at.isoformat(),
            'author': document.author,
            'reviewers': document.reviewers,
            'tags': document.tags,
            'categories': document.categories,
            'dependencies': document.dependencies,
            'translations': document.translations,
            'history': document.history,
            **document.metadata
        }
        
        return frontmatter.dumps(document.content, **frontmatter_data)

    def _format_template(self, template: DocumentTemplate) -> str:
        """Format a template for storage."""
        frontmatter_data = {
            'name': template.name,
            'description': template.description,
            'created_at': template.created_at.isoformat(),
            'updated_at': template.updated_at.isoformat(),
            'author': template.author,
            'tags': template.tags,
            'categories': template.categories,
            'variables': template.variables,
            **template.metadata
        }
        
        return frontmatter.dumps(template.content, **frontmatter_data)

    def _format_version(self, doc_version: Version) -> str:
        """Format a version for storage."""
        frontmatter_data = {
            'document_id': doc_version.document_id,
            'version': doc_version.version,
            'changes': doc_version.changes,
            'created_at': doc_version.created_at.isoformat(),
            'author': doc_version.author,
            'commit_hash': doc_version.commit_hash,
            'status': doc_version.status,
            **doc_version.metadata
        }
        
        return frontmatter.dumps(doc_version.content, **frontmatter_data)

    def _apply_template(self, content: str, template: DocumentTemplate) -> str:
        """Apply a template to content."""
        try:
            # Replace template variables
            for key, value in template.metadata.items():
                if isinstance(value, str):
                    content = content.replace(f"{{{{ {key} }}}}", value)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Template application error: {str(e)}")
            return content

    async def clear_cache(self):
        """Clear the document cache."""
        try:
            self.document_cache.clear()
            self.template_cache.clear()
            self.logger.info("Cleared document and template cache")
            
        except Exception as e:
            self.logger.error(f"Cache clearing error: {str(e)}")

    async def backup_documents(self, backup_path: str):
        """Backup all documents."""
        try:
            # Create backup directory
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup documents
            for doc_id, document in self.documents.items():
                doc_path = backup_dir / f"{doc_id}.md"
                async with aiofiles.open(doc_path, 'w', encoding='utf-8') as f:
                    await f.write(self._format_document(document))
            
            # Backup templates
            template_dir = backup_dir / "templates"
            template_dir.mkdir(exist_ok=True)
            for template_id, template in self.templates.items():
                template_path = template_dir / f"{template_id}.md"
                async with aiofiles.open(template_path, 'w', encoding='utf-8') as f:
                    await f.write(self._format_template(template))
            
            # Backup versions
            version_dir = backup_dir / "versions"
            version_dir.mkdir(exist_ok=True)
            for doc_id, versions in self.version_manager.versions.items():
                doc_version_dir = version_dir / doc_id
                doc_version_dir.mkdir(exist_ok=True)
                for doc_version in versions:
                    version_path = doc_version_dir / f"{doc_version.version}.md"
                    async with aiofiles.open(version_path, 'w', encoding='utf-8') as f:
                        await f.write(self._format_version(doc_version))
            
            self.logger.info(f"Backed up documents to {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Document backup error: {str(e)}")
            raise

    async def restore_documents(self, backup_path: str):
        """Restore documents from backup."""
        try:
            # Clear current documents
            self.documents.clear()
            self.templates.clear()
            self.version_manager.versions.clear()
            
            # Restore documents
            backup_dir = Path(backup_path)
            for doc_path in backup_dir.glob('*.md'):
                document = await self._load_document(doc_path)
                if document:
                    self.documents[document.id] = document
            
            # Restore templates
            template_dir = backup_dir / "templates"
            if template_dir.exists():
                for template_path in template_dir.glob('*.md'):
                    template = await self._load_template(template_path)
                    if template:
                        self.templates[template.id] = template
            
            # Restore versions
            version_dir = backup_dir / "versions"
            if version_dir.exists():
                for doc_version_dir in version_dir.iterdir():
                    if doc_version_dir.is_dir():
                        doc_id = doc_version_dir.name
                        self.version_manager.versions[doc_id] = []
                        for version_path in doc_version_dir.glob('*.md'):
                            doc_version = await self._load_version(version_path)
                            if doc_version:
                                self.version_manager.versions[doc_id].append(doc_version)
            
            self.logger.info(f"Restored documents from {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Document restore error: {str(e)}")
            raise

    # Collaboration features
    def add_reviewer(self, doc_id: str, reviewer: str):
        doc = self.documents.get(doc_id)
        if doc and reviewer not in doc.reviewers:
            doc.reviewers.append(reviewer)
            self.save_document(doc)
            self.logger.info(f"Added reviewer {reviewer} to document {doc_id}")

    def set_status(self, doc_id: str, status: str):
        doc = self.documents.get(doc_id)
        if doc:
            doc.status = status
            self.save_document(doc)
            self.logger.info(f"Set status of document {doc_id} to {status}")

    def save_document(self, doc: Document):
        lang_path = self.docs_path / doc.language
        lang_path.mkdir(parents=True, exist_ok=True)
        doc_path = lang_path / f"{doc.id}.md"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(f"# {doc.title}\n\n{doc.content}\n")
            f.write(f"\n---\n# Metadata\n")
            yaml.dump({
                'id': doc.id,
                'version': doc.version,
                'authors': doc.authors,
                'reviewers': doc.reviewers,
                'created_at': doc.created_at.isoformat(),
                'updated_at': doc.updated_at.isoformat(),
                'tags': doc.tags,
                'metadata': doc.metadata,
                'status': doc.status
            }, f)

    def create_template(self, name: str, content: str, variables: List[str], category: str, tags: List[str]) -> DocumentTemplate:
        template_id = f"tpl_{hash(name + str(datetime.now()))}"
        tpl = DocumentTemplate(
            id=template_id,
            name=name,
            description='',
            content=content,
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            author=self.docs_config.get('default_author', 'system'),
            tags=tags,
            categories=[category],
            variables=variables
        )
        self.templates[template_id] = tpl
        self.save_template(tpl)
        self.logger.info(f"Created template: {template_id}")
        return tpl

    def save_template(self, tpl: DocumentTemplate):
        tpl_path = self.templates_path / f"{tpl.id}.md"
        with open(tpl_path, 'w', encoding='utf-8') as f:
            f.write(f"# {tpl.name}\n\n{tpl.content}\n")
            f.write(f"\n---\n# Variables\n")
            yaml.dump({'variables': tpl.variables, 'category': tpl.categories[0], 'tags': tpl.tags}, f)

    def create_version(self, doc_id: str, content: str, author: str, change_log: str) -> Optional[Version]:
        doc = self.documents.get(doc_id)
        if not doc:
            self.logger.error(f"Document not found: {doc_id}")
            return None
        new_version = self._increment_version(doc.version)
        version_id = f"ver_{hash(doc_id + new_version + str(datetime.now()))}"
        version = Version(
            id=version_id,
            document_id=doc_id,
            version=new_version,
            content=content,
            changes=self._get_changes(doc, content),
            created_at=datetime.now(),
            author=author,
            commit_hash=self._get_commit_hash(),
            status='draft',
            metadata={},
            change_log=change_log
        )
        self.version_manager.versions.setdefault(doc_id, []).append(version)
        doc.version = new_version
        doc.content = content
        doc.updated_at = datetime.now()
        self.save_document(doc)
        self.logger.info(f"Created version {new_version} for document: {doc_id}")
        return version

    def get_versions(self, doc_id: str) -> List[Version]:
        return self.version_manager.versions.get(doc_id, [])

    def backup_documents(self, backup_path: str):
        shutil.make_archive(backup_path, 'zip', self.docs_path)
        self.logger.info(f"Backed up documents to {backup_path}.zip")

    def restore_documents(self, backup_zip: str):
        shutil.unpack_archive(backup_zip, self.docs_path)
        self.logger.info(f"Restored documents from {backup_zip}")

    def _increment_version(self, version: str) -> str:
        parts = version.split('.')
        if len(parts) == 3:
            major, minor, patch = map(int, parts)
            patch += 1
            return f"{major}.{minor}.{patch}"
        return '0.1.0'

    def _get_changes(self, document: Document, new_content: str) -> List[Dict[str, Any]]:
        """Get changes between document versions."""
        try:
            # TODO: Implement diff algorithm
            return []
            
        except Exception as e:
            self.logger.error(f"Change detection error: {str(e)}")
            return []

    def _get_commit_hash(self) -> str:
        """Get current Git commit hash."""
        try:
            return self.repo.head.commit.hexsha
            
        except Exception as e:
            self.logger.error(f"Commit hash retrieval error: {str(e)}")
            return ''

    async def add_comment(
        self,
        doc_id: str,
        content: str,
        user_id: str,
        parent_id: Optional[str] = None
    ) -> Optional[DocumentComment]:
        """Add a comment to a document."""
        try:
            async with self.lock:
                document = await self.get_document(doc_id, user_id)
                if not document:
                    return None
                
                # Create comment
                comment = DocumentComment(
                    id=f"comment_{hash(f'{content}_{datetime.now()}')}",
                    user_id=user_id,
                    content=content,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    parent_id=parent_id
                )
                
                # Add comment
                document.comments.append(comment)
                
                # Update metrics
                document.metrics.comments += 1
                document.metrics.last_commented = datetime.now()
                
                # Save document
                await self._save_document(document)
                
                # Update cache
                self.document_cache[doc_id] = document
                
                return comment
                
        except Exception as e:
            self.logger.error(f"Comment addition error: {str(e)}")
            return None

    async def resolve_comment(
        self,
        doc_id: str,
        comment_id: str,
        user_id: str
    ) -> bool:
        """Resolve a comment on a document."""
        try:
            async with self.lock:
                document = await self.get_document(doc_id, user_id)
                if not document:
                    return False
                
                # Find comment
                for comment in document.comments:
                    if comment.id == comment_id:
                        # Update comment
                        comment.resolved = True
                        comment.resolved_at = datetime.now()
                        comment.resolved_by = user_id
                        
                        # Save document
                        await self._save_document(document)
                        
                        # Update cache
                        self.document_cache[doc_id] = document
                        
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Comment resolution error: {str(e)}")
            return False

    async def create_review(
        self,
        doc_id: str,
        user_id: str,
        status: str = 'pending'
    ) -> Optional[DocumentReview]:
        """Create a review for a document."""
        try:
            async with self.lock:
                document = await self.get_document(doc_id, user_id)
                if not document:
                    return None
                
                # Create review
                review = DocumentReview(
                    id=f"review_{hash(f'{user_id}_{datetime.now()}')}",
                    user_id=user_id,
                    status=status,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Add review
                document.reviews.append(review)
                
                # Update metrics
                document.metrics.reviews += 1
                document.metrics.last_reviewed = datetime.now()
                
                # Save document
                await self._save_document(document)
                
                # Update cache
                self.document_cache[doc_id] = document
                
                return review
                
        except Exception as e:
            self.logger.error(f"Review creation error: {str(e)}")
            return None

    async def update_review(
        self,
        doc_id: str,
        review_id: str,
        user_id: str,
        status: str = None,
        feedback: str = None
    ) -> bool:
        """Update a review for a document."""
        try:
            async with self.lock:
                document = await self.get_document(doc_id, user_id)
                if not document:
                    return False
                
                # Find review
                for review in document.reviews:
                    if review.id == review_id:
                        # Update review
                        if status:
                            review.status = status
                        if feedback:
                            review.feedback = feedback
                        review.updated_at = datetime.now()
                        
                        if status == 'completed':
                            review.completed_at = datetime.now()
                        
                        # Save document
                        await self._save_document(document)
                        
                        # Update cache
                        self.document_cache[doc_id] = document
                        
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Review update error: {str(e)}")
            return False