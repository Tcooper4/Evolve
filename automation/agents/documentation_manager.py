import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime
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
from .documentation_version import DocumentationVersion, Version

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
        self.cache = {}
        self.lock = asyncio.Lock()
        self.docs_path = Path(config.get('documentation', {}).get('docs_path', 'documentation'))
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.templates_path = self.docs_path / 'templates'
        self.templates_path.mkdir(parents=True, exist_ok=True)

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

    async def create_document(
        self,
        title: str,
        content: str,
        language: str = None,
        template_id: str = None,
        metadata: Optional[Dict] = None
    ) -> Document:
        """Create a new document."""
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
                    author=self.docs_config.get('default_author', 'system'),
                    reviewers=[],
                    tags=[],
                    categories=[],
                    dependencies=[],
                    translations={},
                    history=[],
                    authors=[self.docs_config.get('default_author', 'system')]
                )
                
                # Save document
                await self._save_document(document)
                
                # Create initial version
                await self._create_version(document)
                
                self.logger.info(f"Created document: {doc_id}")
                return document
                
        except Exception as e:
            self.logger.error(f"Document creation error: {str(e)}")
            raise

    async def update_document(
        self,
        doc_id: str,
        content: str = None,
        metadata: Optional[Dict] = None
    ) -> Document:
        """Update an existing document."""
        try:
            async with self.lock:
                if doc_id not in self.documents:
                    raise ValueError(f"Document not found: {doc_id}")
                
                document = self.documents[doc_id]
                
                # Update content if provided
                if content:
                    document.content = content
                
                # Update metadata if provided
                if metadata:
                    document.metadata.update(metadata)
                
                # Update timestamp
                document.updated_at = datetime.now()
                
                # Save document
                await self._save_document(document)
                
                # Create new version
                await self._create_version(document)
                
                self.logger.info(f"Updated document: {doc_id}")
                return document
                
        except Exception as e:
            self.logger.error(f"Document update error: {str(e)}")
            raise

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        try:
            async with self.lock:
                if doc_id not in self.documents:
                    raise ValueError(f"Document not found: {doc_id}")
                
                # Delete document file
                doc_path = Path(f"documentation/{self.documents[doc_id].language}/{doc_id}.md")
                if doc_path.exists():
                    doc_path.unlink()
                
                # Remove from memory
                del self.documents[doc_id]
                
                # Remove versions
                if doc_id in self.version_manager.versions:
                    del self.version_manager.versions[doc_id]
                
                self.logger.info(f"Deleted document: {doc_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Document deletion error: {str(e)}")
            return False

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        try:
            if doc_id in self.documents:
                return self.documents[doc_id]
            
            # Try to load from file
            for lang in self.languages:
                doc_path = Path(f"documentation/{lang}/{doc_id}.md")
                if doc_path.exists():
                    document = await self._load_document(doc_path)
                    if document:
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
        categories: List[str] = None
    ) -> List[Document]:
        """List documents with optional filters."""
        try:
            documents = list(self.documents.values())
            
            # Apply filters
            if language:
                documents = [d for d in documents if d.language == language]
            
            if status:
                documents = [d for d in documents if d.status == status]
            
            if tags:
                documents = [d for d in documents if all(tag in d.tags for tag in tags)]
            
            if categories:
                documents = [d for d in documents if all(cat in d.categories for cat in categories)]
            
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
            self.cache.clear()
            self.logger.info("Cleared document cache")
            
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