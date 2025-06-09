import logging
from typing import Dict, List, Optional, Union, Set, Tuple
import json
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import jinja2
import markdown
import yaml
import re
import ast
import inspect
import pdoc
import sphinx
import mkdocs
import docutils
import rst2html
import docstring_parser
from pydantic import BaseModel, Field, validator
from enum import Enum
import hashlib
import aiofiles
import aiohttp
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry
import jwt
from typing_extensions import TypedDict
from .documentation_search import DocumentationSearch, SearchResult
from .documentation_collaboration import DocumentationCollaboration, User, Comment, Review

class DocumentationStatus(str, Enum):
    """Status of documentation."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class DocumentationVersion(BaseModel):
    """Version of documentation."""
    version: int = Field(..., ge=1)
    doc_id: str
    content: str
    format: str
    created_at: datetime
    created_by: str
    changes: List[Dict[str, Any]]
    status: DocumentationStatus
    reviewers: List[str]
    approvers: List[str]
    metadata: Dict[str, Any]

class DocumentationProgress(BaseModel):
    """Progress tracking for documentation generation."""
    doc_id: str
    total_steps: int
    completed_steps: int
    current_step: str
    status: str
    start_time: datetime
    estimated_completion: Optional[datetime]
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class Documentation:
    id: str
    type: str
    title: str
    content: str
    format: str
    created_at: str
    updated_at: str
    metadata: Dict
    tags: List[str]
    status: DocumentationStatus = DocumentationStatus.DRAFT
    version: int = 1
    author: Optional[str] = None
    reviewers: List[str] = None
    approvers: List[str] = None
    security_level: int = 1
    validation_level: int = 1
    audit_level: int = 1
    metrics_level: int = 1
    health_level: int = 1
    logging_level: int = 1

class DocumentationGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_cache()
        self.setup_security()
        self.setup_validation()
        self.setup_audit()
        self.setup_metrics()
        self.setup_health()
        self.docs: Dict[str, Documentation] = {}
        self.versions: Dict[str, List[DocumentationVersion]] = {}
        self.collaborations: Dict[str, DocumentationCollaboration] = {}
        self.progress: Dict[str, DocumentationProgress] = {}
        self.doc_config = config.get('documentation', {})
        self.template_loader = jinja2.FileSystemLoader(self.doc_config.get('templates_path', 'templates'))
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.backup_path = Path(self.doc_config.get('backup_path', 'backups/documentation'))
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.setup_search()
        self.setup_collaboration()
        self.documents: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def setup_logging(self):
        """Configure logging for the documentation generator."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "documentation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_cache(self):
        """Setup caching for documentation."""
        self.cache = TTLCache(
            maxsize=self.doc_config.get('cache_size', 1000),
            ttl=self.doc_config.get('cache_ttl', 3600)
        )

    def setup_security(self):
        """Setup security for documentation."""
        self.jwt_secret = self.doc_config.get('jwt_secret')
        self.allowed_paths = set(self.doc_config.get('allowed_paths', []))
        self.max_file_size = self.doc_config.get('max_file_size', 10 * 1024 * 1024)  # 10MB

    def setup_validation(self):
        """Setup validation for documentation."""
        self.validation_rules = self.doc_config.get('validation_rules', {})
        self.required_fields = set(self.doc_config.get('required_fields', []))

    def setup_audit(self):
        """Setup audit logging for documentation."""
        self.audit_path = Path("automation/logs/documentation/audit")
        self.audit_path.mkdir(parents=True, exist_ok=True)

    def setup_metrics(self):
        """Setup metrics collection for documentation."""
        self.metrics_path = Path("automation/logs/documentation/metrics")
        self.metrics_path.mkdir(parents=True, exist_ok=True)

    def setup_health(self):
        """Setup health monitoring for documentation."""
        self.health_path = Path("automation/logs/documentation/health")
        self.health_path.mkdir(parents=True, exist_ok=True)

    def setup_search(self):
        """Setup search system."""
        self.search = DocumentationSearch(self.config)

    def setup_collaboration(self):
        """Setup collaboration system."""
        self.collaboration = DocumentationCollaboration(self.config)

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def generate_documentation(
        self,
        type: str,
        title: str,
        content: str,
        format: str = 'markdown',
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        security_level: int = 1,
        validation_level: int = 1,
        audit_level: int = 1,
        metrics_level: int = 1,
        health_level: int = 1,
        logging_level: int = 1
    ) -> str:
        """
        Generate documentation for a component with enhanced features.
        """
        try:
            # Validate input
            self._validate_input(type, title, content, format, metadata, tags)
            
            # Check security
            self._check_security(author, security_level)
            
            # Create progress tracker
            progress = DocumentationProgress(
                doc_id=str(len(self.docs) + 1),
                total_steps=5,
                completed_steps=0,
                current_step="Initialization",
                status="In Progress",
                start_time=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(minutes=5),
                errors=[],
                warnings=[],
                metadata={}
            )
            
            # Generate documentation
            doc_id = str(len(self.docs) + 1)
            doc = Documentation(
                id=doc_id,
                type=type,
                title=title,
                content=content,
                format=format,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                metadata=metadata or {},
                tags=tags or [],
                author=author,
                security_level=security_level,
                validation_level=validation_level,
                audit_level=audit_level,
                metrics_level=metrics_level,
                health_level=health_level,
                logging_level=logging_level
            )
            
            # Store documentation
            self.docs[doc_id] = doc
            self.progress[doc_id] = progress
            
            # Create version
            version = DocumentationVersion(
                version=1,
                doc_id=doc_id,
                content=content,
                format=format,
                created_at=datetime.utcnow(),
                created_by=author or "system",
                changes=[{"type": "create", "timestamp": datetime.utcnow()}],
                status=DocumentationStatus.DRAFT,
                reviewers=[],
                approvers=[],
                metadata=metadata or {}
            )
            self.versions[doc_id] = [version]
            
            # Create collaboration
            collaboration = DocumentationCollaboration(
                doc_id=doc_id,
                collaborators=[author] if author else [],
                comments=[],
                suggestions=[],
                review_requests=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.collaborations[doc_id] = collaboration
            
            # Update progress
            progress.completed_steps = 5
            progress.current_step = "Completed"
            progress.status = "Success"
            progress.estimated_completion = datetime.utcnow()
            
            # Cache documentation
            self._cache_documentation(doc_id, doc)
            
            # Backup documentation
            await self._backup_documentation(doc_id)
            
            # Record metrics
            self._record_metrics("documentation_generated", doc_id)
            
            # Audit
            self._audit_action("documentation_generated", doc_id, author)
            
            self.logger.info(f"Generated documentation: {doc_id} - {title}")
            
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Failed to generate documentation: {str(e)}")
            if progress:
                progress.errors.append({"error": str(e), "timestamp": datetime.utcnow()})
                progress.status = "Failed"
            raise

    def _validate_input(
        self,
        type: str,
        title: str,
        content: str,
        format: str,
        metadata: Optional[Dict],
        tags: Optional[List[str]]
    ) -> None:
        """Validate input parameters."""
        if not type or not isinstance(type, str):
            raise ValueError("Type must be a non-empty string")
            
        if not title or not isinstance(title, str):
            raise ValueError("Title must be a non-empty string")
            
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
            
        if format not in ['markdown', 'rst', 'html']:
            raise ValueError("Format must be one of: markdown, rst, html")
            
        if metadata and not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")
            
        if tags and not isinstance(tags, list):
            raise ValueError("Tags must be a list")

    def _check_security(self, author: Optional[str], security_level: int) -> None:
        """Check security requirements."""
        if security_level > 1 and not author:
            raise ValueError("Author is required for high security level")
            
        if security_level > 2:
            # Additional security checks
            pass

    def _cache_documentation(self, doc_id: str, doc: Documentation) -> None:
        """Cache documentation."""
        cache_key = f"doc:{doc_id}"
        self.cache[cache_key] = doc

    async def _backup_documentation(self, doc_id: str) -> None:
        """Backup documentation."""
        doc = self.docs[doc_id]
        backup_file = self.backup_path / f"{doc_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        async with aiofiles.open(backup_file, 'w') as f:
            await f.write(json.dumps(doc.__dict__))

    def _record_metrics(self, action: str, doc_id: str) -> None:
        """Record metrics."""
        metrics_file = self.metrics_path / f"{datetime.utcnow().strftime('%Y%m%d')}.json"
        
        metrics = {
            "action": action,
            "doc_id": doc_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def _audit_action(self, action: str, doc_id: str, author: Optional[str]) -> None:
        """Record audit log."""
        audit_file = self.audit_path / f"{datetime.utcnow().strftime('%Y%m%d')}.json"
        
        audit = {
            "action": action,
            "doc_id": doc_id,
            "author": author,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(audit_file, 'a') as f:
            f.write(json.dumps(audit) + '\n')

    async def search_documentation(self, search: DocumentationSearch) -> List[Documentation]:
        """Search documentation with advanced filtering."""
        results = []
        
        for doc in self.docs.values():
            if self._matches_search(doc, search):
                results.append(doc)
                
        return results[:search.limit] if search.limit else results

    def _matches_search(self, doc: Documentation, search: DocumentationSearch) -> bool:
        """Check if documentation matches search criteria."""
        if search.type and doc.type != search.type:
            return False
            
        if search.tags and not all(tag in doc.tags for tag in search.tags):
            return False
            
        if search.status and doc.status != search.status:
            return False
            
        if search.created_after and datetime.fromisoformat(doc.created_at) < search.created_after:
            return False
            
        if search.created_before and datetime.fromisoformat(doc.created_at) > search.created_before:
            return False
            
        if search.updated_after and datetime.fromisoformat(doc.updated_at) < search.updated_after:
            return False
            
        if search.updated_before and datetime.fromisoformat(doc.updated_at) > search.updated_before:
            return False
            
        if search.author and doc.author != search.author:
            return False
            
        if search.reviewer and not any(reviewer == search.reviewer for reviewer in doc.reviewers):
            return False
            
        if search.approver and not any(approver == search.approver for approver in doc.approvers):
            return False
            
        return True

    async def add_collaborator(
        self,
        doc_id: str,
        collaborator: str,
        role: str = "reviewer"
    ) -> None:
        """Add a collaborator to documentation."""
        if doc_id not in self.collaborations:
            raise ValueError(f"Documentation {doc_id} not found")
            
        collaboration = self.collaborations[doc_id]
        
        if collaborator not in collaboration.collaborators:
            collaboration.collaborators.append(collaborator)
            
        if role == "reviewer" and collaborator not in self.docs[doc_id].reviewers:
            self.docs[doc_id].reviewers.append(collaborator)
            
        if role == "approver" and collaborator not in self.docs[doc_id].approvers:
            self.docs[doc_id].approvers.append(collaborator)
            
        collaboration.updated_at = datetime.utcnow()
        
        self._audit_action("collaborator_added", doc_id, collaborator)

    async def add_comment(
        self,
        doc_id: str,
        author: str,
        comment: str,
        parent_id: Optional[str] = None
    ) -> None:
        """Add a comment to documentation."""
        if doc_id not in self.collaborations:
            raise ValueError(f"Documentation {doc_id} not found")
            
        collaboration = self.collaborations[doc_id]
        
        comment_data = {
            "id": str(len(collaboration.comments) + 1),
            "author": author,
            "comment": comment,
            "parent_id": parent_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        collaboration.comments.append(comment_data)
        collaboration.updated_at = datetime.utcnow()
        
        self._audit_action("comment_added", doc_id, author)

    async def request_review(
        self,
        doc_id: str,
        author: str,
        reviewers: List[str],
        message: Optional[str] = None
    ) -> None:
        """Request review for documentation."""
        if doc_id not in self.collaborations:
            raise ValueError(f"Documentation {doc_id} not found")
            
        collaboration = self.collaborations[doc_id]
        
        review_request = {
            "id": str(len(collaboration.review_requests) + 1),
            "author": author,
            "reviewers": reviewers,
            "message": message,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        collaboration.review_requests.append(review_request)
        collaboration.updated_at = datetime.utcnow()
        
        self._audit_action("review_requested", doc_id, author)

    async def update_documentation_status(
        self,
        doc_id: str,
        status: DocumentationStatus,
        author: str
    ) -> None:
        """Update documentation status."""
        if doc_id not in self.docs:
            raise ValueError(f"Documentation {doc_id} not found")
            
        doc = self.docs[doc_id]
        
        if status == DocumentationStatus.APPROVED and author not in doc.approvers:
            raise ValueError(f"User {author} is not authorized to approve documentation")
            
        doc.status = status
        doc.updated_at = datetime.utcnow().isoformat()
        
        # Create new version
        version = DocumentationVersion(
            version=len(self.versions[doc_id]) + 1,
            doc_id=doc_id,
            content=doc.content,
            format=doc.format,
            created_at=datetime.utcnow(),
            created_by=author,
            changes=[{
                "type": "status_change",
                "from": doc.status,
                "to": status,
                "timestamp": datetime.utcnow()
            }],
            status=status,
            reviewers=doc.reviewers,
            approvers=doc.approvers,
            metadata=doc.metadata
        )
        
        self.versions[doc_id].append(version)
        
        self._audit_action("status_updated", doc_id, author)

    async def generate_api_docs(self, module_path: str) -> str:
        """Generate API documentation with enhanced features."""
        try:
            # Validate module path
            if not Path(module_path).exists():
                raise ValueError(f"Module path {module_path} does not exist")

            # Create progress tracker
            progress = DocumentationProgress(
                doc_id=f"api_{module_path}",
                total_steps=4,
                completed_steps=0,
                current_step="Initialization",
                status="In Progress",
                start_time=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(minutes=10),
                errors=[],
                warnings=[],
                metadata={"module_path": module_path}
            )

            # Extract module information
            progress.current_step = "Extracting Module Info"
            module_info = self._extract_module_info(module_path)
            progress.completed_steps += 1

            # Generate documentation
            progress.current_step = "Generating Documentation"
            content = self._generate_api_content(module_info)
            progress.completed_steps += 1

            # Create documentation
            progress.current_step = "Creating Documentation"
            doc_id = await self.generate_documentation(
                type="api",
                title=f"API Documentation - {module_path}",
                content=content,
                format="markdown",
                metadata={
                    "module_path": module_path,
                    "module_info": module_info
                },
                tags=["api", "documentation", module_path]
            )
            progress.completed_steps += 1

            # Update progress
            progress.completed_steps = 4
            progress.current_step = "Completed"
            progress.status = "Success"
            progress.estimated_completion = datetime.utcnow()

            return doc_id

        except Exception as e:
            self.logger.error(f"Failed to generate API documentation: {str(e)}")
            if progress:
                progress.errors.append({"error": str(e), "timestamp": datetime.utcnow()})
                progress.status = "Failed"
            raise

    def _extract_module_info(self, module_path: str) -> Dict:
        """Extract comprehensive module information."""
        module_info = {
            "name": Path(module_path).name,
            "path": module_path,
            "classes": [],
            "functions": [],
            "imports": [],
            "dependencies": [],
            "metadata": {}
        }

        try:
            with open(module_path, 'r') as f:
                tree = ast.parse(f.read())

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    module_info["imports"].extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module_info["imports"].append(f"{node.module}.{node.names[0].name}")

            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node)
                    module_info["classes"].append(class_info)
                elif isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function_info(node)
                    module_info["functions"].append(func_info)

            # Extract dependencies
            module_info["dependencies"] = self._extract_dependencies(module_path)

            # Extract metadata
            module_info["metadata"] = self._extract_metadata(tree)

        except Exception as e:
            self.logger.error(f"Error extracting module info: {str(e)}")
            raise

        return module_info

    def _extract_class_info(self, node: ast.ClassDef) -> Dict:
        """Extract detailed class information."""
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "properties": [],
            "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
            "bases": [self._get_name(base) for base in node.bases],
            "attributes": []
        }

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item)
                class_info["methods"].append(method_info)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info["attributes"].append({
                            "name": target.id,
                            "value": self._get_value(item.value)
                        })

        return class_info

    def _extract_function_info(self, node: ast.FunctionDef) -> Dict:
        """Extract detailed function information."""
        func_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": self._extract_arguments(node.args),
            "returns": self._extract_returns(node),
            "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
            "async": isinstance(node, ast.AsyncFunctionDef)
        }
        return func_info

    def _extract_arguments(self, args: ast.arguments) -> Dict:
        """Extract function arguments information."""
        return {
            "args": [arg.arg for arg in args.args],
            "defaults": [self._get_value(d) for d in args.defaults],
            "kwonly": [arg.arg for arg in args.kwonlyargs],
            "kw_defaults": [self._get_value(d) for d in args.kw_defaults],
            "vararg": args.vararg.arg if args.vararg else None,
            "kwarg": args.kwarg.arg if args.kwarg else None
        }

    def _extract_returns(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract function return type information."""
        if node.returns:
            return self._get_name(node.returns)
        return None

    def _extract_dependencies(self, module_path: str) -> List[str]:
        """Extract module dependencies."""
        dependencies = []
        try:
            with open(module_path, 'r') as f:
                content = f.read()
                # Look for import statements
                import_pattern = r'^import\s+(\w+)|^from\s+(\w+)\s+import'
                matches = re.finditer(import_pattern, content, re.MULTILINE)
                for match in matches:
                    dep = match.group(1) or match.group(2)
                    if dep and dep not in dependencies:
                        dependencies.append(dep)
        except Exception as e:
            self.logger.error(f"Error extracting dependencies: {str(e)}")
        return dependencies

    def _extract_metadata(self, tree: ast.Module) -> Dict:
        """Extract module metadata."""
        metadata = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ['__version__', '__author__', '__license__']:
                        metadata[target.id] = self._get_value(node.value)
        return metadata

    def _generate_api_content(self, module_info: Dict) -> str:
        """Generate comprehensive API documentation content."""
        template = self.template_env.get_template('api.md.j2')
        return template.render(
            module_info=module_info,
            generated_at=datetime.utcnow().isoformat()
        )

    def _get_name(self, node: ast.AST) -> str:
        """Get the name of an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def _get_value(self, node: ast.AST) -> Any:
        """Get the value of an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return f"{self._get_name(node.func)}()"
        return str(node)

    async def generate_code_docs(self, code_path: str) -> str:
        """Generate code documentation with enhanced features."""
        try:
            # Validate code path
            if not Path(code_path).exists():
                raise ValueError(f"Code path {code_path} does not exist")

            # Create progress tracker
            progress = DocumentationProgress(
                doc_id=f"code_{code_path}",
                total_steps=4,
                completed_steps=0,
                current_step="Initialization",
                status="In Progress",
                start_time=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(minutes=15),
                errors=[],
                warnings=[],
                metadata={"code_path": code_path}
            )

            # Extract code information
            progress.current_step = "Extracting Code Info"
            code_info = self._extract_code_info(code_path)
            progress.completed_steps += 1

            # Generate documentation
            progress.current_step = "Generating Documentation"
            content = self._generate_code_content(code_info)
            progress.completed_steps += 1

            # Create documentation
            progress.current_step = "Creating Documentation"
            doc_id = await self.generate_documentation(
                type="code",
                title=f"Code Documentation - {code_path}",
                content=content,
                format="markdown",
                metadata={
                    "code_path": code_path,
                    "code_info": code_info
                },
                tags=["code", "documentation", code_path]
            )
            progress.completed_steps += 1

            # Update progress
            progress.completed_steps = 4
            progress.current_step = "Completed"
            progress.status = "Success"
            progress.estimated_completion = datetime.utcnow()

            return doc_id

        except Exception as e:
            self.logger.error(f"Failed to generate code documentation: {str(e)}")
            if progress:
                progress.errors.append({"error": str(e), "timestamp": datetime.utcnow()})
                progress.status = "Failed"
            raise

    def _extract_code_info(self, code_path: str) -> Dict:
        """Extract comprehensive code information."""
        code_info = {
            "path": code_path,
            "language": self._detect_language(code_path),
            "structure": self._analyze_code_structure(code_path),
            "complexity": self._analyze_code_complexity(code_path),
            "dependencies": self._extract_dependencies(code_path),
            "metadata": {}
        }

        try:
            with open(code_path, 'r') as f:
                content = f.read()
                code_info["metadata"] = self._extract_code_metadata(content)
        except Exception as e:
            self.logger.error(f"Error extracting code info: {str(e)}")
            raise

        return code_info

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language of the file."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust'
        }
        return language_map.get(ext, 'Unknown')

    def _analyze_code_structure(self, code_path: str) -> Dict:
        """Analyze code structure and organization."""
        structure = {
            "imports": [],
            "classes": [],
            "functions": [],
            "variables": [],
            "comments": []
        }

        try:
            with open(code_path, 'r') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    structure["imports"].extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    structure["imports"].append(f"{node.module}.{node.names[0].name}")
                elif isinstance(node, ast.ClassDef):
                    structure["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            structure["variables"].append(target.id)

        except Exception as e:
            self.logger.error(f"Error analyzing code structure: {str(e)}")

        return structure

    def _analyze_code_complexity(self, code_path: str) -> Dict:
        """Analyze code complexity metrics."""
        complexity = {
            "cyclomatic_complexity": 0,
            "cognitive_complexity": 0,
            "maintainability_index": 0,
            "lines_of_code": 0,
            "comment_ratio": 0
        }

        try:
            with open(code_path, 'r') as f:
                content = f.read()
                lines = content.splitlines()
                complexity["lines_of_code"] = len(lines)
                
                # Calculate comment ratio
                comments = sum(1 for line in lines if line.strip().startswith('#'))
                complexity["comment_ratio"] = comments / len(lines) if lines else 0

                # Calculate cyclomatic complexity
                tree = ast.parse(content)
                complexity["cyclomatic_complexity"] = self._calculate_cyclomatic_complexity(tree)

                # Calculate maintainability index
                complexity["maintainability_index"] = self._calculate_maintainability_index(
                    complexity["cyclomatic_complexity"],
                    complexity["lines_of_code"],
                    complexity["comment_ratio"]
                )

        except Exception as e:
            self.logger.error(f"Error analyzing code complexity: {str(e)}")

        return complexity

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of the code."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.FunctionDef,
                               ast.ClassDef, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_maintainability_index(self, cyclomatic_complexity: int,
                                       lines_of_code: int, comment_ratio: float) -> float:
        """Calculate maintainability index of the code."""
        if lines_of_code == 0:
            return 100.0

        volume = lines_of_code * (1 - comment_ratio)
        if volume == 0:
            return 100.0

        return max(0, min(100, 171 - 5.2 * math.log(volume) - 0.23 * cyclomatic_complexity - 16.2 * math.log(lines_of_code)))

    def _extract_code_metadata(self, content: str) -> Dict:
        """Extract code metadata."""
        metadata = {}
        try:
            # Look for common metadata patterns
            patterns = {
                'version': r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
                'author': r'__author__\s*=\s*[\'"]([^\'"]+)[\'"]',
                'license': r'__license__\s*=\s*[\'"]([^\'"]+)[\'"]',
                'description': r'"""([^"]*)"""'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metadata[key] = match.group(1)

        except Exception as e:
            self.logger.error(f"Error extracting code metadata: {str(e)}")

        return metadata

    def _generate_code_content(self, code_info: Dict) -> str:
        """Generate comprehensive code documentation content."""
        template = self.template_env.get_template('code.md.j2')
        return template.render(
            code_info=code_info,
            generated_at=datetime.utcnow().isoformat()
        )

    async def generate_system_docs(self, system_path: str) -> str:
        """Generate system documentation with enhanced features."""
        try:
            # Validate system path
            if not Path(system_path).exists():
                raise ValueError(f"System path {system_path} does not exist")

            # Create progress tracker
            progress = DocumentationProgress(
                doc_id=f"system_{system_path}",
                total_steps=4,
                completed_steps=0,
                current_step="Initialization",
                status="In Progress",
                start_time=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(minutes=20),
                errors=[],
                warnings=[],
                metadata={"system_path": system_path}
            )

            # Extract system information
            progress.current_step = "Extracting System Info"
            system_info = self._extract_system_info(system_path)
            progress.completed_steps += 1

            # Generate documentation
            progress.current_step = "Generating Documentation"
            content = self._generate_system_content(system_info)
            progress.completed_steps += 1

            # Create documentation
            progress.current_step = "Creating Documentation"
            doc_id = await self.generate_documentation(
                type="system",
                title=f"System Documentation - {system_path}",
                content=content,
                format="markdown",
                metadata={
                    "system_path": system_path,
                    "system_info": system_info
                },
                tags=["system", "documentation", system_path]
            )
            progress.completed_steps += 1

            # Update progress
            progress.completed_steps = 4
            progress.current_step = "Completed"
            progress.status = "Success"
            progress.estimated_completion = datetime.utcnow()

            return doc_id

        except Exception as e:
            self.logger.error(f"Failed to generate system documentation: {str(e)}")
            if progress:
                progress.errors.append({"error": str(e), "timestamp": datetime.utcnow()})
                progress.status = "Failed"
            raise

    def _extract_system_info(self, system_path: str) -> Dict:
        """Extract comprehensive system information."""
        system_info = {
            "path": system_path,
            "components": self._analyze_system_components(system_path),
            "architecture": self._analyze_system_architecture(system_path),
            "dependencies": self._extract_system_dependencies(system_path),
            "configuration": self._extract_system_configuration(system_path),
            "metadata": {}
        }

        try:
            # Extract system metadata
            system_info["metadata"] = self._extract_system_metadata(system_path)
        except Exception as e:
            self.logger.error(f"Error extracting system info: {str(e)}")
            raise

        return system_info

    def _analyze_system_components(self, system_path: str) -> List[Dict]:
        """Analyze system components and their relationships."""
        components = []
        try:
            for path in Path(system_path).rglob('*'):
                if path.is_file() and path.suffix in ['.py', '.js', '.ts', '.java']:
                    component = {
                        "name": path.name,
                        "path": str(path.relative_to(system_path)),
                        "type": self._detect_language(str(path)),
                        "dependencies": self._extract_dependencies(str(path)),
                        "metadata": self._extract_code_metadata(path.read_text())
                    }
                    components.append(component)
        except Exception as e:
            self.logger.error(f"Error analyzing system components: {str(e)}")
        return components

    def _analyze_system_architecture(self, system_path: str) -> Dict:
        """Analyze system architecture and design patterns."""
        architecture = {
            "patterns": [],
            "layers": [],
            "interfaces": [],
            "data_flow": []
        }

        try:
            # Look for common architectural patterns
            for path in Path(system_path).rglob('*'):
                if path.is_file():
                    content = path.read_text()
                    
                    # Detect patterns
                    if 'class Service' in content:
                        architecture["patterns"].append("Service Layer")
                    if 'class Repository' in content:
                        architecture["patterns"].append("Repository Pattern")
                    if 'class Factory' in content:
                        architecture["patterns"].append("Factory Pattern")
                    if 'class Observer' in content:
                        architecture["patterns"].append("Observer Pattern")

                    # Detect layers
                    if 'api/' in str(path):
                        architecture["layers"].append("API Layer")
                    if 'services/' in str(path):
                        architecture["layers"].append("Service Layer")
                    if 'models/' in str(path):
                        architecture["layers"].append("Model Layer")
                    if 'repositories/' in str(path):
                        architecture["layers"].append("Repository Layer")

        except Exception as e:
            self.logger.error(f"Error analyzing system architecture: {str(e)}")

        return architecture

    def _extract_system_dependencies(self, system_path: str) -> Dict:
        """Extract system-wide dependencies."""
        dependencies = {
            "internal": [],
            "external": [],
            "dev": []
        }

        try:
            # Check for package management files
            for file in ['requirements.txt', 'package.json', 'pom.xml']:
                path = Path(system_path) / file
                if path.exists():
                    if file == 'requirements.txt':
                        with open(path) as f:
                            dependencies["external"].extend(line.strip() for line in f if line.strip())
                    elif file == 'package.json':
                        with open(path) as f:
                            data = json.load(f)
                            dependencies["external"].extend(data.get('dependencies', {}).keys())
                            dependencies["dev"].extend(data.get('devDependencies', {}).keys())

            # Extract internal dependencies
            for path in Path(system_path).rglob('*.py'):
                if path.is_file():
                    with open(path) as f:
                        content = f.read()
                        imports = re.findall(r'^import\s+(\w+)|^from\s+(\w+)\s+import', content, re.MULTILINE)
                        for imp in imports:
                            dep = imp[0] or imp[1]
                            if dep and dep not in dependencies["internal"]:
                                dependencies["internal"].append(dep)

        except Exception as e:
            self.logger.error(f"Error extracting system dependencies: {str(e)}")

        return dependencies

    def _extract_system_configuration(self, system_path: str) -> Dict:
        """Extract system configuration information."""
        configuration = {
            "environment": {},
            "settings": {},
            "secrets": {}
        }

        try:
            # Look for configuration files
            config_files = ['config.json', 'settings.json', '.env', 'config.yaml']
            for file in config_files:
                path = Path(system_path) / file
                if path.exists():
                    if file.endswith('.json'):
                        with open(path) as f:
                            data = json.load(f)
                            configuration["settings"].update(data)
                    elif file.endswith('.yaml'):
                        with open(path) as f:
                            data = yaml.safe_load(f)
                            configuration["settings"].update(data)
                    elif file == '.env':
                        with open(path) as f:
                            for line in f:
                                if '=' in line:
                                    key, value = line.strip().split('=', 1)
                                    if 'SECRET' in key or 'KEY' in key:
                                        configuration["secrets"][key] = '***'
                                    else:
                                        configuration["environment"][key] = value

        except Exception as e:
            self.logger.error(f"Error extracting system configuration: {str(e)}")

        return configuration

    def _extract_system_metadata(self, system_path: str) -> Dict:
        """Extract system metadata."""
        metadata = {
            "name": Path(system_path).name,
            "description": "",
            "version": "",
            "authors": [],
            "license": "",
            "created_at": "",
            "updated_at": ""
        }

        try:
            # Look for metadata files
            for file in ['README.md', 'package.json', 'setup.py']:
                path = Path(system_path) / file
                if path.exists():
                    if file == 'README.md':
                        content = path.read_text()
                        metadata["description"] = content.split('\n')[0]
                    elif file == 'package.json':
                        with open(path) as f:
                            data = json.load(f)
                            metadata.update({
                                "name": data.get('name', metadata["name"]),
                                "version": data.get('version', metadata["version"]),
                                "authors": data.get('authors', metadata["authors"]),
                                "license": data.get('license', metadata["license"])
                            })
                    elif file == 'setup.py':
                        content = path.read_text()
                        metadata["version"] = re.search(r'version=["\']([^"\']+)["\']', content).group(1)

            # Get file timestamps
            created = min(p.stat().st_ctime for p in Path(system_path).rglob('*') if p.is_file())
            updated = max(p.stat().st_mtime for p in Path(system_path).rglob('*') if p.is_file())
            metadata["created_at"] = datetime.fromtimestamp(created).isoformat()
            metadata["updated_at"] = datetime.fromtimestamp(updated).isoformat()

        except Exception as e:
            self.logger.error(f"Error extracting system metadata: {str(e)}")

        return metadata

    def _generate_system_content(self, system_info: Dict) -> str:
        """Generate comprehensive system documentation content."""
        template = self.template_env.get_template('system.md.j2')
        return template.render(
            system_info=system_info,
            generated_at=datetime.utcnow().isoformat()
        )

    async def export_documentation(self, doc_id: str, format: str = 'html') -> str:
        """Export documentation in various formats with enhanced features."""
        try:
            if doc_id not in self.docs:
                raise ValueError(f"Documentation {doc_id} not found")

            doc = self.docs[doc_id]

            # Create progress tracker
            progress = DocumentationProgress(
                doc_id=f"export_{doc_id}",
                total_steps=3,
                completed_steps=0,
                current_step="Initialization",
                status="In Progress",
                start_time=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(minutes=5),
                errors=[],
                warnings=[],
                metadata={"doc_id": doc_id, "format": format}
            )

            # Convert content
            progress.current_step = "Converting Content"
            if format == 'html':
                content = markdown.markdown(doc.content)
            elif format == 'pdf':
                content = self._convert_to_pdf(doc.content)
            elif format == 'rst':
                content = self._convert_to_rst(doc.content)
            else:
                raise ValueError(f"Unsupported format: {format}")
            progress.completed_steps += 1

            # Generate export
            progress.current_step = "Generating Export"
            template = self.template_env.get_template(f'{format}_export.html.j2')
            export_content = template.render(
                doc=doc,
                content=content,
                generated_at=datetime.utcnow().isoformat()
            )
            progress.completed_steps += 1

            # Save export
            progress.current_step = "Saving Export"
            export_path = Path(self.doc_config.get('export_path', 'exports')) / f"{doc_id}.{format}"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(export_path, 'w') as f:
                await f.write(export_content)
            progress.completed_steps += 1

            # Update progress
            progress.completed_steps = 3
            progress.current_step = "Completed"
            progress.status = "Success"
            progress.estimated_completion = datetime.utcnow()

            return str(export_path)

        except Exception as e:
            self.logger.error(f"Failed to export documentation: {str(e)}")
            if progress:
                progress.errors.append({"error": str(e), "timestamp": datetime.utcnow()})
                progress.status = "Failed"
            raise

    def _convert_to_pdf(self, content: str) -> str:
        """Convert markdown content to PDF format."""
        # Implementation depends on PDF generation library
        pass

    def _convert_to_rst(self, content: str) -> str:
        """Convert markdown content to RST format."""
        # Implementation depends on RST conversion library
        pass

    def get_documentation(
        self,
        type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        status: Optional[DocumentationStatus] = None,
        author: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        updated_before: Optional[datetime] = None,
        security_level: Optional[int] = None,
        validation_level: Optional[int] = None,
        audit_level: Optional[int] = None,
        metrics_level: Optional[int] = None,
        health_level: Optional[int] = None,
        logging_level: Optional[int] = None
    ) -> List[Documentation]:
        """Get documentation with enhanced filtering and validation."""
        try:
            docs = list(self.docs.values())
            
            # Apply filters
            if type:
                docs = [d for d in docs if d.type == type]
            if tags:
                docs = [d for d in docs if all(tag in d.tags for tag in tags)]
            if status:
                docs = [d for d in docs if d.status == status]
            if author:
                docs = [d for d in docs if d.author == author]
            if created_after:
                docs = [d for d in docs if datetime.fromisoformat(d.created_at) > created_after]
            if created_before:
                docs = [d for d in docs if datetime.fromisoformat(d.created_at) < created_before]
            if updated_after:
                docs = [d for d in docs if datetime.fromisoformat(d.updated_at) > updated_after]
            if updated_before:
                docs = [d for d in docs if datetime.fromisoformat(d.updated_at) < updated_before]
            if security_level is not None:
                docs = [d for d in docs if d.security_level == security_level]
            if validation_level is not None:
                docs = [d for d in docs if d.validation_level == validation_level]
            if audit_level is not None:
                docs = [d for d in docs if d.audit_level == audit_level]
            if metrics_level is not None:
                docs = [d for d in docs if d.metrics_level == metrics_level]
            if health_level is not None:
                docs = [d for d in docs if d.health_level == health_level]
            if logging_level is not None:
                docs = [d for d in docs if d.logging_level == logging_level]
            
            # Sort by updated time (newest first)
            docs.sort(key=lambda x: x.updated_at, reverse=True)
            
            # Apply limit
            if limit:
                docs = docs[:limit]
            
            # Record metrics
            self._record_metrics("documentation_retrieved", str(len(docs)))
            
            return docs
            
        except Exception as e:
            self.logger.error(f"Failed to get documentation: {str(e)}")
            raise

    def update_documentation(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        status: Optional[DocumentationStatus] = None,
        author: Optional[str] = None,
        security_level: Optional[int] = None,
        validation_level: Optional[int] = None,
        audit_level: Optional[int] = None,
        metrics_level: Optional[int] = None,
        health_level: Optional[int] = None,
        logging_level: Optional[int] = None
    ):
        """Update documentation with enhanced validation and tracking."""
        try:
            if doc_id not in self.docs:
                raise ValueError(f"Documentation not found: {doc_id}")
            
            doc = self.docs[doc_id]
            
            # Validate changes
            if content and not isinstance(content, str):
                raise ValueError("Content must be a string")
            if metadata and not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")
            if tags and not isinstance(tags, list):
                raise ValueError("Tags must be a list")
            if status and not isinstance(status, DocumentationStatus):
                raise ValueError("Invalid status")
            if security_level is not None and not isinstance(security_level, int):
                raise ValueError("Security level must be an integer")
            if validation_level is not None and not isinstance(validation_level, int):
                raise ValueError("Validation level must be an integer")
            if audit_level is not None and not isinstance(audit_level, int):
                raise ValueError("Audit level must be an integer")
            if metrics_level is not None and not isinstance(metrics_level, int):
                raise ValueError("Metrics level must be an integer")
            if health_level is not None and not isinstance(health_level, int):
                raise ValueError("Health level must be an integer")
            if logging_level is not None and not isinstance(logging_level, int):
                raise ValueError("Logging level must be an integer")
            
            # Track changes
            changes = []
            
            # Apply changes
            if content:
                changes.append({"field": "content", "old": doc.content, "new": content})
                doc.content = content
            if metadata:
                changes.append({"field": "metadata", "old": doc.metadata, "new": metadata})
                doc.metadata.update(metadata)
            if tags:
                changes.append({"field": "tags", "old": doc.tags, "new": tags})
                doc.tags = tags
            if status:
                changes.append({"field": "status", "old": doc.status, "new": status})
                doc.status = status
            if author:
                changes.append({"field": "author", "old": doc.author, "new": author})
                doc.author = author
            if security_level is not None:
                changes.append({"field": "security_level", "old": doc.security_level, "new": security_level})
                doc.security_level = security_level
            if validation_level is not None:
                changes.append({"field": "validation_level", "old": doc.validation_level, "new": validation_level})
                doc.validation_level = validation_level
            if audit_level is not None:
                changes.append({"field": "audit_level", "old": doc.audit_level, "new": audit_level})
                doc.audit_level = audit_level
            if metrics_level is not None:
                changes.append({"field": "metrics_level", "old": doc.metrics_level, "new": metrics_level})
                doc.metrics_level = metrics_level
            if health_level is not None:
                changes.append({"field": "health_level", "old": doc.health_level, "new": health_level})
                doc.health_level = health_level
            if logging_level is not None:
                changes.append({"field": "logging_level", "old": doc.logging_level, "new": logging_level})
                doc.logging_level = logging_level
            
            # Update timestamps
            doc.updated_at = datetime.utcnow().isoformat()
            
            # Create new version if there are changes
            if changes:
                version = DocumentationVersion(
                    version=len(self.versions[doc_id]) + 1,
                    doc_id=doc_id,
                    content=doc.content,
                    format=doc.format,
                    created_at=datetime.utcnow(),
                    created_by=author or doc.author or "system",
                    changes=changes,
                    status=doc.status,
                    reviewers=doc.reviewers,
                    approvers=doc.approvers,
                    metadata=doc.metadata
                )
                self.versions[doc_id].append(version)
            
            # Cache updated documentation
            self._cache_documentation(doc_id, doc)
            
            # Backup documentation
            asyncio.create_task(self._backup_documentation(doc_id))
            
            # Record metrics
            self._record_metrics("documentation_updated", doc_id)
            
            # Audit
            self._audit_action("documentation_updated", doc_id, author)
            
            self.logger.info(f"Updated documentation: {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update documentation: {str(e)}")
            raise

    def clear_documentation(
        self,
        before: Optional[datetime] = None,
        type: Optional[str] = None,
        status: Optional[DocumentationStatus] = None,
        author: Optional[str] = None,
        security_level: Optional[int] = None,
        validation_level: Optional[int] = None,
        audit_level: Optional[int] = None,
        metrics_level: Optional[int] = None,
        health_level: Optional[int] = None,
        logging_level: Optional[int] = None
    ):
        """Clear documentation with enhanced filtering and validation."""
        try:
            docs_to_keep = {}
            
            for doc_id, doc in self.docs.items():
                # Apply filters
                if before and datetime.fromisoformat(doc.updated_at) <= before:
                    continue
                if type and doc.type != type:
                    continue
                if status and doc.status != status:
                    continue
                if author and doc.author != author:
                    continue
                if security_level is not None and doc.security_level != security_level:
                    continue
                if validation_level is not None and doc.validation_level != validation_level:
                    continue
                if audit_level is not None and doc.audit_level != audit_level:
                    continue
                if metrics_level is not None and doc.metrics_level != metrics_level:
                    continue
                if health_level is not None and doc.health_level != health_level:
                    continue
                if logging_level is not None and doc.logging_level != logging_level:
                    continue
                
                docs_to_keep[doc_id] = doc
            
            # Clear documentation
            cleared_count = len(self.docs) - len(docs_to_keep)
            self.docs = docs_to_keep
            
            # Clear related data
            self.versions = {k: v for k, v in self.versions.items() if k in docs_to_keep}
            self.collaborations = {k: v for k, v in self.collaborations.items() if k in docs_to_keep}
            self.progress = {k: v for k, v in self.progress.items() if k in docs_to_keep}
            
            # Clear cache
            for doc_id in set(self.cache.keys()) - set(docs_to_keep.keys()):
                del self.cache[f"doc:{doc_id}"]
            
            # Record metrics
            self._record_metrics("documentation_cleared", str(cleared_count))
            
            # Audit
            self._audit_action("documentation_cleared", str(cleared_count), None)
            
            self.logger.info(f"Cleared {cleared_count} documentation entries")
            
        except Exception as e:
            self.logger.error(f"Failed to clear documentation: {str(e)}")
            raise

    async def export_documentation(
        self,
        format: str = 'html',
        output_path: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        type: Optional[str] = None,
        status: Optional[DocumentationStatus] = None,
        author: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        updated_before: Optional[datetime] = None,
        security_level: Optional[int] = None,
        validation_level: Optional[int] = None,
        audit_level: Optional[int] = None,
        metrics_level: Optional[int] = None,
        health_level: Optional[int] = None,
        logging_level: Optional[int] = None
    ) -> str:
        """Export documentation with enhanced filtering and validation."""
        try:
            if not output_path:
                output_path = f"docs/export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get documentation to export
            docs = self.get_documentation(
                type=type,
                status=status,
                author=author,
                created_after=created_after,
                created_before=created_before,
                updated_after=updated_after,
                updated_before=updated_before,
                security_level=security_level,
                validation_level=validation_level,
                audit_level=audit_level,
                metrics_level=metrics_level,
                health_level=health_level,
                logging_level=logging_level
            )
            
            if doc_ids:
                docs = [d for d in docs if d.id in doc_ids]
            
            if not docs:
                raise ValueError("No documentation to export")
            
            # Create progress tracker
            progress = DocumentationProgress(
                doc_id=f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                total_steps=len(docs),
                completed_steps=0,
                current_step="Initialization",
                status="In Progress",
                start_time=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(minutes=len(docs)),
                errors=[],
                warnings=[],
                metadata={
                    "format": format,
                    "output_path": str(output_path),
                    "doc_count": len(docs)
                }
            )
            
            # Export documentation
            try:
                if format == 'html':
                    await self._export_html(output_path, docs, progress)
                elif format == 'pdf':
                    await self._export_pdf(output_path, docs, progress)
                elif format == 'markdown':
                    await self._export_markdown(output_path, docs, progress)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                
                # Update progress
                progress.completed_steps = len(docs)
                progress.current_step = "Completed"
                progress.status = "Success"
                progress.estimated_completion = datetime.utcnow()
                
                # Record metrics
                self._record_metrics("documentation_exported", str(len(docs)))
                
                # Audit
                self._audit_action("documentation_exported", str(len(docs)), None)
                
                self.logger.info(f"Exported {len(docs)} documentation entries to {output_path}")
                return str(output_path)
                
            except Exception as e:
                progress.errors.append({"error": str(e), "timestamp": datetime.utcnow()})
                progress.status = "Failed"
                raise
            
        except Exception as e:
            self.logger.error(f"Failed to export documentation: {str(e)}")
            raise 