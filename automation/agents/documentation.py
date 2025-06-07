import logging
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
from datetime import datetime
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

class DocumentationGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.docs: Dict[str, Documentation] = {}
        self.doc_config = config.get('documentation', {})
        self.template_loader = jinja2.FileSystemLoader(self.doc_config.get('templates_path', 'templates'))
        self.template_env = jinja2.Environment(loader=self.template_loader)

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

    async def generate_documentation(
        self,
        type: str,
        title: str,
        content: str,
        format: str = 'markdown',
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Generate documentation for a component.
        
        Args:
            type: Documentation type (api, code, system, etc.)
            title: Documentation title
            content: Documentation content
            format: Content format (markdown, rst, html)
            metadata: Additional metadata
            tags: Documentation tags
        
        Returns:
            str: Documentation ID
        """
        doc_id = str(len(self.docs) + 1)
        
        doc = Documentation(
            id=doc_id,
            type=type,
            title=title,
            content=content,
            format=format,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata=metadata or {},
            tags=tags or []
        )
        
        self.docs[doc_id] = doc
        self.logger.info(f"Generated documentation: {doc_id} - {title}")
        
        return doc_id

    async def generate_api_docs(self, module_path: str) -> str:
        """Generate API documentation from a Python module."""
        try:
            # Parse module
            module = ast.parse(open(module_path).read())
            
            # Extract docstrings and function signatures
            api_docs = []
            for node in ast.walk(module):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        api_docs.append({
                            'name': node.name,
                            'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                            'docstring': docstring,
                            'signature': self._get_signature(node)
                        })
            
            # Generate markdown
            content = self._generate_api_markdown(api_docs)
            
            # Create documentation
            return await self.generate_documentation(
                type='api',
                title=f"API Documentation - {Path(module_path).stem}",
                content=content,
                format='markdown',
                metadata={'module': module_path},
                tags=['api', 'code']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate API docs: {str(e)}")
            raise

    def _get_signature(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> str:
        """Get function/class signature."""
        if isinstance(node, ast.FunctionDef):
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            return f"{node.name}({', '.join(args)})"
        return node.name

    def _generate_api_markdown(self, api_docs: List[Dict]) -> str:
        """Generate markdown from API documentation."""
        template = self.template_env.get_template('api.md.j2')
        return template.render(api_docs=api_docs)

    async def generate_system_docs(self) -> str:
        """Generate system documentation."""
        try:
            # Collect system information
            system_info = {
                'components': self._get_system_components(),
                'architecture': self._get_system_architecture(),
                'configuration': self._get_system_configuration(),
                'dependencies': self._get_system_dependencies()
            }
            
            # Generate markdown
            content = self._generate_system_markdown(system_info)
            
            # Create documentation
            return await self.generate_documentation(
                type='system',
                title='System Documentation',
                content=content,
                format='markdown',
                metadata=system_info,
                tags=['system', 'architecture']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate system docs: {str(e)}")
            raise

    def _get_system_components(self) -> List[Dict]:
        """Get system components information."""
        components = []
        for component in self.doc_config.get('components', []):
            components.append({
                'name': component['name'],
                'description': component.get('description', ''),
                'type': component.get('type', ''),
                'dependencies': component.get('dependencies', [])
            })
        return components

    def _get_system_architecture(self) -> Dict:
        """Get system architecture information."""
        return self.doc_config.get('architecture', {})

    def _get_system_configuration(self) -> Dict:
        """Get system configuration information."""
        return self.doc_config.get('configuration', {})

    def _get_system_dependencies(self) -> List[Dict]:
        """Get system dependencies information."""
        return self.doc_config.get('dependencies', [])

    def _generate_system_markdown(self, system_info: Dict) -> str:
        """Generate markdown from system information."""
        template = self.template_env.get_template('system.md.j2')
        return template.render(**system_info)

    async def generate_code_docs(self, code_path: str) -> str:
        """Generate code documentation."""
        try:
            # Parse code
            with open(code_path) as f:
                code = f.read()
            
            # Extract code information
            code_info = {
                'path': code_path,
                'language': self._detect_language(code_path),
                'classes': self._extract_classes(code),
                'functions': self._extract_functions(code),
                'imports': self._extract_imports(code),
                'dependencies': self._extract_dependencies(code)
            }
            
            # Generate markdown
            content = self._generate_code_markdown(code_info)
            
            # Create documentation
            return await self.generate_documentation(
                type='code',
                title=f"Code Documentation - {Path(code_path).name}",
                content=content,
                format='markdown',
                metadata=code_info,
                tags=['code', 'implementation']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate code docs: {str(e)}")
            raise

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        languages = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust'
        }
        return languages.get(ext, 'Unknown')

    def _extract_classes(self, code: str) -> List[Dict]:
        """Extract class information from code."""
        classes = []
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'methods': self._extract_methods(node),
                    'bases': [base.id for base in node.bases if isinstance(base, ast.Name)]
                })
        return classes

    def _extract_functions(self, code: str) -> List[Dict]:
        """Extract function information from code."""
        functions = []
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'args': [arg.arg for arg in node.args.args],
                    'returns': self._extract_return_type(node)
                })
        return functions

    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
        return imports

    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract dependencies from code."""
        dependencies = set()
        for imp in self._extract_imports(code):
            # Extract base package name
            base = imp.split('.')[0]
            if base not in ['os', 'sys', 'datetime', 'json', 'typing']:
                dependencies.add(base)
        return list(dependencies)

    def _extract_methods(self, class_node: ast.ClassDef) -> List[Dict]:
        """Extract method information from a class."""
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'args': [arg.arg for arg in node.args.args],
                    'returns': self._extract_return_type(node)
                })
        return methods

    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type from function."""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Subscript):
                return f"{node.returns.value.id}[{node.returns.slice.value.id}]"
        return None

    def _generate_code_markdown(self, code_info: Dict) -> str:
        """Generate markdown from code information."""
        template = self.template_env.get_template('code.md.j2')
        return template.render(**code_info)

    def get_documentation(
        self,
        type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Documentation]:
        """Get documentation with optional filtering."""
        docs = list(self.docs.values())
        
        if type:
            docs = [d for d in docs if d.type == type]
        if tags:
            docs = [d for d in docs if all(tag in d.tags for tag in tags)]
        
        # Sort by updated time (newest first)
        docs.sort(key=lambda x: x.updated_at, reverse=True)
        
        if limit:
            docs = docs[:limit]
        
        return docs

    def update_documentation(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ):
        """Update documentation."""
        if doc_id not in self.docs:
            raise ValueError(f"Documentation not found: {doc_id}")
        
        doc = self.docs[doc_id]
        
        if content:
            doc.content = content
        if metadata:
            doc.metadata.update(metadata)
        if tags:
            doc.tags = tags
        
        doc.updated_at = datetime.now().isoformat()
        self.logger.info(f"Updated documentation: {doc_id}")

    def clear_documentation(self, before: Optional[datetime] = None):
        """Clear old documentation."""
        if before:
            self.docs = {
                id: d for id, d in self.docs.items()
                if datetime.fromisoformat(d.updated_at) > before
            }
        else:
            self.docs.clear()
        
        self.logger.info(f"Cleared documentation before {before}")

    async def export_documentation(
        self,
        format: str = 'html',
        output_path: Optional[str] = None
    ) -> str:
        """Export documentation to a specific format."""
        if not output_path:
            output_path = f"docs/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'html':
                await self._export_html(output_path)
            elif format == 'pdf':
                await self._export_pdf(output_path)
            elif format == 'markdown':
                await self._export_markdown(output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported documentation to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export documentation: {str(e)}")
            raise

    async def _export_html(self, output_path: Path):
        """Export documentation to HTML."""
        template = self.template_env.get_template('html_export.html.j2')
        
        for doc in self.docs.values():
            # Convert content to HTML if needed
            if doc.format == 'markdown':
                content = markdown.markdown(doc.content)
            elif doc.format == 'rst':
                content = docutils.core.publish_parts(doc.content, writer_name='html')['html_body']
            else:
                content = doc.content
            
            # Generate HTML
            html = template.render(
                title=doc.title,
                content=content,
                metadata=doc.metadata,
                tags=doc.tags,
                created_at=doc.created_at,
                updated_at=doc.updated_at
            )
            
            # Save HTML file
            file_path = output_path / f"{doc.id}.html"
            file_path.write_text(html)

    async def _export_pdf(self, output_path: Path):
        """Export documentation to PDF."""
        # First export to HTML
        html_path = output_path / 'html'
        await self._export_html(html_path)
        
        # Convert HTML to PDF
        for html_file in html_path.glob('*.html'):
            pdf_file = output_path / f"{html_file.stem}.pdf"
            subprocess.run(
                ['wkhtmltopdf', str(html_file), str(pdf_file)],
                check=True
            )

    async def _export_markdown(self, output_path: Path):
        """Export documentation to Markdown."""
        for doc in self.docs.values():
            # Convert content to Markdown if needed
            if doc.format == 'rst':
                content = docutils.core.publish_parts(doc.content, writer_name='markdown')['body']
            elif doc.format == 'html':
                # Use a simple HTML to Markdown converter
                content = self._html_to_markdown(doc.content)
            else:
                content = doc.content
            
            # Save Markdown file
            file_path = output_path / f"{doc.id}.md"
            file_path.write_text(content)

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown."""
        # This is a simple implementation
        # For production, use a proper HTML to Markdown converter
        markdown = html
        
        # Replace common HTML tags
        replacements = [
            (r'<h1>(.*?)</h1>', r'# \1'),
            (r'<h2>(.*?)</h2>', r'## \1'),
            (r'<h3>(.*?)</h3>', r'### \1'),
            (r'<p>(.*?)</p>', r'\1\n\n'),
            (r'<strong>(.*?)</strong>', r'**\1**'),
            (r'<em>(.*?)</em>', r'*\1*'),
            (r'<code>(.*?)</code>', r'`\1`'),
            (r'<pre>(.*?)</pre>', r'```\n\1\n```'),
            (r'<a href="(.*?)">(.*?)</a>', r'[\2](\1)'),
            (r'<ul>(.*?)</ul>', r'\1'),
            (r'<li>(.*?)</li>', r'- \1\n'),
            (r'<ol>(.*?)</ol>', r'\1'),
            (r'<li>(.*?)</li>', r'1. \1\n')
        ]
        
        for pattern, replacement in replacements:
            markdown = re.sub(pattern, replacement, markdown, flags=re.DOTALL)
        
        return markdown 