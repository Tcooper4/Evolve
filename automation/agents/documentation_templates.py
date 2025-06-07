import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime
import asyncio
import aiofiles
from dataclasses import dataclass
import jinja2
import yaml
import markdown
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape
import mistune
from mistune import Markdown
import frontmatter
import slugify
import pygments
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

@dataclass
class Template:
    id: str
    name: str
    description: str
    content: str
    variables: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class DocumentationTemplates:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.template_config = config.get('documentation', {}).get('templates', {})
        self.setup_jinja()
        self.setup_markdown()
        self.templates: Dict[str, Template] = {}
        self.load_templates()

    def setup_logging(self):
        """Configure logging for the documentation template system."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "templates.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_jinja(self):
        """Setup Jinja2 environment."""
        template_path = Path(self.template_config.get('path', 'templates'))
        template_path.mkdir(parents=True, exist_ok=True)
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_path)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def setup_markdown(self):
        """Setup Markdown parser."""
        self.markdown = Markdown(
            renderer=mistune.HTMLRenderer(
                escape=False,
                hard_wrap=True
            )
        )

    def load_templates(self):
        """Load templates from filesystem."""
        try:
            template_path = Path(self.template_config.get('path', 'templates'))
            for template_file in template_path.glob('**/*.md'):
                self._load_template(template_file)
        except Exception as e:
            self.logger.error(f"Template loading error: {str(e)}")

    def _load_template(self, template_file: Path):
        """Load a single template from file."""
        try:
            # Read template file
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse frontmatter
            post = frontmatter.loads(content)
            metadata = post.metadata

            # Create template
            template = Template(
                id=slugify.slugify(template_file.stem),
                name=metadata.get('name', template_file.stem),
                description=metadata.get('description', ''),
                content=post.content,
                variables=metadata.get('variables', {}),
                metadata=metadata,
                created_at=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(metadata.get('updated_at', datetime.now().isoformat()))
            )

            self.templates[template.id] = template
            self.logger.info(f"Loaded template: {template.id}")

        except Exception as e:
            self.logger.error(f"Template loading error for {template_file}: {str(e)}")

    async def create_template(
        self,
        name: str,
        content: str,
        variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Template:
        """Create a new template."""
        try:
            # Generate template ID
            template_id = slugify.slugify(name)

            # Create template
            template = Template(
                id=template_id,
                name=name,
                description=metadata.get('description', '') if metadata else '',
                content=content,
                variables=variables or {},
                metadata=metadata or {},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # Save template
            await self._save_template(template)

            self.templates[template_id] = template
            self.logger.info(f"Created template: {template_id}")

            return template

        except Exception as e:
            self.logger.error(f"Template creation error: {str(e)}")
            raise

    async def _save_template(self, template: Template):
        """Save template to filesystem."""
        try:
            template_path = Path(self.template_config.get('path', 'templates'))
            template_file = template_path / f"{template.id}.md"

            # Prepare frontmatter
            frontmatter_data = {
                'name': template.name,
                'description': template.description,
                'variables': template.variables,
                'created_at': template.created_at.isoformat(),
                'updated_at': template.updated_at.isoformat()
            }
            frontmatter_data.update(template.metadata)

            # Create content with frontmatter
            content = frontmatter.dumps(
                frontmatter.Post(
                    template.content,
                    **frontmatter_data
                )
            )

            # Write to file
            async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
                await f.write(content)

        except Exception as e:
            self.logger.error(f"Template save error: {str(e)}")
            raise

    async def render_template(
        self,
        template_id: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render a template with variables."""
        try:
            # Get template
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")

            # Merge variables
            merged_variables = template.variables.copy()
            if variables:
                merged_variables.update(variables)

            # Create Jinja template
            jinja_template = self.jinja_env.from_string(template.content)

            # Render template
            content = jinja_template.render(**merged_variables)

            # Convert Markdown to HTML
            html = self.markdown(content)

            return html

        except Exception as e:
            self.logger.error(f"Template rendering error: {str(e)}")
            raise

    async def update_template(
        self,
        template_id: str,
        content: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Template:
        """Update an existing template."""
        try:
            # Get template
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")

            # Update template
            if content is not None:
                template.content = content
            if variables is not None:
                template.variables = variables
            if metadata is not None:
                template.metadata.update(metadata)
                if 'description' in metadata:
                    template.description = metadata['description']

            template.updated_at = datetime.now()

            # Save template
            await self._save_template(template)

            self.logger.info(f"Updated template: {template_id}")
            return template

        except Exception as e:
            self.logger.error(f"Template update error: {str(e)}")
            raise

    async def delete_template(self, template_id: str):
        """Delete a template."""
        try:
            # Get template
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")

            # Delete template file
            template_path = Path(self.template_config.get('path', 'templates'))
            template_file = template_path / f"{template_id}.md"
            if template_file.exists():
                template_file.unlink()

            # Remove from templates
            del self.templates[template_id]

            self.logger.info(f"Deleted template: {template_id}")

        except Exception as e:
            self.logger.error(f"Template deletion error: {str(e)}")
            raise

    def get_template(self, template_id: str) -> Optional[Template]:
        """Get template by ID."""
        return self.templates.get(template_id)

    def get_templates(
        self,
        name: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None
    ) -> List[Template]:
        """Get templates with optional filtering."""
        templates = list(self.templates.values())
        
        if name:
            templates = [t for t in templates if name.lower() in t.name.lower()]
        if created_after:
            templates = [t for t in templates if t.created_at >= created_after]
        if created_before:
            templates = [t for t in templates if t.created_at <= created_before]
        
        return templates

    async def validate_template(
        self,
        template_id: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a template and its variables."""
        try:
            # Get template
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")

            # Check required variables
            missing_vars = []
            for var_name, var_info in template.variables.items():
                if var_info.get('required', False):
                    if not variables or var_name not in variables:
                        missing_vars.append(var_name)

            # Validate variable types
            type_errors = []
            if variables:
                for var_name, var_value in variables.items():
                    if var_name in template.variables:
                        expected_type = template.variables[var_name].get('type', 'string')
                        if not self._validate_type(var_value, expected_type):
                            type_errors.append({
                                'variable': var_name,
                                'expected_type': expected_type,
                                'actual_type': type(var_value).__name__
                            })

            # Try rendering template
            render_error = None
            try:
                await self.render_template(template_id, variables)
            except Exception as e:
                render_error = str(e)

            return {
                'valid': not (missing_vars or type_errors or render_error),
                'missing_variables': missing_vars,
                'type_errors': type_errors,
                'render_error': render_error
            }

        except Exception as e:
            self.logger.error(f"Template validation error: {str(e)}")
            raise

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type."""
        try:
            if expected_type == 'string':
                return isinstance(value, str)
            elif expected_type == 'number':
                return isinstance(value, (int, float))
            elif expected_type == 'boolean':
                return isinstance(value, bool)
            elif expected_type == 'array':
                return isinstance(value, list)
            elif expected_type == 'object':
                return isinstance(value, dict)
            else:
                return True
        except Exception:
            return False

    async def export_template(
        self,
        template_id: str,
        format: str = 'markdown'
    ) -> str:
        """Export template in specified format."""
        try:
            # Get template
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")

            if format == 'markdown':
                # Export as Markdown with frontmatter
                return frontmatter.dumps(
                    frontmatter.Post(
                        template.content,
                        name=template.name,
                        description=template.description,
                        variables=template.variables,
                        created_at=template.created_at.isoformat(),
                        updated_at=template.updated_at.isoformat(),
                        **template.metadata
                    )
                )
            elif format == 'html':
                # Export as HTML
                return await self.render_template(template_id)
            elif format == 'json':
                # Export as JSON
                return json.dumps({
                    'id': template.id,
                    'name': template.name,
                    'description': template.description,
                    'content': template.content,
                    'variables': template.variables,
                    'metadata': template.metadata,
                    'created_at': template.created_at.isoformat(),
                    'updated_at': template.updated_at.isoformat()
                }, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            self.logger.error(f"Template export error: {str(e)}")
            raise

    async def import_template(
        self,
        content: str,
        format: str = 'markdown'
    ) -> Template:
        """Import template from specified format."""
        try:
            if format == 'markdown':
                # Import from Markdown with frontmatter
                post = frontmatter.loads(content)
                return await self.create_template(
                    name=post.metadata.get('name', ''),
                    content=post.content,
                    variables=post.metadata.get('variables', {}),
                    metadata=post.metadata
                )
            elif format == 'json':
                # Import from JSON
                data = json.loads(content)
                return await self.create_template(
                    name=data['name'],
                    content=data['content'],
                    variables=data['variables'],
                    metadata=data['metadata']
                )
            else:
                raise ValueError(f"Unsupported import format: {format}")

        except Exception as e:
            self.logger.error(f"Template import error: {str(e)}")
            raise

    def get_template_variables(self, template_id: str) -> Dict[str, Any]:
        """Get template variables and their metadata."""
        try:
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            return template.variables
        except Exception as e:
            self.logger.error(f"Template variables error: {str(e)}")
            raise

    async def render_preview(
        self,
        template_id: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render template preview with sample data."""
        try:
            # Get template
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")

            # Generate sample data
            sample_data = {}
            for var_name, var_info in template.variables.items():
                sample_data[var_name] = self._generate_sample_value(var_info)

            # Merge with provided variables
            if variables:
                sample_data.update(variables)

            # Render preview
            return await self.render_template(template_id, sample_data)

        except Exception as e:
            self.logger.error(f"Template preview error: {str(e)}")
            raise

    def _generate_sample_value(self, var_info: Dict[str, Any]) -> Any:
        """Generate sample value for variable."""
        var_type = var_info.get('type', 'string')
        if var_type == 'string':
            return var_info.get('example', 'Sample text')
        elif var_type == 'number':
            return var_info.get('example', 42)
        elif var_type == 'boolean':
            return var_info.get('example', True)
        elif var_type == 'array':
            return var_info.get('example', ['Item 1', 'Item 2'])
        elif var_type == 'object':
            return var_info.get('example', {'key': 'value'})
        else:
            return None 