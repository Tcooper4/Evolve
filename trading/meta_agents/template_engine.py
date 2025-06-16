"""
Template Engine

This module implements template rendering functionality.

Note: This module was adapted from the legacy automation/core/template_engine.py file.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

class TemplateEngine:
    """Handles template rendering."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize template engine."""
        self.config = config
        self.template_dir = Path(config.get('template_dir', 'templates'))
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for template engine."""
        log_path = Path("logs/templates")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "template_engine.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def render_template(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Render a template with the given context."""
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            self.logger.error(f"Error rendering template {template_name}: {str(e)}")
            raise
    
    def render_string(
        self,
        template_string: str,
        context: Dict[str, Any]
    ) -> str:
        """Render a template string with the given context."""
        try:
            template = self.env.from_string(template_string)
            return template.render(**context)
        except Exception as e:
            self.logger.error(f"Error rendering template string: {str(e)}")
            raise
    
    def load_template(self, template_name: str) -> str:
        """Load a template file."""
        try:
            template_path = self.template_dir / template_name
            with open(template_path, 'r') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error loading template {template_name}: {str(e)}")
            raise
    
    def save_template(
        self,
        template_name: str,
        content: str
    ) -> None:
        """Save a template file."""
        try:
            template_path = self.template_dir / template_name
            template_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(template_path, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Saved template: {template_name}")
        except Exception as e:
            self.logger.error(f"Error saving template {template_name}: {str(e)}")
            raise
    
    def list_templates(self) -> List[str]:
        """List all available templates."""
        try:
            return [
                str(p.relative_to(self.template_dir))
                for p in self.template_dir.rglob('*')
                if p.is_file()
            ]
        except Exception as e:
            self.logger.error(f"Error listing templates: {str(e)}")
            raise
    
    def delete_template(self, template_name: str) -> None:
        """Delete a template file."""
        try:
            template_path = self.template_dir / template_name
            if template_path.exists():
                template_path.unlink()
                self.logger.info(f"Deleted template: {template_name}")
            else:
                raise FileNotFoundError(f"Template {template_name} not found")
        except Exception as e:
            self.logger.error(f"Error deleting template {template_name}: {str(e)}")
            raise
    
    def validate_template(
        self,
        template_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate a template with optional context."""
        try:
            template = self.env.get_template(template_name)
            if context:
                template.render(**context)
            return True
        except Exception as e:
            self.logger.error(f"Error validating template {template_name}: {str(e)}")
            return False 