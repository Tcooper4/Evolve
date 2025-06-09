"""
Template Engine

This module implements a template engine for rendering notification templates.
It supports both string-based templates and file-based templates.
"""

import os
import re
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
from datetime import datetime

class TemplateEngine:
    """Template engine for rendering notification templates."""
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize the template engine."""
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir) if template_dir else None,
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.env.filters["format_datetime"] = self._format_datetime
        self.env.filters["format_number"] = self._format_number
        self.env.filters["truncate"] = self._truncate

    def render(self, template: str, data: Dict[str, Any]) -> str:
        """Render a template with the given data."""
        try:
            # Check if template is a file path
            if os.path.isfile(template):
                with open(template, "r") as f:
                    template_content = f.read()
                template_obj = Template(template_content)
            else:
                # Assume template is a string
                template_obj = Template(template)
            
            return template_obj.render(**data)
            
        except Exception as e:
            raise ValueError(f"Template rendering failed: {str(e)}")

    def render_file(self, template_name: str, data: Dict[str, Any]) -> str:
        """Render a template from a file."""
        if not self.template_dir:
            raise ValueError("Template directory not set")
            
        try:
            template = self.env.get_template(template_name)
            return template.render(**data)
            
        except Exception as e:
            raise ValueError(f"Template rendering failed: {str(e)}")

    def _format_datetime(self, value: Any, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format a datetime value."""
        if isinstance(value, datetime):
            return value.strftime(format)
        return str(value)

    def _format_number(self, value: Any, precision: int = 2) -> str:
        """Format a number with the given precision."""
        try:
            num = float(value)
            return f"{num:.{precision}f}"
        except (ValueError, TypeError):
            return str(value)

    def _truncate(self, value: str, length: int = 100, suffix: str = "...") -> str:
        """Truncate a string to the given length."""
        if not isinstance(value, str):
            value = str(value)
        if len(value) <= length:
            return value
        return value[:length - len(suffix)] + suffix

class NotificationTemplate:
    """Helper class for notification templates."""
    def __init__(self, template_engine: TemplateEngine):
        """Initialize the notification template helper."""
        self.template_engine = template_engine

    def render_email(self, template: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Render an email template."""
        try:
            # Render subject and body
            subject = self.template_engine.render(template + ".subject", data)
            body = self.template_engine.render(template + ".body", data)
            
            return {
                "subject": subject,
                "body": body
            }
            
        except Exception as e:
            raise ValueError(f"Email template rendering failed: {str(e)}")

    def render_slack(self, template: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a Slack message template."""
        try:
            # Render message and attachments
            message = self.template_engine.render(template + ".message", data)
            attachments = self.template_engine.render(template + ".attachments", data)
            
            return {
                "text": message,
                "attachments": attachments
            }
            
        except Exception as e:
            raise ValueError(f"Slack template rendering failed: {str(e)}")

    def render_webhook(self, template: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a webhook payload template."""
        try:
            # Render payload
            payload = self.template_engine.render(template + ".payload", data)
            
            return {
                "payload": payload
            }
            
        except Exception as e:
            raise ValueError(f"Webhook template rendering failed: {str(e)}") 