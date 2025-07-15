"""
Prompt Formatter

Formats and validates prompts with JSON input handling and fallback mechanisms.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FormattedPrompt:
    """Formatted prompt with metadata."""
    original_prompt: str
    formatted_prompt: str
    format_type: str
    validation_passed: bool
    errors: List[str]
    metadata: Dict[str, Any]


class PromptFormatter:
    """Formats prompts with validation and fallback handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the prompt formatter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_format = self.config.get("default_format", "text")
        self.max_prompt_length = self.config.get("max_prompt_length", 1000)
        self.enable_json_validation = self.config.get("enable_json_validation", True)
        
    def format_prompt(self, prompt: Union[str, Dict[str, Any]]) -> FormattedPrompt:
        """
        Format a prompt with validation and fallback handling.
        
        Args:
            prompt: Input prompt (string or dictionary)
            
        Returns:
            FormattedPrompt with results and metadata
        """
        original_prompt = str(prompt)
        errors = []
        format_type = "text"
        formatted_prompt = original_prompt
        
        try:
            # Try to parse as JSON if validation is enabled
            if self.enable_json_validation and self._looks_like_json(original_prompt):
                try:
                    parsed_json = json.loads(original_prompt)
                    formatted_prompt = self._format_json_prompt(parsed_json)
                    format_type = "json"
                except json.JSONDecodeError as e:
                    errors.append(f"JSON decode error: {str(e)}")
                    formatted_prompt = self._get_default_format(original_prompt)
                    format_type = "fallback"
                    
            # Validate prompt length
            if len(formatted_prompt) > self.max_prompt_length:
                errors.append(f"Prompt too long ({len(formatted_prompt)} chars, max {self.max_prompt_length})")
                formatted_prompt = formatted_prompt[:self.max_prompt_length] + "..."
                
            # Validate prompt content
            content_errors = self._validate_prompt_content(formatted_prompt)
            errors.extend(content_errors)
            
            # Clean up the prompt
            formatted_prompt = self._clean_prompt(formatted_prompt)
            
        except Exception as e:
            errors.append(f"Formatting error: {str(e)}")
            formatted_prompt = self._get_default_format(original_prompt)
            format_type = "error_fallback"
            
        return FormattedPrompt(
            original_prompt=original_prompt,
            formatted_prompt=formatted_prompt,
            format_type=format_type,
            validation_passed=len(errors) == 0,
            errors=errors,
            metadata={
                "length": len(formatted_prompt),
                "format_type": format_type,
                "has_errors": len(errors) > 0
            }
        )
        
    def _looks_like_json(self, text: str) -> bool:
        """Check if text looks like JSON."""
        text = text.strip()
        return (text.startswith('{') and text.endswith('}')) or \
               (text.startswith('[') and text.endswith(']'))
               
    def _format_json_prompt(self, parsed_json: Dict[str, Any]) -> str:
        """Format JSON prompt into structured text."""
        if isinstance(parsed_json, dict):
            # Handle dictionary format
            if "prompt" in parsed_json:
                base_prompt = parsed_json["prompt"]
            elif "message" in parsed_json:
                base_prompt = parsed_json["message"]
            elif "text" in parsed_json:
                base_prompt = parsed_json["text"]
            else:
                # Use the first string value found
                base_prompt = str(parsed_json)
                
            # Add context if available
            context_parts = []
            for key, value in parsed_json.items():
                if key not in ["prompt", "message", "text"] and isinstance(value, (str, int, float)):
                    context_parts.append(f"{key}: {value}")
                    
            if context_parts:
                base_prompt += f" [Context: {', '.join(context_parts)}]"
                
            return base_prompt
        else:
            return str(parsed_json)
            
    def _get_default_format(self, prompt: str) -> str:
        """Get default formatted prompt when JSON parsing fails."""
        # Remove JSON-like characters and clean up
        cleaned = re.sub(r'[{}[\]"]', '', prompt)
        cleaned = re.sub(r',', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if not cleaned:
            return "Please provide a valid prompt"
            
        return cleaned
        
    def _validate_prompt_content(self, prompt: str) -> List[str]:
        """Validate prompt content for issues."""
        errors = []
        
        # Check for empty or whitespace-only prompts
        if not prompt or not prompt.strip():
            errors.append("Prompt is empty or contains only whitespace")
            
        # Check for potentially harmful content
        harmful_patterns = [
            r"\b(delete|remove|drop|truncate)\b",
            r"\b(password|secret|key)\b",
            r"\b(exec|eval|system)\b",
            r"\b(rm -rf|format|wipe)\b"
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, prompt.lower()):
                errors.append("Prompt contains potentially harmful content")
                break
                
        # Check for excessive repetition
        words = prompt.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:  # More than 30% repetition
                errors.append("Prompt contains excessive word repetition")
                
        return errors
        
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize the prompt."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', prompt).strip()
        
        # Remove common formatting artifacts
        cleaned = re.sub(r'^\s*["\']\s*', '', cleaned)  # Remove leading quotes
        cleaned = re.sub(r'\s*["\']\s*$', '', cleaned)  # Remove trailing quotes
        
        # Normalize line breaks
        cleaned = re.sub(r'\n+', ' ', cleaned)
        
        return cleaned
        
    def format_prompt_with_template(self, prompt: str, template: str) -> FormattedPrompt:
        """
        Format prompt using a template.
        
        Args:
            prompt: Input prompt
            template: Template string with placeholders
            
        Returns:
            FormattedPrompt with template applied
        """
        try:
            # Simple template replacement
            formatted = template.replace("{prompt}", prompt)
            
            # Handle additional placeholders
            formatted = formatted.replace("{length}", str(len(prompt)))
            formatted = formatted.replace("{timestamp}", "now")
            
            return FormattedPrompt(
                original_prompt=prompt,
                formatted_prompt=formatted,
                format_type="template",
                validation_passed=True,
                errors=[],
                metadata={
                    "template_used": True,
                    "length": len(formatted)
                }
            )
            
        except Exception as e:
            return FormattedPrompt(
                original_prompt=prompt,
                formatted_prompt=prompt,
                format_type="template_error",
                validation_passed=False,
                errors=[f"Template formatting error: {str(e)}"],
                metadata={"template_used": False}
            )
            
    def validate_json_input(self, json_str: str) -> Tuple[bool, List[str]]:
        """
        Validate JSON input string.
        
        Args:
            json_str: JSON string to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            parsed = json.loads(json_str)
            
            # Check for required fields if it's a prompt object
            if isinstance(parsed, dict):
                if "prompt" not in parsed and "message" not in parsed and "text" not in parsed:
                    errors.append("JSON missing required 'prompt', 'message', or 'text' field")
                    
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            return False, [f"JSON decode error: {str(e)}"]
            
    def extract_prompt_from_json(self, json_str: str) -> Optional[str]:
        """
        Extract prompt text from JSON string.
        
        Args:
            json_str: JSON string containing prompt
            
        Returns:
            Extracted prompt text or None if extraction fails
        """
        try:
            parsed = json.loads(json_str)
            
            if isinstance(parsed, dict):
                # Try different possible field names
                for field in ["prompt", "message", "text", "content"]:
                    if field in parsed and isinstance(parsed[field], str):
                        return parsed[field]
                        
            elif isinstance(parsed, str):
                return parsed
                
            return None
            
        except json.JSONDecodeError:
            return None
            
    def get_formatting_statistics(self) -> Dict[str, Any]:
        """Get formatting statistics and configuration."""
        return {
            "max_prompt_length": self.max_prompt_length,
            "enable_json_validation": self.enable_json_validation,
            "default_format": self.default_format,
            "supported_formats": ["text", "json", "template", "fallback"]
        } 