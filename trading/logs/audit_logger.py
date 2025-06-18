"""Audit logging for tracking agent actions and system events."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from .logger import get_logger

# Get base logger
logger = get_logger(__name__)

class AuditLogger:
    """Audit logger for tracking agent actions and system events."""
    
    def __init__(self, log_file: str = "logs/audit.log"):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file
        self._ensure_log_dir()
    
    def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def _format_entry(
        self,
        action: str,
        details: Dict[str, Any],
        agent_id: Optional[str] = None,
        module: Optional[str] = None
    ) -> str:
        """Format audit log entry.
        
        Args:
            action: Action being logged
            details: Additional details about the action
            agent_id: Optional agent identifier
            module: Optional module name
            
        Returns:
            Formatted log entry
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        
        if agent_id:
            entry["agent_id"] = agent_id
        if module:
            entry["module"] = module
            
        return json.dumps(entry)
    
    def log_prompt(
        self,
        prompt: str,
        response: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log prompt and response.
        
        Args:
            prompt: Input prompt
            response: Model response
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata
        """
        details = {
            "prompt": prompt,
            "response": response
        }
        if metadata:
            details["metadata"] = metadata
            
        self._write_log("prompt", details, agent_id, module)
    
    def log_strategy(
        self,
        strategy_name: str,
        action: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log strategy activation.
        
        Args:
            strategy_name: Name of the strategy
            action: Action taken (e.g., "activated", "triggered")
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata
        """
        details = {
            "strategy": strategy_name,
            "action": action
        }
        if metadata:
            details["metadata"] = metadata
            
        self._write_log("strategy", details, agent_id, module)
    
    def log_rule(
        self,
        rule_id: str,
        action: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log rule activation.
        
        Args:
            rule_id: ID of the rule
            action: Action taken (e.g., "activated", "triggered")
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata
        """
        details = {
            "rule_id": rule_id,
            "action": action
        }
        if metadata:
            details["metadata"] = metadata
            
        self._write_log("rule", details, agent_id, module)
    
    def log_llm(
        self,
        model_name: str,
        action: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log LLM selection and usage.
        
        Args:
            model_name: Name of the model
            action: Action taken (e.g., "selected", "fallback")
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata
        """
        details = {
            "model": model_name,
            "action": action
        }
        if metadata:
            details["metadata"] = metadata
            
        self._write_log("llm", details, agent_id, module)
    
    def log_error(
        self,
        error: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error occurrence.
        
        Args:
            error: Error message
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata
        """
        details = {
            "error": error
        }
        if metadata:
            details["metadata"] = metadata
            
        self._write_log("error", details, agent_id, module)
    
    def _write_log(
        self,
        action: str,
        details: Dict[str, Any],
        agent_id: Optional[str] = None,
        module: Optional[str] = None
    ) -> None:
        """Write log entry to file.
        
        Args:
            action: Action being logged
            details: Additional details about the action
            agent_id: Optional agent identifier
            module: Optional module name
        """
        try:
            entry = self._format_entry(action, details, agent_id, module)
            with open(self.log_file, "a") as f:
                f.write(entry + "\n")
        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")

# Create singleton instance
audit_logger = AuditLogger() 