"""Audit logging for tracking agent actions and system events."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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

    def _ensure_log_dir(self) -> Dict[str, Any]:
        """Ensure log directory exists.

        Returns:
            Dictionary with directory creation status
        """
        try:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "log_dir": str(Path(self.log_file).parent),
                "exists": Path(self.log_file).parent.exists(),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "log_dir": str(Path(self.log_file).parent)}

    def _format_entry(
        self, action: str, details: Dict[str, Any], agent_id: Optional[str] = None, module: Optional[str] = None
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
        entry = {"timestamp": datetime.now().isoformat(), "action": action, "details": details}

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log prompt and response.

        Args:
            prompt: Input prompt
            response: Model response
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata

        Returns:
            Dictionary with logging status
        """
        details = {"prompt": prompt, "response": response}
        if metadata:
            details["metadata"] = metadata

        result = self._write_log("prompt", details, agent_id, module)
        return {
            "success": True,
            "action": "prompt_logged",
            "agent_id": agent_id,
            "module": module,
            "prompt_length": len(prompt),
            "response_length": len(response),
        }

    def log_strategy(
        self,
        strategy_name: str,
        action: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log strategy activation.

        Args:
            strategy_name: Name of the strategy
            action: Action taken (e.g., "activated", "triggered")
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata

        Returns:
            Dictionary with logging status
        """
        details = {"strategy": strategy_name, "action": action}
        if metadata:
            details["metadata"] = metadata

        result = self._write_log("strategy", details, agent_id, module)
        return {
            "success": True,
            "action": "strategy_logged",
            "strategy_name": strategy_name,
            "action_type": action,
            "agent_id": agent_id,
            "module": module,
        }

    def log_rule(
        self,
        rule_id: str,
        action: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log rule activation.

        Args:
            rule_id: ID of the rule
            action: Action taken (e.g., "activated", "triggered")
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata

        Returns:
            Dictionary with logging status
        """
        details = {"rule_id": rule_id, "action": action}
        if metadata:
            details["metadata"] = metadata

        result = self._write_log("rule", details, agent_id, module)
        return {
            "success": True,
            "action": "rule_logged",
            "rule_id": rule_id,
            "action_type": action,
            "agent_id": agent_id,
            "module": module,
        }

    def log_llm(
        self,
        model_name: str,
        action: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log LLM selection and usage.

        Args:
            model_name: Name of the model
            action: Action taken (e.g., "selected", "fallback")
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata

        Returns:
            Dictionary with logging status
        """
        details = {"model": model_name, "action": action}
        if metadata:
            details["metadata"] = metadata

        result = self._write_log("llm", details, agent_id, module)
        return {
            "success": True,
            "action": "llm_logged",
            "model_name": model_name,
            "action_type": action,
            "agent_id": agent_id,
            "module": module,
        }

    def log_error(
        self,
        error: str,
        agent_id: Optional[str] = None,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log error occurrence.

        Args:
            error: Error message
            agent_id: Optional agent identifier
            module: Optional module name
            metadata: Optional additional metadata

        Returns:
            Dictionary with logging status
        """
        details = {"error": error}
        if metadata:
            details["metadata"] = metadata

        result = self._write_log("error", details, agent_id, module)
        return {
            "success": True,
            "action": "error_logged",
            "error_length": len(error),
            "agent_id": agent_id,
            "module": module,
        }

    def _write_log(
        self, action: str, details: Dict[str, Any], agent_id: Optional[str] = None, module: Optional[str] = None
    ) -> Dict[str, Any]:
        """Write log entry to file.

        Args:
            action: Action being logged
            details: Additional details about the action
            agent_id: Optional agent identifier
            module: Optional module name

        Returns:
            Dictionary with write status
        """
        try:
            entry = self._format_entry(action, details, agent_id, module)
            with open(self.log_file, "a") as f:
                f.write(entry + "\n")
            return {"success": True, "action": action, "entry_length": len(entry), "log_file": self.log_file}
        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")
            return {"success": False, "error": str(e), "action": action, "log_file": self.log_file}


# Create singleton instance
audit_logger = AuditLogger()
