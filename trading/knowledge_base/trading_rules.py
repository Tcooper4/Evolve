import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleStatus(Enum):
    """Enum for rule status."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SNOOZED = "SNOOZED"
    ARCHIVED = "ARCHIVED"


class LogicalOperator(Enum):
    """Enum for logical operators in rule chaining."""

    AND = "AND"
    OR = "OR"


@dataclass
class RuleMetadata:
    """Metadata for a trading rule."""

    version: int = 1
    created_at: datetime = None
    updated_at: datetime = None
    created_by: str = None
    modified_by: str = None
    agent_id: Optional[str] = None
    agent_confidence: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


class TradingRules:
    """A class to manage trading rules and strategies."""

    def __init__(self):
        """Initialize the TradingRules class."""
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.rule_audit_log: List[Dict[str, Any]] = []

    def add_rule(
        self,
        rule_name: str,
        rule_description: str,
        category: str = "General",
        priority: int = 0,
        dependencies: List[str] = None,
        tags: List[str] = None,
        logical_operator: LogicalOperator = LogicalOperator.AND,
        action: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        agent_confidence: Optional[float] = None,
        created_by: str = "system",
    ) -> str:
        """Add a new trading rule.

        Args:
            rule_name (str): The name of the rule.
            rule_description (str): A description of the rule.
            category (str): The category of the rule.
            priority (int): The priority of the rule.
            dependencies (List[str]): List of rule IDs this rule depends on.
            tags (List[str]): List of tags for the rule.
            logical_operator (LogicalOperator): Operator for rule dependencies.
            action (Dict[str, Any]): Action to take when rule is triggered.
            agent_id (Optional[str]): ID of the agent that created the rule.
            agent_confidence (Optional[float]): Confidence score of the agent.
            created_by (str): Identifier of who created the rule.

        Returns:
            str: The ID of the created rule.
        """
        if not rule_name or not rule_description:
            raise ValueError("Rule name and description cannot be empty.")

        rule_id = str(uuid4())
        metadata = RuleMetadata(
            created_by=created_by,
            modified_by=created_by,
            agent_id=agent_id,
            agent_confidence=agent_confidence,
        )

        self.rules[rule_id] = {
            "name": rule_name,
            "description": rule_description,
            "category": category,
            "priority": priority,
            "dependencies": dependencies or [],
            "tags": tags or [],
            "logical_operator": logical_operator.value,
            "action": action,
            "status": RuleStatus.ACTIVE.value,
            "metadata": asdict(metadata),
            "snooze_until": None,
        }

        self._log_rule_change(rule_id, "created")
        logger.info(f"Added new rule: {rule_name} (ID: {rule_id})")
        return rule_id

    def update_rule(
        self, rule_id: str, updates: Dict[str, Any], modified_by: str
    ) -> None:
        """Update an existing rule.

        Args:
            rule_id (str): The ID of the rule to update.
            updates (Dict[str, Any]): Dictionary of fields to update.
            modified_by (str): Identifier of who modified the rule.
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule with ID {rule_id} not found.")

        rule = self.rules[rule_id]
        metadata = RuleMetadata(**rule["metadata"])
        metadata.version += 1
        metadata.updated_at = datetime.now()
        metadata.modified_by = modified_by

        # Update fields
        for key, value in updates.items():
            if key != "metadata":  # Don't allow direct metadata updates
                rule[key] = value

        rule["metadata"] = asdict(metadata)
        self._log_rule_change(rule_id, "updated")
        logger.info(f"Updated rule: {rule['name']} (ID: {rule_id})")

    def filter_rules_by_tag(self, tag: str) -> Dict[str, Dict[str, Any]]:
        """Filter rules by tag.

        Args:
            tag (str): The tag to filter by.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of rules with the specified tag.
        """
        return {
            rule_id: rule
            for rule_id, rule in self.rules.items()
            if tag in rule.get("tags", [])
        }

    def snooze_rule(self, rule_id: str, duration_minutes: int) -> None:
        """Snooze a rule for a specified duration.

        Args:
            rule_id (str): The ID of the rule to snooze.
            duration_minutes (int): Duration in minutes to snooze the rule.
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule with ID {rule_id} not found.")

        rule = self.rules[rule_id]
        rule["status"] = RuleStatus.SNOOZED.value
        rule["snooze_until"] = datetime.now() + timedelta(minutes=duration_minutes)
        self._log_rule_change(rule_id, "snoozed")
        logger.info(
            f"Snoozed rule: {rule['name']} (ID: {rule_id}) for {duration_minutes} minutes"
        )

    def archive_rule(self, rule_id: str) -> None:
        """Archive a rule.

        Args:
            rule_id (str): The ID of the rule to archive.
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule with ID {rule_id} not found.")

        rule = self.rules[rule_id]
        rule["status"] = RuleStatus.ARCHIVED.value
        self._log_rule_change(rule_id, "archived")
        logger.info(f"Archived rule: {rule['name']} (ID: {rule_id})")

    def is_rule_active(self, rule_id: str) -> bool:
        """Check if a rule is currently active.

        Args:
            rule_id (str): The ID of the rule to check.

        Returns:
            bool: True if the rule is active, False otherwise.
        """
        if rule_id not in self.rules:
            return False

        rule = self.rules[rule_id]

        # Check basic status
        if rule["status"] != RuleStatus.ACTIVE.value:
            if rule["status"] == RuleStatus.SNOOZED.value:
                if rule["snooze_until"] and datetime.now() > rule["snooze_until"]:
                    rule["status"] = RuleStatus.ACTIVE.value
                    return True
            return False

        # Check dependencies if any
        if rule["dependencies"]:
            dependency_results = [
                self.is_rule_active(dep_id) for dep_id in rule["dependencies"]
            ]

            if rule["logical_operator"] == LogicalOperator.AND.value:
                return all(dependency_results)
            else:  # OR
                return any(dependency_results)

        return True

    def get_triggered_actions(self) -> List[Dict[str, Any]]:
        """Get all currently triggered rule actions.

        Returns:
            List[Dict[str, Any]]: List of triggered actions.
        """
        triggered_actions = []
        for rule_id, rule in self.rules.items():
            if self.is_rule_active(rule_id) and rule.get("action"):
                triggered_actions.append(
                    {
                        "rule_id": rule_id,
                        "rule_name": rule["name"],
                        "action": rule["action"],
                    }
                )
        return triggered_actions

    def test_rule(self, rule_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Test a rule against historical data.

        Args:
            rule_id (str): The ID of the rule to test.
            df (pd.DataFrame): Historical data to test against.

        Returns:
            Dict[str, Any]: Test results including accuracy and hit rate.
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule with ID {rule_id} not found.")

        # Placeholder for actual rule testing logic
        # This should be implemented based on specific rule types
        results = {
            "rule_id": rule_id,
            "rule_name": self.rules[rule_id]["name"],
            "accuracy": 0.0,
            "hit_rate": 0.0,
            "total_signals": 0,
            "correct_signals": 0,
            "test_period": {"start": df.index[0], "end": df.index[-1]},
        }

        logger.info(f"Tested rule: {self.rules[rule_id]['name']} (ID: {rule_id})")
        return results

    def export_rule_audit_log(self, path: str) -> None:
        """Export the rule audit log to a file.

        Args:
            path (str): Path to save the audit log.
        """
        with open(path, "w") as f:
            json.dump(self.rule_audit_log, f, default=str)
        logger.info(f"Exported rule audit log to {path}")

    def _log_rule_change(self, rule_id: str, action: str) -> None:
        """Log a change to a rule.

        Args:
            rule_id (str): The ID of the rule that changed.
            action (str): The type of change.
        """
        self.rule_audit_log.append(
            {
                "timestamp": datetime.now(),
                "rule_id": rule_id,
                "action": action,
                "rule_state": self.rules[rule_id],
            }
        )

    def __str__(self) -> str:
        """Return a string representation of the trading rules."""
        return f"TradingRules with {len(self.rules)} rules"

    def __repr__(self) -> str:
        """Return a detailed string representation of the trading rules."""
        return f"TradingRules(rules={self.rules})"

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get a trading rule by ID.

        Args:
            rule_id (str): The ID of the rule to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The rule details, or None if not found.
        """
        return self.rules.get(rule_id)

    def list_rules(self) -> Dict[str, Dict[str, Any]]:
        """List all trading rules.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of all trading rules.
        """
        return self.rules

    def activate_rule(self, rule_id: str) -> None:
        """Activate a trading rule.

        Args:
            rule_id (str): The ID of the rule to activate.
        """
        if rule_id in self.rules:
            self.rules[rule_id]["status"] = RuleStatus.ACTIVE.value
            self._log_rule_change(rule_id, "activated")
            logger.info(
                f"Activated rule: {self.rules[rule_id]['name']} (ID: {rule_id})"
            )

    def deactivate_rule(self, rule_id: str) -> None:
        """Deactivate a trading rule.

        Args:
            rule_id (str): The ID of the rule to deactivate.
        """
        if rule_id in self.rules:
            self.rules[rule_id]["status"] = RuleStatus.INACTIVE.value
            self._log_rule_change(rule_id, "deactivated")
            logger.info(
                f"Deactivated rule: {self.rules[rule_id]['name']} (ID: {rule_id})"
            )

    def export_rules(self, filename: str) -> None:
        """Export rules to a file.

        Args:
            filename (str): The name of the file to export rules to.
        """
        with open(filename, "w") as f:
            json.dump(self.rules, f, default=str)
        logger.info(f"Exported rules to {filename}")

    def import_rules(self, filename: str) -> None:
        """Import rules from a file.

        Args:
            filename (str): The name of the file to import rules from.
        """
        with open(filename, "r") as f:
            imported_rules = json.load(f)

        # Convert imported rules to new format if needed
        for rule_id, rule in imported_rules.items():
            if "metadata" not in rule:
                # Convert old format to new format
                metadata = RuleMetadata(
                    created_at=rule.get("created_at", datetime.now()),
                    updated_at=rule.get("modified_at", datetime.now()),
                    created_by=rule.get("created_by", "system"),
                    modified_by=rule.get("modified_by", "system"),
                )
                rule["metadata"] = asdict(metadata)
                rule["status"] = (
                    RuleStatus.ACTIVE.value
                    if rule.get("active", True)
                    else RuleStatus.INACTIVE.value
                )
                rule["tags"] = rule.get("tags", [])
                rule["logical_operator"] = rule.get(
                    "logical_operator", LogicalOperator.AND.value
                )
                rule["action"] = rule.get("action", None)
                rule["snooze_until"] = None

        self.rules = imported_rules
        logger.info(f"Imported rules from {filename}")

    def load_rules_from_json(self, json_file: str) -> None:
        """Load trading rules from a JSON file.

        Args:
            json_file (str): The path to the JSON file containing trading rules.
        """
        with open(json_file, "r") as f:
            rules_data = json.load(f)

        def process_rules(data: Dict, prefix: str = "") -> None:
            """Process rules recursively."""
            for key, value in data.items():
                if isinstance(value, dict):
                    if "description" in value:
                        # This is a rule
                        rule_name = f"{prefix}{key}" if prefix else key
                        rule_id = self.add_rule(
                            rule_name=rule_name,
                            rule_description=value["description"],
                            category=value.get("category", "General"),
                            priority=value.get("priority", 0),
                            dependencies=value.get("dependencies", []),
                            tags=value.get("tags", []),
                            logical_operator=LogicalOperator(
                                value.get("logical_operator", LogicalOperator.AND.value)
                            ),
                            action=value.get("action", None),
                            agent_id=value.get("agent_id"),
                            agent_confidence=value.get("agent_confidence"),
                            created_by=value.get("created_by", "system"),
                        )
                    # Process nested rules
                    process_rules(value, f"{prefix}{key}.")
                elif isinstance(value, list) and key == "rules":
                    # Process list of rules
                    for i, rule in enumerate(value):
                        rule_name = f"{prefix}rule_{i+1}"
                        self.add_rule(
                            rule_name=rule_name,
                            rule_description=rule,
                            category="General",
                            priority=0,
                            created_by="system",
                        )

        process_rules(rules_data)
        logger.info(f"Loaded rules from {json_file}")
