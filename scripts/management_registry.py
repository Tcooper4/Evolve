"""
Management Registry Module

This module handles management-related functionality and registry management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ManagementRegistry:
    """Registry for management components and configurations."""

    def __init__(self):
        """Initialize the management registry."""
        self.managers = {}
        self.manager_configs = {}
        self.health_status = {}

    def register_manager(
        self, manager_name: str, manager_config: Dict[str, Any]
    ) -> bool:
        """Register a management component.

        Args:
            manager_name: Name of the manager
            manager_config: Manager configuration

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.managers[manager_name] = {
                "config": manager_config,
                "registered_at": datetime.now().isoformat(),
                "status": "active",
            }
            logger.info(f"Registered manager: {manager_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register manager {manager_name}: {e}")
            return False

    def get_manager(self, manager_name: str) -> Optional[Dict[str, Any]]:
        """Get a manager configuration.

        Args:
            manager_name: Name of the manager

        Returns:
            Dict: Manager configuration or None if not found
        """
        return self.managers.get(manager_name)

    def list_managers(self) -> List[str]:
        """List all registered managers.

        Returns:
            List: Names of registered managers
        """
        return list(self.managers.keys())

    def update_health_status(
        self, manager_name: str, status: str, details: Dict[str, Any] = None
    ) -> bool:
        """Update health status for a manager.

        Args:
            manager_name: Name of the manager
            status: Health status ('healthy', 'warning', 'error')
            details: Additional status details

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.health_status[manager_name] = {
                "status": status,
                "details": details or {},
                "timestamp": datetime.now().isoformat(),
            }
            logger.info(f"Updated health status for manager {manager_name}: {status}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to update health status for manager {manager_name}: {e}"
            )
            return False

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status.

        Returns:
            Dict: System health information
        """
        healthy_count = 0
        warning_count = 0
        error_count = 0

        for status_info in self.health_status.values():
            status = status_info["status"]
            if status == "healthy":
                healthy_count += 1
            elif status == "warning":
                warning_count += 1
            elif status == "error":
                error_count += 1

        total_managers = len(self.managers)

        return {
            "total_managers": total_managers,
            "healthy": healthy_count,
            "warning": warning_count,
            "error": error_count,
            "health_percentage": (
                (healthy_count / total_managers * 100) if total_managers > 0 else 0
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def get_manager_status(self, manager_name: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific manager.

        Args:
            manager_name: Name of the manager

        Returns:
            Dict: Manager status or None
        """
        return self.health_status.get(manager_name)


# Global instance
management_registry = ManagementRegistry()
