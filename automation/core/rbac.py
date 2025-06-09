"""
Role-Based Access Control (RBAC)

This module implements a comprehensive RBAC system that handles:
- Role management
- Permission management
- Access control
- Policy enforcement
"""

import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Permission(Enum):
    """Permission enumeration."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

class Resource(Enum):
    """Resource enumeration."""
    WORKFLOW = "workflow"
    SERVICE = "service"
    TASK = "task"
    USER = "user"
    ROLE = "role"
    SYSTEM = "system"

@dataclass
class Policy:
    """Access control policy."""
    resource: Resource
    permissions: Set[Permission]
    conditions: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None

@dataclass
class Role:
    """Role definition."""
    name: str
    description: str
    policies: List[Policy] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class User:
    """User definition."""
    username: str
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class RBAC:
    """Role-Based Access Control system."""
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RBAC system."""
        self.config = config
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self._load_roles()
        self._load_users()

    def _load_roles(self):
        """Load roles from configuration."""
        try:
            roles_config = self.config.get("roles", {})
            for role_name, role_data in roles_config.items():
                policies = []
                for policy_data in role_data.get("policies", []):
                    policy = Policy(
                        resource=Resource(policy_data["resource"]),
                        permissions={Permission(p) for p in policy_data["permissions"]},
                        conditions=policy_data.get("conditions", {}),
                        expires_at=datetime.fromisoformat(policy_data["expires_at"]) if "expires_at" in policy_data else None
                    )
                    policies.append(policy)
                
                role = Role(
                    name=role_name,
                    description=role_data.get("description", ""),
                    policies=policies,
                    metadata=role_data.get("metadata", {})
                )
                self.roles[role_name] = role
                
        except Exception as e:
            logger.error(f"Failed to load roles: {str(e)}")
            raise

    def _load_users(self):
        """Load users from configuration."""
        try:
            users_config = self.config.get("users", {})
            for username, user_data in users_config.items():
                user = User(
                    username=username,
                    roles=user_data.get("roles", []),
                    metadata=user_data.get("metadata", {})
                )
                self.users[username] = user
                
        except Exception as e:
            logger.error(f"Failed to load users: {str(e)}")
            raise

    def create_role(self, role: Role) -> bool:
        """Create a new role."""
        try:
            if role.name in self.roles:
                logger.warning(f"Role {role.name} already exists")
                return False
            
            self.roles[role.name] = role
            logger.info(f"Role created: {role.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create role {role.name}: {str(e)}")
            return False

    def delete_role(self, role_name: str) -> bool:
        """Delete a role."""
        try:
            if role_name not in self.roles:
                logger.warning(f"Role {role_name} does not exist")
                return False
            
            # Check if role is assigned to any users
            for user in self.users.values():
                if role_name in user.roles:
                    logger.warning(f"Cannot delete role {role_name}: assigned to user {user.username}")
                    return False
            
            del self.roles[role_name]
            logger.info(f"Role deleted: {role_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete role {role_name}: {str(e)}")
            return False

    def get_role(self, role_name: str) -> Optional[Role]:
        """Get role information."""
        return self.roles.get(role_name)

    def create_user(self, user: User) -> bool:
        """Create a new user."""
        try:
            if user.username in self.users:
                logger.warning(f"User {user.username} already exists")
                return False
            
            # Validate roles
            for role_name in user.roles:
                if role_name not in self.roles:
                    logger.warning(f"Invalid role {role_name} for user {user.username}")
                    return False
            
            self.users[user.username] = user
            logger.info(f"User created: {user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create user {user.username}: {str(e)}")
            return False

    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        try:
            if username not in self.users:
                logger.warning(f"User {username} does not exist")
                return False
            
            del self.users[username]
            logger.info(f"User deleted: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user {username}: {str(e)}")
            return False

    def get_user(self, username: str) -> Optional[User]:
        """Get user information."""
        return self.users.get(username)

    def assign_role(self, username: str, role_name: str) -> bool:
        """Assign a role to a user."""
        try:
            user = self.get_user(username)
            if not user:
                logger.warning(f"User {username} does not exist")
                return False
            
            if role_name not in self.roles:
                logger.warning(f"Role {role_name} does not exist")
                return False
            
            if role_name in user.roles:
                logger.warning(f"User {username} already has role {role_name}")
                return False
            
            user.roles.append(role_name)
            logger.info(f"Role {role_name} assigned to user {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role {role_name} to user {username}: {str(e)}")
            return False

    def revoke_role(self, username: str, role_name: str) -> bool:
        """Revoke a role from a user."""
        try:
            user = self.get_user(username)
            if not user:
                logger.warning(f"User {username} does not exist")
                return False
            
            if role_name not in user.roles:
                logger.warning(f"User {username} does not have role {role_name}")
                return False
            
            user.roles.remove(role_name)
            logger.info(f"Role {role_name} revoked from user {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke role {role_name} from user {username}: {str(e)}")
            return False

    def check_permission(self, username: str, resource: Resource, permission: Permission) -> bool:
        """Check if a user has permission for a resource."""
        try:
            user = self.get_user(username)
            if not user:
                logger.warning(f"User {username} does not exist")
                return False
            
            # Check each role's policies
            for role_name in user.roles:
                role = self.get_role(role_name)
                if not role:
                    continue
                
                for policy in role.policies:
                    if policy.resource != resource:
                        continue
                    
                    # Check if policy has expired
                    if policy.expires_at and datetime.now() > policy.expires_at:
                        continue
                    
                    # Check conditions
                    if not self._check_conditions(policy.conditions, user):
                        continue
                    
                    # Check permission
                    if permission in policy.permissions or Permission.ADMIN in policy.permissions:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check permission for user {username}: {str(e)}")
            return False

    def _check_conditions(self, conditions: Dict[str, Any], user: User) -> bool:
        """Check if conditions are met."""
        try:
            for key, value in conditions.items():
                if key not in user.metadata:
                    return False
                if user.metadata[key] != value:
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to check conditions: {str(e)}")
            return False

    def get_user_permissions(self, username: str) -> Dict[Resource, Set[Permission]]:
        """Get all permissions for a user."""
        try:
            user = self.get_user(username)
            if not user:
                return {}
            
            permissions: Dict[Resource, Set[Permission]] = {}
            
            # Check each role's policies
            for role_name in user.roles:
                role = self.get_role(role_name)
                if not role:
                    continue
                
                for policy in role.policies:
                    # Skip expired policies
                    if policy.expires_at and datetime.now() > policy.expires_at:
                        continue
                    
                    # Skip if conditions not met
                    if not self._check_conditions(policy.conditions, user):
                        continue
                    
                    # Add permissions
                    if policy.resource not in permissions:
                        permissions[policy.resource] = set()
                    permissions[policy.resource].update(policy.permissions)
            
            return permissions
            
        except Exception as e:
            logger.error(f"Failed to get permissions for user {username}: {str(e)}")
            return {}

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration."""
        try:
            config = {
                "roles": {},
                "users": {}
            }
            
            # Export roles
            for role_name, role in self.roles.items():
                config["roles"][role_name] = {
                    "description": role.description,
                    "policies": [
                        {
                            "resource": policy.resource.value,
                            "permissions": [p.value for p in policy.permissions],
                            "conditions": policy.conditions,
                            "expires_at": policy.expires_at.isoformat() if policy.expires_at else None
                        }
                        for policy in role.policies
                    ],
                    "metadata": role.metadata
                }
            
            # Export users
            for username, user in self.users.items():
                config["users"][username] = {
                    "roles": user.roles,
                    "metadata": user.metadata
                }
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {str(e)}")
            return {}

    def save_config(self, file_path: str):
        """Save configuration to file."""
        try:
            config = self.export_config()
            with open(file_path, "w") as f:
                if file_path.endswith(".json"):
                    json.dump(config, f, indent=2)
                elif file_path.endswith((".yml", ".yaml")):
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    raise ValueError("Unsupported file format")
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise 