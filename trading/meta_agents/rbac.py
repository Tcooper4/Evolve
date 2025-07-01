"""
Role-Based Access Control (RBAC)

This module implements role-based access control functionality.

Note: This module was adapted from the legacy automation/core/rbac.py file.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json

class Role:
    """Represents a role in the system."""
    
    def __init__(
        self,
        name: str,
        description: str,
        permissions: List[str]):
        """Initialize role."""
        self.name = name
        self.description = description
        self.permissions = permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert role to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'permissions': self.permissions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """Create role from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            permissions=data['permissions']
        )

class User:
    """Represents a user in the system."""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        roles: List[str]):
        """Initialize user."""
        self.user_id = user_id
        self.username = username
        self.roles = roles
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'roles': self.roles
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary."""
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            roles=data['roles']
        )

class RBACManager:
    """Manages role-based access control."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RBAC manager."""
        self.config = config
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.setup_logging()
        self.load_default_roles()
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logging(self):
        """Configure logging for RBAC manager."""
        log_path = Path("logs/rbac")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "rbac.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def load_default_roles(self):
        """Load default roles."""
        try:
            default_roles = {
                'admin': Role(
                    name='admin',
                    description='Administrator with full access',
                    permissions=['*']
                ),
                'user': Role(
                    name='user',
                    description='Regular user with basic access',
                    permissions=[
                        'read:own',
                        'write:own',
                        'delete:own'
                    ]
                ),
                'viewer': Role(
                    name='viewer',
                    description='View-only access',
                    permissions=['read:own']
                )
            }
            
            self.roles.update(default_roles)
            self.logger.info("Loaded default roles")
        except Exception as e:
            self.logger.error(f"Error loading default roles: {str(e)}")
            raise

    def create_role(
        self,
        name: str,
        description: str,
        permissions: List[str]
    ) -> Role:
        """Create a new role."""
        try:
            if name in self.roles:
                raise ValueError(f"Role {name} already exists")
            
            role = Role(name, description, permissions)
            self.roles[name] = role
            
            self.logger.info(f"Created role: {name}")
            return role
        except Exception as e:
            self.logger.error(f"Error creating role: {str(e)}")
            raise
    
    def update_role(
        self,
        name: str,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None
    ) -> Role:
        """Update an existing role."""
        try:
            if name not in self.roles:
                raise ValueError(f"Role {name} does not exist")
            
            role = self.roles[name]
            
            if description is not None:
                role.description = description
            
            if permissions is not None:
                role.permissions = permissions
            
            self.logger.info(f"Updated role: {name}")
            return role
        except Exception as e:
            self.logger.error(f"Error updating role: {str(e)}")
            raise
    
    def delete_role(self, name: str) -> None:
        """Delete a role."""
        try:
            if name not in self.roles:
                raise ValueError(f"Role {name} does not exist")
            
            del self.roles[name]
            self.logger.info(f"Deleted role: {name}")
        except Exception as e:
            self.logger.error(f"Error deleting role: {str(e)}")
            raise

    def create_user(
        self,
        user_id: str,
        username: str,
        roles: List[str]
    ) -> User:
        """Create a new user."""
        try:
            if user_id in self.users:
                raise ValueError(f"User {user_id} already exists")
            
            # Validate roles
            for role in roles:
                if role not in self.roles:
                    raise ValueError(f"Role {role} does not exist")
            
            user = User(user_id, username, roles)
            self.users[user_id] = user
            
            self.logger.info(f"Created user: {user_id}")
            return user
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            raise
    
    def update_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        roles: Optional[List[str]] = None
    ) -> User:
        """Update an existing user."""
        try:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} does not exist")
            
            user = self.users[user_id]
            
            if username is not None:
                user.username = username
            
            if roles is not None:
                # Validate roles
                for role in roles:
                    if role not in self.roles:
                        raise ValueError(f"Role {role} does not exist")
                user.roles = roles
            
            self.logger.info(f"Updated user: {user_id}")
            return user
        except Exception as e:
            self.logger.error(f"Error updating user: {str(e)}")
            raise
    
    def delete_user(self, user_id: str) -> None:
        """Delete a user."""
        try:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} does not exist")
            
            del self.users[user_id]
            self.logger.info(f"Deleted user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error deleting user: {str(e)}")
            raise

    def check_permission(
        self,
        user_id: str,
        permission: str
    ) -> bool:
        """Check if a user has a specific permission."""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            
            # Check each role's permissions
            for role_name in user.roles:
                role = self.roles[role_name]
                if '*' in role.permissions or permission in role.permissions:
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error checking permission: {str(e)}")
            return False
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user."""
        try:
            if user_id not in self.users:
                return []
            
            user = self.users[user_id]
            permissions = set()
            
            # Collect permissions from all roles
            for role_name in user.roles:
                role = self.roles[role_name]
                permissions.update(role.permissions)
            
            return list(permissions)
        except Exception as e:
            self.logger.error(f"Error getting user permissions: {str(e)}")
            return []