"""
Tests for the RBAC system.
"""

import pytest
from datetime import datetime, timedelta
from automation.core.rbac import (
    RBAC,
    Role,
    User,
    Policy,
    Permission,
    Resource
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        "roles": {
            "admin": {
                "description": "Administrator role",
                "policies": [
                    {
                        "resource": "system",
                        "permissions": ["admin"],
                        "conditions": {},
                        "expires_at": None
                    }
                ],
                "metadata": {}
            },
            "workflow_manager": {
                "description": "Workflow management role",
                "policies": [
                    {
                        "resource": "workflow",
                        "permissions": ["read", "write", "execute"],
                        "conditions": {},
                        "expires_at": None
                    }
                ],
                "metadata": {}
            },
            "service_manager": {
                "description": "Service management role",
                "policies": [
                    {
                        "resource": "service",
                        "permissions": ["read", "write"],
                        "conditions": {},
                        "expires_at": None
                    }
                ],
                "metadata": {}
            }
        },
        "users": {
            "admin_user": {
                "roles": ["admin"],
                "metadata": {}
            },
            "workflow_user": {
                "roles": ["workflow_manager"],
                "metadata": {}
            },
            "service_user": {
                "roles": ["service_manager"],
                "metadata": {}
            }
        }
    }

@pytest.fixture
def rbac(config):
    """Test RBAC instance."""
    return RBAC(config)

def test_rbac_initialization(config):
    """Test RBAC initialization."""
    rbac = RBAC(config)
    assert len(rbac.roles) == 3
    assert len(rbac.users) == 3

def test_create_role(rbac):
    """Test role creation."""
    role = Role(
        name="test_role",
        description="Test role",
        policies=[
            Policy(
                resource=Resource.WORKFLOW,
                permissions={Permission.READ, Permission.WRITE}
            )
        ]
    )
    success = rbac.create_role(role)
    assert success
    assert "test_role" in rbac.roles
    assert rbac.roles["test_role"] == role

def test_delete_role(rbac):
    """Test role deletion."""
    # Try to delete non-existent role
    success = rbac.delete_role("non_existent")
    assert not success
    
    # Try to delete role assigned to user
    success = rbac.delete_role("admin")
    assert not success
    
    # Create and delete unassigned role
    role = Role(
        name="test_role",
        description="Test role",
        policies=[]
    )
    rbac.create_role(role)
    success = rbac.delete_role("test_role")
    assert success
    assert "test_role" not in rbac.roles

def test_create_user(rbac):
    """Test user creation."""
    user = User(
        username="test_user",
        roles=["workflow_manager"]
    )
    success = rbac.create_user(user)
    assert success
    assert "test_user" in rbac.users
    assert rbac.users["test_user"] == user

def test_delete_user(rbac):
    """Test user deletion."""
    # Try to delete non-existent user
    success = rbac.delete_user("non_existent")
    assert not success
    
    # Create and delete user
    user = User(
        username="test_user",
        roles=["workflow_manager"]
    )
    rbac.create_user(user)
    success = rbac.delete_user("test_user")
    assert success
    assert "test_user" not in rbac.users

def test_assign_role(rbac):
    """Test role assignment."""
    # Try to assign non-existent role
    success = rbac.assign_role("workflow_user", "non_existent")
    assert not success
    
    # Try to assign role to non-existent user
    success = rbac.assign_role("non_existent", "workflow_manager")
    assert not success
    
    # Assign role
    success = rbac.assign_role("workflow_user", "service_manager")
    assert success
    assert "service_manager" in rbac.users["workflow_user"].roles

def test_revoke_role(rbac):
    """Test role revocation."""
    # Try to revoke non-existent role
    success = rbac.revoke_role("workflow_user", "non_existent")
    assert not success
    
    # Try to revoke role from non-existent user
    success = rbac.revoke_role("non_existent", "workflow_manager")
    assert not success
    
    # Revoke role
    success = rbac.revoke_role("workflow_user", "workflow_manager")
    assert success
    assert "workflow_manager" not in rbac.users["workflow_user"].roles

def test_check_permission(rbac):
    """Test permission checking."""
    # Admin permissions
    assert rbac.check_permission("admin_user", Resource.SYSTEM, Permission.ADMIN)
    assert rbac.check_permission("admin_user", Resource.WORKFLOW, Permission.ADMIN)
    
    # Workflow manager permissions
    assert rbac.check_permission("workflow_user", Resource.WORKFLOW, Permission.READ)
    assert rbac.check_permission("workflow_user", Resource.WORKFLOW, Permission.WRITE)
    assert rbac.check_permission("workflow_user", Resource.WORKFLOW, Permission.EXECUTE)
    assert not rbac.check_permission("workflow_user", Resource.SERVICE, Permission.WRITE)
    
    # Service manager permissions
    assert rbac.check_permission("service_user", Resource.SERVICE, Permission.READ)
    assert rbac.check_permission("service_user", Resource.SERVICE, Permission.WRITE)
    assert not rbac.check_permission("service_user", Resource.SERVICE, Permission.EXECUTE)

def test_policy_conditions(rbac):
    """Test policy conditions."""
    # Create role with conditions
    role = Role(
        name="conditional_role",
        description="Role with conditions",
        policies=[
            Policy(
                resource=Resource.WORKFLOW,
                permissions={Permission.READ},
                conditions={"department": "engineering"}
            )
        ]
    )
    rbac.create_role(role)
    
    # Create user with matching condition
    user = User(
        username="conditional_user",
        roles=["conditional_role"],
        metadata={"department": "engineering"}
    )
    rbac.create_user(user)
    
    # Create user with non-matching condition
    user = User(
        username="non_matching_user",
        roles=["conditional_role"],
        metadata={"department": "marketing"}
    )
    rbac.create_user(user)
    
    # Check permissions
    assert rbac.check_permission("conditional_user", Resource.WORKFLOW, Permission.READ)
    assert not rbac.check_permission("non_matching_user", Resource.WORKFLOW, Permission.READ)

def test_policy_expiration(rbac):
    """Test policy expiration."""
    # Create role with expiring policy
    role = Role(
        name="expiring_role",
        description="Role with expiring policy",
        policies=[
            Policy(
                resource=Resource.WORKFLOW,
                permissions={Permission.READ},
                expires_at=datetime.now() - timedelta(days=1)  # Expired
            )
        ]
    )
    rbac.create_role(role)
    
    # Create user with expiring role
    user = User(
        username="expiring_user",
        roles=["expiring_role"]
    )
    rbac.create_user(user)
    
    # Check permissions
    assert not rbac.check_permission("expiring_user", Resource.WORKFLOW, Permission.READ)

def test_get_user_permissions(rbac):
    """Test getting user permissions."""
    # Get admin permissions
    admin_permissions = rbac.get_user_permissions("admin_user")
    assert Resource.SYSTEM in admin_permissions
    assert Permission.ADMIN in admin_permissions[Resource.SYSTEM]
    
    # Get workflow manager permissions
    workflow_permissions = rbac.get_user_permissions("workflow_user")
    assert Resource.WORKFLOW in workflow_permissions
    assert Permission.READ in workflow_permissions[Resource.WORKFLOW]
    assert Permission.WRITE in workflow_permissions[Resource.WORKFLOW]
    assert Permission.EXECUTE in workflow_permissions[Resource.WORKFLOW]
    
    # Get service manager permissions
    service_permissions = rbac.get_user_permissions("service_user")
    assert Resource.SERVICE in service_permissions
    assert Permission.READ in service_permissions[Resource.SERVICE]
    assert Permission.WRITE in service_permissions[Resource.SERVICE]

def test_export_config(rbac):
    """Test configuration export."""
    config = rbac.export_config()
    assert "roles" in config
    assert "users" in config
    assert "admin" in config["roles"]
    assert "admin_user" in config["users"]

def test_save_config(rbac, tmp_path):
    """Test configuration saving."""
    # Save as JSON
    json_path = tmp_path / "config.json"
    rbac.save_config(str(json_path))
    assert json_path.exists()
    
    # Save as YAML
    yaml_path = tmp_path / "config.yaml"
    rbac.save_config(str(yaml_path))
    assert yaml_path.exists()
    
    # Try invalid format
    with pytest.raises(ValueError):
        rbac.save_config(str(tmp_path / "config.txt")) 