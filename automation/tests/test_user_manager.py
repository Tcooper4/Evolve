import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime
import jwt
import bcrypt

from ..auth.user_manager import UserManager

@pytest.fixture
def mock_redis():
    redis_mock = Mock()
    redis_mock.hget.return_value = None
    redis_mock.hset.return_value = True
    redis_mock.hdel.return_value = True
    redis_mock.hexists.return_value = False
    redis_mock.hgetall.return_value = {}
    return redis_mock

@pytest.fixture
def user_manager(mock_redis):
    return UserManager(mock_redis, "test_secret_key")

def test_create_user(user_manager, mock_redis):
    # Test successful user creation
    user = user_manager.create_user("testuser", "password123", "test@example.com")
    assert user["username"] == "testuser"
    assert user["email"] == "test@example.com"
    assert "password" not in user
    
    # Test duplicate user
    mock_redis.hexists.return_value = True
    with pytest.raises(ValueError):
        user_manager.create_user("testuser", "password123", "test@example.com")

def test_authenticate(user_manager, mock_redis):
    # Test successful authentication
    hashed_password = bcrypt.hashpw("password123".encode(), bcrypt.gensalt())
    user_data = {
        "username": "testuser",
        "password": hashed_password.decode(),
        "email": "test@example.com",
        "role": "user",
        "is_active": True
    }
    mock_redis.hget.return_value = json.dumps(user_data)
    
    token = user_manager.authenticate("testuser", "password123")
    assert token is not None
    
    # Test invalid password
    assert user_manager.authenticate("testuser", "wrongpassword") is None
    
    # Test inactive user
    user_data["is_active"] = False
    mock_redis.hget.return_value = json.dumps(user_data)
    assert user_manager.authenticate("testuser", "password123") is None

def test_get_user(user_manager, mock_redis):
    # Test getting existing user
    user_data = {
        "username": "testuser",
        "password": "hashed_password",
        "email": "test@example.com",
        "role": "user"
    }
    mock_redis.hget.return_value = json.dumps(user_data)
    
    user = user_manager.get_user("testuser")
    assert user["username"] == "testuser"
    assert "password" not in user
    
    # Test getting non-existent user
    mock_redis.hget.return_value = None
    assert user_manager.get_user("nonexistent") is None

def test_update_user(user_manager, mock_redis):
    # Test updating user
    user_data = {
        "username": "testuser",
        "password": "old_password",
        "email": "old@example.com",
        "role": "user"
    }
    mock_redis.hget.return_value = json.dumps(user_data)
    
    updates = {
        "email": "new@example.com",
        "password": "new_password"
    }
    
    updated_user = user_manager.update_user("testuser", updates)
    assert updated_user["email"] == "new@example.com"
    assert "password" not in updated_user
    
    # Test updating non-existent user
    mock_redis.hget.return_value = None
    assert user_manager.update_user("nonexistent", updates) is None

def test_delete_user(user_manager, mock_redis):
    # Test deleting existing user
    mock_redis.hexists.return_value = True
    assert user_manager.delete_user("testuser") is True
    
    # Test deleting non-existent user
    mock_redis.hexists.return_value = False
    assert user_manager.delete_user("nonexistent") is False

def test_list_users(user_manager, mock_redis):
    # Test listing users
    users_data = {
        "user1": json.dumps({
            "username": "user1",
            "password": "hashed_password1",
            "email": "user1@example.com",
            "role": "user"
        }),
        "user2": json.dumps({
            "username": "user2",
            "password": "hashed_password2",
            "email": "user2@example.com",
            "role": "admin"
        })
    }
    mock_redis.hgetall.return_value = users_data
    
    users = user_manager.list_users()
    assert len(users) == 2
    assert all("password" not in user for user in users)

def test_verify_token(user_manager):
    # Test valid token
    payload = {
        "username": "testuser",
        "role": "user",
        "exp": datetime.utcnow().timestamp() + 3600
    }
    token = jwt.encode(payload, "test_secret_key", algorithm="HS256")
    
    verified = user_manager.verify_token(token)
    assert verified["username"] == "testuser"
    assert verified["role"] == "user"
    
    # Test invalid token
    assert user_manager.verify_token("invalid_token") is None
    
    # Test expired token
    payload["exp"] = datetime.utcnow().timestamp() - 3600
    expired_token = jwt.encode(payload, "test_secret_key", algorithm="HS256")
    assert user_manager.verify_token(expired_token) is None 