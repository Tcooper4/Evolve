import pytest
from unittest.mock import Mock, patch
import json
import jwt
from datetime import datetime, timedelta

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def mock_user_manager():
    with patch('automation.web.app.user_manager') as mock:
        yield mock

def test_login_page(client):
    """Test login page rendering."""
    response = client.get('/login')
    assert response.status_code == 200
    assert b'Automation System' in response.data
    assert b'Please sign in to continue' in response.data

def test_login_success(client, mock_user_manager):
    """Test successful login."""
    # Mock successful authentication
    mock_user_manager.authenticate.return_value = "valid_token"
    
    response = client.post('/api/auth/login', json={
        'username': 'testuser',
        'password': 'password123'
    })
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'token' in data
    assert data['token'] == "valid_token"

def test_login_missing_credentials(client):
    """Test login with missing credentials."""
    response = client.post('/api/auth/login', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_login_invalid_credentials(client, mock_user_manager):
    """Test login with invalid credentials."""
    # Mock failed authentication
    mock_user_manager.authenticate.return_value = None
    
    response = client.post('/api/auth/login', json={
        'username': 'testuser',
        'password': 'wrongpassword'
    })
    
    assert response.status_code == 401
    data = json.loads(response.data)
    assert 'error' in data

def test_register_success(client, mock_user_manager):
    """Test successful user registration."""
    # Mock user creation
    mock_user_manager.create_user.return_value = {
        'username': 'newuser',
        'email': 'new@example.com',
        'role': 'user'
    }
    
    response = client.post('/api/auth/register', json={
        'username': 'newuser',
        'password': 'password123',
        'email': 'new@example.com'
    }, headers={'Authorization': 'Bearer admin_token'})
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['username'] == 'newuser'
    assert data['email'] == 'new@example.com'

def test_register_unauthorized(client):
    """Test registration without admin access."""
    response = client.post('/api/auth/register', json={
        'username': 'newuser',
        'password': 'password123',
        'email': 'new@example.com'
    })
    
    assert response.status_code == 401

def test_list_users_success(client, mock_user_manager):
    """Test successful user listing."""
    # Mock user list
    mock_user_manager.list_users.return_value = [
        {'username': 'user1', 'email': 'user1@example.com', 'role': 'user'},
        {'username': 'user2', 'email': 'user2@example.com', 'role': 'admin'}
    ]
    
    response = client.get('/api/auth/users', headers={'Authorization': 'Bearer admin_token'})
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 2
    assert data[0]['username'] == 'user1'
    assert data[1]['username'] == 'user2'

def test_get_user_success(client, mock_user_manager):
    """Test successful user retrieval."""
    # Mock user data
    mock_user_manager.get_user.return_value = {
        'username': 'testuser',
        'email': 'test@example.com',
        'role': 'user'
    }
    
    response = client.get('/api/auth/users/testuser', headers={'Authorization': 'Bearer valid_token'})
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['username'] == 'testuser'
    assert data['email'] == 'test@example.com'

def test_get_user_not_found(client, mock_user_manager):
    """Test getting non-existent user."""
    # Mock user not found
    mock_user_manager.get_user.return_value = None
    
    response = client.get('/api/auth/users/nonexistent', headers={'Authorization': 'Bearer valid_token'})
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data

def test_update_user_success(client, mock_user_manager):
    """Test successful user update."""
    # Mock updated user data
    mock_user_manager.update_user.return_value = {
        'username': 'testuser',
        'email': 'updated@example.com',
        'role': 'user'
    }
    
    response = client.put('/api/auth/users/testuser', json={
        'email': 'updated@example.com'
    }, headers={'Authorization': 'Bearer valid_token'})
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['email'] == 'updated@example.com'

def test_delete_user_success(client, mock_user_manager):
    """Test successful user deletion."""
    # Mock successful deletion
    mock_user_manager.delete_user.return_value = True
    
    response = client.delete('/api/auth/users/testuser', headers={'Authorization': 'Bearer admin_token'})
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data

def test_delete_user_not_found(client, mock_user_manager):
    """Test deleting non-existent user."""
    # Mock user not found
    mock_user_manager.delete_user.return_value = False
    
    response = client.delete('/api/auth/users/nonexistent', headers={'Authorization': 'Bearer admin_token'})
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data 