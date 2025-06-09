# API Documentation

## Overview

The Automation System API provides a RESTful interface for managing all aspects of the system. This documentation covers the available endpoints, request/response formats, authentication, and usage examples.

## Base URL

```
https://api.example.com/v1
```

## Authentication

### API Key Authentication

All API requests require an API key to be included in the request header:

```
Authorization: Bearer your-api-key
```

### OAuth 2.0 Authentication

For OAuth 2.0 authentication, use the following endpoints:

#### Get Access Token
```http
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&client_id=your-client-id&client_secret=your-client-secret
```

Response:
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

## Rate Limiting

- Rate limit: 1000 requests per hour
- Rate limit header: `X-RateLimit-Limit`
- Remaining requests: `X-RateLimit-Remaining`
- Reset time: `X-RateLimit-Reset`

## Endpoints

### Service Management

#### List Services
```http
GET /services
```

Response:
```json
{
    "services": [
        {
            "id": "service-1",
            "name": "API Gateway",
            "status": "running",
            "health": "healthy",
            "last_updated": "2024-03-20T10:00:00Z"
        }
    ],
    "total": 1,
    "page": 1,
    "per_page": 10
}
```

#### Get Service
```http
GET /services/{service_id}
```

Response:
```json
{
    "id": "service-1",
    "name": "API Gateway",
    "status": "running",
    "health": "healthy",
    "configuration": {
        "port": 8000,
        "host": "0.0.0.0"
    },
    "metrics": {
        "cpu_usage": 45.2,
        "memory_usage": 1024,
        "request_count": 1000
    },
    "last_updated": "2024-03-20T10:00:00Z"
}
```

#### Create Service
```http
POST /services
Content-Type: application/json

{
    "name": "New Service",
    "type": "api",
    "configuration": {
        "port": 8000,
        "host": "0.0.0.0"
    }
}
```

Response:
```json
{
    "id": "service-2",
    "name": "New Service",
    "status": "created",
    "health": "unknown",
    "last_updated": "2024-03-20T10:00:00Z"
}
```

#### Update Service
```http
PUT /services/{service_id}
Content-Type: application/json

{
    "name": "Updated Service",
    "configuration": {
        "port": 8001
    }
}
```

Response:
```json
{
    "id": "service-2",
    "name": "Updated Service",
    "status": "running",
    "health": "healthy",
    "last_updated": "2024-03-20T10:00:00Z"
}
```

#### Delete Service
```http
DELETE /services/{service_id}
```

Response:
```json
{
    "message": "Service deleted successfully"
}
```

### Monitoring

#### Get System Metrics
```http
GET /monitoring/metrics
```

Response:
```json
{
    "cpu_usage": 45.2,
    "memory_usage": 8192,
    "disk_usage": 51200,
    "network_traffic": {
        "in": 1024,
        "out": 2048
    },
    "timestamp": "2024-03-20T10:00:00Z"
}
```

#### Get Service Metrics
```http
GET /monitoring/services/{service_id}/metrics
```

Response:
```json
{
    "service_id": "service-1",
    "metrics": {
        "cpu_usage": 25.5,
        "memory_usage": 1024,
        "request_count": 1000,
        "error_count": 5,
        "response_time": 150
    },
    "timestamp": "2024-03-20T10:00:00Z"
}
```

### Logging

#### Get Logs
```http
GET /logs
Query Parameters:
- service_id (optional)
- level (optional)
- start_time (optional)
- end_time (optional)
- limit (optional, default: 100)
```

Response:
```json
{
    "logs": [
        {
            "id": "log-1",
            "service_id": "service-1",
            "level": "INFO",
            "message": "Service started successfully",
            "timestamp": "2024-03-20T10:00:00Z",
            "metadata": {
                "request_id": "req-1",
                "user_id": "user-1"
            }
        }
    ],
    "total": 1,
    "page": 1,
    "per_page": 100
}
```

#### Create Log
```http
POST /logs
Content-Type: application/json

{
    "service_id": "service-1",
    "level": "INFO",
    "message": "Custom log message",
    "metadata": {
        "request_id": "req-1",
        "user_id": "user-1"
    }
}
```

Response:
```json
{
    "id": "log-2",
    "service_id": "service-1",
    "level": "INFO",
    "message": "Custom log message",
    "timestamp": "2024-03-20T10:00:00Z",
    "metadata": {
        "request_id": "req-1",
        "user_id": "user-1"
    }
}
```

### Data Processing

#### Process Data
```http
POST /data/process
Content-Type: application/json

{
    "data_type": "logs",
    "content": {
        "logs": [
            {
                "level": "ERROR",
                "message": "Service failed to start"
            }
        ]
    },
    "options": {
        "aggregate": true,
        "format": "json"
    }
}
```

Response:
```json
{
    "job_id": "job-1",
    "status": "processing",
    "created_at": "2024-03-20T10:00:00Z"
}
```

#### Get Processing Status
```http
GET /data/process/{job_id}
```

Response:
```json
{
    "job_id": "job-1",
    "status": "completed",
    "result": {
        "error_count": 1,
        "processed_logs": 1,
        "aggregated_data": {
            "errors": [
                {
                    "level": "ERROR",
                    "message": "Service failed to start",
                    "count": 1
                }
            ]
        }
    },
    "created_at": "2024-03-20T10:00:00Z",
    "completed_at": "2024-03-20T10:00:01Z"
}
```

### Notifications

#### Send Notification
```http
POST /notifications
Content-Type: application/json

{
    "type": "alert",
    "level": "critical",
    "message": "Service is down",
    "recipients": ["admin@example.com"],
    "metadata": {
        "service_id": "service-1",
        "error": "Connection timeout"
    }
}
```

Response:
```json
{
    "id": "notification-1",
    "status": "sent",
    "created_at": "2024-03-20T10:00:00Z"
}
```

#### Get Notification Status
```http
GET /notifications/{notification_id}
```

Response:
```json
{
    "id": "notification-1",
    "type": "alert",
    "level": "critical",
    "message": "Service is down",
    "status": "delivered",
    "recipients": ["admin@example.com"],
    "created_at": "2024-03-20T10:00:00Z",
    "delivered_at": "2024-03-20T10:00:01Z"
}
```

### RBAC (Role-Based Access Control)

#### List Roles
```http
GET /roles
```

Response:
```json
{
    "roles": [
        {
            "id": "role-1",
            "name": "admin",
            "permissions": ["read", "write", "delete"],
            "created_at": "2024-03-20T10:00:00Z"
        }
    ],
    "total": 1,
    "page": 1,
    "per_page": 10
}
```

#### Create Role
```http
POST /roles
Content-Type: application/json

{
    "name": "operator",
    "permissions": ["read", "write"]
}
```

Response:
```json
{
    "id": "role-2",
    "name": "operator",
    "permissions": ["read", "write"],
    "created_at": "2024-03-20T10:00:00Z"
}
```

#### Update Role
```http
PUT /roles/{role_id}
Content-Type: application/json

{
    "permissions": ["read", "write", "monitor"]
}
```

Response:
```json
{
    "id": "role-2",
    "name": "operator",
    "permissions": ["read", "write", "monitor"],
    "updated_at": "2024-03-20T10:00:00Z"
}
```

#### Delete Role
```http
DELETE /roles/{role_id}
```

Response:
```json
{
    "message": "Role deleted successfully"
}
```

## Error Handling

### Error Response Format
```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Error description",
        "details": {
            "field": "Additional error details"
        }
    }
}
```

### Common Error Codes
- `INVALID_REQUEST`: Invalid request parameters
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `CONFLICT`: Resource conflict
- `INTERNAL_ERROR`: Server error

## WebSocket API

### Connection
```
wss://api.example.com/v1/ws
```

### Authentication
Send authentication message after connection:
```json
{
    "type": "auth",
    "token": "your-api-key"
}
```

### Event Types

#### Service Status Updates
```json
{
    "type": "service_status",
    "data": {
        "service_id": "service-1",
        "status": "running",
        "health": "healthy",
        "timestamp": "2024-03-20T10:00:00Z"
    }
}
```

#### Monitoring Alerts
```json
{
    "type": "alert",
    "data": {
        "level": "critical",
        "message": "High CPU usage",
        "service_id": "service-1",
        "timestamp": "2024-03-20T10:00:00Z"
    }
}
```

#### Log Updates
```json
{
    "type": "log",
    "data": {
        "service_id": "service-1",
        "level": "ERROR",
        "message": "Service failed to start",
        "timestamp": "2024-03-20T10:00:00Z"
    }
}
```

## Versioning

The API version is included in the URL path. The current version is v1.

## Changelog

### v1.0.0 (2024-03-20)
- Initial release
- Basic CRUD operations for all resources
- Authentication and authorization
- Monitoring and logging
- Data processing
- Notifications
- RBAC system

## Support

For API support, contact:
- Email: api-support@example.com
- Documentation: https://docs.example.com/api
- Status Page: https://status.example.com 